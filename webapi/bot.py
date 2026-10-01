"""
webapi/bot.py — ครอบ chat_interaction() ของ chatbot_v5 เดิม โดยไม่แก้ gate / dialogue

สัญญาที่ใช้ (อ่านจาก run_testcases.py ซึ่งเทสต์ผ่าน 35/35 และ 33/33):

    chat_interaction(text, image, history, last_pids, session_id, last_vit, last_graph)
      -> (history, _state_out, refs, last_pids, session_id, last_vit, last_graph)

    history = [{"role": "user"|"assistant", "content": str | ...}, ...]  (Gradio messages)

Backend เก็บ 5 ค่าที่บอทต้องใช้ต่อเทิร์น (history, last_pids, session_id, last_vit, last_graph)
ไว้ฝั่งเซิร์ฟเวอร์ต่อ session — เบราว์เซอร์ถือแค่ session id
refs (แผง debug เดิม) ไม่ส่งไปหน้าเว็บ — เก็บแค่ Decision ไว้ใน log
"""
from __future__ import annotations

import importlib
import logging
import os
import re
import secrets
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

log = logging.getLogger("beltbot")


# ═══════════════════════════════════════════════════════════════
# session
# ═══════════════════════════════════════════════════════════════

@dataclass
class BotSession:
    id: str
    history: list = field(default_factory=list)
    last_pids: list = field(default_factory=list)
    bot_session_id: str = ""
    last_vit: Any = None
    last_graph: Any = None
    turns: int = 0
    touched: float = field(default_factory=time.time)
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)


class SessionStore:
    """LRU + TTL ในหน่วยความจำ · id สุ่ม 128 บิต เดาไม่ได้"""

    def __init__(self, ttl_s: int, max_sessions: int):
        self.ttl_s, self.max_sessions = ttl_s, max_sessions
        self._items: "OrderedDict[str, BotSession]" = OrderedDict()
        self._lock = threading.Lock()

    def get_or_create(self, sid: Optional[str]) -> tuple[BotSession, bool]:
        now = time.time()
        with self._lock:
            s = self._items.get(sid or "")
            if s is not None and now - s.touched <= self.ttl_s:
                s.touched = now
                self._items.move_to_end(s.id)
                return s, False
            if s is not None:
                del self._items[s.id]
            s = BotSession(id=secrets.token_urlsafe(16))
            self._items[s.id] = s
            while len(self._items) > self.max_sessions:
                self._items.popitem(last=False)
            return s, True

    def drop(self, sid: str) -> bool:
        with self._lock:
            return self._items.pop(sid, None) is not None

    def sweep(self) -> int:
        now = time.time()
        with self._lock:
            dead = [k for k, s in self._items.items() if now - s.touched > self.ttl_s]
            for k in dead:
                del self._items[k]
            return len(dead)

    def __len__(self) -> int:
        return len(self._items)


# ═══════════════════════════════════════════════════════════════
# แปลงข้อความของบอท → ข้อความสำหรับหน้าเว็บ
# ═══════════════════════════════════════════════════════════════

_IMG_SRC = re.compile(r"""<img[^>]*\bsrc\s*=\s*["']([^"']+)["']""", re.IGNORECASE)
_TAG = re.compile(r"<[^>]+>")
# รหัสภายในของ Movex เช่น movex_chain_LF820_K325 / movex_sprocket_57801 / Movex_Sprocket_880_z10_b25
_INTERNAL_ID = re.compile(r"\b[Mm]ovex_(?:chain|sprocket|Sprocket|Chain)_([A-Za-z0-9_]+)\b")


def scrub_internal_ids(text: str) -> str:
    """
    GRAPH-001: บอทหลุดรหัสภายในบางครั้ง — แก้ที่ชั้นแสดงผลเท่านั้น (ไม่แตะบอท)
    movex_chain_LF880_TAB_K325 → LF880 TAB K325 · movex_sprocket_57801 → 57801
    """
    return _INTERNAL_ID.sub(lambda m: m.group(1).replace("_", " "), text)


class MediaRegistry:
    """
    บอทส่งรูป drawing กลับเป็น <img src="path"> หรือ {"path": ...}
    เบราว์เซอร์ต้องไม่เห็น path จริง → ออก token แทน และเสิร์ฟได้เฉพาะไฟล์ใต้ MEDIA_ROOTS
    """

    def __init__(self, roots: list[str], base_dir: str):
        self.roots = [os.path.realpath(os.path.join(base_dir, r)) for r in roots]
        self.base_dir = base_dir
        self._tokens: dict[str, str] = {}
        self._by_path: dict[str, str] = {}
        self._lock = threading.Lock()

    def register(self, src: str) -> Optional[str]:
        if src.startswith(("http://", "https://", "data:image/")):
            return src                                   # ส่งต่อได้ตรง ๆ
        path = src[len("file="):] if src.startswith("file=") else src
        path = os.path.realpath(path if os.path.isabs(path) else os.path.join(self.base_dir, path))
        if not any(path == r or path.startswith(r + os.sep) for r in self.roots) or not os.path.isfile(path):
            log.warning("media ไม่อยู่ใน MEDIA_ROOTS หรือไม่มีไฟล์ — ไม่ส่งไปหน้าเว็บ: %s", src)
            return None
        with self._lock:
            tok = self._by_path.get(path)
            if tok is None:
                tok = secrets.token_urlsafe(12)
                self._tokens[tok], self._by_path[path] = path, tok
        return f"/media/{tok}"

    def resolve(self, token: str) -> Optional[str]:
        return self._tokens.get(token)


def to_web_messages(msgs: list, media: MediaRegistry, scrub: bool) -> list[dict]:
    out: list[dict] = []
    for m in msgs:
        if not isinstance(m, dict) or m.get("role") != "assistant":
            continue
        c = m.get("content")
        if isinstance(c, dict) and c.get("path"):
            url = media.register(str(c["path"]))
            if url:
                out.append({"type": "image", "url": url, "alt": c.get("alt_text") or ""})
            continue
        if isinstance(c, (list, tuple)) and c and isinstance(c[0], str) and not c[0].startswith("<"):
            url = media.register(c[0])                   # รูปแบบ tuple (path, alt) ของ Gradio รุ่นเก่า
            if url:
                out.append({"type": "image", "url": url, "alt": c[1] if len(c) > 1 else ""})
            continue
        if not isinstance(c, str):
            continue
        for src in _IMG_SRC.findall(c):
            url = media.register(src)
            if url:
                out.append({"type": "image", "url": url, "alt": ""})
        text = _TAG.sub("", c) if "<img" in c.lower() else c
        text = text.strip()
        if text:
            out.append({"type": "text", "text": scrub_internal_ids(text) if scrub else text})
    return out


_DECISION = re.compile(r"\*\*Decision:\*\*\s*(\w+)")


# ═══════════════════════════════════════════════════════════════
# adapter
# ═══════════════════════════════════════════════════════════════

class BotAdapter:
    def __init__(self, module_name: str, chat_fn: Optional[Callable] = None):
        self.module_name = module_name
        self._fn = chat_fn
        self._load_lock = threading.Lock()
        self.load_error: Optional[str] = None

    @property
    def ready(self) -> bool:
        return self._fn is not None

    def load(self) -> None:
        """import chatbot_v5 ครั้งเดียว (โหลดโมเดล ViT/SigLIP/Jina + Qdrant) — ช้า จึงทำตอนเริ่ม"""
        with self._load_lock:
            if self._fn is not None:
                return
            t0 = time.time()
            try:
                mod = importlib.import_module(self.module_name)
                self._fn = getattr(mod, "chat_interaction")
                log.info("โหลด %s เสร็จใน %.1f วินาที", self.module_name, time.time() - t0)
            except Exception as e:
                self.load_error = f"{type(e).__name__}: {e}"
                log.exception("โหลดบอทไม่สำเร็จ")
                raise

    def turn(self, s: BotSession, text: str, image) -> tuple[list, str]:
        """เรียกบอทหนึ่งเทิร์น คืน (ข้อความใหม่ของ assistant ในเทิร์นนี้, decision)"""
        if self._fn is None:
            self.load()
        before = len(s.history)
        result = self._fn(text, image, list(s.history), list(s.last_pids),
                          s.bot_session_id, s.last_vit, s.last_graph)
        history, _state_out, refs, last_pids, bot_sid, last_vit, last_graph = result
        history = list(history or [])
        new = history[before:] if len(history) >= before else history[-2:]
        s.history, s.last_pids, s.bot_session_id = history, list(last_pids or []), bot_sid or ""
        s.last_vit, s.last_graph = last_vit, last_graph
        s.turns += 1
        m = _DECISION.search(refs or "") if isinstance(refs, str) else None
        return new, (m.group(1) if m else "")
