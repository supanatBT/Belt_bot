"""
app.py — FastAPI สำหรับ Hugging Face Spaces (Docker, พอร์ต 7860)

ครอบ chat_interaction() ของ chatbot_v5 เดิม ไม่แก้ gate / dialogue
หน้าเว็บ Next.js บน Vercel เรียกผ่าน /api/chat ของตัวเอง (proxy) ซึ่งแนบ PROXY_SECRET และ IP จริงของผู้ใช้

Endpoints
  GET  /health                 สถานะ (บอทโหลดเสร็จหรือยัง)
  POST /api/chat               multipart: text, image (ไม่บังคับ), session_id (ไม่บังคับ)
  POST /api/session/reset      form: session_id → เริ่มบทสนทนาใหม่
  GET  /media/{token}          รูป drawing ที่บอทส่งกลับ (เฉพาะไฟล์ใต้ MEDIA_ROOTS)

รันในเครื่อง:  uvicorn app:app --port 7860
"""
from __future__ import annotations

import asyncio
import hmac
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import Depends, FastAPI, File, Form, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from starlette.concurrency import run_in_threadpool

from webapi.bot import BotAdapter, MediaRegistry, SessionStore, to_web_messages
from webapi.image_guard import ImageRejected, load_image, read_limited
from webapi.ratelimit import MESSAGES_TH, RateLimiter
from webapi.settings import Settings, settings as default_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("beltbot")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class ApiError(Exception):
    def __init__(self, status: int, code: str, message_th: str, retry_after: int = 0):
        self.status, self.code, self.message_th, self.retry_after = status, code, message_th, retry_after


def create_app(cfg: Settings = default_settings, bot: Optional[BotAdapter] = None) -> FastAPI:
    bot = bot or BotAdapter(cfg.bot_module)
    sessions = SessionStore(cfg.session_ttl_s, cfg.max_sessions)
    limiter = RateLimiter(cfg.rl_per_minute, cfg.rl_per_hour, cfg.rl_images_per_hour, cfg.global_per_day)
    media = MediaRegistry(cfg.media_roots, BASE_DIR)
    gate = asyncio.Semaphore(max(1, cfg.max_concurrent))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if cfg.warm_on_start and not bot.ready:
            threading.Thread(target=_safe_load, args=(bot,), daemon=True).start()
        task = asyncio.create_task(_janitor(sessions, limiter))
        yield
        task.cancel()

    app = FastAPI(title="Movex / Modutech assistant", lifespan=lifespan, docs_url=None, redoc_url=None)
    app.state.bot, app.state.sessions, app.state.limiter, app.state.media = bot, sessions, limiter, media

    if cfg.allowed_origins:
        app.add_middleware(CORSMiddleware, allow_origins=cfg.allowed_origins,
                           allow_methods=["GET", "POST"], allow_headers=["*"])

    @app.exception_handler(ApiError)
    async def _api_error(_: Request, e: ApiError):
        headers = {"Retry-After": str(e.retry_after)} if e.retry_after else None
        return JSONResponse({"error": e.code, "message_th": e.message_th}, status_code=e.status, headers=headers)

    # ── การยืนยันว่ามาจาก proxy ของเรา + หา IP จริง ──
    def client_ip(request: Request) -> str:
        if cfg.proxy_secret:
            sent = request.headers.get("x-proxy-secret", "")
            if not hmac.compare_digest(sent.encode(), cfg.proxy_secret.encode()):
                raise ApiError(403, "forbidden", "ไม่อนุญาต")
            ip = request.headers.get("x-client-ip", "").strip()
            if ip:
                return ip
        fwd = request.headers.get("x-forwarded-for", "")
        if fwd:
            return fwd.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    @app.get("/health")
    async def health():
        return {"status": "ok" if bot.ready else ("error" if bot.load_error else "loading"),
                "bot_ready": bot.ready, "sessions": len(sessions)}

    @app.post("/api/chat")
    async def chat(text: str = Form(""), session_id: str = Form(""),
                   image: Optional[UploadFile] = File(None), ip: str = Depends(client_ip)):
        text = (text or "").strip()
        has_image = image is not None and bool(image.filename or image.size)
        if not text and not has_image:
            raise ApiError(400, "empty", "พิมพ์คำถามหรือแนบรูปก่อนส่งครับ")
        if len(text) > cfg.max_text_chars:
            raise ApiError(400, "text_too_long", f"ข้อความยาวเกิน {cfg.max_text_chars} ตัวอักษรครับ รบกวนแบ่งถาม")
        if not bot.ready:
            if bot.load_error:
                raise ApiError(503, "bot_error", "ระบบขัดข้องชั่วคราวครับ รบกวนติดต่อฝ่ายขายโดยตรง")
            raise ApiError(503, "warming_up", "ระบบกำลังเริ่มต้น (โหลดโมเดล) รบกวนรอสักครู่แล้วส่งใหม่ครับ", 15)

        verdict = limiter.check(ip, has_image)
        if not verdict.ok:
            raise ApiError(429, verdict.reason, MESSAGES_TH[verdict.reason], verdict.retry_after_s)

        pil = None
        if has_image:
            try:
                raw = await read_limited(image, cfg.max_image_bytes)
                pil = (await run_in_threadpool(load_image, raw, cfg.max_image_bytes,
                                               cfg.max_image_pixels, cfg.image_max_side)).image
            except ImageRejected as e:
                raise ApiError(400, f"image_{e.code}", e.message_th)

        s, _new = sessions.get_or_create(session_id)
        if s.turns >= cfg.max_turns:
            raise ApiError(409, "session_full", "บทสนทนานี้ยาวมากแล้วครับ กด “เริ่มแชตใหม่” เพื่อถามต่อ")
        if not s.lock.acquire(blocking=False):
            raise ApiError(409, "busy_session", "กำลังตอบข้อความก่อนหน้าอยู่ครับ รอสักครู่")
        try:
            try:
                await asyncio.wait_for(gate.acquire(), timeout=cfg.queue_wait_s)
            except asyncio.TimeoutError:
                raise ApiError(503, "busy", "ตอนนี้มีผู้ใช้งานพร้อมกันหลายคนครับ รบกวนส่งใหม่อีกครั้ง", 10)
            t0 = time.perf_counter()
            try:
                new, decision = await run_in_threadpool(bot.turn, s, text, pil)
            except Exception:
                log.exception("บอทตอบไม่สำเร็จ session=%s", s.id)
                raise ApiError(500, "bot_failed", "ขออภัยครับ ระบบตอบไม่สำเร็จ รบกวนลองใหม่อีกครั้ง")
            finally:
                gate.release()
            ms = round((time.perf_counter() - t0) * 1000)
            log.info("turn session=%s ip=%s image=%s decision=%s %dms", s.id[:8], _mask_ip(ip),
                     has_image, decision or "-", ms)
        finally:
            s.lock.release()

        msgs = to_web_messages(new, media, cfg.scrub_internal_ids)
        if not msgs:
            msgs = [{"type": "text", "text": "ขออภัยครับ ระบบยังไม่มีคำตอบสำหรับคำถามนี้ รบกวนถามใหม่อีกครั้ง"}]
        return {"session_id": s.id, "messages": msgs, "turns_left": max(0, cfg.max_turns - s.turns)}

    @app.post("/api/session/reset")
    async def reset(session_id: str = Form(""), _ip: str = Depends(client_ip)):
        if session_id:
            sessions.drop(session_id)
        s, _ = sessions.get_or_create(None)
        return {"session_id": s.id}

    @app.get("/media/{token}")
    async def media_file(token: str, _ip: str = Depends(client_ip)):
        path = media.resolve(token)
        if not path or not os.path.isfile(path):
            raise ApiError(404, "not_found", "ไม่พบไฟล์")
        return FileResponse(path, headers={"Cache-Control": "public, max-age=86400"})

    return app


def _safe_load(bot: BotAdapter) -> None:
    try:
        bot.load()
    except Exception:
        pass                                        # log แล้วใน bot.load — /health จะแสดง error


async def _janitor(sessions: SessionStore, limiter: RateLimiter) -> None:
    while True:
        await asyncio.sleep(300)
        sessions.sweep()
        limiter.sweep()


def _mask_ip(ip: str) -> str:
    """ไม่เก็บ IP เต็มใน log"""
    if "." in ip:
        return ".".join(ip.split(".")[:2] + ["x", "x"])
    return ip[:9] + "…" if ":" in ip else ip


app = create_app()
