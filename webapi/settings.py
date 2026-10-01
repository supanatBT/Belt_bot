"""
webapi/settings.py — ค่าตั้งทั้งหมดของ backend อ่านจาก environment (HF Space → Settings → Variables/Secrets)

ไม่มีค่าไหนผูกกับแบรนด์ — แบรนด์และโมเดลรูปอยู่ใน image_models.yaml
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def _list(name: str, default: str) -> list[str]:
    return [x.strip() for x in os.environ.get(name, default).split(",") if x.strip()]


@dataclass
class Settings:
    # ── บอท ──
    bot_module: str = field(default_factory=lambda: os.environ.get("BOT_MODULE", "chatbot_v5"))
    warm_on_start: bool = field(default_factory=lambda: os.environ.get("WARM_ON_START", "1") == "1")

    # ── ความปลอดภัย ──
    # ตั้ง PROXY_SECRET ให้ตรงกับฝั่ง Vercel → Space รับเฉพาะคำขอที่ผ่าน Vercel
    # และเชื่อ X-Client-IP ที่ Vercel ส่งมา (ไม่ตั้ง = ใช้ X-Forwarded-For ของ HF และไม่เชื่อ X-Client-IP)
    proxy_secret: str = field(default_factory=lambda: os.environ.get("PROXY_SECRET", ""))
    allowed_origins: list[str] = field(default_factory=lambda: _list("ALLOWED_ORIGINS", ""))

    # ── rate limit ต่อ IP ──
    rl_per_minute: int = field(default_factory=lambda: _int("RL_PER_MINUTE", 8))
    rl_per_hour: int = field(default_factory=lambda: _int("RL_PER_HOUR", 60))
    rl_images_per_hour: int = field(default_factory=lambda: _int("RL_IMAGES_PER_HOUR", 15))
    # เพดานรวมทั้งระบบต่อวัน (กันโควตา Gemini หมด) 0 = ไม่จำกัด
    global_per_day: int = field(default_factory=lambda: _int("GLOBAL_PER_DAY", 1500))
    # จำนวนเทิร์นที่ประมวลผลพร้อมกัน (CPU ฟรีของ HF มี 2 คอร์)
    max_concurrent: int = field(default_factory=lambda: _int("MAX_CONCURRENT", 2))
    queue_wait_s: int = field(default_factory=lambda: _int("QUEUE_WAIT_S", 20))

    # ── ข้อความ / รูป ──
    max_text_chars: int = field(default_factory=lambda: _int("MAX_TEXT_CHARS", 1000))
    max_image_bytes: int = field(default_factory=lambda: _int("MAX_IMAGE_BYTES", 5 * 1024 * 1024))
    max_image_pixels: int = field(default_factory=lambda: _int("MAX_IMAGE_PIXELS", 40_000_000))
    image_max_side: int = field(default_factory=lambda: _int("IMAGE_MAX_SIDE", 1600))

    # ── session ──
    session_ttl_s: int = field(default_factory=lambda: _int("SESSION_TTL_S", 2 * 3600))
    max_sessions: int = field(default_factory=lambda: _int("MAX_SESSIONS", 3000))
    max_turns: int = field(default_factory=lambda: _int("MAX_TURNS", 40))

    # ── รูปที่บอทส่งกลับ (drawing ของ Movex) — เสิร์ฟได้เฉพาะไฟล์ใต้โฟลเดอร์เหล่านี้ ──
    media_roots: list[str] = field(default_factory=lambda: _list("MEDIA_ROOTS", "image,images"))

    # แทนรหัสภายใน movex_chain_… ในคำตอบด้วยรหัสที่ลูกค้าอ่านได้ (GRAPH-001) — ชั้นแสดงผลเท่านั้น
    scrub_internal_ids: bool = field(default_factory=lambda: os.environ.get("SCRUB_INTERNAL_IDS", "1") == "1")


settings = Settings()
