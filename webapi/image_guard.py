"""
webapi/image_guard.py — ตรวจรูปที่ผู้ใช้อัปโหลดก่อนถึงบอท

  · จำกัดขนาดไฟล์ (ตรวจระหว่างอ่าน ไม่อ่านทั้งไฟล์ก่อนค่อยตรวจ)
  · ยอมรับเฉพาะ JPEG / PNG / WEBP ดูจากเนื้อไฟล์ ไม่เชื่อนามสกุลหรือ content-type
  · กัน decompression bomb (จำนวนพิกเซล)
  · หมุนตาม EXIF · ย่อด้านยาวสุดไม่เกิน max_side · แปลงเป็น RGB (เหมือน run_testcases.py)
"""
from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Optional

from PIL import Image, ImageOps, UnidentifiedImageError

ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}


class ImageRejected(ValueError):
    def __init__(self, code: str, message_th: str):
        super().__init__(message_th)
        self.code = code
        self.message_th = message_th


@dataclass
class CleanImage:
    image: Image.Image
    original_format: str
    original_size: tuple[int, int]
    bytes_in: int


def load_image(data: bytes, max_bytes: int, max_pixels: int, max_side: int) -> CleanImage:
    if len(data) > max_bytes:
        raise ImageRejected("too_large", f"รูปใหญ่เกิน {max_bytes // (1024 * 1024)} MB ครับ รบกวนย่อรูปแล้วส่งใหม่")
    if not data:
        raise ImageRejected("empty", "ไฟล์รูปว่างครับ")
    try:
        probe = Image.open(io.BytesIO(data))
        fmt = probe.format or ""
        w, h = probe.size
    except (UnidentifiedImageError, OSError):
        raise ImageRejected("not_image", "ไฟล์นี้ไม่ใช่รูปภาพที่ระบบอ่านได้ครับ รองรับ JPG, PNG, WEBP")
    if fmt not in ALLOWED_FORMATS:
        raise ImageRejected("format", f"ยังไม่รองรับรูปแบบ {fmt or 'นี้'} ครับ รองรับ JPG, PNG, WEBP")
    if w * h > max_pixels:
        raise ImageRejected("too_many_pixels", "รูปมีความละเอียดสูงเกินไปครับ รบกวนย่อรูปแล้วส่งใหม่")
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except Exception:
        raise ImageRejected("corrupt", "ไฟล์รูปเสียหรืออ่านไม่ครบครับ รบกวนส่งใหม่")
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")
    if max(img.size) > max_side:
        img.thumbnail((max_side, max_side), Image.LANCZOS)
    return CleanImage(img, fmt, (w, h), len(data))


async def read_limited(upload, max_bytes: int) -> bytes:
    """อ่าน UploadFile ทีละก้อน หยุดทันทีเมื่อเกินขนาด"""
    buf = bytearray()
    while True:
        chunk = await upload.read(64 * 1024)
        if not chunk:
            break
        buf += chunk
        if len(buf) > max_bytes:
            raise ImageRejected("too_large", f"รูปใหญ่เกิน {max_bytes // (1024 * 1024)} MB ครับ รบกวนย่อรูปแล้วส่งใหม่")
    return bytes(buf)
