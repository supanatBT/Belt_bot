"""
webapi/ratelimit.py — rate limit ต่อ IP แบบหน้าต่างเลื่อน (sliding window) ในหน่วยความจำ

Space รันโปรเซสเดียว จึงเก็บในหน่วยความจำได้ รีสตาร์ตแล้วนับใหม่ (ยอมรับได้)
ไม่มี I/O ไม่มี dependency — เทสต์ได้ด้วย python ล้วน โดยส่ง now เข้าไปเอง
"""
from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Optional


@dataclass
class Verdict:
    ok: bool
    retry_after_s: int = 0
    reason: str = ""          # per_minute | per_hour | images_per_hour | global_day


class RateLimiter:
    def __init__(self, per_minute: int, per_hour: int, images_per_hour: int, global_per_day: int):
        self.limits = {"per_minute": (per_minute, 60), "per_hour": (per_hour, 3600),
                       "images_per_hour": (images_per_hour, 3600)}
        self.global_per_day = global_per_day
        self._hits: dict[tuple[str, str], deque] = defaultdict(deque)
        self._global: deque = deque()
        self._lock = threading.Lock()

    @staticmethod
    def _prune(q: deque, window: int, now: float) -> None:
        while q and q[0] <= now - window:
            q.popleft()

    def check(self, ip: str, has_image: bool, now: Optional[float] = None) -> Verdict:
        """ตรวจและนับในครั้งเดียว — ถ้าไม่ผ่านจะไม่นับครั้งนี้"""
        now = time.time() if now is None else now
        with self._lock:
            keys = ["per_minute", "per_hour"] + (["images_per_hour"] if has_image else [])
            for k in keys:
                limit, window = self.limits[k]
                if limit <= 0:
                    continue
                q = self._hits[(ip, k)]
                self._prune(q, window, now)
                if len(q) >= limit:
                    return Verdict(False, max(1, int(q[0] + window - now) + 1), k)
            if self.global_per_day > 0:
                self._prune(self._global, 86400, now)
                if len(self._global) >= self.global_per_day:
                    return Verdict(False, max(1, int(self._global[0] + 86400 - now) + 1), "global_day")
                self._global.append(now)
            for k in keys:
                if self.limits[k][0] > 0:
                    self._hits[(ip, k)].append(now)
            return Verdict(True)

    def sweep(self, now: Optional[float] = None) -> None:
        """ลบ IP ที่ไม่มีการใช้งานในชั่วโมงที่ผ่านมา กันหน่วยความจำโต"""
        now = time.time() if now is None else now
        with self._lock:
            for key in list(self._hits):
                q = self._hits[key]
                self._prune(q, 3600, now)
                if not q:
                    del self._hits[key]


MESSAGES_TH = {
    "per_minute": "ส่งข้อความถี่เกินไปครับ รบกวนรอสักครู่แล้วลองใหม่",
    "per_hour": "ใช้งานครบจำนวนต่อชั่วโมงแล้วครับ รบกวนลองใหม่ภายหลัง",
    "images_per_hour": "อัปโหลดรูปครบจำนวนต่อชั่วโมงแล้วครับ พิมพ์รหัสรุ่นแทนได้เลย หรือลองส่งรูปใหม่ภายหลัง",
    "global_day": "ระบบมีผู้ใช้งานเต็มโควตาของวันนี้แล้วครับ รบกวนติดต่อฝ่ายขายโดยตรง หรือลองใหม่พรุ่งนี้",
}
