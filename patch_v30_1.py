#!/usr/bin/env python3
"""modutech_v30.json → modutech_v30_1.json — แพตช์ก่อน launch prototype เท่านั้น

แก้แค่สองอย่าง ไม่ใช่ v31

  1. เติม specifications.notes ของ 8 รุ่นที่ว่างเปล่าใน v30 ทั้งที่เล่มพิมพ์หมายเหตุไว้ครบ
     ดึงจาก PDF ตรง ๆ ตัดคำนำหน้า "- " ออกอย่างเดียว ไม่เรียบเรียงใหม่
     ทำไมต้องเติม: catalogue_increment_status() อ่าน specifications.notes เท่านั้น
     ถ้า notes ว่าง gate จะตรวจค่า "เพิ่มทีละ" ของรุ่นนั้นไม่ได้เลย

  2. ตั้ง temperature_trust = "unconfirmed" ให้ variant md254-rr--pph-pph ตัวเดียว
     พร้อมข้อความเตือนใน temperature_unconfirmed_th
     บอทยังตอบ 93 °C (ค่าที่เข้มกว่า) แต่แนบคำเตือนไปด้วย แทนที่จะตอบเหมือนเป็นค่ายืนยันแล้ว

ไม่แตะ: ค่าตัวเลขทุกตัว · width_effective · temperature_effective · answering · จำนวนนับ

รัน: python patch_v30_1.py modutech_v30.json <path ของ PDF> [modutech_v30_1.json]
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
import sys

# ── สิ่งที่ตั้งใจจะแก้ ประกาศไว้ล่วงหน้าเพื่อให้ shadow copy ตรวจได้ ──
NOTES_PAGES = {
    "SM127 C": 75, "HC127 C": 67, "EC254 GT": 133, "MD254 FG-RT": 157,
    "MD254 GT": 163, "MD254 RR": 165, "XP254 C": 113, "XP254 PR22%": 117,
}

UNCONFIRMED_VARIANT = "md254-rr--pph-pph"
UNCONFIRMED_TH = (
    "ค่าสูงสุด 93 °C ยังไม่ยืนยัน — แคตตาล็อกหน้า 165 พิมพ์ +5/+93 °C คู่กับ +40/+230 °F "
    "ซึ่ง 230 °F เท่ากับ 110 °C ไม่ใช่ 93 °C และ 93 °C เป็นค่าเดียวกับเกรด POM ที่พิมพ์อยู่สามแถวเหนือขึ้นไป "
    "ในเล่มเดียวกันเกรด PPH ของรุ่น HC508 PR22 ระบุ 110 °C และ HP508 RR (ผิว Raised Rib เหมือนกัน) ระบุ 118 °C "
    "ระบบเลือกใช้ค่าที่เข้มกว่าคือ 93 °C ไว้ก่อน ถ้างานจริงต้องเกิน 93 °C ต้องยืนยันกับฝ่ายเทคนิคก่อนเสนอ"
)


def fingerprint(obj) -> str:
    return hashlib.sha256(
        json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str).encode()
    ).hexdigest()


def notes_from_page(doc, page: int) -> list[str]:
    """บรรทัดใต้หัวข้อ Important Notes ที่ขึ้นต้นด้วย '-' ติดกัน หยุดที่บรรทัดแรกที่ไม่ขึ้นต้นด้วย '-'"""
    lines = doc[page - 1].get_text().split("\n")
    try:
        i = next(k for k, l in enumerate(lines) if l.strip() == "Important Notes")
    except StopIteration:
        raise SystemExit(f"หน้า {page}: ไม่พบหัวข้อ Important Notes — หยุด ไม่เดา")
    out = []
    for l in lines[i + 1:]:
        s = l.strip()
        if not s:
            continue
        if not s.startswith("-"):
            break
        out.append(s.lstrip("- ").strip())
    if not out:
        raise SystemExit(f"หน้า {page}: อ่านหมายเหตุไม่ได้สักบรรทัด — หยุด")
    return out


def scrub(doc: dict) -> dict:
    """ลบเฉพาะฟิลด์ที่ตั้งใจจะแก้ออกจากทั้งสองฝั่ง เพื่อให้ที่เหลือต้องเหมือนกันเป๊ะ"""
    d = copy.deepcopy(doc)
    for p in d.get("products", []):
        if p.get("belt_code") in NOTES_PAGES:
            p.get("specifications", {}).pop("notes", None)
        for v in p.get("specifications", {}).get("variants") or []:
            if v.get("variant_id") == UNCONFIRMED_VARIANT:
                v.pop("temperature_trust", None)
                v.pop("temperature_unconfirmed_th", None)
    return d


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    src, pdf = sys.argv[1], sys.argv[2]
    dst = sys.argv[3] if len(sys.argv) > 3 else "modutech_v30_1.json"

    import pymupdf
    doc = pymupdf.open(pdf)

    with open(src, encoding="utf-8") as f:
        data = json.load(f)
    shadow = copy.deepcopy(data)          # ของดิบไว้เทียบ

    by_code = {p.get("belt_code"): p for p in data.get("products", [])}

    print("1) เติม specifications.notes")
    for code, page in NOTES_PAGES.items():
        p = by_code.get(code)
        if p is None:
            raise SystemExit(f"ไม่พบรุ่น {code} ในไฟล์ — หยุด")
        spec = p.setdefault("specifications", {})
        if spec.get("page") != page:
            raise SystemExit(f"{code}: ไฟล์ระบุหน้า {spec.get('page')} แต่สคริปต์ตั้งไว้ {page} — หยุด")
        if spec.get("notes"):
            raise SystemExit(f"{code}: notes ไม่ว่าง ({len(spec['notes'])} บรรทัด) — หยุด ไม่ทับของเดิม")
        notes = notes_from_page(doc, page)
        spec["notes"] = notes
        m = re.search(r"(?<!Non-)Standard\s*b\s?elt\s*increme\s?nts?\s*([\d,\.]+)",
                      " | ".join(notes), re.IGNORECASE)
        printed = float(m.group(1).replace(",", ".").rstrip(".")) if m else None
        table = (spec.get("width_effective") or {}).get("increment_mm")
        flag = "ตรงกัน" if (printed is not None and table is not None
                            and abs(printed - table) < 0.05) else "⚠️ ไม่ตรง"
        print(f"   {code:<14} p{page}  {len(notes)} บรรทัด · เล่ม={printed} ตาราง={table} {flag}")

    print("\n2) ตั้ง temperature_trust = unconfirmed")
    hits = 0
    for p in data.get("products", []):
        for v in p.get("specifications", {}).get("variants") or []:
            if v.get("variant_id") != UNCONFIRMED_VARIANT:
                continue
            if v.get("temperature_trust") != "ok":
                raise SystemExit(f"{UNCONFIRMED_VARIANT}: temperature_trust เดิมคือ "
                                 f"{v.get('temperature_trust')!r} ไม่ใช่ 'ok' — หยุด")
            v["temperature_trust"] = "unconfirmed"
            v["temperature_unconfirmed_th"] = UNCONFIRMED_TH
            hits += 1
    if hits != 1:
        raise SystemExit(f"เจอ variant {UNCONFIRMED_VARIANT} {hits} ตัว ต้องเป็น 1 — หยุด")
    print(f"   {UNCONFIRMED_VARIANT} · ok → unconfirmed (ค่าตัวเลขไม่เปลี่ยน)")

    print("\n3) เทียบกับ shadow copy")
    a, b = fingerprint(scrub(shadow)), fingerprint(scrub(data))
    if a != b:
        raise SystemExit("❌ มีอย่างอื่นเปลี่ยนนอกเหนือจากที่ประกาศไว้ — ไม่เขียนไฟล์")
    print("   ✅ นอกจากฟิลด์ที่ประกาศไว้ ไม่มีอะไรเปลี่ยน")

    print("\n4) จำนวนนับต้องเท่าเดิม")
    n_prod = len(data["products"])
    n_var = sum(len(p.get("specifications", {}).get("variants") or []) for p in data["products"])
    n_app = sum(len(a.get("applications") or [])
                for p in data["products"] for a in p.get("industries") or [])
    n_pair = sum(len(p.get("industries") or []) for p in data["products"])
    print(f"   สินค้า {n_prod} · คู่สายพาน-อุตสาหกรรม {n_pair} · การใช้งาน {n_app} · variant {n_var}")
    if (n_prod, n_pair, n_app) != (68, 211, 651):
        raise SystemExit("❌ จำนวนนับเปลี่ยน — ไม่เขียนไฟล์")

    with open(dst, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    print(f"\nเขียน {dst} แล้ว · {src} ไม่ถูกแตะต้อง")
    return 0


if __name__ == "__main__":
    sys.exit(main())
