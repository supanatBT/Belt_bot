#!/usr/bin/env python3
"""พิสูจน์ว่าการแยก unreadable ออกจาก not_printed ทำงานถูก และไม่เปลี่ยนพฤติกรรมกับ v30

รัน: python test_increment_status.py modutech_v30.json
"""
import copy
import sys

from modutech_gates import (Catalogue, Slots, catalogue_increment_status,
                            catalogue_standard_increment, width_gate)

TOPIC_VARIANTS = [
    ("Standard belt increments 76.2 mm", "found", 76.2),
    ("Standard Belt Increments 76,2", "found", 76.2),
    ("- Standard belt increments 76,2 mm.", "found", 76.2),
    ("Standard belt increments: 76.2 mm", "unreadable", None),
    ("Standard belt increments = 76,2 mm", "unreadable", None),
    # หน้า 157 (MD254 FG-RT) พิมพ์เอกพจน์จริง ต้องอ่านออก ไม่ใช่เงียบ
    ("Standard belt increment 50 mm.", "found", 50.0),
    ("Standard b elt increme nts 101,6 m m.", "found", 101.6),
    ("Standard belt increments are 76.2 mm", "unreadable", None),
    ("Non-standard belt increments 16,6 mm", "not_printed", None),
    ("Belt is FDA approved", "not_printed", None),
]


def unit():
    bad = 0
    for text, want_status, want_val in TOPIC_VARIANTS:
        p = {"specifications": {"notes": [text]}}
        got_status, got_val = catalogue_increment_status(p)
        ok = got_status == want_status and got_val == want_val
        bad += not ok
        print(f"   {'ok ' if ok else 'ผิด'} {got_status:<11} {str(got_val):<6} | {text}")
    # ไม่มีหมายเหตุเลย ≠ เล่มไม่ได้พิมพ์ไว้ — ต้องเป็น no_notes เพื่อให้ gate เงียบ
    for notes, want in (([], "no_notes"), (None, "no_notes"), (["x"], "not_printed")):
        got = catalogue_increment_status({"specifications": {"notes": notes}})[0]
        ok = got == want
        bad += not ok
        print(f"   {'ok ' if ok else 'ผิด'} {got:<11} {'':<6} | notes = {notes!r}")
    return bad


def rewritten_by_gate(product, slots=Slots()):
    """คืน (gate เขียนข้อความใหม่ไหม, ข้อความที่ได้) — ดูจาก Decision ไม่ใช่เดาจากข้อความ

    width_gate เขียน width_note_th ใหม่เฉพาะ 15 รุ่นที่ค่าต่ำสุดแคบกว่าขนาดมาตรฐาน
    รุ่นอื่นข้อความเดิมของ v30 ผ่านไปทั้งอย่างนั้น ซึ่งบางรุ่นก็มีคำว่า "เพิ่มทีละ" อยู่แล้ว
    ถ้าไม่แยกตรงนี้จะนับปนกัน
    """
    res = width_gate([product], slots)
    if not res.products:
        return False, ""
    note = ((res.products[0].get("specifications", {}).get("width_effective") or {})
            .get("width_note_th") or "")
    # ดูจาก action ไม่ใช่ข้อความ เพราะถ้อยคำเปลี่ยนได้ตามการตัดสินใจ (ทางเลือก ข → ค)
    touched = any(d.gate == "width" and d.action == "rewrite" for d in res.decisions)
    return touched, note


# ข้อความใหม่มีคำว่า "เพิ่มทีละ" ได้สองที่ — ขั้นมาตรฐาน กับ "ขนาดนอกมาตรฐานเพิ่มทีละ"
# ที่ต้องตรวจคือขั้นมาตรฐานเท่านั้น ค่านอกมาตรฐานอ่านตรงจากเล่มไม่ได้คำนวณจากตาราง
_STD_CLAUSE_RE = __import__("re").compile(r"(?<!ขนาดนอกมาตรฐาน)เพิ่มทีละ\s*[\d.]+")


def shows_increment(product, slots=Slots()):
    touched, note = rewritten_by_gate(product, slots)
    return touched and bool(_STD_CLAUSE_RE.search(note))


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "modutech_v30_1.json"
    print("1) ทดสอบรูปแบบข้อความ")
    bad = unit()

    cat = Catalogue(path)
    print(f"\n2) สถานะของทั้ง {len(cat.products)} รุ่นใน {path}")
    counts = {}
    for p in cat.products:
        s, _ = catalogue_increment_status(p)
        counts[s] = counts.get(s, 0) + 1
    print("   ", counts)
    if counts.get("unreadable"):
        print("   ⚠️ มีรุ่นที่อ่านไม่ออกอยู่แล้ววันนี้ ต้องดูด้วยตา:")
        for p in cat.products:
            if catalogue_increment_status(p)[0] == "unreadable":
                print("      ", p["belt_code"],
                      p.get("specifications", {}).get("notes"))

    print("\n3) พฤติกรรมกับ v30 ต้องไม่เปลี่ยน (นับเฉพาะรุ่นที่ gate เขียนข้อความใหม่)")
    rewritten = [p for p in cat.products if rewritten_by_gate(copy.deepcopy(p))[0]]
    shown = [p["belt_code"] for p in rewritten if shows_increment(copy.deepcopy(p))]
    hidden = [p["belt_code"] for p in rewritten if not shows_increment(copy.deepcopy(p))]
    print(f"   รุ่นที่ gate เขียนใหม่: {len(rewritten)}")
    print(f"   แสดงค่าเพิ่มทีละ {len(shown)}: {', '.join(shown)}")
    print(f"   ซ่อนไว้ {len(hidden)}: {', '.join(hidden) if hidden else '(ไม่มี)'}")

    print("\n4) จำลองวันที่ข้อความในเล่มเปลี่ยนรูป (เติมโคลอน)")
    target = next((p for p in cat.products
                   if catalogue_increment_status(p)[0] == "found"
                   and (p.get("specifications", {}).get("width_effective") or {})
                   .get("standard_min_mm") is not None), None)
    if target is None:
        print("   ไม่มีรุ่นที่เข้าเงื่อนไขทดสอบ")
        return bad

    code = target["belt_code"]
    before_val = catalogue_standard_increment(target)
    # แทรกโคลอนตรงตำแหน่งที่ตัวอ่านจับได้จริง ไม่ใช่ replace ข้อความตรง ๆ
    # เพราะหมายเหตุในเล่มมีคำแตก เช่น "Standard b elt increme nts 101,6 m m."
    import re as _re
    from modutech_gates import _STD_INCREMENT_RE
    broken = copy.deepcopy(target)
    notes = broken["specifications"].get("notes") or []
    for i, n in enumerate(notes):
        m = _STD_INCREMENT_RE.search(n)
        if m:
            notes[i] = n[:m.start(1)] + ": " + n[m.start(1):]
            break
    broken["specifications"]["notes"] = notes
    st, _ = catalogue_increment_status(broken)
    print(f"   ใช้รุ่น {code} (เดิมอ่านได้ {before_val})")
    print(f"   หลังเติมโคลอน สถานะ = {st}")
    old_behaviour = "แสดง (เพราะ None ถูกตีความว่าเล่มไม่ได้ระบุ)"
    new_behaviour = "แสดง" if shows_increment(broken) else "ไม่แสดง"
    print(f"   โค้ดเดิมจะ : {old_behaviour}")
    print(f"   โค้ดใหม่จะ : {new_behaviour}")
    if st != "unreadable" or new_behaviour != "ไม่แสดง":
        print("   ❌ ไม่เป็นไปตามที่ตั้งใจ")
        bad += 1
    else:
        print("   ✅ เงียบไว้ตามหลัก ไม่เดา เตือนแทน")

    print(f"\nสรุป: {'ผ่านทุกข้อ' if bad == 0 else f'มีข้อผิดพลาด {bad} จุด'}")
    return bad


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
