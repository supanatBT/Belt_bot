#!/usr/bin/env python3
"""ชุดทดสอบ 120 เคสสำหรับชั้นที่ตัดสินด้วยโค้ด (router → dialogue → gate)

ทำไมไม่เทสต์ทั้งระบบ
  ชั้นนี้ไม่มีความสุ่มจาก LLM ผลลัพธ์เหมือนเดิมทุกครั้ง จึงใช้เป็นเกณฑ์ตัดสินได้
  ส่วนถ้อยคำที่ LLM เรียบเรียงต้องดูด้วยตาอีกชั้น (ไฟล์ CSV ที่ export ไว้ให้)

ค่าคาดหวังส่วนใหญ่ "คำนวณจากข้อมูลจริง" ไม่ได้พิมพ์มือ
  เช่น ค่าเพิ่มทีละที่ควรแสดง อ่านจากหมายเหตุในเล่มผ่าน catalogue_increment_status()
  ถ้าข้อมูลเปลี่ยน เทสต์จะเปลี่ยนตาม ไม่ค้างอยู่กับเลขเก่า

รัน: python testsuite_120.py [modutech_v30_1.json]
"""
from __future__ import annotations

import copy
import csv
import json
import sys

from dialogue_state import DialogueManager, DialogState
from modutech_gates import (Catalogue, Slots, catalogue_increment_status,
                            catalogue_nonstandard_increment, run_gates)
from modutech_router import ExploratoryQuery

PATH = sys.argv[1] if len(sys.argv) > 1 else "modutech_v30_1.json"
CAT = Catalogue(PATH)
DM = DialogueManager({}, CAT)
BY = {p["belt_code"]: p for p in CAT.products}

_SLOT_KEYS = ("wet_or_dry", "material", "line_shape", "operating_temp_c")


def play(turns: list[str]) -> list[dict]:
    """เล่นบทสนทนาหนึ่งชุด คืนผลของทุกเทิร์น"""
    st, out = DialogState(), []
    for t in turns:
        d = DM.decide(t, state=st)
        st = d.state
        row = {"d": d, "payload": [], "search": None}
        if d.action == "answer":
            codes = d.modutech.get("codes") or []
            slots = Slots(**{k: v for k, v in (d.modutech.get("slots") or {}).items()
                             if k in _SLOT_KEYS})
            recs = CAT.records(codes) if codes else []
            if recs:
                row["payload"] = run_gates(copy.deepcopy(recs), slots).products
        elif d.action == "search":
            q = d.modutech.get("search_query") or {}
            row["search"] = DM.search.search(ExploratoryQuery(**q))
        out.append(row)
    return out


# ═══════════════════════════════════════════════════════════════
# ตัวช่วยเขียนเงื่อนไข
# ═══════════════════════════════════════════════════════════════

def act(i, want):
    return lambda r: None if r[i]["d"].action == want else f"action={r[i]['d'].action} ควรเป็น {want}"


def brand(i, want):
    return lambda r: None if r[i]["d"].brand == want else f"brand={r[i]['d'].brand} ควรเป็น {want}"


def codes(i, want):
    def f(r):
        got = r[i]["d"].modutech.get("codes") or []
        return None if sorted(got) == sorted(want) else f"codes={got} ควรเป็น {want}"
    return f


def note_has(i, *subs):
    def f(r):
        n = (r[i]["payload"][0].get("width_note_th") if r[i]["payload"] else "") or ""
        miss = [s for s in subs if s not in n]
        return None if not miss else f"width_note ขาด {miss} · ได้: {n[:120]}"
    return f


def note_hasnt(i, *subs):
    def f(r):
        n = (r[i]["payload"][0].get("width_note_th") if r[i]["payload"] else "") or ""
        bad = [s for s in subs if s in n]
        return None if not bad else f"width_note ไม่ควรมี {bad} · ได้: {n[:120]}"
    return f


import re as _re

# "เพิ่มทีละ" ที่ไม่ได้นำหน้าด้วย "ขนาดนอกมาตรฐาน" = การประกาศขั้นมาตรฐาน ซึ่งเป็นสิ่งที่ห้ามเมื่อยังยืนยันไม่ได้
# ต้องตามด้วยตัวเลขถึงจะนับเป็นการ "ประกาศค่า" — ประโยคเตือน "ยังยืนยันค่าเพิ่มทีละมาตรฐานไม่ได้" ไม่ใช่การประกาศ
_STD_CLAIM_RE = _re.compile(r"(?<!ขนาดนอกมาตรฐาน)เพิ่มทีละ\s*[\d]")


def no_std_increment(i):
    def f(r):
        n = (r[i]["payload"][0].get("width_note_th") if r[i]["payload"] else "") or ""
        return None if not _STD_CLAIM_RE.search(n) else f"ยังประกาศขั้นมาตรฐานอยู่: {n[:120]}"
    return f


def all_(*fns):
    def f(r):
        for fn in fns:
            e = fn(r)
            if e:
                return e
        return None
    return f


def temp_of(i, material, key, want, tol=0.01):
    def f(r):
        if not r[i]["payload"]:
            return "ไม่มี payload"
        vs = [v for v in r[i]["payload"][0]["variants"] if v.get("belt_material") == material]
        if not vs:
            return f"ไม่มีเกรด {material}"
        got = (vs[0].get("temperature_answer") or {}).get(key)
        if isinstance(want, str):
            return None if got == want else f"{material}.{key}={got} ควรเป็น {want}"
        if got is None or abs(got - want) > tol:
            return f"{material}.{key}={got} ควรเป็น {want}"
        return None
    return f


def gate_note_has(i, sub):
    def f(r):
        ns = (r[i]["payload"][0].get("gate_notes_th") if r[i]["payload"] else []) or []
        return None if any(sub in n for n in ns) else f"gate_notes ไม่มี '{sub}' · ได้ {ns}"
    return f


def search_count(i, n):
    def f(r):
        got = (r[i]["search"] or {}).get("matched")
        return None if got == n else f"matched={got} ควรเป็น {n}"
    return f


def search_has(i, *belts):
    def f(r):
        got = {m["belt_code"] for m in (r[i]["search"] or {}).get("matches", [])}
        miss = [b for b in belts if b not in got]
        return None if not miss else f"ผลค้นขาด {miss} · ได้ {sorted(got)}"
    return f


def inch_ok(i):
    """ค่านิ้วต้องมีและตรงกับ มม. ทุกตัว"""
    def f(r):
        if not r[i]["payload"]:
            return "ไม่มี payload"
        p = r[i]["payload"][0]
        mm, inch = p.get("belt_widths_mm") or [], p.get("belt_widths_inch") or []
        if not inch:
            return "ไม่มี belt_widths_inch"
        if len(mm) != len(inch):
            return f"จำนวนไม่เท่ากัน mm={len(mm)} inch={len(inch)}"
        bad = [(a, b) for a, b in zip(mm, inch) if abs(a / 25.4 - b) > 0.001]
        return None if not bad else f"แปลงผิด {bad[:3]}"
    return f


# ═══════════════════════════════════════════════════════════════
# เคส
# ═══════════════════════════════════════════════════════════════

CASES: list[tuple] = []


def case(cid, cat_, turns, check, desc=""):
    CASES.append((cid, cat_, turns, check, desc))


# ── A. เส้นทางแบรนด์ / บทสนทนา (18) ────────────────────────────
case("RT-001", "routing", ["MP80 C พิทช์เท่าไหร่"], all_(act(0, "answer"), brand(0, "modutech")), "รหัส Modutech")
case("RT-002", "routing", ["LF 820 K325 กว้างเท่าไหร่"], brand(0, "movex"), "รหัส Movex")
case("RT-003", "routing", ["สวัสดีครับ"], act(0, "smalltalk"), "ทักทาย")
case("RT-004", "routing", ["อากาศวันนี้เป็นยังไง"], act(0, "refuse_offtopic"), "นอกเรื่อง เทิร์นแรก")
case("RT-005", "routing", ["ช่วยแปลภาษาอังกฤษให้หน่อย"], act(0, "refuse_offtopic"), "นอกเรื่อง")
case("RT-006", "routing", ["อยากได้สายพาน Modutech แนะนำหน่อย"], act(0, "ask"), "กว้างเกินไป ต้องถามกลับ")
case("RT-007", "routing", ["มีสายพานที่เลี้ยวได้ไหม"], act(0, "ask"), "ไม่รู้แบรนด์")
case("RT-008", "routing", ["สายพาน SS มีอะไรบ้าง"], act(0, "cross_brand"), "SS ตอบสองแบรนด์")
case("RT-009", "routing", ["HC127 C กว้างเท่าไหร่", "หนาเท่าไหร่"],
     all_(act(1, "answer"), brand(1, "modutech")), "ถามต่อไม่มีรหัส")
case("RT-010", "routing", ["HC127 C กว้างเท่าไหร่", "กี่นิ้ว"], act(1, "answer"), "ถามต่อเรื่องหน่วย")
case("RT-011", "routing", ["HC127 C กว้างเท่าไหร่", "ขอให้ตอบเป็นหน่วยนิ้วได้ไหม"], act(1, "answer"), "ขอเปลี่ยนหน่วย")
case("RT-012", "routing", ["MP80 C ใช้กับเบเกอรี่ได้ไหม", "แล้ว LF820 K325 กว้างเท่าไหร่"],
     brand(1, "movex"), "สลับแบรนด์กลางบทสนทนา")
case("RT-013", "routing", ["สายพานเลาะกระดูกไก่"], brand(0, "modutech"), "คำอุตสาหกรรม → Modutech")
case("RT-014", "routing", ["โซ่ลำเลียงขวด PET"], brand(0, "movex"), "คำงานขวด → Movex")
case("RT-015", "routing", ["MP80 C กว้างเท่าไหร่", "อันนี้ทนร้อนกี่องศา"], act(1, "ask"), "คำอ้างอิง + ถามอุณหภูมิ")
case("RT-016", "routing", ["ขอบคุณครับ"], act(0, "smalltalk"), "ขอบคุณ")
case("RT-017", "routing", ["EC254 C พิทช์เท่าไหร่", "แล้วผิวแบบไหน"], act(1, "answer"), "ถามต่อเรื่องผิว")
case("RT-018", "routing", ["HC508 ใช้กับเนื้อสัตว์ได้ไหม"], act(0, "ask"), "ซีรีส์กำกวม")

# ── B. อ่านรหัส (12) ──────────────────────────────────────────
case("CD-001", "code", ["mp80c พิทช์เท่าไหร่"], codes(0, ["MP80 C"]), "พิมพ์เล็กติดกัน")
case("CD-002", "code", ["MP8 0 C พิทช์เท่าไหร่"], codes(0, ["MP80 C"]), "รหัสแตกช่องว่าง (OCR)")
case("CD-003", "code", ["MD254-RR พิทช์เท่าไหร่"], codes(0, ["MD254 RR"]), "ขีดกลาง")
case("CD-004", "code", ["EC254 R-GT พิทช์เท่าไหร่"], codes(0, ["EC254 R-GT"]), "ห้ามตัดเหลือ EC254 R")
case("CD-005", "code", ["EC254 R พิทช์เท่าไหร่"], codes(0, ["EC254 R"]), "รหัสสั้นกว่า")
case("CD-006", "code", ["XP254 PR22% พิทช์เท่าไหร่"], codes(0, ["XP254 PR22%"]), "มีเครื่องหมาย %")
case("CD-007", "code", ["MD127 GAP50% กว้างเท่าไหร่"], codes(0, ["MD127 GAP50%"]), "GAP + %")
case("CD-008", "code", ["HC508 C-MTW ทนร้อนไหม"], codes(0, ["HC508 C-MTW"]), "ต่อท้ายหลายชั้น")
case("CD-009", "code", ["hc127 c กว้างเท่าไหร่"], codes(0, ["HC127 C"]), "พิมพ์เล็กมีช่องว่าง")
case("CD-010", "code", ["MD508 NS แรงดึงเท่าไหร่"], codes(0, ["MD508 NS"]), "รหัสปกติ")
case("CD-011", "code", ["HD635 C กว้างเท่าไหร่"], codes(0, ["HD635 C"]), "รุ่นเดี่ยว")
case("CD-012", "code", ["MP80 C กับ MD254 RR ต่างกันยังไง"],
     lambda r: None if set(r[0]["d"].modutech.get("codes") or []) == {"MP80 C", "MD254 RR"}
     else f"codes={r[0]['d'].modutech.get('codes')}", "สองรหัสในประโยคเดียว")


# ── C. ความกว้าง (สร้างจากข้อมูล) ──────────────────────────────
def _width_groups():
    silent, shown, hidden_min = [], [], []
    for p in CAT.products:
        we = p["specifications"].get("width_effective") or {}
        inc, lo, std = we.get("increment_mm"), we.get("orderable_min_mm"), we.get("standard_min_mm")
        stt, cat_inc = catalogue_increment_status(p)
        ok = inc is not None and (stt == "not_printed" or (stt == "found" and abs(cat_inc - inc) < 0.05))
        if std is not None and lo is not None and lo < std:
            hidden_min.append(p["belt_code"])
        (shown if ok else silent).append((p["belt_code"], inc))
    return silent, shown, hidden_min


SILENT, SHOWN, HIDDEN_MIN = _width_groups()

for n, (code_, inc) in enumerate([s for s in SILENT if s[1] is not None][:16], 1):
    case(f"WD-S{n:02d}", "width_silent", [f"{code_} กว้างเท่าไหร่"],
         all_(no_std_increment(0), gate_note_has(0, "ยังยืนยันค่าเพิ่มทีละมาตรฐาน")),
         f"{code_}: เล่มกับตารางไม่ตรง ต้องไม่ประกาศขั้นมาตรฐาน")

# รุ่นที่ไม่มีค่าเพิ่มทีละในข้อมูลเลย — ไม่มีอะไรให้เตือน แค่ต้องไม่ประกาศ
for n, code_ in enumerate([c for c, i in SILENT if i is None][:3], 1):
    case(f"WD-N{n:02d}", "width_silent", [f"{code_} กว้างเท่าไหร่"], no_std_increment(0),
         f"{code_}: ไม่มีค่าเพิ่มทีละในข้อมูล ต้องไม่ประกาศ")

for n, (code_, inc) in enumerate(SHOWN[:10], 1):
    case(f"WD-P{n:02d}", "width_shown", [f"{code_} กว้างเท่าไหร่"],
         note_has(0, f"เพิ่มทีละ {inc:g}"), f"{code_}: ตรงกับเล่ม ต้องแสดง {inc:g}")

for n, code_ in enumerate(HIDDEN_MIN[:8], 1):
    we = BY[code_]["specifications"]["width_effective"]
    case(f"WD-M{n:02d}", "width_min", [f"{code_} กว้างเท่าไหร่"],
         all_(note_has(0, f"{we['orderable_min_mm']:g}", "ห้ามรับปากว่าผลิตได้"),
              note_hasnt(0, "สั่งพิเศษ")),
         f"{code_}: บอกค่าต่ำสุดได้ แต่ห้ามรับปาก")

# เคสตรึงตัวเลข — กลุ่มข้างบนสร้างจากไฟล์ที่กำลังทดสอบ จึงเปลี่ยนตามข้อมูลและจับ "ข้อมูลหาย" ไม่ได้
# สี่ข้อนี้ตรึงค่าที่อ่านจากเล่มด้วยตาไว้ ถ้า specifications.notes ของ 8 รุ่นหายไปอีก เคสนี้จะแดงทันที
for cid_, code_, inc_ in [("WD-F01", "MD254 RR", "50"), ("WD-F02", "SM127 C", "76.2"),
                          ("WD-F03", "XP254 C", "76.2"), ("WD-F04", "MD254 GT", "50")]:
    case(cid_, "width_pinned", [f"{code_} กว้างเท่าไหร่"], note_has(0, f"เพิ่มทีละ {inc_}"),
         f"{code_}: เล่มพิมพ์ {inc_} มม. ต้องแสดงได้ (ต้องมี notes ในข้อมูล)")

case("WD-F05", "width_pinned", ["HC127 C กว้างเท่าไหร่"], no_std_increment(0),
     "HC127 C: เล่มพิมพ์ 101.6 แต่ตารางก้าว 50.8 ต้องไม่ประกาศ")

case("WD-I01", "width_inch", ["HC127 C กว้างกี่นิ้ว"], all_(inch_ok(0), note_has(0, "นิ้ว")), "นิ้วของ HC127 C")
case("WD-I02", "width_inch", ["MP80 C กว้างกี่นิ้ว"], inch_ok(0), "นิ้วของ MP80 C")
case("WD-I03", "width_inch", ["EC254 R กว้างกี่นิ้ว"], all_(inch_ok(0), note_has(0, "นิ้ว")), "นิ้วของ EC254 R")
case("WD-X01", "width_special", ["HC127 C ขอกว้าง 150 มม. ได้ไหม"],
     note_has(0, "ฝ่ายขาย"), "ขนาดนอกมาตรฐาน ต้องส่งฝ่ายขาย")
case("WD-X02", "width_special", ["EC254 C กว้าง 76.2 มม. ได้ไหม"],
     note_has(0, "ยืนยันกับฝ่ายเทคนิค"), "ตารางแคบกว่าค่าต่ำสุด ต้องเตือน")
case("WD-X03", "width_special", ["XP254 EVO CR กว้างเท่าไหร่"],
     note_has(0, "ยืนยันกับฝ่ายเทคนิค"), "กลุ่ม disputed")

# ── D. อุณหภูมิ (20) ──────────────────────────────────────────
case("TP-001", "temp", ["MD254 RR ทนร้อนกี่องศา"], act(0, "ask"), "ต้องถามเปียก/แห้งก่อน")
case("TP-002", "temp", ["MD254 RR ทนร้อนกี่องศา", "ไลน์แห้ง"],
     all_(act(1, "answer"), temp_of(1, "PPH", "max_c", 93.0),
          temp_of(1, "PPH", "status", "ok_unconfirmed"),
          gate_note_has(1, "ยังไม่ยืนยัน")), "PPH ต้องตอบ 93 พร้อมคำเตือน")
case("TP-003", "temp", ["MD254 RR ทนร้อนกี่องศา", "ไลน์แห้ง"],
     temp_of(1, "PP", "max_c", 105.0), "PP ของรุ่นเดียวกัน 105")
case("TP-004", "temp", ["MD254 RR ทนร้อนกี่องศา", "ไลน์แห้ง"],
     temp_of(1, "PPH", "max_f", 199.4), "°F คำนวณจาก °C")
case("TP-005", "temp", ["MP80 C ทนร้อนกี่องศา", "ไลน์เปียก"],
     temp_of(1, "POM", "max_c", 60.0), "POM เปียก 60")
case("TP-006", "temp", ["MP80 C ทนร้อนกี่องศา", "ไลน์แห้ง"],
     temp_of(1, "POM", "max_c", 93.0), "POM แห้ง 93")
case("TP-007", "temp", ["SM254 FG50% เกรด PA6 ใช้ไลน์เปียกได้ไหม"],
     temp_of(0, "PA6", "status", "not_recommended_wet"), "PA6 ห้ามใช้ไลน์เปียก")
case("TP-008", "temp", ["HC508 C-MTW ไลน์เปียกทนได้กี่องศา"],
     temp_of(0, "ICR", "max_c", 58.0), "ICR เปียก 58")
case("TP-009", "temp", ["EC254 C ไลน์แห้ง ทนเย็นกี่องศา"],
     temp_of(0, "PE", "min_c", -40.0), "PE แห้ง -40")
case("TP-010", "temp", ["EC254 C ไลน์แห้ง ทนเย็นกี่องศา"],
     temp_of(0, "PE", "min_f", -40.0), "-40 °C = -40 °F")
# ตารางสเปกหน้า 275 ระบุ 118 แต่ทะเบียนวัสดุหน้า 347 ระบุเพดาน PPH ที่ 115
# ระบบต้องใช้ค่าที่ต่ำกว่า (115) และต้องเตือนว่าสองแหล่งไม่ตรงกัน ไม่ใช่เลือกข้างเงียบ ๆ
case("TP-011", "temp", ["HP508 RR ทนร้อนกี่องศา", "ไลน์แห้ง"],
     all_(temp_of(1, "PPH", "max_c", 115.0),
          gate_note_has(1, "ตารางสเปกระบุอุณหภูมิสูงกว่าคุณสมบัติวัสดุ")),
     "HP508 RR: เล่ม 118 vs ทะเบียน 115 → ใช้ 115 พร้อมคำเตือน")
case("TP-012", "temp", ["HC508 PR22 ทนร้อนกี่องศา", "ไลน์แห้ง"],
     temp_of(1, "PPH", "max_c", 110.0), "PPH ของ HC508 PR22 = 110")
case("TP-013", "temp", ["EC127 GT ทนร้อนกี่องศา"],
     temp_of(0, "PP", "max_c", 60.0), "รุ่นไม่มี slot ตอบได้เลย")
case("TP-014", "temp", ["MD254 RR ใช้ที่ 100 องศา ไลน์แห้ง ได้ไหม"],
     gate_note_has(0, "ยังไม่ยืนยัน"), "PPH ถูกตัดที่ 100 แต่ต้องเตือน")
case("TP-015", "temp", ["MD254 RR ใช้ที่ 100 องศา ไลน์แห้ง ได้ไหม"],
     lambda r: None if [v["belt_material"] for v in r[0]["payload"][0]["variants"]] == ["PP"]
     else f"เกรดที่เหลือ {[v['belt_material'] for v in r[0]['payload'][0]['variants']]}",
     "ที่ 100 °C เหลือแต่ PP")
case("TP-016", "temp", ["MP80 C ใช้ที่ 80 องศา ไลน์เปียก ได้ไหม"],
     lambda r: None if not r[0]["payload"] else "ไม่ควรเหลือรุ่นไหนเลย", "80 °C เปียก ต้องตัดหมด")
case("TP-017", "temp", ["EC127 C ทนร้อนกี่องศา", "แห้ง"],
     temp_of(1, "PP", "max_c", 105.0), "PP แห้ง 105")
# เกรด PE ทนเปียกได้ 65 แต่พิน POM ได้แค่ 60 — ต้องใช้ค่าของชั้นที่แคบที่สุด ไม่ใช่ค่าของเนื้อสายพาน
case("TP-018", "temp", ["MD508 C ทนร้อนกี่องศา", "ไลน์เปียก"],
     temp_of(1, "PE", "max_c", 60.0), "PE/พิน POM เปียก → พินเป็นตัวจำกัดที่ 60")
case("TP-019", "temp", ["MD254 RR ทนร้อนกี่องศา", "ไลน์เปียก"],
     temp_of(1, "PPH", "status", "ok_unconfirmed"), "PPH เปียกก็ต้องเตือนเหมือนกัน")
case("TP-020", "temp", ["HC152 FG ทนร้อนกี่องศา", "แห้ง"],
     temp_of(1, "PE", "max_c", 65.0), "PE แห้ง 65")

# ── E. แรงดึง / โค้ง (10) ─────────────────────────────────────
CURVE = [p["belt_code"] for p in CAT.products if p["specifications"].get("has_curve_rating")]
for n, code_ in enumerate(CURVE[:6], 1):
    case(f"ST-C{n:02d}", "curve", [f"{code_} รับแรงดึงเท่าไหร่"], act(0, "ask"),
         f"{code_} วิ่งโค้งได้ ต้องถามรูปไลน์ก่อน")
case("ST-001", "strength", ["EC254 R รับแรงดึงเท่าไหร่", "ไลน์มีโค้ง"],
     lambda r: None if all("belt_strength_curve" in v for v in r[1]["payload"][0]["variants"])
     else "ไม่มีค่าแรงดึงตอนโค้ง", "ไลน์โค้ง → ใช้ค่าโค้ง")
case("ST-002", "strength", ["EC254 R รับแรงดึงเท่าไหร่", "ไลน์ตรง"],
     lambda r: None if all("belt_strength_curve" not in v for v in r[1]["payload"][0]["variants"])
     else "ไลน์ตรงไม่ควรมีค่าโค้ง", "ไลน์ตรง → ตัดค่าโค้งทิ้ง")
case("ST-003", "strength", ["MD508 C รับแรงดึงเท่าไหร่"],
     lambda r: None if any((v.get("belt_strength") or {}).get("n_per_m") == 60000.0
                           for v in r[0]["payload"][0]["variants"]) else "ไม่เจอ 60000",
     "MD508 C POM 60000 N/m")
case("ST-004", "strength", ["HD635 C รับแรงดึงเท่าไหร่"],
     lambda r: None if any((v.get("belt_strength") or {}).get("n_per_m") == 100000.0
                           for v in r[0]["payload"][0]["variants"]) else "ไม่เจอ 100000",
     "HD635 C 100000 N/m")

# ── F. วัสดุ / ปฏิเสธ (8) ─────────────────────────────────────
case("MT-001", "material", ["SM127 FG มีเกรด PE ไหม"],
     gate_note_has(0, "ไม่ได้ผลิตเกรด PE"), "แนะนำไว้แต่ไม่ผลิต")
case("MT-002", "material", ["HC127 C มีเกรด PP ไหม"],
     gate_note_has(0, "ไม่ได้ผลิตเกรด PP"), "HC127 C ไม่มี PP")
case("MT-003", "material", ["MP80 C ทำจากวัสดุอะไร"],
     lambda r: None if {v["belt_material"] for v in r[0]["payload"][0]["variants"]} == {"POM"}
     else "วัสดุไม่ตรง", "MP80 C มี POM อย่างเดียว")
case("MT-004", "material", ["HD635 C ทำจากวัสดุอะไร"],
     lambda r: None if "POM-NL" in {v["belt_material"] for v in r[0]["payload"][0]["variants"]}
     else "ไม่เจอ POM-NL", "HD635 C ใช้ POM-NL")
case("MT-005", "material", ["MD254 GAP48%-EHT ทำจากวัสดุอะไร"],
     lambda r: None if "EHT" in {v["belt_material"] for v in r[0]["payload"][0]["variants"]}
     else "ไม่เจอ EHT", "เกรด EHT")
case("MT-006", "material", ["XP254 C ทำจากวัสดุอะไร"], act(0, "answer"), "ตอบวัสดุได้")
case("MT-007", "material", ["MD508 FG-RT ทำจากวัสดุอะไร"], act(0, "answer"), "รุ่นผิวลูกกลิ้ง")
case("MT-008", "material", ["SM254 FG50% ทำจากวัสดุอะไร"], act(0, "answer"), "PA6")

# ── G. ส่งฝ่ายขาย (10) ────────────────────────────────────────
case("ES-001", "escalate", ["MP80 C ราคาเท่าไหร่"], act(0, "escalate"), "ราคา")
case("ES-002", "escalate", ["HC508 C lead time กี่วัน"], act(0, "escalate"), "lead time")
case("ES-003", "escalate", ["MP80 C สั่งขั้นต่ำเท่าไหร่"], act(0, "escalate"), "MOQ")
case("ES-004", "escalate", ["MD254 RR มีสต็อกไหม"], act(0, "escalate"), "สต็อก")
case("ES-005", "escalate", ["สเตอร์ของ MP80 C รูเพลาขนาดเท่าไหร่"], act(0, "escalate"), "ขนาดเพลา")
case("ES-006", "escalate", ["MP80 C มีอะไหล่ขายไหม"], act(0, "escalate"), "อะไหล่")
case("ES-007", "escalate", ["MP80 C กว้างสุดเท่าไหร่ แล้วราคาเมตรละเท่าไหร่"],
     all_(act(0, "answer"), lambda r: None if r[0]["d"].sales_handoff_th else "ไม่มี handoff"),
     "คำถามผสม: ตอบสเปก + ต่อท้ายฝ่ายขาย")
case("ES-008", "escalate", ["MD508 C ราคาเท่าไหร่ แรงดึงเท่าไหร่"],
     all_(act(0, "answer"), lambda r: None if r[0]["d"].sales_handoff_th else "ไม่มี handoff"),
     "ผสม: แรงดึง + ราคา")
case("ES-009", "escalate", ["MP80 C ราคาเท่าไหร่ ทนร้อนกี่องศา"],
     all_(act(0, "ask"), lambda r: None if r[0]["d"].reply_th.count("ฝ่ายขาย") else "ไม่มีฝ่ายขายในคำถามกลับ"),
     "ผสม: ถามกลับ + ฝ่ายขาย")
case("ES-010", "escalate", ["HC127 C ส่งของกี่วัน"], act(0, "escalate"), "lead time อีกสำนวน")

# ── H. ค้นทั้งฐาน (10) ────────────────────────────────────────
case("EX-001", "explore", ["ต้องการสายพานรับแรงดึงอย่างน้อย 50000 N/m"],
     all_(act(0, "search"), search_count(0, 3), search_has(0, "HD635 C", "MD508 C", "MD508 NS")),
     "แรงดึง ≥ 50000 → 3 รุ่น")
case("EX-002", "explore", ["สายพาน Modutech ผิวลูกกลิ้งมีรุ่นไหนบ้าง"],
     all_(act(0, "search"), search_has(0, "MD254 C-RT", "MD508 FG-RT")), "ผิวลูกกลิ้ง")
case("EX-003", "explore", ["สายพาน Modutech ที่ใช้ก้านสแตนเลสมีรุ่นไหนบ้าง"],
     all_(act(0, "search"), search_count(0, 2), search_has(0, "HD635 C", "MD254 GAP48%-EHT")),
     "พินสแตนเลส → 2 รุ่น")
# คำว่า "สายพาน" เฉย ๆ ใช้ได้ทั้งสองแบรนด์ ระบบจึงต้องถามแบรนด์ก่อน แล้วค่อยค้นเมื่อรู้
case("EX-004", "explore", ["มีสายพานพื้นที่เปิดอย่างน้อย 45% ไหม", "Modutech"],
     all_(act(0, "ask"), act(1, "search"), search_has(1, "MD127 GAP50%", "SM254 FG50%")),
     "open area ≥ 45% (ถามแบรนด์ก่อน)")
case("EX-005", "explore", ["มีสายพานพิทช์ไม่เกิน 10 มม. ไหม", "Modutech"],
     all_(act(0, "ask"), act(1, "search"), search_has(1, "MP80 C")), "พิทช์ ≤ 10")
case("EX-006", "explore", ["ต้องการสายพานน้ำหนักไม่เกิน 5 kg/m2", "Modutech"],
     all_(act(0, "ask"), act(1, "search")), "น้ำหนัก ≤ 5")
case("EX-007", "explore", ["สายพานผิว Flush Grid มีรุ่นไหนบ้าง"], act(0, "search"), "ค้นตามผิว")
case("EX-008", "explore", ["มีสายพานรับแรงดึงเกิน 200000 N/m ไหม"],
     all_(act(0, "search"), search_count(0, 0)), "ไม่มีจริง ต้องตอบว่าไม่มี")
case("EX-009", "explore", ["สายพานผิวกันลื่นมีรุ่นไหนบ้าง"], act(0, "search"), "Non Slip")
case("EX-010", "explore", ["ต้องการสายพานรับแรงดึงอย่างน้อย 50000 N/m"],
     lambda r: None if "ค้นครบทั้ง 68 รุ่น" in (r[0]["search"] or {}).get("note_th", "")
     else "ไม่มีข้อความยืนยันว่าค้นครบ", "ต้องบอกว่าค้นครบทั้งฐาน")

# ── I. งานและอุตสาหกรรม (8) ───────────────────────────────────
case("AP-001", "application", ["สายพานสำหรับไลน์ตัดแต่งเลาะกระดูกไก่ ใช้รุ่นไหนดี"], act(0, "answer"), "เลาะกระดูกไก่")
case("AP-002", "application", ["สายพานทางเข้าเตาอบขนมปัง ใช้รุ่นไหนได้บ้าง"], act(0, "answer"), "เตาอบเบเกอรี่")
case("AP-003", "application", ["MP80 C ใช้กับอุตสาหกรรมอะไรบ้าง"], act(0, "answer"), "รุ่นเดียว หลายอุตสาหกรรม")
case("AP-004", "application", ["สายพานสำหรับไลน์แล่ปลา"], act(0, "answer"), "อาหารทะเล")
case("AP-005", "application", ["สายพานสำหรับล้างผัก"], act(0, "answer"), "ผักผลไม้")
case("AP-006", "application", ["สายพานสำหรับลำเลียงกล่องลูกฟูก"], act(0, "answer"), "กล่องลูกฟูก")
case("AP-007", "application", ["สายพานเข้าเครื่องตรวจจับโลหะ"], act(0, "answer"), "ตรวจจับโลหะ")
case("AP-008", "application", ["MP80C ใช้กับเบเกอรี่ได้ไหม"], act(0, "answer"), "ถามเจาะจงรุ่น+อุตสาหกรรม")


# ═══════════════════════════════════════════════════════════════
# รัน
# ═══════════════════════════════════════════════════════════════

def main() -> int:
    print(f"ชุดทดสอบ {len(CASES)} เคส · ข้อมูล {PATH}\n")
    fails, bycat = [], {}
    for cid, cat_, turns, check, desc in CASES:
        bycat.setdefault(cat_, [0, 0])
        try:
            r = play(turns)
            err = check(r)
        except Exception as e:                      # noqa: BLE001
            err = f"ข้อผิดพลาดตอนรัน: {type(e).__name__}: {e}"
        bycat[cat_][1] += 1
        if err:
            fails.append((cid, cat_, turns, desc, err))
        else:
            bycat[cat_][0] += 1

    for c, (ok, tot) in sorted(bycat.items()):
        mark = "✅" if ok == tot else "❌"
        print(f"  {mark} {c:<16} {ok:>3}/{tot}")

    total = sum(t for _, t in bycat.values())
    passed = sum(o for o, _ in bycat.values())
    print(f"\nรวม {passed}/{total}")
    if fails:
        print(f"\nเคสที่ไม่ผ่าน {len(fails)} ข้อ")
        for cid, cat_, turns, desc, err in fails:
            print(f"\n  [{cid}] {desc}")
            print(f"    ถาม : {' → '.join(turns)}")
            print(f"    ผล  : {err}")
    return len(fails)


CSV_COLS = ["id", "category", "subcategory", "turn_type", "turn_number", "user_text", "image",
            "must_contain", "must_contain_any", "must_not_contain", "must_ask_back",
            "expected_value", "expected_unit", "reply_max_length", "metrics", "description"]

SEP = " | "

# run_answer_eval.py ตัดสินว่า "ถามกลับจริงไหม" ด้วยคำใบ้ชุดนี้ (looks_like_question)
# ข้อความถามกลับบางแบบของเราไม่มีคำใบ้เลย เช่น ASK_MODUTECH_NEED_TH ที่ขึ้นต้นว่า
# "ขอข้อมูลเพิ่มนิดนะครับ" — ถ้าตั้ง must_ask_back ไว้จะแดงทั้งที่ระบบถามกลับถูกต้อง
# จึงตั้งเฉพาะเมื่อข้อความจริงมีคำใบ้ ที่เหลือใช้ must_contain_any แทน
ASK_BACK_HINTS = ("?", "ไหม", "หรือไม่", "หรือเปล่า", "ระบุ", "ขอทราบ",
                  "ช่วยบอก", "รบกวนแจ้ง", "รุ่นไหน", "แบบไหน", "กรุณา")


def _fmt_num(x) -> str:
    return f"{x:g}"


def _assertions(cat_, res: dict) -> dict:
    """สร้างเงื่อนไขตรวจจากผลของเกตจริง — เงื่อนไขจึงไม่มีทางขัดกับสิ่งที่ระบบส่งให้ LLM

    ตรวจแค่ "ตัวเลข/ชื่อรุ่นที่ต้องมี" กับ "สิ่งที่ห้ามพูด" ไม่ตรวจสำนวน
    เพราะ LLM เรียบเรียงคำได้หลายแบบ การบังคับถ้อยคำจะทำให้เทสต์แดงทั้งที่คำตอบถูก
    """
    d, pay, sr = res["d"], res["payload"], res["search"]
    a = {"must_contain": [], "must_contain_any": [], "must_not_contain": [], "must_ask_back": ""}

    def any_once(*words):
        """ตั้ง must_contain_any ได้ครั้งเดียว — เงื่อนไขแรกคือเรื่องหลักของเคสนั้น
        ถ้าปล่อยให้ทับกัน เงื่อนไขสำคัญจะถูกเงื่อนไขรองแทนที่ แล้วเทสต์จะหลวมลงเงียบ ๆ"""
        if not a["must_contain_any"]:
            a["must_contain_any"] = list(words)

    if d.brand == "movex":
        # ฝั่ง Movex ตรวจไม่ได้จากที่นี่ (ฐานข้อมูล Movex ไม่ได้โหลดในชุดนี้) — ปล่อยให้ตรวจด้วยตา
        return a

    if d.action == "ask":
        if any(h in (d.reply_th or "") for h in ASK_BACK_HINTS):
            a["must_ask_back"] = "True"
        if d.topic == "temperature":
            any_once("เปียก", "แห้ง")
        elif d.topic == "strength":
            any_once("ตรง", "โค้ง")
        elif d.topic == "code":
            any_once("รุ่นไหน", "หมายถึง")
        elif d.topic == "brand":
            any_once("Movex", "Modutech")
        elif d.topic == "recommend":
            any_once("อุตสาหกรรม", "เปียก", "จุดที่ใช้งาน")
    elif d.action == "escalate":
        a["must_contain"] = ["ฝ่ายขาย"]
    elif d.action == "refuse_offtopic":
        any_once("ตอบได้เฉพาะ", "ขออภัย")
    elif d.action == "smalltalk":
        any_once("ยินดี", "มีเรื่อง")
    elif d.action == "cross_brand":
        a["must_contain"] = ["Movex", "Modutech"]
    elif d.action == "search" and sr:
        a["must_contain"] = [m["belt_code"] for m in sr["matches"][:3]]
        if sr["matched"] == 0:
            any_once("ไม่มี", "ไม่พบ")
    elif d.action == "answer" and not pay:
        # ตอบจากการจับคู่งาน ไม่ได้ระบุรหัสในคำถาม — ตรวจว่ารุ่นที่เล่มระบุไว้ต้องอยู่ในคำตอบ
        am = DM.search.match_application(d.effective_text)
        if am and am.belt_codes:
            a["must_contain"] = list(am.belt_codes[:3])
        # ไม่เข้าคำในตารางงาน (เช่น "แล่ปลา") → ระบบยังตอบได้ผ่านเวกเตอร์เสิร์ชและเครื่องมือค้น
        # แต่เราทำนายคำตอบไม่ได้จากที่นี่ จึงไม่ตั้งเงื่อนไข ดีกว่าตั้งเงื่อนไขมั่วแล้วแดงปลอม
    elif d.action == "answer" and pay:
        p = pay[0]
        a["must_contain"] = [p["belt_code"]]
        note = p.get("width_note_th") or ""
        widths = [_fmt_num(x) for x in (p.get("belt_widths_mm") or [])]
        if cat_.startswith("width"):
            m = _STD_CLAIM_RE.search(note)
            if m:
                num = _re.search(r"เพิ่มทีละ\s*([\d.]+)", note[m.start():])
                # ใส่ค่าขั้นเฉพาะเมื่อไม่ใช่สตริงย่อยของความกว้างตัวอื่น
                # ("50" ของ MD254 RR ซ่อนอยู่ใน 150/1150 อยู่แล้ว ใส่ไปก็ผ่านฟรี ไม่ได้ตรวจอะไร)
                if num and not any(num.group(1) in w for w in widths if w != num.group(1)):
                    a["must_contain"].append(num.group(1))
            elif ((BY.get(p["belt_code"], {}).get("specifications", {})
                   .get("width_effective") or {}).get("increment_mm")) is not None:
                # เรียกร้องคำอธิบายเฉพาะรุ่นที่ "มีค่าอยู่แต่ถูกกลั้นไว้"
                # HD635 C มีความกว้างเดียว ไม่มีค่าเพิ่มทีละตั้งแต่ต้น จึงไม่มีอะไรให้อธิบาย
                any_once("ยืนยัน", "ไม่ตรงกัน", "ไม่ได้ระบุ")
            if widths:
                a["must_contain"].append(widths[0])
            if cat_ == "width_inch":
                any_once("นิ้ว", "inch")
        if cat_ == "temp":
            # ถามเรื่องความเย็นต้องตรวจค่าต่ำสุด ไม่ใช่ค่าสูงสุด
            # (EC254 C "ทนเย็นกี่องศา" บอตตอบค่าต่ำสุดถูกต้อง แต่เงื่อนไขเดิมไปเรียกค่าสูงสุด)
            cold = _re.search(r"เย็น|ติดลบ|แช่แข็ง|ฟรีซ|ต่ำสุด|cold|freez", d.effective_text or "", _re.I)
            key = "min_c" if cold else "max_c"
            for v in p["variants"]:
                ta = v.get("temperature_answer") or {}
                if ta.get(key) is not None:
                    a["must_contain"].append(_fmt_num(ta[key]))
                if str(ta.get("status", "")).endswith("unconfirmed"):
                    # ตรวจเชิงบวกเท่านั้น — ต้องตอบค่าที่เข้มกว่าและต้องบอกว่ายังไม่ยืนยัน
                    #
                    # เคยตั้ง must_not_contain = "110 °C" ไว้ แต่คิดผิด: คำตอบจริงอ้างถึง 110
                    # ในฐานะ "เหตุผลว่าทำไมยังยืนยันไม่ได้" ซึ่งเป็นพฤติกรรมที่เราต้องการพอดี
                    # สิ่งที่อยากกันคือ "ตอบว่ารุ่นนี้ทนได้ 110" ซึ่งแยกไม่ออกด้วยการจับสตริง
                    # เงื่อนไขแบบนั้นจึงกันของผิดไม่ได้ แต่พร้อมจะแดงใส่ของถูกเมื่อ LLM สลับย่อหน้า
                    any_once("ยังไม่ยืนยัน", "ยืนยัน")
                if ta.get("status") == "not_recommended_wet":
                    any_once("ไม่แนะนำ", "ไม่ควร")
        if cat_ in ("strength", "curve"):
            for v in p["variants"]:
                for k in ("belt_strength_curve", "belt_strength"):
                    n = (v.get(k) or {}).get("n_per_m")
                    if n:
                        a["must_contain"].append(_fmt_num(n))
                        break
        if cat_ == "material":
            for n in (p.get("gate_notes_th") or []):
                if "ไม่ได้ผลิตเกรด" in n:
                    any_once("ไม่มี", "ไม่ได้ผลิต")
    # ห้ามซ้ำ และจำกัดจำนวน — บังคับให้ LLM พูดครบทุกค่าของทุกเกรดคือเกณฑ์ที่เปราะเกินไป
    # เอาแค่ 4 ค่าแรกก็พอชี้ได้แล้วว่าหยิบข้อมูลถูกชุด
    caps = {"must_contain": 4, "must_contain_any": 4, "must_not_contain": 3}
    for k, cap in caps.items():
        seen, out = set(), []
        for x in a[k]:
            if x not in seen:
                seen.add(x)
                out.append(x)
        a[k] = out[:cap]
    return a


def export_csv(path="testcases_modutech_136.csv") -> None:
    rows = 0
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS)
        w.writeheader()
        for cid, cat_, turns, _c, desc in CASES:
            try:
                res = play(turns)
            except Exception:                       # noqa: BLE001
                res = None
            for i, t in enumerate(turns, 1):
                a = _assertions(cat_, res[i - 1]) if res else {}
                w.writerow({
                    "id": cid, "category": cat_, "subcategory": cat_.split("_")[-1],
                    "turn_type": "multi" if len(turns) > 1 else "single",
                    "turn_number": i, "user_text": t, "image": "",
                    "must_contain": SEP.join(a.get("must_contain") or []),
                    "must_contain_any": SEP.join(a.get("must_contain_any") or []),
                    "must_not_contain": SEP.join(a.get("must_not_contain") or []),
                    "must_ask_back": a.get("must_ask_back", ""),
                    "expected_value": "", "expected_unit": "", "reply_max_length": "",
                    "metrics": cat_, "description": desc if i == 1 else "",
                })
                rows += 1
    print(f"\nเขียน {path} ({rows} เทิร์น) — สคีมาเดียวกับ testcases_v2.csv รันกับบอตตัวเต็มได้เลย")


if __name__ == "__main__":
    n = main()
    export_csv()
    sys.exit(1 if n else 0)
