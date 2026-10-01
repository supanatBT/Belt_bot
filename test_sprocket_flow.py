#!/usr/bin/env python3
"""เทสต์ทางเดินคำถามสเตอร์ตั้งแต่ข้อความลูกค้าถึง context ที่ LLM เห็น — ไม่เรียก LLM

คุมสามอย่างที่พลาดง่ายและไม่มี error ให้เห็นเวลาพลาด

  1. ขนาดสเตอร์กับรูเพลาต้องไม่หลุดเข้า prompt
     sprocket_gate ตัด dimensions/bore ออกเพราะเล่มไม่นิยาม Di/Do/A/B และ & ในรูเพลา
     แต่ context["sprockets"] ไม่ผ่าน project_for_llm (whitelist ใช้กับ products เท่านั้น)
     ถ้าวันไหนมีคนเพิ่มฟิลด์หรือข้ามเกต ค่าที่ยังตีความไม่ได้จะถึงลูกค้าโดยไม่มีใครรู้

  2. รหัสสเตอร์ที่ลูกค้าพิมพ์มาต้องแปลงกลับเป็นสายพานได้
     ก่อน 30 ก.ย. router จับรหัสได้แต่ไม่มีใครใช้ ผลคือ products = [] และ LLM ไม่มีข้อมูล

  3. วัสดุที่ต่อท้ายรหัสสเตอร์ต้องไม่ถูกอ่านเป็นความต้องการของลูกค้า
     EC127SQZ24*PA เคยทำให้ slots material = "PA" แล้วไปกรองสายพานผิด

รัน: python test_sprocket_flow.py [modutech_v30_1.json]
"""
from __future__ import annotations

import json
import os
import sys

import modutech_gates as mg
import modutech_router as mr
import modutech_turn as mt
from dialogue_state import DialogueManager
from modutech_kg import ModutechGraph, load_modutech_kg

PATH = sys.argv[1] if len(sys.argv) > 1 else "modutech_v30_1.json"
cat = mg.Catalogue(PATH)
search = mr.CatalogueSearch(cat)
kg = ModutechGraph.from_catalogue(cat)

ok = fail = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global ok, fail
    if cond:
        ok += 1
        print(f"  ✓ {name}")
    else:
        fail += 1
        print(f"  ✗ {name}{('  — ' + detail) if detail else ''}")


def turn(text: str, with_kg: bool = True):
    """เทิร์นใหม่ทุกครั้ง — DialogueManager ใหม่ เพื่อไม่ให้ state ของเคสก่อนรั่วมา"""
    dm = DialogueManager({}, cat, movex_materials={})
    d = dm.decide(text)
    return mt.build_modutech_turn(d, cat, search, vector_search=None,
                                  kg=kg if with_kg else None)


print("1) สายพานที่เล่มผูกสเตอร์ไว้")
t = turn("HC508 C ใช้สเตอร์ตัวไหน")
sp = t.context.get("sprockets") or []
codes = [x["sprocket_code"] for x in sp if "sprocket_code" in x]
check("มีคีย์ sprockets ใน context", "sprockets" in t.context)
check("ได้สเตอร์ 9 ตัว", len(codes) == 9, str(len(codes)))
check("ไม่ส่งสเตอร์ตัวเดียวกันซ้ำ", len(codes) == len(set(codes)))
check("มี link ของสายพานติดมาด้วย", len(sp) == len(codes) + 1, str(len(sp)))

print("\n2) สายพานที่เล่มไม่ได้ผูกสเตอร์ไว้ — ต้องบอก ไม่ใช่เงียบ ไม่ใช่เดา")
for code in ("MD508 C", "MD508 FG", "SM127 FG-GT", "EC254 TR-TAB"):
    t = turn(f"{code} ใช้สเตอร์ตัวไหน")
    sp = t.context.get("sprockets") or []
    check(f"{code}: สเตอร์ 0 ตัว + มีคำเตือน",
          not sp and bool(t.context.get("sprocket_note_th")),
          f"sprockets={len(sp)} note={t.context.get('sprocket_note_th')!r}")

print("\n3) ลูกค้าพิมพ์รหัสสเตอร์มาเดี่ยว ๆ")
t = turn("EC127SQZ24*PA ใช้กับสายพานรุ่นไหน")
sq = t.context.get("sprocket_query") or {}
check("แปลงกลับได้เป็น EC127 C/FG/GT",
      sq.get("resolved_belt_codes") == ["EC127 C", "EC127 FG", "EC127 GT"],
      str(sq.get("resolved_belt_codes")))
check("แนบ record ของสายพานทั้งสามรุ่น", len(t.context.get("products") or []) == 3,
      str(len(t.context.get("products") or [])))
check("เหลือเฉพาะสเตอร์ตัวที่ถาม ไม่เทมาทั้งซีรีส์",
      [x["sprocket_code"] for x in (t.context.get("sprockets") or [])
       if "sprocket_code" in x] == ["EC127SQZ24*PA"])
check("ไม่ตั้ง slot material จากวัสดุต่อท้ายรหัส",
      "material" not in (t.context.get("slots_known") or {}),
      str(t.context.get("slots_known")))
check("prompt มีกฎกำกับการใช้ sprocket_query", "sprocket_query" in t.system_prompt)

t = turn("EC127SQZ24*PA ใช้กับสายพานรุ่นไหน", with_kg=False)
check("ไม่ส่ง kg มาก็ไม่พัง (แค่ตอบไม่ได้)", isinstance(t.context, dict))

print("\n4) ขนาดสเตอร์และรูเพลาต้องไม่หลุดเข้า prompt — ตรวจทุกรุ่นที่มีสเตอร์")
LEAK = ("dimensions", "bore", "dimension_note_th", "Di", "Do",
        "141,6", "158,2", "1.5 & 2.5", "40 & 60")
leaked: list[str] = []
for belt in sorted(cat.links):
    t = turn(f"{belt} ใช้สเตอร์ตัวไหน")
    blob = json.dumps(t.context.get("sprockets") or [], ensure_ascii=False)
    hit = [k for k in LEAK if f'"{k}"' in blob or (k[0].isdigit() and k in blob)]
    if hit:
        leaked.append(f"{belt}:{hit}")
check(f"ทั้ง {len(cat.links)} รุ่นไม่มีขนาดหลุด", not leaked, str(leaked[:3]))
check("แต่ละสเตอร์ยังมีคำเตือนให้ขอขนาดจากฝ่ายขาย",
      all(x.get("_gate_notes_th") for x in (turn("HC508 C ใช้สเตอร์ตัวไหน")
          .context.get("sprockets") or []) if "sprocket_code" in x))

print("\n5) คำถามที่ไม่ได้ถามสเตอร์ ต้องไม่แนบสเตอร์มาให้เปลือง")
for q in ("HC508 C ทนอุณหภูมิเท่าไหร่", "HC508 C กว้างเท่าไหร่"):
    check(f"{q!r} ไม่มีคีย์ sprockets", "sprockets" not in turn(q).context)

print("\n6) ใช้ได้กับกราฟทั้งสองแบบ — สร้างสด และโหลดจากไฟล์ .db")
# ถ้า KnowledgeGraph จากไฟล์ใช้แทนกันไม่ได้ ระบบจะได้ products = [] ซึ่งดูเหมือน
# "เล่มไม่มีข้อมูล" ทั้งที่จริงคือคีย์ไม่ตรงกัน (ไฟล์ใช้ belt_code_id สร้างสดใช้ belt_code)
DB = os.environ.get("MODUTECH_KG_DB", "./modutech_kg.db")
if os.path.exists(DB):
    from modutech_kg import assert_no_blocked_fields, db_fingerprint, catalogue_fingerprint
    kg_db = load_modutech_kg(cat, DB, quiet=True)
    check("ไฟล์ .db ไม่มีฟิลด์ที่ sprocket_gate บล็อก",
          not assert_no_blocked_fields(DB), str(assert_no_blocked_fields(DB)))
    check("ลายนิ้วมือในไฟล์ตรงกับ JSON ปัจจุบัน",
          db_fingerprint(DB) == catalogue_fingerprint(cat.data),
          f"{db_fingerprint(DB)} != {catalogue_fingerprint(cat.data)}")
    check("shim แปลงรหัสสเตอร์ได้ผลเท่ากันทั้งสองแบบ",
          mt.sprocket_to_belts(kg_db, cat, ["EC127SQZ24*PA"]) ==
          mt.sprocket_to_belts(kg, cat, ["EC127SQZ24*PA"]) ==
          ["EC127 C", "EC127 FG", "EC127 GT"])
    dm = DialogueManager({}, cat, movex_materials={})
    t_db = mt.build_modutech_turn(dm.decide("EC127SQZ24*PA ใช้กับสายพานรุ่นไหน"),
                                  cat, search, vector_search=None, kg=kg_db)
    check("ได้ record สายพานเท่ากันทั้งสองแบบ",
          len(t_db.context.get("products") or []) == 3,
          str(len(t_db.context.get("products") or [])))
else:
    print(f"  – ข้าม (ยังไม่มี {DB} — สร้างด้วย python build_modutech_graph.py)")

print("\n7) BELONGS_TO — รุ่นอื่นในซีรีส์เดียวกัน (ตอบได้จากกราฟเท่านั้น)")
t = turn("HC508 C มีรุ่นอื่นในซีรีส์เดียวกันไหม")
ss = t.context.get("same_series") or {}
check("HC508 C ได้ 21 รุ่นในซีรีส์เดียวกัน", len(ss.get("belt_codes") or []) == 21,
      str(len(ss.get("belt_codes") or [])))
check("ไม่มีตัวเองอยู่ในรายการ", "HC508 C" not in (ss.get("belt_codes") or []))
check("ทุกรุ่นในรายการอยู่ในซีรีส์เดียวกันจริง",
      all(cat.by_code[c].get("belt_series") == cat.by_code["HC508 C"].get("belt_series")
          for c in ss.get("belt_codes") or []))
check("มีคำกำกับห้ามสรุปว่าใช้แทนกันได้",
      "ห้ามสรุปว่าใช้แทนกันได้" in (ss.get("note_th") or ""))
check("คำถามที่ไม่ได้ถามซีรีส์ ไม่แนบ same_series มาให้เปลือง",
      "same_series" not in turn("HC508 C กว้างเท่าไหร่").context)
check("ไม่ส่ง kg มา ก็ไม่มีคีย์นี้ ไม่ใช่พัง",
      "same_series" not in turn("HC508 C มีรุ่นอื่นในซีรีส์เดียวกันไหม",
                                with_kg=False).context)
check("prompt มีกฎกำกับการใช้ same_series", "same_series" in t.system_prompt)

if os.path.exists(DB):
    # บั๊กที่เจอ 1 ต.ค.: KnowledgeGraph มี get_same_series เหมือนกัน การเดาชนิดกราฟ
    # จาก hasattr จึงเข้าสาขาผิดแล้วคืน [] เงียบ ๆ ต้องแยกด้วยธง accepts_belt_codes
    t_db = mt.build_modutech_turn(
        DialogueManager({}, cat, movex_materials={}).decide("HC508 C มีรุ่นอื่นในซีรีส์เดียวกันไหม"),
        cat, search, vector_search=None, kg=load_modutech_kg(cat, DB, quiet=True))
    check("กราฟจากไฟล์ .db ให้รายชื่อเท่ากับกราฟที่สร้างสด",
          sorted((t_db.context.get("same_series") or {}).get("belt_codes") or []) ==
          sorted(ss.get("belt_codes") or []),
          str(len((t_db.context.get("same_series") or {}).get("belt_codes") or [])))

print("\n8) รหัสสเตอร์เดี่ยว ๆ ต้องไม่ถูกตีเป็นคำถามแนะนำรุ่นแบบเปิด")
# บั๊กที่เจอจาก graph_run1 (1 ต.ค.): "EC127SQZ19*PA ใช้กับสายพานรุ่นไหน" มีคำว่า "สายพาน"
# จึงเข้าเงื่อนไข MOD_RECOMMEND แล้วถูกถามกลับว่าทำงานอะไร ทั้งที่ลูกค้าระบุของมาชัดที่สุดแล้ว
# มันเคยรอดเพราะบั๊กอีกตัว — *PA ถูกอ่านเป็น slot material ทำให้ has_need เป็นจริง
# พอแก้ slot อาการนี้จึงโผล่ คือเคยทำงานได้ด้วยเหตุผลที่ผิด
for code, want in (("EC127SQZ19*PA", ["EC127 C", "EC127 FG", "EC127 GT"]),
                   ("HD635SQZ10*POM", ["HD635 C"])):
    dm2 = DialogueManager({}, cat, movex_materials={})
    d2 = dm2.decide(f"{code} ใช้กับสายพานรุ่นไหน")
    check(f"{code}: ได้ action=answer ไม่ใช่ถามกลับ", d2.action == "answer", repr(d2.action))
    t2 = mt.build_modutech_turn(d2, cat, search, vector_search=None, kg=kg)
    got = (t2.context.get("sprocket_query") or {}).get("resolved_belt_codes")
    check(f"{code}: แปลงกลับได้ {want}", got == want, str(got))

for q in ("อยากได้สายพาน Modutech แนะนำหน่อย", "สายพานสำหรับไลน์เปียก"):
    d3 = DialogueManager({}, cat, movex_materials={}).decide(q)
    check(f"{q[:26]!r} ยังทำงานเหมือนเดิม", d3.action in ("ask", "answer", "search"), repr(d3.action))
check("คำถามแนะนำรุ่นแบบเปิดยังถามกลับ",
      DialogueManager({}, cat, movex_materials={}).decide(
          "อยากได้สายพาน Modutech แนะนำหน่อย").action == "ask")

print("\n9) คำกำกับขนาดสเตอร์ต้องบอกเหตุผลจริง ไม่ปล่อยให้ LLM เติมเอง")
# graph_run2 เคส SPK-D01: LLM ตอบว่า "แคตตาล็อกไม่ได้ระบุไว้" ซึ่งไม่จริง
# เล่มพิมพ์ Di 68,5 / Do 78,3 / รูเพลา 25-40 ไว้ที่หน้า 52 เหตุผลจริงคือไม่มีนิยามสัญลักษณ์
notes = [n for x in (turn("HC508 C ใช้สเตอร์ตัวไหน").context.get("sprockets") or [])
         if "sprocket_code" in x for n in (x.get("_gate_notes_th") or [])]
check("คำกำกับบอกว่าเล่มพิมพ์ค่าไว้", any("เล่มพิมพ์ตัวเลขขนาดไว้" in n for n in notes),
      str(notes[:1]))
check("คำกำกับบอกเหตุผลว่าไม่มีนิยามสัญลักษณ์",
      any("ไม่ได้นิยามว่า Di Do A B" in n for n in notes))
check("คำกำกับห้าม LLM อ้างว่าแคตตาล็อกไม่มีข้อมูล",
      any("ห้ามบอกว่าแคตตาล็อกไม่มีข้อมูลขนาด" in n for n in notes))
check("ยังไม่มีตัวเลขขนาดหลุดมาพร้อมคำกำกับ",
      not any(k in json.dumps(turn("HC508 C ใช้สเตอร์ตัวไหน").context, ensure_ascii=False)
              for k in ("141,6", "158,2", "1.5 & 2.5")))

print(f"\nรวม {ok}/{ok + fail}")
sys.exit(1 if fail else 0)