#!/usr/bin/env python3
"""เทสต์กราฟ Modutech — ยืนยันสามอย่าง

  1. ความสัมพันธ์ตรงกับ modutech_v30_1.json ทุกเส้น (ไม่ขาด ไม่เกิน)
  2. กราฟไม่มีตัวเลขสเปกอยู่ในตัวมันเอง — Di/Do/Bore/teeth รั่วออกจากกราฟไม่ได้
  3. ถ้าข้อมูลขาด กราฟหยุด ไม่เดินต่อแบบเงียบ ๆ

รัน: python test_modutech_kg.py [modutech_v30_1.json]
"""
from __future__ import annotations

import copy
import json
import sys

from modutech_kg import BELT, COMPATIBLE_WITH, MATERIAL, SPROCKET, ModutechGraph

PATH = sys.argv[1] if len(sys.argv) > 1 else "modutech_v30_1.json"
with open(PATH, encoding="utf-8") as f:
    DATA = json.load(f)

ok = fail = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global ok, fail
    if cond:
        ok += 1
        print(f"  ✓ {name}")
    else:
        fail += 1
        print(f"  ✗ {name}{('  — ' + detail) if detail else ''}")


print("1) สร้างกราฟและนับ")
g = ModutechGraph(DATA)
st = g.stats()
check("สายพาน 68 ปม", st["nodes"][BELT] == 68, str(st["nodes"]))
check("สเตอร์ 142 ปม", st["nodes"][SPROCKET] == 142, str(st["nodes"]))
check("ไม่มีปมวัสดุลอย", all(
    any(nid in r.get("MADE_OF", ()) for r in g._out.values())
    for nid, n in g.nodes.items() if n.node_type == MATERIAL))

print("\n2) ความสัมพันธ์ตรงกับไฟล์")
want = {(l["belt_code"], sc)
        for l in DATA["belt_sprocket_links"]
        for sc in l.get("sprocket_codes") or []}
got = {(g.nodes[b].label, g.nodes[s].label)
       for b, rels in g._out.items()
       for s in rels.get(COMPATIBLE_WITH, ())
       if g.nodes[b].node_type == BELT}
check(f"คู่สายพาน-สเตอร์ {len(want)} คู่ ตรงกันหมด", want == got,
      f"ขาด {len(want - got)} เกิน {len(got - want)}")
check("เส้นเป็นสองทาง ถามกลับได้",
      all(g.belts_for_sprocket(sc) for _, sc in list(want)[:20]))

# 11 รุ่นที่เล่มไม่ได้ผูกสเตอร์ไว้ — ต้องคืน [] ไม่ใช่เดาจากรหัสซีรีส์
miss = g.belts_without_sprocket()
check("สายพานที่ไม่มีคู่สเตอร์ 11 รุ่น", len(miss) == 11, str(miss))
check("MD508 ทั้ง 6 รุ่นอยู่ในกลุ่มนั้น",
      all(f"MD508 {s}" in miss for s in ("C", "C-RT", "FG", "FG-RT", "NS", "PR25")))
check("MD508 C ถามสเตอร์ได้ [] ไม่ใช่เดา", g.sprockets_for_belt("MD508 C") == [])

print("\n3) กราฟไม่ถือตัวเลข — ค่าที่เกตบล็อกรั่วออกไม่ได้")
blob = json.dumps([n.as_dict() for n in g.nodes.values()], ensure_ascii=False)
for f_ in ("Di_mm", "Do_mm", "Bore_mm", "Bore_inch", "pitch_diameter",
           "Max_Working_Load_N", "Weight_kg_m", "temperature"):
    check(f"ไม่มี {f_} ในกราฟ", f_ not in blob)
nums = [(n.node_id, k, v) for n in g.nodes.values() for k, v in n.attrs
        if isinstance(v, (int, float))]
check("ไม่มีค่าตัวเลขใน attrs เลย", not nums, str(nums[:3]))
# ปมสเตอร์เก็บ teeth_printed ("Z10") เป็นสตริงได้ แต่ห้ามเก็บ teeth เป็นตัวเลข
check("teeth เก็บเป็นข้อความที่เล่มพิมพ์ ไม่ใช่ตัวเลขให้คำนวณ",
      '"teeth":' not in blob and "teeth_printed" in blob)

print("\n4) ไม่แก้ข้อมูลที่รับมา")
before = json.dumps(DATA, sort_keys=True)
ModutechGraph(DATA).expand_products(["HC508 C", "MD508 C", "ไม่มีรุ่นนี้"])
check("dict ต้นทางไม่ถูกแก้", json.dumps(DATA, sort_keys=True) == before)

print("\n5) ข้อมูลขาด = หยุด ไม่เดินต่อ")
bad = copy.deepcopy(DATA)
bad["sprockets"] = bad["sprockets"][:-1]          # ทำให้จำนวนไม่ตรง
try:
    ModutechGraph(bad)
    check("สเตอร์หายไปหนึ่งตัวต้อง raise", False, "ไม่ raise")
except AssertionError:
    check("สเตอร์หายไปหนึ่งตัวต้อง raise", True)

bad2 = copy.deepcopy(DATA)
bad2["belt_sprocket_links"].append(
    {"belt_code": "ไม่มีรุ่นนี้ในฐาน", "sprocket_codes": ["ไม่มีรหัสนี้"]})
try:
    ModutechGraph(bad2)
    check("ผูกกับรุ่นที่ไม่มีต้อง raise", False, "ไม่ raise")
except AssertionError:
    check("ผูกกับรุ่นที่ไม่มีต้อง raise", True)

print("\n6) ส่วนที่ RetrieverV4 เรียก")
ctx = g.expand_products(["HC508 C"])
check("expand_products คืนสเตอร์ของรุ่นนั้น", len(ctx.compatible_products) == 9,
      str(len(ctx.compatible_products)))
check(".compatible กับ .compatible_products ชี้ของเดียวกัน",
      ctx.compatible == ctx.compatible_products)
check("รหัสที่ไม่รู้จักได้ context ว่าง ไม่ใช่ error",
      g.expand_products(["ไม่มีรุ่นนี้"]).compatible_products == [])
check("ไม่ส่ง product_ids มาเลยก็ไม่พัง", g.expand_products().seed_products == [])
check("รับคีย์เวิร์ดชื่ออื่นได้",
      g.expand_products(seed_products=["HC508 C"]).compatible_products != [])

print("\n7) ตรงกับที่ RetrieverV4 เรียกจริง (retriever_v4.py บรรทัด 712)")
# ชื่อคีย์เวิร์ดต้องเป็น seed_product_ids ไม่ใช่ product_ids
# ถ้าชื่อไม่ตรง Python จะโยนเข้า **kwargs แล้วกราฟคืนว่างแบบไม่มี error ให้เห็น
real = g.expand_products(
    seed_product_ids=["HC508 C"],
    include_same_series=True, include_compatible=True, include_same_material=False)
check("seed_product_ids= ใช้ได้", len(real.compatible_products) == 9,
      str(len(real.compatible_products)))
check("include_same_series=True ได้รุ่นในซีรีส์", len(real.same_series) == 21,
      str(len(real.same_series)))
check("include_same_material=False ไม่ดึงวัสดุ", real.material_details == {})
check("include_compatible=False ไม่ดึงสเตอร์",
      g.expand_products(seed_product_ids=["HC508 C"], include_compatible=False)
      .compatible_products == [])
mat = g.expand_products(seed_product_ids=["HC508 C"], include_same_material=True).material_details
check("material_details มี full_name ให้ to_llm_context ใช้",
      all("full_name" in v for v in mat.values()), str(list(mat)[:3]))
check("process_details ว่างเสมอ (เล่มไม่มีข้อมูลกระบวนการผลิต)",
      g.get_process_details("HC508 C") == {})
# hop2 ของ retriever ส่งรหัสสเตอร์กลับเข้ามาเป็น seed ต้องรับได้
check("รับรหัสสเตอร์เป็น seed ได้ (hop 2 ของ retriever)",
      g.expand_products(seed_product_ids=["EC-HC508SQZ10*PA"],
                        include_same_material=True).material_details != {})

print("\n8) ถามกลับจากรหัสสเตอร์")
check("EC127SQZ24*PA → EC127 C/FG/GT",
      g.resolve_sprocket_to_belts(["EC127SQZ24*PA"]) == ["EC127 C", "EC127 FG", "EC127 GT"],
      str(g.resolve_sprocket_to_belts(["EC127SQZ24*PA"])))
check("รหัสที่ไม่มีในเล่มคืน [] ไม่ใช่เดา",
      g.resolve_sprocket_to_belts(["ไม่มีรหัสนี้"]) == [])
check("ส่งหลายรหัสแล้วไม่ซ้ำ",
      len(set(g.resolve_sprocket_to_belts(["EC127SQZ24*PA", "EC127SRZ24*PA"]))) ==
      len(g.resolve_sprocket_to_belts(["EC127SQZ24*PA", "EC127SRZ24*PA"])))

print(f"\nรวม {ok}/{ok + fail}")
sys.exit(1 if fail else 0)
