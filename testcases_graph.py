#!/usr/bin/env python3
"""สร้างชุดทดสอบของกราฟ — ครอบทุก relation ทั้งขามีและขาไม่มี

ทำไมต้อง generate ไม่เขียนมือ
  เคยพลาดมาแล้วตอนชุด 136 เคส: เขียนค่าที่คาดด้วยมือแล้วค่าดันขัดกับที่ระบบตั้งใจตอบ
  ต้องไล่แก้ทีหลัง 11 จุด ไฟล์นี้จึงอ่านค่าที่คาดจากกราฟกับเกตตรง ๆ
  ชุดทดสอบจึงไม่มีทางขัดกับสิ่งที่ระบบตั้งใจทำได้

relation ที่ครอบ
  COMPATIBLE_WITH ขาไป     สายพาน → สเตอร์           SPK-F*
  COMPATIBLE_WITH ขากลับ   รหัสสเตอร์ → สายพาน       SPK-R*
  COMPATIBLE_WITH ขาไม่มี  11 รุ่นที่เล่มไม่ผูกไว้     SPK-N*
  BELONGS_TO               รุ่นอื่นในซีรีส์เดียวกัน   SER-*
  MADE_OF                  วัสดุของรุ่น                MAT-G*
  ขนาดสเตอร์ที่เกตบล็อก    Di/Do/รูเพลา → ฝ่ายขาย     SPK-D*

ทุกเคสที่เกี่ยวกับสเตอร์มี must_not_contain เป็นตัวเลขขนาดจริงจากไฟล์
เพราะกราฟเป็นทางเดียวที่เลี่ยง sprocket_gate ได้ — ถ้าวันไหนรั่ว ชุดนี้จับได้

รัน: python testcases_graph.py [modutech_v30_1.json] [-o testcases_graph_relations.csv]
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict

import modutech_gates as mg
import modutech_router as mr
import modutech_turn as mt
from dialogue_state import DialogueManager
from modutech_kg import ModutechGraph

FIELDS = ["id", "category", "subcategory", "turn_type", "turn_number", "user_text",
          "image", "must_contain", "must_contain_any", "must_not_contain",
          "must_ask_back", "expected_value", "expected_unit", "reply_max_length",
          "metrics", "description"]

SALES_WORDS = "ฝ่ายขาย|ติดต่อ|สอบถาม"          # คำใดคำหนึ่งก็ผ่าน
CONFIRM_WORDS = "ยืนยัน|ตรวจสอบ|สอบถาม|ฝ่ายขาย"


def row(**kw) -> dict:
    base = {f: "" for f in FIELDS}
    base.update(turn_type="single", turn_number="1", image="",
                must_ask_back="", expected_value="", expected_unit="",
                reply_max_length="", metrics="")
    base.update(kw)
    return base


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    path = args[0] if args else "modutech_v30_1.json"
    out_path = "testcases_graph_relations.csv"
    if "-o" in sys.argv:
        out_path = sys.argv[sys.argv.index("-o") + 1]

    cat = mg.Catalogue(path)
    kg = ModutechGraph.from_catalogue(cat)
    search = mr.CatalogueSearch(cat)
    codes_all = sorted(cat.by_code)

    def prefix_safe(code: str) -> bool:
        """ใช้ must_contain ได้ไหม — ไม่ได้ถ้ารหัสนี้เป็นต้นสตริงของรหัสอื่น

        "EC508 C" เป็นสตริงย่อยของ "EC508 C-RT" การตรวจแบบ substring จะผ่าน
        ทั้งที่คำตอบพูดถึงอีกรุ่น ต้องเลี่ยงเคสแบบนี้ ไม่ใช่ปล่อยให้ผ่านฟรี
        """
        return not any(o != code and o.startswith(code) for o in codes_all)

    def ctx_of(text: str) -> dict:
        dm = DialogueManager({}, cat, movex_materials={})
        return mt.build_modutech_turn(dm.decide(text), cat, search,
                                      vector_search=None, kg=kg).context

    # ตัวเลขขนาดของสเตอร์ที่ต้องไม่ปรากฏในคำตอบ — ดึงจากไฟล์ ไม่พิมพ์เอง
    def dim_strings(sprocket_codes: list) -> list:
        out: set = set()
        for c in sprocket_codes:
            s = cat.sprockets.get(c) or {}
            for k, v in (s.get("dimensions") or {}).items():
                mm = v.get("mm")
                if isinstance(mm, (int, float)):
                    out.add(f"{mm:g}".replace(".", ","))     # เล่มใช้จุลภาค
                    out.add(f"{mm:g}")
            for unit in ("mm", "inch"):
                pr = ((s.get("bore") or {}).get(unit) or {}).get("printed")
                if pr and not str(pr).strip("-").isdigit():
                    out.add(str(pr))                         # เช่น "1.5 & 2.5"
        # ตัวเลขสั้นเกินไปอาจตรงกับอย่างอื่นในคำตอบ เช่น "40" ในความกว้าง
        return sorted(x for x in out if len(x) >= 5)

    rows: list[dict] = []

    # ── COMPATIBLE_WITH ขาไป: สายพาน → สเตอร์ ─────────────────
    linked = sorted(cat.links)
    by_series: dict[str, list[str]] = defaultdict(list)
    for c in linked:
        by_series[cat.by_code[c].get("belt_series") or "?"].append(c)
    picks = [v[0] for v in by_series.values()][:6]           # ซีรีส์ละรุ่น

    for i, code in enumerate(picks, 1):
        ctx = ctx_of(f"{code} ใช้สเตอร์ตัวไหน")
        spr = [x["sprocket_code"] for x in (ctx.get("sprockets") or [])
               if "sprocket_code" in x]
        if not spr:
            continue
        rows.append(row(
            id=f"SPK-F{i:02d}", category="graph_compatible", subcategory="belt_to_sprocket",
            user_text=f"{code} ใช้สเตอร์ตัวไหน",
            must_contain=code,
            must_contain_any="|".join(spr[:4]),
            must_not_contain="|".join(dim_strings(spr)) or "",
            description=f"COMPATIBLE_WITH ขาไป — เล่มผูก {len(spr)} รหัสกับรุ่นนี้ "
                        f"ต้องบอกรหัสได้ แต่ห้ามบอกขนาดหรือรูเพลา"))

    # ── COMPATIBLE_WITH ขากลับ: รหัสสเตอร์ → สายพาน ────────────
    rev_picks: list[str] = []
    for code in picks:
        for sc in (cat.links.get(code) or {}).get("sprocket_codes") or []:
            belts = kg.belts_for_sprocket(sc)
            if belts and all(prefix_safe(b) for b in belts) and len(belts) <= 6:
                rev_picks.append(sc)
                break
    for i, sc in enumerate(rev_picks[:5], 1):
        belts = kg.belts_for_sprocket(sc)
        rows.append(row(
            id=f"SPK-R{i:02d}", category="graph_compatible", subcategory="sprocket_to_belt",
            user_text=f"{sc} ใช้กับสายพานรุ่นไหน",
            must_contain="|".join(belts),
            description=f"COMPATIBLE_WITH ขากลับ — ถามจากรหัสสเตอร์ ต้องได้ครบ "
                        f"{len(belts)} รุ่นตามตารางความเข้ากันได้"))

    # ── COMPATIBLE_WITH ขาไม่มี: 11 รุ่นที่เล่มไม่ผูกไว้ ────────
    # ห้ามเดารหัสจากซีรีส์ ต้องส่งฝ่ายขาย — จุดที่พลาดง่ายที่สุดของทั้งชุด
    nolink = kg.belts_without_sprocket()
    for i, code in enumerate(nolink, 1):
        series_pref = sorted({sc.split("Z")[0] for b in linked
                              if (cat.by_code[b].get("belt_series") ==
                                  cat.by_code[code].get("belt_series"))
                              for sc in (cat.links.get(b) or {}).get("sprocket_codes") or []})
        rows.append(row(
            id=f"SPK-N{i:02d}", category="graph_compatible", subcategory="no_link",
            user_text=f"{code} ใช้สเตอร์ตัวไหน",
            must_contain=code,
            must_contain_any=SALES_WORDS,
            must_not_contain="|".join(series_pref) or "",
            description=f"COMPATIBLE_WITH ขาไม่มี — เล่มไม่ผูกสเตอร์กับรุ่นนี้ "
                        f"ต้องบอกว่ายืนยันไม่ได้ ห้ามเดารหัสจากซีรีส์เดียวกัน"))

    # ── BELONGS_TO: รุ่นอื่นในซีรีส์เดียวกัน ───────────────────
    ser_picks = [v[0] for v in by_series.values()]
    n = 0
    for code in ser_picks:
        ctx = ctx_of(f"{code} มีรุ่นอื่นในซีรีส์เดียวกันไหม")
        sib = (ctx.get("same_series") or {}).get("belt_codes") or []
        if not sib:
            continue
        other = sorted({c for c in codes_all
                        if cat.by_code[c].get("belt_series") != cat.by_code[code].get("belt_series")
                        and prefix_safe(c)})
        n += 1
        rows.append(row(
            id=f"SER-{n:02d}", category="graph_series", subcategory="belongs_to",
            user_text=f"{code} มีรุ่นอื่นในซีรีส์เดียวกันไหม",
            must_contain_any="|".join(c for c in sib if prefix_safe(c))[:400],
            must_not_contain="|".join(other[:6]),
            description=f"BELONGS_TO — ซีรีส์นี้มี {len(sib)} รุ่นอื่น "
                        f"ต้องบอกจากรายการ ห้ามหยิบรุ่นจากซีรีส์อื่นมาปน"))
        if n >= 5:
            break

    # ── MADE_OF: วัสดุของรุ่น ─────────────────────────────────
    for i, code in enumerate(ser_picks[:5], 1):
        mats = sorted((cat.by_code[code].get("retrieval_card") or {}).get("materials") or [])
        if not mats:
            continue
        rows.append(row(
            id=f"MAT-G{i:02d}", category="graph_material", subcategory="made_of",
            user_text=f"{code} ทำจากวัสดุอะไรได้บ้าง",
            must_contain="|".join(mats),
            description=f"MADE_OF — เล่มระบุ {len(mats)} เกรดสำหรับรุ่นนี้ ต้องบอกครบ"))

    # ── ขนาดสเตอร์: เกตบล็อกเสมอ ──────────────────────────────
    for i, (code, q) in enumerate([
            (picks[0], "สเตอร์ของ {} มีขนาดเท่าไหร่"),
            (picks[0], "รูเพลาของสเตอร์ {} กี่มิลลิเมตร"),
            (picks[1] if len(picks) > 1 else picks[0], "เส้นผ่านศูนย์กลางสเตอร์ของ {} เท่าไหร่")], 1):
        text = q.format(code)
        ctx = ctx_of(text)
        spr = [x["sprocket_code"] for x in (ctx.get("sprockets") or [])
               if "sprocket_code" in x] or \
              (cat.links.get(code) or {}).get("sprocket_codes") or []
        rows.append(row(
            id=f"SPK-D{i:02d}", category="graph_blocked", subcategory="sprocket_dimension",
            user_text=text,
            must_contain_any=CONFIRM_WORDS,
            must_not_contain="|".join(dim_strings(spr)) or "",
            description="ขนาดสเตอร์กับรูเพลาถูก sprocket_gate บล็อกเสมอ เพราะเล่มไม่นิยาม "
                        "Di/Do/A/B และ & ในช่องรูเพลายังตีความไม่ได้ ต้องส่งฝ่ายขาย"))

    # ── ตรวจความเป็นธรรมของเคส ก่อนเขียนไฟล์ ──────────────────
    # เคสที่ไม่เป็นธรรมแย่กว่าไม่มีเคส เพราะมันฟ้องว่าระบบผิดทั้งที่ระบบทำถูก
    #   · คำที่ห้ามตอบ ต้องไม่อยู่ใน context ตั้งแต่ต้น ไม่งั้นเป็นการขอให้ LLM
    #     มองข้ามสิ่งที่เราเพิ่งยื่นให้มันเอง
    #   · คำที่ต้องตอบ ต้องหาได้ใน context ไม่งั้นเคสนี้ตอบไม่ได้ ไม่ว่า LLM เก่งแค่ไหน
    import json as _json
    unfair: list[str] = []
    for r in rows:
        blob = _json.dumps(ctx_of(r["user_text"]), ensure_ascii=False)
        for w in filter(None, r["must_not_contain"].split("|")):
            if w in blob:
                unfair.append(f'{r["id"]}: ห้ามตอบ {w!r} แต่มันอยู่ใน context แล้ว')
        for w in filter(None, r["must_contain"].split("|")):
            if w not in blob:
                unfair.append(f'{r["id"]}: ต้องตอบ {w!r} แต่ไม่มีใน context')
    if unfair:
        print("❌ เคสไม่เป็นธรรม ไม่เขียนไฟล์:")
        for u in unfair[:10]:
            print("  " + u)
        sys.exit(1)
    print(f"✓ ตรวจแล้ว {len(rows)} เคสเป็นธรรมทั้งหมด "
          f"(คำที่ห้ามไม่อยู่ใน context · คำที่ต้องมีหาได้ใน context)")

    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    cnt: dict[str, int] = defaultdict(int)
    for r in rows:
        cnt[r["category"]] += 1
    print(f"เขียน {out_path} — {len(rows)} เคส")
    for k in sorted(cnt):
        print(f"  {k:<20} {cnt[k]}")
    print(f"\nรันกับบอตตัวเต็ม:")
    print(f"  python run_answer_eval.py --module chatbot_v5 --testcases {out_path} -o graph_run1.json")


if __name__ == "__main__":
    main()
