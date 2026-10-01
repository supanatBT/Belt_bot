"""
run_answer_eval.py — harness ชั้น B
==================================
รันชุด test case ผ่าน chat_interaction() จริง แล้วตรวจ "ถ้อยคำของคำตอบ"

    ชั้น A  run_retrieval_eval.py   วัดว่าค้นเจอของที่ถูกไหม   ไม่ใช้ LLM
    ชั้น B  ไฟล์นี้                  วัดว่าเรียบเรียงถูกไหม     ใช้ Gemini

ต่างจากชั้น A ตรงไหน
-------------------
  · เรียก chat_interaction() เต็มรูป ผ่าน Gemini จริง
  · ต่อบทสนทนาข้ามเทิร์นได้ (chat_history + last_pids + vit/graph carry-over)
    เคส multi_turn จึงวัดได้ ต่างจากชั้น A ที่รันแต่ละเทิร์นแยกกัน
  · เปลืองโควตา API และผลไม่คงที่ทุกรอบ จึงควรรันหลายรอบแล้วดูความสม่ำเสมอ

metric ที่ชั้นนี้ตรวจ
--------------------
  answer_has_required      must_contain ครบทุกคำ
  answer_has_any           must_contain_any อย่างน้อยหนึ่งคำ
  answer_avoids_forbidden  must_not_contain ไม่มีสักคำ
  asked_back               must_ask_back = TRUE แล้วตอบกลับเป็นคำถามจริง
  value_correct            expected_value ปรากฏในคำตอบ
  unit_correct             expected_unit ปรากฏในคำตอบ
  reply_within_limit       ความยาวไม่เกิน reply_max_length
  no_error                 ไม่มี exception ระหว่างเรียก

รัน
---
  # เส้นฐาน Movex — ต้องตั้ง GEMINI_API_KEY ก่อน
  python run_answer_eval.py -o baseline_answer_movex.json

  # รันเฉพาะบางเคสตอนดีบัก
  python run_answer_eval.py --only SPEC-001,GRAPH-001 -o probe.json

  # เทียบสองรอบ
  python run_answer_eval.py --compare baseline_answer_movex.json after_answer_movex.json
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

# คำที่บ่งว่าคำตอบเป็นการถามกลับ ไม่ใช่การตอบ
ASK_BACK_HINTS = ("?", "ไหม", "หรือไม่", "หรือเปล่า", "ระบุ", "ขอทราบ",
                  "ช่วยบอก", "รบกวนแจ้ง", "รุ่นไหน", "แบบไหน", "กรุณา")


# คำปฏิเสธที่วางไว้หน้าคำยืนยันในภาษาไทย
# "เลี้ยวได้ครับ" เป็นสตริงย่อยของ "ไม่สามารถเลี้ยวได้ครับ" พอดี
# การหาแบบ substring จึงเจอทั้งที่ความหมายตรงข้าม ต้องดูคำนำหน้าด้วย
NEGATIONS = ("ไม่สามารถ", "ไม่ได้", "ไม่", "ยังไม่", "มิได้", "ปราศจาก",
             "cannot", "can not", "can't", "not ", "no ")
# ขอบเขตประโยคภาษาไทย — ไม่มีจุดปิดประโยค จึงใช้คำลงท้ายกับขึ้นบรรทัดใหม่
# ไม่ตัดที่จุด เพราะภาษาไทยใช้จุดในคำย่อ (มม. ซม. น.) การตัดที่จุดจะแยก
# "ไม่สามารถสั่งความกว้าง 500 มม." ออกจาก "ได้ครับ" แล้วคำปฏิเสธหลุดไปคนละประโยค
SENTENCE_SPLIT = re.compile(r"(?<=ครับ)|(?<=ค่ะ)|(?<=คะ)|[\n。!?]|(?<=แต่)|(?<=อย่างไรก็ตาม)")


def _sentences(text: str) -> list[str]:
    return [s for s in SENTENCE_SPLIT.split(norm_text(text) or "") if s and s.strip()]


def mentions_positively(token: str, text: str) -> bool:
    """
    คำต้องห้ามนี้ถูกพูดถึงในเชิงยืนยันจริงหรือไม่

    ดูคำปฏิเสธ "ทั้งประโยค" ที่มีคำนั้นอยู่ ไม่ใช่นับตัวอักษรก่อนหน้า
    เพราะภาษาไทยวางคำปฏิเสธไว้ต้นประโยคแล้วมีส่วนขยายคั่นก่อนถึงคำลงท้าย

        "ไม่สามารถสั่งความกว้าง 500 มม. ได้ครับ"
         ↑ ปฏิเสธ                        ↑ token ห่าง 26 ตัวอักษร

    การนับหน้าต่างแบบเดิม (14 ตัวอักษร) จึงพลาดกรณีนี้ และทำให้คำตอบที่ถูกต้อง
    ถูกนับเป็นละเมิดแบบสุ่ม ขึ้นกับว่าบอทใส่ส่วนขยายยาวแค่ไหน
    """
    token = norm_text(token)
    if not token:
        return False
    for sentence in _sentences(text):
        if token in sentence and not any(neg in sentence for neg in NEGATIONS):
            return True                       # เจอแบบยืนยัน = ละเมิดจริง
    return False


def load_image(name: str, img_dir: Path):
    """
    เปิดไฟล์รูปจากคอลัมน์ image ของ test case

    chat_interaction รับ PIL Image (เห็นได้จาก image_to_base64_html(pil_image))
    คืน (image, error) — ถ้าไม่เจอไฟล์จะคืน error เพื่อให้รายงานว่า "ข้ามเพราะไม่มีรูป"
    ไม่ใช่นับเป็นตก เพราะระบบไม่ได้ผิด
    """
    path = img_dir / name
    if not path.exists():
        return None, f"ไม่พบรูป {path}"
    try:
        from PIL import Image
        return Image.open(path).convert("RGB"), None
    except Exception as exc:                            # noqa: BLE001
        return None, f"เปิดรูปไม่ได้ {path}: {type(exc).__name__}"


def split_tokens(raw: str) -> list[str]:
    return [t.strip() for t in re.split(r"\s*\|\s*", raw or "") if t.strip()]


def norm_text(text: str) -> str:
    """ตัดตัวคั่นหลักพันและช่องว่างซ้ำ เพื่อให้ 1,000 กับ 1000 เทียบกันได้"""
    text = (text or "").lower()
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)
    return re.sub(r"\s+", " ", text)


def number_in_text(value: str, text: str) -> bool:
    """
    หาตัวเลขในคำตอบแบบยอมรับรูปแบบต่างกัน

    100000 · 100,000 · 100000.0 ถือว่าเจอเหมือนกัน
    แต่ 1000 ต้องไม่ไปแมตช์กับ 10000 จึงบังคับขอบเขตตัวเลข
    """
    try:
        num = float(value)
    except (TypeError, ValueError):
        return str(value).lower() in norm_text(text)
    body = f"{int(num)}" if num.is_integer() else f"{num}"
    return re.search(rf"(?<!\d){re.escape(body)}(?:\.0+)?(?!\d)", norm_text(text)) is not None


def looks_like_question(reply: str) -> bool:
    return any(hint in reply for hint in ASK_BACK_HINTS)


# ===============================================================
# ตรวจคำตอบหนึ่งเทิร์น
# ===============================================================


def check_reply(case: dict, reply: str, error: str | None) -> dict:
    checks: dict[str, bool] = {"no_error": error is None}
    if error is not None:
        return {"checks": checks, "passed": False, "reply": "", "error": error}

    body = norm_text(reply)

    required = split_tokens(case.get("must_contain", ""))
    if required:
        checks["answer_has_required"] = all(norm_text(t) in body for t in required)

    any_of = split_tokens(case.get("must_contain_any", ""))
    if any_of:
        checks["answer_has_any"] = any(norm_text(t) in body for t in any_of)

    forbidden = split_tokens(case.get("must_not_contain", ""))
    if forbidden:
        # ตรวจเฉพาะย่อหน้าแรก ซึ่งเป็นส่วนที่ตอบว่าได้หรือไม่ได้
        #
        # บอทมักตอบคำตัดสินก่อน แล้วค่อยแนะนำรุ่นทางเลือกในย่อหน้าถัดไป
        # เช่น "LF820 เลี้ยวไม่ได้ครับ ... แนะนำ LF880 TAB K325 รัศมี 457 mm"
        # ค่า 457 เป็นของรุ่นที่แนะนำ ไม่ใช่ของรุ่นที่ถูกถาม การตรวจทั้งคำตอบ
        # จึงลงโทษพฤติกรรมที่ดี เจตนาของเกณฑ์คือกันการอ้างค่าผิดรุ่นในคำตัดสิน
        verdict = (reply or "").split("\n\n")[0]
        checks["answer_avoids_forbidden"] = not any(
            mentions_positively(t, verdict) for t in forbidden)
        # เก็บผลแบบตรวจทั้งคำตอบไว้เป็นข้อมูล ไม่ใช้ตัดสิน
        info_forbidden_anywhere = not any(
            mentions_positively(t, reply) for t in forbidden)
    else:
        info_forbidden_anywhere = None

    if (case.get("must_ask_back") or "").strip().upper() == "TRUE":
        checks["asked_back"] = looks_like_question(reply)

    value = (case.get("expected_value") or "").strip()
    if value:
        checks["value_correct"] = number_in_text(value, reply)

    unit = (case.get("expected_unit") or "").strip()
    if unit:
        # หน่วยเขียนได้หลายแบบ เช่น C กับ องศา · N/m กับ นิวตันต่อเมตร
        variants = {unit.lower(), unit.lower().replace("/", " / ")}
        if unit.upper() == "C":
            variants |= {"°c", "องศา"}
        if unit.lower() == "mm":
            variants |= {"มม.", "มิลลิเมตร"}
        if unit.lower() == "n/m":
            variants |= {"n/m", "นิวตัน"}
        if unit == "%":
            variants |= {"เปอร์เซ็นต์"}
        checks["unit_correct"] = any(v in body for v in variants)

    limit = (case.get("reply_max_length") or "").strip()
    if limit.isdigit():
        checks["reply_within_limit"] = len(reply) <= int(limit)

    return {"checks": checks, "passed": all(checks.values()), "reply": reply,
            "error": None,
            "info": {"forbidden_absent_anywhere": info_forbidden_anywhere}}


# ===============================================================
# รันทั้งชุด
# ===============================================================


def extract_reply(chat_history: list) -> str:
    for msg in reversed(chat_history or []):
        if msg.get("role") == "assistant":
            content = msg.get("content") or ""
            if content.startswith("<img"):      # datasheet ไม่ใช่คำตอบ
                continue
            return content
    return ""


def run(module_name: str, testcases: str, only: set[str] | None,
        sleep: float, limit: int | None, img_dir: Path) -> dict:
    print(f"⏳ โหลด {module_name} ... (โหลดโมเดลและต่อ Gemini)")
    bot = importlib.import_module(module_name)
    if not hasattr(bot, "chat_interaction"):
        sys.exit(f"❌ {module_name} ไม่มีฟังก์ชัน chat_interaction")
    print("✅ พร้อม")

    rows = list(csv.DictReader(open(testcases, encoding="utf-8-sig")))
    if only:
        rows = [r for r in rows if r["id"] in only]
    if limit:
        rows = rows[:limit]

    # จัดกลุ่มตาม id แล้วเรียงตามเทิร์น เพื่อเล่นบทสนทนาต่อเนื่อง
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["id"]].append(row)
    for case_id in grouped:
        grouped[case_id].sort(key=lambda r: int(r.get("turn_number") or 1))

    results = []
    started = time.time()
    for case_id, turns in grouped.items():
        # state ของบทสนทนา รีเซ็ตทุกเคส
        history: list = []
        last_pids: list = []
        session_id = ""
        vit_signal = None
        graph_result = None

        for case in turns:
            query = case.get("user_text", "")
            img_name = (case.get("image") or "").strip()

            # เทิร์นที่เป็นรูปล้วน (user_text ว่าง) ต้องไม่ข้าม
            # ไม่งั้นบริบทของบทสนทนาจะไม่ถูกตั้ง แล้วเทิร์นถัดไปตอบผิดรุ่นตามกันหมด
            if not query and not img_name:
                continue

            image = None
            if img_name:
                image, img_err = load_image(img_name, img_dir)
                if img_err:
                    results.append({
                        "checks": {}, "passed": None, "skipped": True,
                        "skip_reason": "image_missing", "reply": "", "error": None,
                        "id": case_id, "turn": case.get("turn_number", "1"),
                        "category": case.get("category", ""), "query": query or "[รูป]",
                        "metrics": split_tokens(case.get("metrics", "")),
                        "seconds": 0.0, "note": img_err,
                    })
                    print(f"  ⬜ {case_id:<16} t{case.get('turn_number','1')} {img_err}")
                    # บริบทขาดไปแล้ว เทิร์นที่เหลือของเคสนี้วัดไม่ได้
                    break

            error = None
            reply = ""
            t0 = time.perf_counter()
            try:
                out = bot.chat_interaction(
                    query, image, history, last_pids, session_id, vit_signal, graph_result)
                history, _state, _refs, last_pids, session_id, vit_signal, graph_result = out
                reply = extract_reply(history)
            except Exception as exc:                    # noqa: BLE001
                error = f"{type(exc).__name__}: {exc}"

            outcome = check_reply(case, reply, error)
            outcome["skipped"] = False
            outcome["used_image"] = bool(img_name)
            outcome.update({
                "id": case_id,
                "turn": case.get("turn_number", "1"),
                "category": case.get("category", ""),
                "query": query,
                "metrics": split_tokens(case.get("metrics", "")),
                "seconds": round(time.perf_counter() - t0, 1),
            })
            results.append(outcome)

            mark = "⚠️ " if error else ("✅ " if outcome["passed"] else "❌ ")
            print(f"  {mark}{case_id:<16} t{outcome['turn']} {query[:38]:<40} "
                  f"{outcome['seconds']}s")
            if sleep:
                time.sleep(sleep)

    graded = [r for r in results if not r.get("skipped")]
    passed = [r for r in graded if r["passed"]]
    per_cat: dict[str, list] = defaultdict(list)
    for r in graded:
        per_cat[r["category"]].append(r["passed"])

    return {
        "module": module_name,
        "testcases": testcases,
        "total_turns": len(results),
        "graded": len(graded),
        "passed": len(passed),
        "failed": len(graded) - len(passed),
        "skipped_no_image": sum(1 for r in results if r.get("skipped")),
        "turns_with_image": sum(1 for r in graded if r.get("used_image")),
        "errors": sum(1 for r in graded if r["error"]),
        "seconds": round(time.time() - started, 1),
        "by_category": {k: f"{sum(v)}/{len(v)}" for k, v in sorted(per_cat.items())},
        "by_check": dict(Counter(
            k for r in graded for k, v in r["checks"].items() if not v)),
        "results": results,
    }


def print_summary(s: dict) -> None:
    print("\n" + "=" * 62)
    print(f"  โมดูล        {s['module']}")
    print(f"  เทิร์นทั้งหมด  {s['total_turns']}   วัดได้ {s.get('graded', s['total_turns'])}"
          f"   ผ่าน {s['passed']} · ตก {s['failed']}")
    if s.get("skipped_no_image"):
        print(f"  ⬜ ข้ามเพราะไม่มีรูป  {s['skipped_no_image']}")
    if s.get("turns_with_image"):
        print(f"  เทิร์นที่ใช้รูป      {s['turns_with_image']}")
    if s["errors"]:
        print(f"  ⚠️ error      {s['errors']}")
    print(f"  เวลา         {s['seconds']} วินาที")
    if s["by_category"]:
        print("\n  แยกตามหมวด:")
        for k, v in s["by_category"].items():
            print(f"    {k:<16} {v}")
    if s["by_check"]:
        print("\n  เกณฑ์ที่ตกบ่อยสุด:")
        for k, v in sorted(s["by_check"].items(), key=lambda x: -x[1]):
            print(f"    {k:<24} {v} ครั้ง")
    fails = [r for r in s["results"] if not r.get("skipped") and not r["passed"]]
    if fails:
        print("\n  เทิร์นที่ตก:")
        for r in fails:
            why = r["error"] or ", ".join(k for k, v in r["checks"].items() if not v)
            print(f"    {r['id']:<16} t{r['turn']}  {why}")
            print(f"        ถาม  {r['query'][:60]}")
            print(f"        ตอบ  {(r['reply'] or '')[:100]}")
    print("=" * 62)


def compare(path_a: str, path_b: str) -> None:
    a = json.loads(Path(path_a).read_text(encoding="utf-8"))
    b = json.loads(Path(path_b).read_text(encoding="utf-8"))
    ra = {r["id"] + str(r["turn"]): r for r in a["results"]}
    rb = {r["id"] + str(r["turn"]): r for r in b["results"]}

    print(f"\n{'=' * 62}")
    print(f"  ก่อน  {path_a}   ผ่าน {a['passed']}/{a['total_turns']}")
    print(f"  หลัง  {path_b}   ผ่าน {b['passed']}/{b['total_turns']}")

    changed = [(k, f"{'ผ่าน' if ra[k]['passed'] else 'ตก'} → "
                   f"{'ผ่าน' if rb[k]['passed'] else 'ตก'}")
               for k in sorted(set(ra) & set(rb)) if ra[k]["passed"] != rb[k]["passed"]]
    only_a = sorted(set(ra) - set(rb))
    only_b = sorted(set(rb) - set(ra))

    if not changed and not only_a and not only_b:
        print("\n  ✅ ไม่มีเทิร์นไหนเปลี่ยนสถานะ")
    else:
        for k, what in changed:
            print(f"    {k:<18} {what}")
        for k in only_a:
            print(f"    {k:<18} มีเฉพาะรอบแรก")
        for k in only_b:
            print(f"    {k:<18} มีเฉพาะรอบหลัง")
    print("\n  หมายเหตุ: คำตอบของ LLM ไม่คงที่ทุกรอบ การเปลี่ยนสถานะ 1-2 เทิร์น")
    print("  อาจเป็นความผันผวนปกติ ไม่ใช่ผลของการแก้โค้ดเสมอไป")
    print("=" * 62)


def main() -> None:
    ap = argparse.ArgumentParser(description="harness ชั้น B — วัดถ้อยคำของคำตอบ")
    ap.add_argument("--module", default="chatbot_v4",
                    help="โมดูลที่มี chat_interaction (เช่น chatbot_modutech)")
    ap.add_argument("--testcases", default="./testcases_v2.csv")
    ap.add_argument("--only", help="รันเฉพาะ id ที่ระบุ คั่นด้วยจุลภาค")
    ap.add_argument("--sleep", type=float, default=1.0,
                    help="พักระหว่างเทิร์น กัน rate limit")
    ap.add_argument("--img-dir", default="./img",
                    help="โฟลเดอร์เก็บรูปที่ test case อ้างถึง")
    ap.add_argument("--limit", type=int)
    ap.add_argument("-o", "--out")
    ap.add_argument("--compare", nargs=2, metavar=("ก่อน", "หลัง"))
    args = ap.parse_args()

    if args.compare:
        compare(*args.compare)
        return

    if not Path(args.testcases).exists():
        sys.exit(f"❌ ไม่พบ {args.testcases}")

    only = {x.strip() for x in args.only.split(",")} if args.only else None
    img_dir = Path(args.img_dir)
    if not img_dir.exists():
        print(f"⚠️  ไม่พบโฟลเดอร์ {img_dir} — เคสที่ต้องใช้รูปจะถูกข้าม")
    summary = run(args.module, args.testcases, only, args.sleep, args.limit, img_dir)
    print_summary(summary)

    if args.out:
        Path(args.out).write_text(json.dumps(summary, ensure_ascii=False, indent=1),
                                  encoding="utf-8")
        print(f"\n📄 {args.out}")


if __name__ == "__main__":
    main()