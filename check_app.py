#!/usr/bin/env python3
"""ตรวจโฟลเดอร์เดียวก่อนรัน — ใช้กับผังแบบรวมทุกอย่างไว้ที่โฟลเดอร์นี้

ต่างจาก check_local.py เดิมที่แยก space/ กับ web/ ตัวนี้คาดว่าไฟล์ของบอตอยู่แบน ๆ
ที่โฟลเดอร์นี้ และมีแต่ web/ ที่ยังเป็นโฟลเดอร์ย่อย (เพราะ Next.js ต้องมี package.json
ของตัวเอง) เหตุผลที่ไฟล์บอตต้องอยู่ "ที่ราก" คือ chatbot_v5.py อ้างพาธแบบ ./xxx
ซึ่งนับจากโฟลเดอร์ที่สั่งรัน ไม่ใช่ที่ไฟล์วางอยู่ — วางลึกกว่านี้แล้วเปิดจากที่อื่นจะหาไม่เจอ

สิ่งที่ตัวนี้จับได้และคนมักพลาด
  1. เอา zip เก่า (27 ก.ย.) ทับไฟล์ที่แก้วันที่ 29–30 ก.ย. — ไม่มี error บอตยังรัน แค่ตอบผิดเดิม
  2. ลืมคัดลอกฐานข้อมูลหรือโมเดล — พังตอน import ไม่ใช่ตอนตอบ จะดูเหมือนโค้ดเสีย
  3. python คนละตัวกับรอบที่รันได้ (30 ก.ย. เจอมาแล้ว prompt ขึ้น (.venv) แต่เรียกตัวหลัก)
  4. PROXY_SECRET สองฝั่งไม่ตรง — หน้าเว็บขึ้นปกติแต่พิมพ์แล้ว error

รัน:  python check_app.py
"""
from __future__ import annotations

import json
import os
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WEB = ROOT / "web"

ok: list[str] = []
warn: list[str] = []
bad: list[str] = []


def check(cond: bool, good: str, fail: str, hard: bool = True) -> bool:
    (ok if cond else (bad if hard else warn)).append(good if cond else fail)
    return cond


def read(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""


def human(n: int) -> str:
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024 or u == "GB":
            return f"{n:.0f} {u}" if u == "B" else f"{n:.1f} {u}"
        n /= 1024
    return f"{n:.1f} GB"


def dir_size(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


# ── 0. สภาพแวดล้อม ───────────────────────────────────────────
# 30 ก.ย. ระบบพังเพราะ prompt ขึ้น (.venv) แต่ python ไปเรียก interpreter ตัวหลัก
# ซึ่งมี transformers คนละเวอร์ชัน โค้ดไม่ได้เปลี่ยนเลยแต่พังตั้งแต่ import
# ที่เสียเวลาที่สุดคือการไล่หาว่าอะไรเปลี่ยน ทั้งที่ traceback บอกอยู่แล้ว
# ตัวตรวจเวอร์ชันนี้ต้องแยกสามกรณีให้ออก ไม่ใช่สองกรณี
#   · import ไม่ได้        = ไม่ได้ติดตั้งจริง → ต้องแก้
#   · import ได้ ไม่มี __version__ = ติดตั้งแล้ว แค่แพ็กเกจไม่ประกาศ → ไม่ใช่ปัญหา
#   · import ได้ มีเวอร์ชัน  = ปกติ
# รุ่นแรกเขียน __import__(mod).__version__ ใน try เดียว แล้วจับ Exception กว้าง
# qdrant_client ไม่มี __version__ (ยืนยันแล้วกับ 1.19.1) จึงโดนรายงานว่า "ไม่ได้ติดตั้ง"
# แล้วขึ้น ✗ บล็อกการรัน ทั้งที่แพ็กเกจอยู่ครบและบอตรันได้ — ตัวตรวจผิด ไม่ใช่เครื่องผิด
CHECKER_VERSION = "2"          # ขึ้นเลขนี้เมื่อวิธีอ่านเวอร์ชันเปลี่ยน
DIST_NAME = {"qdrant_client": "qdrant-client"}   # ชื่อแพ็กเกจ ≠ ชื่อโมดูล


def probe(mod: str) -> tuple[str, bool]:
    """คืน (ข้อความเวอร์ชัน, ติดตั้งแล้วหรือไม่)"""
    try:
        m = __import__(mod)
    except ImportError:
        return "(ไม่ได้ติดตั้ง)", False
    except Exception as e:                              # noqa: BLE001
        return f"(import พังแล้ว: {type(e).__name__})", False
    v = getattr(m, "__version__", None)
    if isinstance(v, str) and v:
        return v, True
    try:
        from importlib.metadata import version
        return version(DIST_NAME.get(mod, mod)), True
    except Exception:                                   # noqa: BLE001
        return "(ติดตั้งแล้ว อ่านเวอร์ชันไม่ได้)", True


print("0) สภาพแวดล้อม")
env: dict[str, str] = {"python": sys.executable}
missing_pkg: list[str] = []
for mod in ("transformers", "torch", "qdrant_client", "peft", "gradio"):
    env[mod], installed = probe(mod)
    if not installed:
        missing_pkg.append(mod)
for k, v in env.items():
    print(f"   {k:<14} {v}")

check(not missing_pkg,
      "   ✓ ไลบรารีหลักครบ",
      f"   ✗ ยังไม่ได้ติดตั้ง {', '.join(missing_pkg)} ใน python ตัวนี้\n"
      "       ถ้าเคยรันได้แปลว่ากำลังใช้ python ผิดตัว ไม่ใช่ลืมติดตั้ง")

LOCK = ROOT / "env_lock.json"
_stamp = dict(env, _checker=CHECKER_VERSION)
if LOCK.is_file():
    try:
        was = json.loads(LOCK.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        was = {}
    # ถ้าล็อกถูกบันทึกด้วยตัวตรวจรุ่นก่อน ค่าที่ต่างอาจมาจากวิธีอ่านที่เปลี่ยน
    # ไม่ใช่สภาพแวดล้อมที่เปลี่ยน — เขียนทับเงียบ ๆ ดีกว่าเตือนผิดแล้วคนเลิกเชื่อคำเตือน
    if was.get("_checker") != CHECKER_VERSION:
        LOCK.write_text(json.dumps(_stamp, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"   อัปเดต {LOCK.name} ให้ตรงกับวิธีอ่านเวอร์ชันแบบใหม่")
        was = _stamp
    diff = [k for k in env if was.get(k) != env[k]]
    check(not diff,
          "   ✓ ตรงกับสภาพแวดล้อมที่บันทึกไว้",
          "   ⚠ สภาพแวดล้อมเปลี่ยนจากที่บันทึกไว้:\n" +
          "\n".join(f"       {k}: {was.get(k)} → {env[k]}" for k in diff) +
          "\n       ถ้าพฤติกรรมเปลี่ยน ให้สงสัยตรงนี้ก่อนสงสัยโค้ด", hard=False)
else:
    LOCK.write_text(json.dumps(_stamp, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"   บันทึกไว้ที่ {LOCK.name} แล้ว รอบหน้าจะเทียบให้")

# ── 1. โค้ด ──────────────────────────────────────────────────
# ชื่อพวกนี้มาจากบรรทัด import ของ chatbot_v5.py ขาดตัวไหนพังตอน import ทันที
CODE = {
    "chatbot_v5.py":     "ตัวบอต — โหลดโมเดล ต่อ Gemini คุมบทสนทนา",
    "app.py":            "FastAPI ที่หน้าเว็บเรียก (มาจาก beltbot_web_deploy.zip)",
    "retriever_v4.py":   "ค้นข้อมูลจาก Qdrant",
    "knowledge_graph.py": "กราฟความเข้ากันได้ของ Movex",
    "pipeline_logger.py": "บันทึกทุกขั้นของการตอบ",
    "modutech_gates.py": "เกตกรองคำตอบ Modutech",
    "modutech_router.py": "แยกว่าคำถามเป็น Movex หรือ Modutech",
    "modutech_turn.py":  "ประกอบคำตอบหนึ่งเทิร์นของ Modutech",
    "dialogue_state.py": "จำบริบทข้ามเทิร์น",
    "modutech_kg.py":    "กราฟความสัมพันธ์ + ตัวโหลดที่ตรวจลายนิ้วมือ",
    "knowledge_graph.py": "คลาสกราฟที่ใช้ร่วมกับ Movex (โหลดไฟล์ .db)",
    "sales_contact.py":  "เบอร์/อีเมลฝ่ายขายตัวจริง",
}

print("\n1) ไฟล์โค้ด (ต้องอยู่แบน ๆ ที่โฟลเดอร์นี้)")
for f, why in CODE.items():
    check((ROOT / f).is_file(), f"   ✓ {f} — {why}", f"   ✗ ไม่พบ {f} ({why})")

# ── 2. ข้อมูลและโมเดลในเครื่อง ────────────────────────────────
# พาธพวกนี้อ่านจากค่าคงที่ใน chatbot_v5.py ไม่ได้เขียนทับด้วยมือ
src5 = read(ROOT / "chatbot_v5.py")


def const(name: str, default: str) -> str:
    m = re.search(rf'^{name}\s*=\s*["\']([^"\']+)', src5, re.M)
    return m.group(1) if m else default


NEEDED = [
    (const("QDRANT_PATH", "./qdrant_db"), "dir", "ฐานเวกเตอร์ Movex — ห้ามแก้ ห้ามรัน build ทับ"),
    (const("JSON_PATH", "./extracted_data_v3_machine_ready.json"), "file", "ข้อมูลสินค้า Movex"),
    (const("KG_DB_PATH", "./movex_kg.db"), "file", "กราฟ Movex"),
    (const("VIT_MODEL_PATH", "./movex-lora-tuned-3"), "dir", "ViT LoRA ที่เทรนไว้ (ใช้ตอนส่งรูป)"),
    (const("MODUTECH_JSON", "./modutech_v30_1.json"), "file", "ข้อมูล Modutech"),
    ("./qdrant_modutech_db", "dir", "ฐานเวกเตอร์ Modutech (collection แยกจาก Movex)"),
]

print("\n2) ข้อมูลและโมเดลในเครื่อง")
for rel, kind, why in NEEDED:
    p = (ROOT / rel).resolve()
    hit = p.is_dir() if kind == "dir" else p.is_file()
    size = f" [{human(dir_size(p) if kind == 'dir' else p.stat().st_size)}]" if hit else ""
    check(hit, f"   ✓ {rel}{size} — {why}", f"   ✗ ไม่พบ {rel} ({why})")

# collection ของ Qdrant แบบไฟล์ในเครื่องอยู่ใต้ <path>/collection/<ชื่อ>
# ถ้าคัดลอกมาไม่ครบ บอตจะขึ้นมาได้แต่ค้นไม่เจออะไรเลย ซึ่งดูเหมือนโมเดลโง่ ไม่เหมือนไฟล์หาย
for rel, expect in ((const("QDRANT_PATH", "./qdrant_db"), ("movex_text_chunks", "movex_images")),
                    ("./qdrant_modutech_db", ())):
    cdir = (ROOT / rel / "collection")
    if cdir.is_dir():
        got = sorted(d.name for d in cdir.iterdir() if d.is_dir())
        check(bool(got), f"   ✓ {rel} มี collection: {', '.join(got)}",
              f"   ✗ {rel}/collection ว่าง — คัดลอกมาไม่ครบ")
        miss = [c for c in expect if c not in got]
        if expect:
            check(not miss, f"   ✓ collection ของ Movex ครบ",
                  f"   ✗ {rel} ขาด {', '.join(miss)} — Movex จะค้นไม่เจอ")
    elif (ROOT / rel).is_dir():
        warn.append(f"   ⚠ {rel} ไม่มีโฟลเดอร์ collection/ — อาจเป็นผังของ Qdrant คนละรุ่น")

# ── 3. เวอร์ชันของไฟล์ที่แก้ 29–30 ก.ย. ───────────────────────
# ตรวจด้วยข้อความเฉพาะในโค้ด ไม่ใช่วันที่ไฟล์ ซึ่งเปลี่ยนตอนคัดลอกอยู่แล้ว
MARKERS = {
    "modutech_gates.py": [
        ("inch_basis", "ค่านิ้วคำนวณจาก มม. ทุกฟิลด์ (30 ก.ย.)"),
        ("NO_INCREMENT_TH", "ไม่ประกาศค่าเพิ่มทีละที่ยืนยันไม่ได้ (29 ก.ย.)"),
        ("catalogue_nonstandard_increment", "อ่านขนาดนอกมาตรฐานจากเล่ม (29 ก.ย.)"),
        ("temperature_unconfirmed_th", "คำเตือน MD254 RR (29 ก.ย.)"),
    ],
    "dialogue_state.py": [
        ("FOLLOWUP_MAX_CHARS", "คำถามต่อเนื่องสั้น ๆ ไม่ถูกตีเป็นนอกเรื่อง (29 ก.ย.)"),
        ("นิ้ว", "คำหน่วยวัดใน DOMAIN_WORDS (29 ก.ย.)"),
    ],
}

print("\n3) เวอร์ชันของไฟล์ที่แก้ 29–30 ก.ย.")
for fname, marks in MARKERS.items():
    src = read(ROOT / fname)
    if not src:
        continue
    for token, why in marks:
        check(token in src, f"   ✓ {fname}: {why}",
              f"   ✗ {fname} เป็นรุ่นเก่า — ขาด {why} (เอา zip เก่าทับ?)")

if src5:
    got = const("MODUTECH_JSON", "(ไม่พบบรรทัด)")
    check(got.endswith("modutech_v30_1.json"),
          f"   ✓ chatbot_v5.py ชี้ไป {got}",
          f"   ✗ chatbot_v5.py ชี้ไป {got} — ต้องเป็น ./modutech_v30_1.json ไม่ใช่ v30 หรือ v29")
    # torch_dtype= พังกับ transformers บางเวอร์ชัน (TypeError: hasattr) แล้วกลับมารันได้เอง
    # จึงเตือนอย่างเดียว ไม่บล็อก
    check("torch_dtype=" not in src5,
          "   ✓ ไม่ได้ส่ง torch_dtype เข้า from_pretrained",
          "   ⚠ chatbot_v5.py ส่ง torch_dtype= เข้า Jina v3 — เคยพังเมื่อ transformers อัปเดต\n"
          "       ถ้าขึ้น TypeError: hasattr(): attribute name must be string ให้ตัดออก\n"
          "       แล้วใช้ .to(device=DEVICE, dtype=...) ต่อท้ายแทน", hard=False)

# ── 4. เนื้อข้อมูล ───────────────────────────────────────────
print("\n4) เนื้อข้อมูล Modutech")
jp = ROOT / const("MODUTECH_JSON", "./modutech_v30_1.json").lstrip("./")
if jp.is_file():
    try:
        data = json.loads(jp.read_text(encoding="utf-8"))
        prods = data.get("products", [])
        n_pair = sum(len(p.get("industries") or []) for p in prods)
        n_app = sum(len(a.get("applications") or [])
                    for p in prods for a in p.get("industries") or [])
        check((len(prods), n_pair, n_app) == (68, 211, 651),
              f"   ✓ นับได้ {len(prods)} สายพาน · {n_pair} คู่ · {n_app} การใช้งาน",
              f"   ✗ นับได้ {len(prods)} · {n_pair} · {n_app} ต้องเป็น 68 · 211 · 651")
        empty = [p["belt_code"] for p in prods
                 if not (p.get("specifications", {}).get("notes") or [])]
        check(not empty, "   ✓ ทุกรุ่นมีหมายเหตุจากเล่มครบ",
              f"   ✗ มี {len(empty)} รุ่นที่ notes ว่าง — ไฟล์นี้คือ v30 เก่า ไม่ใช่ v30_1")
        md = next((p for p in prods if p["belt_code"] == "MD254 RR"), None)
        if md:
            # เกรดอยู่ใต้ specifications.variants ไม่ใช่ที่รากของสินค้า
            v = [x for x in md.get("specifications", {}).get("variants") or []
                 if x.get("variant_id") == "md254-rr--pph-pph"]
            check(bool(v) and v[0].get("temperature_trust") == "unconfirmed",
                  "   ✓ MD254 RR เกรด PPH ตั้ง temperature_trust = unconfirmed แล้ว",
                  "   ✗ MD254 RR เกรด PPH ยังไม่ได้ตั้ง unconfirmed — จะตอบ 93 °C โดยไม่เตือน")
    except json.JSONDecodeError as e:
        bad.append(f"   ✗ อ่าน {jp.name} ไม่ได้: {e}")

sc = read(ROOT / "sales_contact.py")
if sc:
    check(bool(re.search(r"\d{2,3}-\d{3,4}-?\d{3,4}|@", sc)),
          "   ✓ sales_contact.py มีช่องทางติดต่อจริง",
          "   ✗ sales_contact.py ว่าง — น่าจะโดน zip เก่าทับ ต้องใส่เบอร์/อีเมลกลับ", hard=False)

# ── 4.5 กราฟความสัมพันธ์สินค้า ─────────────────────────────────
# ไฟล์ .db เป็นสำเนาที่สองของความสัมพันธ์ใน JSON ความเสี่ยงคือแก้ JSON แล้วลืมสร้างใหม่
# ลายนิ้วมือใน metadata จับเรื่องนี้ได้ ตรวจที่นี่จะรู้ก่อนรัน ไม่ต้องไปอ่าน log ตอนสตาร์ต
print("\n4.5) กราฟความสัมพันธ์สินค้า")
KG_DB = ROOT / const("MODUTECH_KG_DB", "./modutech_kg.db").lstrip("./")
if not KG_DB.is_file():
    warn.append(f"   ⚠ ไม่พบ {KG_DB.name} — ระบบจะสร้างกราฟจาก JSON ทุกครั้งที่สตาร์ต (ยังตอบถูก)\n"
                f"       สร้างไฟล์ด้วย: python build_modutech_graph.py")
else:
    try:
        sys.path.insert(0, str(ROOT))
        from modutech_kg import (assert_no_blocked_fields, catalogue_fingerprint,
                                 db_fingerprint)
        want = catalogue_fingerprint(json.loads(jp.read_text(encoding="utf-8"))) if jp.is_file() else None
        got = db_fingerprint(str(KG_DB))
        if got is None:
            check(False, "", f"   ⚠ {KG_DB.name} ไม่มีลายนิ้วมือ — สร้างด้วย builder รุ่นเก่า\n"
                             f"       สร้างใหม่: python build_modutech_graph.py", hard=False)
        elif want and got != want:
            check(False, "", f"   ⚠ {KG_DB.name} ล้าสมัย — ไม่ตรงกับ {jp.name} ปัจจุบัน\n"
                             f"       ไฟล์ {got[:12]} · ข้อมูล {want[:12]}\n"
                             f"       สร้างใหม่: python build_modutech_graph.py", hard=False)
        else:
            ok.append(f"   ✓ {KG_DB.name} ลายนิ้วมือตรงกับข้อมูลปัจจุบัน")
        blocked = assert_no_blocked_fields(str(KG_DB))
        check(not blocked, "   ✓ กราฟไม่มีฟิลด์ที่ sprocket_gate บล็อก",
              f"   ⚠ กราฟมีฟิลด์ที่ sprocket_gate บล็อกไว้: {', '.join(blocked)}\n"
              f"       สร้างใหม่ด้วย build_modutech_graph.py รุ่นที่ตัดฟิลด์พวกนี้ออก", hard=False)
    except ImportError as e:
        warn.append(f"   ⚠ ตรวจกราฟไม่ได้: {e}")

check((ROOT / "build_modutech_graph.py").is_file(),
      "   ✓ มี build_modutech_graph.py ไว้สร้างกราฟใหม่",
      "   ⚠ ไม่มี build_modutech_graph.py — แก้ข้อมูลแล้วสร้างกราฟใหม่ไม่ได้", hard=False)

# ── 5. หน้าเว็บ ──────────────────────────────────────────────
print("\n5) หน้าเว็บ (web/)")
for f in ("package.json", "app/page.tsx", "app/layout.tsx", "app/globals.css"):
    check((WEB / f).is_file(), f"   ✓ web/{f}", f"   ✗ ไม่พบ web/{f}")
check((WEB / "node_modules").is_dir(), "   ✓ ติดตั้ง node_modules แล้ว",
      "   ✗ ยังไม่ได้ npm install ใน web/ (สคริปต์รันจะทำให้ถ้าไม่มี)", hard=False)
check(shutil.which("node") is not None, "   ✓ มี node ในเครื่อง",
      "   ✗ ไม่พบ node — ติดตั้ง Node.js 18+ ก่อน")

# ── 5.5 ชุดทดสอบที่ควรมีติดไว้ ────────────────────────────────
print("\n5.5) ชุดทดสอบ")
for f, why in (("testsuite_120.py", "136 เคส ชั้นเกต"),
               ("test_sprocket_flow.py", "ทางเดินคำถามสเตอร์ + กันขนาดหลุด"),
               ("test_modutech_kg.py", "กราฟตรงกับ JSON")):
    check((ROOT / f).is_file(), f"   ✓ {f} — {why}",
          f"   ⚠ ไม่มี {f} ({why}) รันได้แต่ไม่มีตัวจับการถดถอย", hard=False)

# ── 6. ค่าตั้ง ───────────────────────────────────────────────
print("\n6) ค่าตั้ง")


def env_of(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in read(path).splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


se, we = env_of(ROOT / ".env"), env_of(WEB / ".env.local")
check(bool(se), "   ✓ .env มีอยู่ที่รากโฟลเดอร์", "   ✗ ไม่พบ .env — คัดลอกจาก .env.example")
check(bool(we), "   ✓ web/.env.local มีอยู่", "   ✗ ไม่พบ web/.env.local — คัดลอกจาก .env.local.example")

s_secret, w_secret = se.get("PROXY_SECRET", ""), we.get("PROXY_SECRET", "")
check(bool(s_secret) and s_secret == w_secret,
      "   ✓ PROXY_SECRET สองฝั่งตรงกัน",
      "   ✗ PROXY_SECRET ไม่ตรงกันหรือยังว่าง — หน้าเว็บจะขึ้นปกติแต่พิมพ์แล้ว error")
check(we.get("SPACE_URL", "").startswith("http://127.0.0.1"),
      f"   ✓ SPACE_URL = {we.get('SPACE_URL')}",
      f"   ✗ SPACE_URL = {we.get('SPACE_URL') or '(ว่าง)'} รันในเครื่องต้องเป็น http://127.0.0.1:7860")
check(bool(se.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")),
      "   ✓ มี GEMINI_API_KEY", "   ✗ ยังไม่มี GEMINI_API_KEY ใน .env")

# ── สรุป ─────────────────────────────────────────────────────
for line in ok:
    print(line)
print()
for line in warn:
    print(line.replace("✗", "⚠"))
for line in bad:
    print(line)

print()
if bad:
    print(f"❌ ยังรันไม่ได้ — ต้องแก้ {len(bad)} จุดข้างบนก่อน")
    sys.exit(1)
print("✅ พร้อมรัน")
print("   สำคัญ: ต้องสั่ง uvicorn จาก 'โฟลเดอร์นี้' เพราะพาธในโค้ดเป็นแบบ ./xxx")
print(f"   {ROOT}")
if warn:
    print(f"   (มีข้อเตือน {len(warn)} จุด รันได้แต่ควรดู)")
sys.exit(0)