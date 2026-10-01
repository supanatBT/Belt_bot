"""
modutech_gates.py
=================
ย้าย answering rules ของ Modutech จาก prompt มาบังคับในโค้ด

หลัก: ตัดที่ context ไม่ใช่สั่งที่ prompt — ถ้าโมเดลไม่เห็นตัวเลข มันแต่งตัวเลขไม่ได้

สัญญาของทุก gate
----------------
    gate(products: list[dict], slots: Slots, ...) -> GateResult

  · products = record สินค้าดิบจาก modutech_v30.json (ที่ retriever คืนมา)
  · ไม่แก้ input — ทำบน deepcopy และเทียบ fingerprint ก่อน/หลัง (shadow copy)
  · ไม่เรียก LLM ไม่โหลดโมเดล ไม่แตะ Qdrant → แชต C เทสต์ด้วย python ล้วนได้
  · ตัดสินจากฟิลด์โครงสร้างเท่านั้น ไม่อ่านข้อความใน rules_th / forbidden_th
    (forbidden_th ใช้แค่ตรวจว่าทุกข้อมี gate รองรับ — ดู audit_forbidden_coverage)

ลำดับใน run_gates (ห้ามสลับ — temperature_gate ตัด disqualifiers ทิ้ง จึงต้องมาหลังสุด)
    disqualifier → material → width → curve → strength → source → temperature → project_for_llm

ข้อตัดสินที่ผู้ใช้ยืนยันแล้ว (2026-09-25)
  · variant ที่ wet_status = partial_data → ผ่านพร้อมคำเตือน (ไม่บล็อก)
  · ตัดวัสดุตามอุณหภูมิ → ใช้ temperature_effective ราย variant ตัดสิน
    disqualifiers ใช้ตรวจเทียบ + ยกคำเตือน registry_conflict_th / registry_text ขึ้นมา

  · rules_th ข้อ 3 vs คำถามเชิงสำรวจ → STRENGTH_POLICY = "labeled"
  · ความกว้าง 15 รุ่นที่ต่ำสุด < มาตรฐาน → ทางเลือก ข (ซ่อนค่าต่ำสุด ส่งฝ่ายขาย)
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Callable, Iterable, Literal, Optional

Condition = Literal["wet", "dry"]
LineShape = Literal["straight", "curve"]

EXPECT = {"products": 68, "pairs": 211, "applications": 651,
          "sprockets": 142, "belt_sprocket_links": 57}


# ═══════════════════════════════════════════════════════════════
# CATALOGUE — โหลดอ่านอย่างเดียว + ตรวจเวอร์ชัน + shadow copy
# ═══════════════════════════════════════════════════════════════

def fingerprint(obj) -> str:
    blob = json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class CatalogueError(RuntimeError):
    pass


class Catalogue:
    """
    ถือ modutech_v30.json ไว้ในหน่วยความจำ ไม่มีเมธอดไหนเขียนไฟล์
    ตรวจว่าเป็น v30 จากเนื้อไฟล์ (มีคีย์สเตอร์) ไม่ใช่จากชื่อไฟล์ — v29 ถูกปฏิเสธ
    """

    def __init__(self, path: str = "modutech_v30.json"):
        with open(path, encoding="utf-8") as f:
            self.data = json.load(f)
        self.path = path
        self._verify()
        self._fp = fingerprint(self.data)

        self.products: list[dict] = self.data["products"]
        self.by_code = {p["belt_code"]: p for p in self.products}
        self.by_id = {p["belt_code_id"]: p for p in self.products}
        self.sprockets = {s["sprocket_code"]: s for s in self.data["sprockets"]}
        self.links = {l["belt_code"]: l for l in self.data["belt_sprocket_links"]}

    def _verify(self) -> None:
        d = self.data
        if not ("sprockets" in d and "belt_sprocket_links" in d):
            raise CatalogueError("ไฟล์นี้ไม่ใช่ v30 (ไม่มีคีย์สเตอร์) — ห้ามใช้ v29 หรือเก่ากว่า")
        ps = d.get("products", [])
        got = {
            "products": len(ps),
            "pairs": sum(len(p.get("industries", [])) for p in ps),
            "applications": sum(len(i.get("applications", []))
                                for p in ps for i in p.get("industries", [])),
            "sprockets": len(d["sprockets"]),
            "belt_sprocket_links": len(d["belt_sprocket_links"]),
        }
        if got != EXPECT:
            raise CatalogueError(f"จำนวนไม่ตรง ได้ {got} คาดหวัง {EXPECT}")

    def assert_unchanged(self) -> None:
        if fingerprint(self.data) != self._fp:
            raise CatalogueError("ข้อมูล catalogue ในหน่วยความจำถูกแก้ระหว่างรัน")

    def records(self, codes: Iterable[str]) -> list[dict]:
        """คืน record ตาม belt_code หรือ belt_code_id (ไม่ copy — gate จะ copy เอง)"""
        out = []
        for c in codes:
            p = self.by_code.get(c) or self.by_id.get(c)
            if p is not None:
                out.append(p)
        return out


# ═══════════════════════════════════════════════════════════════
# ชนิดข้อมูลร่วม
# ═══════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Slots:
    """สิ่งที่รู้จากบทสนทนาแล้ว ตัวเติม slot เป็นคนใส่ gate ไม่เดาเอง"""
    wet_or_dry: Optional[Condition] = None
    material: Optional[str] = None            # เกรดสายพาน เช่น "POM"
    line_shape: Optional[LineShape] = None    # ไลน์ตรง / มีโค้ง
    operating_temp_c: Optional[float] = None  # อุณหภูมิใช้งานของลูกค้า (ถ้าบอก)


@dataclass
class Decision:
    gate: str
    belt_code: str
    variant_id: Optional[str]
    action: str          # strip | keep | block_variant | drop_product | pass_with_warning | mismatch
    detail: str = ""


@dataclass
class GateResult:
    products: list[dict]
    decisions: list[Decision] = field(default_factory=list)
    ask_back: list[str] = field(default_factory=list)
    notes_th: list[str] = field(default_factory=list)

    def ask(self, slot: str) -> None:
        if slot not in self.ask_back:
            self.ask_back.append(slot)

    def merge(self, other: "GateResult") -> "GateResult":
        self.products = other.products
        self.decisions += other.decisions
        for s in other.ask_back:
            self.ask(s)
        for n in other.notes_th:
            if n not in self.notes_th:
                self.notes_th.append(n)
        return self


def _gate(fn: Callable) -> Callable:
    """deepcopy input + ยืนยันว่า input ไม่ถูกแก้ (shadow copy) ครอบทุก gate"""
    def wrapper(products: list[dict], *args, **kwargs) -> GateResult:
        before = fingerprint(products)
        result = fn(copy.deepcopy(products), *args, **kwargs)
        if fingerprint(products) != before:
            raise AssertionError(f"{fn.__name__} แก้ input — ห้าม")
        return result
    wrapper.__name__ = fn.__name__
    wrapper.__doc__ = fn.__doc__
    return wrapper


def _variants(p: dict) -> list[dict]:
    return p.get("specifications", {}).get("variants", [])


def _note(p: dict, text: str) -> None:
    """คำเตือนระดับรุ่นที่ต้องไปถึง LLM แทนตัวเลขที่ถูกตัด"""
    notes = p.setdefault("_gate_notes_th", [])
    if text not in notes:
        notes.append(text)


def _in_range(t: float, lo: Optional[float], hi: Optional[float]) -> Optional[bool]:
    if lo is None or hi is None:
        return None
    return lo <= t <= hi


# ═══════════════════════════════════════════════════════════════
# GATE: disqualifiers (ทางเลือก ค)
# ═══════════════════════════════════════════════════════════════

@_gate
def disqualifier_gate(products: list[dict], slots: Slots) -> GateResult:
    """
    ตัด variant ที่อุณหภูมิใช้งานของลูกค้าอยู่นอกช่วง ก่อนส่งให้ LLM

    ตัดสินด้วย specifications.variants[].temperature_effective[สภาวะ] เท่านั้น
    answering.disqualifiers ใช้ตรวจเทียบ — ถ้าผลต่างกันบันทึก Decision(action="mismatch")
    ตรวจแล้ว: temperature_effective ไม่เคยกว้างกว่า disqualifiers (0 กรณี)
    ทางนี้จึงไม่มีวันเสนอสิ่งที่ disqualifiers ตัดทิ้ง

    ทำงานเฉพาะเมื่อรู้อุณหภูมิใช้งาน ถ้ารุ่นต้องการ wet_or_dry แต่ยังไม่รู้ → ไม่ตัด ถามกลับ
    """
    res = GateResult(products=[])
    t = slots.operating_temp_c
    if t is None:
        res.products = products
        return res

    for p in products:
        code = p["belt_code"]
        needs = "wet_or_dry" in (p.get("answering", {}).get("required_slots") or [])
        cond = slots.wet_or_dry
        if cond is None and needs:
            res.ask("wet_or_dry")
            res.products.append(p)
            continue
        cond = cond or "dry"
        disq = {x["exclude_material"]: x for x in p.get("answering", {}).get("disqualifiers") or []}
        wet_unknown = set(p.get("data_quality", {}).get("wet_condition_unknown") or [])

        kept = []
        for v in _variants(p):
            vid, mat = v.get("variant_id"), v.get("belt_material")
            eff = (v.get("temperature_effective") or {})
            rng = eff.get(cond) or {}
            ok = _in_range(t, rng.get("min_c"), rng.get("max_c"))
            unconfirmed = (cond == "wet" and (mat in wet_unknown
                                              or eff.get("wet_status") == "not_recommended"))

            # ── ตรวจเทียบกับ disqualifiers ──
            x = disq.get(mat) or {}
            xr = x.get(cond) or {}
            x_ok = _in_range(t, xr.get("below_c"), xr.get("above_c"))
            if x_ok is not None and ok is not None and x_ok != ok:
                res.decisions.append(Decision(
                    "disqualifier", code, vid, "mismatch",
                    f"{cond} {t}°C: temperature_effective={ok} disqualifiers={x_ok}"))

            if unconfirmed:
                res.decisions.append(Decision("disqualifier", code, vid, "block_variant",
                                              f"{mat} ไม่มีข้อมูลสภาวะเปียกที่ยืนยันได้"))
                _note(p, f"เกรด {mat} ยังไม่มีข้อมูลสภาวะเปียกที่ยืนยันได้ ต้องยืนยันกับฝ่ายขายก่อนเสนอ")
                continue
            if ok is False:
                # ถ้าเกรดที่ถูกตัดเป็นเกรดที่ค่ายังไม่ยืนยัน ต้องบอก ไม่ใช่เงียบ ๆ ตัดทิ้ง
                # มิฉะนั้นงานที่รับได้จริงจะหายไปโดยไม่มีใครรู้ว่าเคยมีทางเลือกนี้
                if v.get("temperature_trust") == "unconfirmed" and v.get("temperature_unconfirmed_th"):
                    _note(p, f"เกรด {mat}: {v['temperature_unconfirmed_th']}")
                res.decisions.append(Decision("disqualifier", code, vid, "block_variant",
                                              f"{t}°C อยู่นอกช่วง {cond} ของ variant นี้"))
                continue

            # ── ยกคำเตือนจาก disqualifiers ขึ้นมา (ตัวเลขใน disqualifiers จะถูกตัดทีหลัง) ──
            # registry_conflict_th ไม่ใช้ — 13 จาก 17 ข้อความเป็นกรณีตารางสเปกต่ำกว่าหน้า 347
            # ซึ่ง v30 เองบอกว่าไม่ใช่ข้อขัดแย้ง ดู real_temperature_conflict()
            if real_temperature_conflict(v):
                _note(p, CONFLICT_NOTE_TH.format(mat=mat))
            if cond == "wet" and (x.get("wet") or {}).get("registry_text"):
                _note(p, f"เกรด {mat}: แคตตาล็อกระบุสภาวะเปียกว่า '{x['wet']['registry_text']}'")
            kept.append(v)

        if not kept:
            res.decisions.append(Decision("disqualifier", code, None, "drop_product",
                                          f"ไม่มี variant ไหนผ่าน {cond} {t}°C"))
            continue
        p["specifications"]["variants"] = kept
        res.products.append(p)
    return res


# ═══════════════════════════════════════════════════════════════
# GATE: วัสดุที่แนะนำแต่ไม่ได้ผลิต (blocking_for = material)
# ═══════════════════════════════════════════════════════════════

@_gate
def material_gate(products: list[dict], slots: Slots) -> GateResult:
    """
    ลบวัสดุใน data_quality.materials_recommended_not_manufactured ออกจากทุกรายการ
    ที่ LLM จะเห็น — บล็อกเฉพาะวัสดุนั้น ไม่บล็อกทั้งรุ่น (ข้อ C)

    ถ้าลูกค้าระบุวัสดุ (slots.material) → เหลือเฉพาะ variant เกรดนั้น
    """
    res = GateResult(products=products)
    for p in products:
        code = p["belt_code"]
        dq = p.get("data_quality", {})
        not_made = set(dq.get("materials_recommended_not_manufactured") or [])
        if "material" in (dq.get("blocking_for") or []) and not not_made:
            # blocking_for บอกให้ตรวจ แต่ไม่มีรายการจริง → ไม่เดา บันทึกไว้
            res.decisions.append(Decision("material", code, None, "mismatch",
                                          "blocking_for มี material แต่รายการว่าง"))
        if not_made:
            for ind in p.get("industries", []):
                for app in ind.get("applications", []):
                    app["materials"] = [m for m in app.get("materials") or [] if m not in not_made]
            der = p.get("derived", {})
            der["materials_recommended"] = [m for m in der.get("materials_recommended") or []
                                            if m not in not_made]
            rc = p.get("retrieval_card", {})
            rc["materials"] = [m for m in rc.get("materials") or [] if m not in not_made]
            _note(p, f"รุ่นนี้ไม่ได้ผลิตเกรด {', '.join(sorted(not_made))} "
                     f"(โบรชัวร์แนะนำไว้แต่ไม่มีในสายผลิต) ห้ามเสนอ")
            res.decisions.append(Decision("material", code, None, "strip",
                                          f"ลบ {sorted(not_made)} ออกจากรายการแนะนำ"))

        if slots.material:
            want = slots.material.upper()
            if want in not_made:
                _note(p, f"ลูกค้าถามเกรด {want} ซึ่งรุ่นนี้ไม่ได้ผลิต ต้องเสนอเกรดอื่นหรือส่งฝ่ายขาย")
            vs = [v for v in _variants(p) if (v.get("belt_material") or "").upper() == want]
            if vs:
                p["specifications"]["variants"] = vs
            else:
                res.decisions.append(Decision("material", code, None, "keep",
                                              f"ไม่มี variant เกรด {want} คงทุก variant ไว้"))
                _note(p, f"รุ่นนี้ไม่มีเกรด {want}")
    return res


# ═══════════════════════════════════════════════════════════════
# GATE: ความกว้าง (blocking_for = width)
# ═══════════════════════════════════════════════════════════════

# ข้อความในเล่มมีคำแตกเป็นช่องว่างคนละตำแหน่งกันในแต่ละหน้า ที่เจอจริง:
#   "Standard belt increments 101,6 mm."      หน้า 67   ไม่แตก
#   "Standard b elt increme nts 101,6 m m."   หน้า 59   แตกที่ b|elt และ increme|nts
#   "Standard belt inc rements 50,8 mm ."     หน้า 219  แตกที่ inc|rements
#   "Standard belt increment 50 mm."          หน้า 157  เอกพจน์
# การไล่เดารูปแบบทีละแบบทำให้พลาดเงียบ ๆ (เคยพลาด 4 รุ่นกลุ่ม HC508 มาแล้ว)
# จึงยอมให้มีช่องว่างแทรกได้ทุกตำแหน่งในคำ แล้วคัดคำว่า Non-standard ออกด้วยการดูข้อความข้างหน้า
def _loose(word: str) -> str:
    return r"\s*".join(re.escape(c) for c in word)


#   "Standa rd belt in crement s 50,8 mm ."   หน้า 215  แตกที่ crement|s ด้วย จึงต้องมี \s* หน้า s
_STD_PHRASE = _loose("Standard") + r"\s*" + _loose("belt") + r"\s*" + _loose("increment") + r"\s*s?"
# ตัวเลขก็แตกได้เหมือนกัน หน้า 215/221/223 พิมพ์ "incre ments 5 0,8 mm" จึงยอมให้มีช่องว่างในตัวเลข
# แล้วค่อยเอาช่องว่างออกก่อนแปลง ถ้าแปลงไม่ได้หรือได้ค่านอกช่วงที่เป็นไปได้ถือว่าอ่านไม่ออก
_STD_INCREMENT_RE = re.compile(_STD_PHRASE + r"\s*([\d][\d,\.\s]*\d|\d)", re.IGNORECASE)
_INCREMENT_MIN_MM, _INCREMENT_MAX_MM = 1.0, 500.0

# ตรวจว่า "หัวข้อ" ค่าเพิ่มทีละมาตรฐานมีอยู่ในหมายเหตุไหม โดยไม่บังคับว่าต้องอ่านตัวเลขออก
# ใช้แยก "เล่มไม่ได้พิมพ์ไว้" ออกจาก "เล่มพิมพ์ไว้แต่เราอ่านไม่ออก" ซึ่งสองอย่างนี้ต้องทำคนละแบบ
_STD_INCREMENT_TOPIC_RE = re.compile(_STD_PHRASE, re.IGNORECASE)


# ช่องว่างในข้อความคือที่ที่ LLM เติมเอง — เจอจริงตอนเทส 2026-09-29
#   HC127 C ซ่อนค่าเพิ่มทีละมาตรฐานไว้ แต่บอตไปหยิบค่าขนาดนอกมาตรฐาน 50.8 มาเรียกว่า
#   "เพิ่มขึ้นทีละช่วงละ 50.8 มม." ซึ่งเป็นข้อความที่เราตั้งใจกันไม่ให้เกิดพอดี
#   (และต่อให้ไม่มีเลข ก็ยังคำนวณเอาเองจาก belt_widths_mm ที่ห่างกัน 50.8 ได้อยู่ดี)
# จึงต้องบอกให้ชัดว่า "ไม่รู้" พร้อมห้ามเอาค่าอื่นมาแทน ไม่ใช่แค่เอาตัวเลขออกเฉย ๆ
NO_INCREMENT_TH = ("ยังยืนยันค่าเพิ่มทีละมาตรฐานของรุ่นนี้ไม่ได้ เพราะหมายเหตุในเล่มกับ"
                   "ตารางความกว้างระบุไม่ตรงกัน ให้บอกเฉพาะรายการความกว้างที่มี "
                   "ห้ามประกาศค่าเพิ่มทีละ ห้ามคำนวณเองจากรายการ")
NON_STD_CAVEAT_TH = "คนละค่ากับค่าเพิ่มทีละมาตรฐาน ห้ามใช้แทนกัน"


def _preceded_by_non(text: str, start: int) -> bool:
    """จับ "Non-standard" ทุกแบบ รวมถึงที่มีช่องว่างหรือขีดแทรก เช่น "Non- stan dard"."""
    before = re.sub(r"[\s\-‐-―]", "", text[max(0, start - 10):start]).lower()
    return before.endswith("non")

# ข้อความ "เพิ่มทีละ N มม." ใน width_note_th ของ v30 — ใช้ถอดออกเมื่อค่าไม่ผ่านการตรวจ
_INC_CLAUSE_RE = re.compile(r"\s*เพิ่มทีละ\s*[\d.,]+\s*มม\.?")

IncrementStatus = Literal["found", "not_printed", "unreadable", "no_notes"]


def _fmt_mm(x: float) -> str:
    return f"{x:g}"


def _fmt_in(x: float) -> str:
    """นิ้วในข้อความ ใช้ทศนิยมพอดูรู้เรื่อง ค่าเต็มความละเอียดอยู่ใน belt_widths_inch"""
    return f"{round(x / 25.4, 2):g}"


def catalogue_standard_increment(p: dict) -> Optional[float]:
    """ค่า "Standard belt increments" ที่เล่มพิมพ์ใน specifications.notes (None ถ้าไม่มี)

    คงไว้เพื่อความเข้ากันได้ย้อนหลัง ตัวที่ gate ใช้จริงคือ catalogue_increment_status()
    ซึ่งแยกได้ว่า None มาจาก "เล่มไม่ได้พิมพ์" หรือ "อ่านไม่ออก"
    """
    return catalogue_increment_status(p)[1]


def catalogue_increment_status(p: dict) -> tuple[IncrementStatus, Optional[float]]:
    """คืน (สถานะ, ค่า) ของ "Standard belt increments" ในหมายเหตุของเล่ม

      found       อ่านตัวเลขออก
      not_printed มีหมายเหตุ แต่ไม่มีหัวข้อนี้อยู่ในนั้น
      unreadable  มีหัวข้อนี้ แต่รูปแบบข้อความไม่ตรงกับที่ตัวอ่านรองรับ
      no_notes    รุ่นนี้ไม่มีหมายเหตุในข้อมูลเลย — ตรวจไม่ได้ ไม่ใช่ "เล่มไม่ได้พิมพ์"

    ทำไมต้องแยก unreadable ออกมา
      เดิมทั้งสองกรณีคืน None เหมือนกัน แล้ว width_gate ตีความ None ว่า "เล่มไม่ได้ระบุ"
      จึงยอมให้แสดงค่าเพิ่มทีละที่คำนวณจากตาราง ถ้าวันหนึ่งข้อความในเล่มเปลี่ยนรูป
      (เติมโคลอน เปลี่ยนเป็นเอกพจน์ แทรกคำ) ตัวอ่านจะเงียบ ๆ กลายเป็น not_printed
      แล้วบอทจะเริ่มบอกค่าเพิ่มทีละที่อาจไม่ตรงกับเล่ม โดยไม่มีสัญญาณเตือนใด ๆ

      ทิศทางความผิดพลาดแบบนั้นสวนกับหลัก "ไม่เดา เตือนแทน" — อ่านไม่ออกต้องเงียบ ไม่ใช่ปล่อยผ่าน
    """
    notes = p.get("specifications", {}).get("notes") or []
    if not notes:
        # v30 มี 8 รุ่นที่ notes ว่างทั้งที่ในเล่มพิมพ์หมายเหตุไว้ครบ
        # ถ้าคืน not_printed จะกลายเป็นการอนุญาตให้แสดงค่าเพิ่มทีละโดยไม่มีใครตรวจ
        return "no_notes", None
    topic = False
    for n in notes:
        for m in _STD_INCREMENT_RE.finditer(n):
            if _preceded_by_non(n, m.start()):
                continue
            raw = re.sub(r"\s+", "", m.group(1)).rstrip(".,").replace(",", ".")
            try:
                val = float(raw)
            except ValueError:
                topic = True          # เจอหัวข้อแล้วแต่ตัวเลขใช้ไม่ได้ → อ่านไม่ออก
                continue
            if not (_INCREMENT_MIN_MM <= val <= _INCREMENT_MAX_MM):
                topic = True
                continue
            return "found", val
        for m in _STD_INCREMENT_TOPIC_RE.finditer(n):
            if not _preceded_by_non(n, m.start()):
                topic = True
                break
    return ("unreadable" if topic else "not_printed"), None


def catalogue_nonstandard_increment(p: dict) -> Optional[float]:
    """ค่า "Non-standard belt increments" ที่เล่มพิมพ์ (None ถ้าไม่มีหรืออ่านไม่ออก)

    ใช้เป็นหลักฐานว่าเล่มยอมรับขนาดนอกมาตรฐานจริง จึงอ้างถึงได้โดยไม่ต้องเดา
    """
    for n in p.get("specifications", {}).get("notes") or []:
        for m in _STD_INCREMENT_RE.finditer(n):
            if not _preceded_by_non(n, m.start()):
                continue
            raw = re.sub(r"\s+", "", m.group(1)).rstrip(".,").replace(",", ".")
            try:
                val = float(raw)
            except ValueError:
                continue
            if _INCREMENT_MIN_MM <= val <= _INCREMENT_MAX_MM:
                return val
    return None


@_gate
def width_gate(products: list[dict], slots: Slots) -> GateResult:
    """
    rules_th ข้อ 10: ตอบความกว้างจาก width_note_th — gate เรียบเรียงใหม่เฉพาะกรณีที่จำเป็น

      · disputed  (6 รุ่น)  → width_note_th ของ v30 ใช้ได้ ตัดค่าที่ต่ำกว่าขั้นต่ำออกจาก belt_widths_mm
      · ต่ำสุด < ขนาดมาตรฐาน (15 รุ่น, confidence = consistent) — ทางเลือก ค (ผู้ใช้ยืนยัน 2026-09-29)
          บอกตัวเลขที่เล่มพิมพ์ทั้งหมด แต่ไม่รับปากว่าสั่งได้ ให้ฝ่ายขายยืนยัน
          ค่าเพิ่มทีละ Z แสดงเฉพาะเมื่อตรงกับ "Standard belt increments" ในเล่ม (หรือเล่มไม่ได้ระบุ)

          ประวัติการตัดสินใจ — สำคัญเวลาเขียนลงเล่ม ว่าเปลี่ยนเพราะหลักฐานเปลี่ยน ไม่ใช่เปลี่ยนใจ
            25 ก.ย. เลือกทางเลือก ข = ซ่อนค่าต่ำสุด W ทั้งหมด
              เหตุผลตอนนั้น: ข้อความ "ผลิตได้ต่ำสุด W แต่เป็นการสั่งพิเศษ" ของ v30 เป็นการตีความ
              และเชื่อว่า HC127 C กับ MD254 RR ไม่มีหลักฐานเรื่องขนาดนอกมาตรฐานเลย
            29 ก.ย. เปลี่ยนเป็นทางเลือก ค หลังเติม specifications.notes ที่ v30 ทิ้งว่างไว้ 8 รุ่น
              กลายเป็นว่า 14 จาก 15 รุ่นมีบรรทัด "Non-standard belt increments" อยู่ในเล่มจริง
              (รวม HC127 C และ MD254 RR ที่เคยคิดว่าไม่มี) และขั้นนอกมาตรฐานไล่ลงมาถึง W พอดี
              เช่น HC127 C: 254 − 50.8 × 3 = 101.6 · EC508 C: 200 − 20 × 5 = 100
              W จึงไม่ใช่ตัวเลขลอย ๆ แต่เป็นค่าที่เล่มพิมพ์และมีเส้นทางไปถึงอธิบายได้
            สิ่งที่ยังห้ามพูดเหมือนเดิมคือคำว่า "ผลิตได้" และ "สั่งพิเศษ" — เล่มไม่ได้เขียนทั้งคู่
              อันแรกคือรับปากแทนโรงงาน อันที่สองคือกำหนดเงื่อนไขการสั่งแทนฝ่ายขาย

    การตรวจค่าเพิ่มทีละใช้กับ "ทุกรุ่น" ไม่ใช่เฉพาะ 15 รุ่นที่ถูกเขียนข้อความใหม่
      เดิมการตรวจอยู่ในสาขา 15 รุ่น อีก 53 รุ่นข้อความของ v30 จึงผ่านไปโดยไม่มีใครตรวจ
      ทำให้ 7 รุ่นตอบค่าที่ไม่ตรงกับ "Standard belt increments" ที่พิมพ์ในเล่ม เช่น
        · HC127 C ตอบ 50.8 ซึ่งเป็นค่า Non-standard ในเล่ม (ค่ามาตรฐานคือ 101.6)
        · HC508 C/NT/MR/PR22 ตอบ 76.2 ทั้งที่เล่มระบุค่ามาตรฐาน 50.8 (ทำให้ปฏิเสธขนาดที่สั่งได้จริง)
        · EC508 TR ตอบ 50.5 และ XP254 EVO FLT ตอบ 76.0 ซึ่งเกิดจากตัวเลขพิมพ์ผิดในตารางของเล่ม
      รายการความกว้างจริงยังส่งไปทาง belt_widths_mm ตามเดิม ตัดเฉพาะการประกาศ "ขั้น"

    EC508 TR — ผู้ใช้ตัดสิน 30 ก.ย. ว่า "ไม่ต้องแก้" ข้อมูลดิบ
      belt_widths_mm ตัวที่สามเป็น 609.9 ทั้งที่คอลัมน์นิ้วในเล่มพิมพ์ 24.00 (= 609.6)
      และบันไดของรุ่นนี้คือ 508.0 → 558.8 → 609.6 → 660.4 ทีละ 50.8 ตรงกับหมายเหตุในเล่ม
      เสนอให้แก้ 609.9 → 609.6 ใน modutech_v30_1.json ผู้ใช้ไม่รับ จึงคงของดิบไว้
      ผลที่ตามมา ยอมรับแล้ว ไม่ใช่บั๊ก
        · ค่าเพิ่มทีละของรุ่นนี้ถูกซ่อนตลอด (50.5 ที่คำนวณจากตารางไม่ตรง 50.8 ที่เล่มพิมพ์)
        · ลูกค้าที่ถามเป็นนิ้วเห็น 24.012 ปนอยู่ในรายการที่เหลือเป็นจำนวนเต็ม
      ถ้าจะแก้ในอนาคต แก้ที่ v31 ที่เดียว แล้วค่าเพิ่มทีละจะกลับมาเองโดยไม่ต้องแตะ gate
    ของดิบของ width_note_th เก็บไว้ใน Decision.detail
    """
    res = GateResult(products=products)
    for p in products:
        code = p["belt_code"]
        spec = p.get("specifications", {})
        we = spec.get("width_effective") or {}
        lo = we.get("orderable_min_mm")
        std = we.get("standard_min_mm")
        floor = std if (std is not None and lo is not None and lo < std) else lo

        widths = spec.get("belt_widths_mm") or []
        removed = [w for w in widths if floor is not None and w < floor]
        spec["belt_widths_mm"] = [w for w in widths if floor is None or w >= floor]
        # ลูกค้าไทยบางรายคิดเป็นนิ้ว ให้โค้ดแปลงให้ อย่าให้ LLM หรือลูกค้าแปลงเอง
        if spec["belt_widths_mm"]:
            spec["belt_widths_inch"] = [mm_to_inch(w) for w in spec["belt_widths_mm"]]
            spec["belt_widths_inch_basis"] = INCH_BASIS_TH
        spec.pop("minimum_width", None)
        p.get("data_quality", {}).pop("width_conflict", None)

        inc = we.get("increment_mm")
        inc_status, cat_inc = catalogue_increment_status(p)
        # แสดงค่าเพิ่มทีละได้สองกรณีเท่านั้น: เล่มมีหมายเหตุแต่ไม่ได้พิมพ์หัวข้อนี้ หรือพิมพ์ไว้แล้วตรงกัน
        # unreadable (อ่านไม่ออก) และ no_notes (ไม่มีหมายเหตุให้ตรวจ) ถือว่าไม่รู้ ต้องเงียบ
        inc_ok = inc is not None and (
            inc_status == "not_printed"
            or (inc_status == "found" and abs(cat_inc - inc) < 0.05))

        if not inc_ok and inc is not None:   # inc = None คือไม่มีค่าให้แสดงอยู่แล้ว ไม่ต้องบันทึกว่าขัดกัน
            detail = {
                "unreadable": f"อ่าน Standard belt increments ในหมายเหตุไม่ออก (ซ่อนค่า {inc} จากตารางไว้ก่อน)",
                "no_notes": f"ไม่มีหมายเหตุของรุ่นนี้ในข้อมูล ตรวจค่า {inc} จากตารางไม่ได้",
            }.get(inc_status,
                  f"เพิ่มทีละ {inc} (จากตาราง) ≠ Standard belt increments {cat_inc} (ในเล่ม)")
            res.decisions.append(Decision("width", code, None, "mismatch", detail))
            _note(p, NO_INCREMENT_TH)   # ย้ำอีกทางหนึ่ง เผื่อ LLM อ่าน width_note_th ข้าม

        non_inc = catalogue_nonstandard_increment(p)
        if floor is not None and floor != lo:          # 15 รุ่น: ต่ำสุดแคบกว่าขนาดมาตรฐาน
            parts = [f"ขนาดมาตรฐาน {_fmt_mm(std)} ถึง {_fmt_mm(we['max_mm'])} มม. "
                     f"({_fmt_in(std)}–{_fmt_in(we['max_mm'])} นิ้ว)"]
            if inc_ok:
                parts.append(f"เพิ่มทีละ {_fmt_mm(inc)} มม.")
            else:
                parts.append(NO_INCREMENT_TH)
            # ค่าต่ำสุดเป็นตัวเลขที่เล่มพิมพ์ไว้จริง บอกได้ แต่ห้ามแปลว่าสั่งได้
            below = (f"เล่มระบุความกว้างต่ำสุดของรุ่นนี้ไว้ที่ {_fmt_mm(lo)} มม. "
                     f"({_fmt_in(lo)} นิ้ว)")
            if non_inc is not None:
                below += (f" และระบุขนาดนอกมาตรฐานเพิ่มทีละ {_fmt_mm(non_inc)} มม. "
                          f"({NON_STD_CAVEAT_TH})")
            parts.append(below)
            parts.append(f"แต่เล่มไม่ได้ระบุเงื่อนไขการสั่งช่วง {_fmt_mm(lo)}–{_fmt_mm(std)} มม. "
                         f"ถ้าลูกค้าต้องการช่วงนี้ ต้องให้ฝ่ายขายยืนยันก่อน ห้ามรับปากว่าผลิตได้")
            raw = we.get("width_note_th")
            we["width_note_th"] = " ".join(parts)
            we.pop("orderable_min_mm", None)
            res.decisions.append(Decision("width", code, None, "rewrite",
                                          f"ค่าต่ำสุด {lo} บอกได้แต่ไม่รับปาก "
                                          f"(non-std ในเล่ม = {non_inc}) · ของดิบ: {raw}"))
        else:
            # 53 รุ่นที่ใช้ข้อความของ v30 ตามเดิม — ถอดเฉพาะวรรค "เพิ่มทีละ N มม." ที่ตรวจไม่ผ่าน
            note = we.get("width_note_th") or ""
            if not inc_ok and _INC_CLAUSE_RE.search(note):
                we["width_note_th"] = f"{_INC_CLAUSE_RE.sub('', note).strip()} {NO_INCREMENT_TH}"
                res.decisions.append(Decision("width", code, None, "strip",
                                              f"ถอดวรรคเพิ่มทีละออก · ของดิบ: {note}"))
            if removed:
                _note(p, "ความกว้างที่แคบกว่าค่าขั้นต่ำที่ยืนยันได้ต้องยืนยันกับฝ่ายเทคนิคก่อน ห้ามรับปาก")
                res.decisions.append(Decision("width", code, None, "strip",
                                              f"ตัดความกว้าง {removed} ที่ต่ำกว่า {lo}"))
    return res


# ═══════════════════════════════════════════════════════════════
# GATE: ความแข็งแรงตอนโค้ง (forbidden_th "รุ่นนี้วิ่งโค้งได้ ...")
# ═══════════════════════════════════════════════════════════════

@_gate
def curve_gate(products: list[dict], slots: Slots) -> GateResult:
    """
    รุ่นที่มี belt_strength_curve (9 รุ่น) ต้องรู้ก่อนว่าไลน์ตรงหรือมีโค้ง
      ไม่รู้  → ตัดทั้ง belt_strength และ belt_strength_curve + ถามกลับ line_shape
      ตรง    → เหลือ belt_strength
      โค้ง    → เหลือ belt_strength_curve (ย้ายมาแทนที่ belt_strength ไม่ได้ — ตั้งชื่อแยก)
    """
    res = GateResult(products=products)
    for p in products:
        vs = _variants(p)
        if not any("belt_strength_curve" in v for v in vs):
            continue
        code = p["belt_code"]
        shape = slots.line_shape
        for v in vs:
            if shape is None:
                v.pop("belt_strength", None)
                v.pop("belt_strength_curve", None)
            elif shape == "straight":
                v.pop("belt_strength_curve", None)
            else:
                v.pop("belt_strength", None)
        p.get("derived", {}).pop("strength_n_per_m", None)
        if shape is None:
            res.ask("line_shape")
            _note(p, "รุ่นนี้วิ่งโค้งได้ ค่าความแข็งแรงตอนโค้งต่ำกว่าวิ่งตรงมาก ต้องรู้ว่าไลน์ตรงหรือมีโค้งก่อน")
        res.decisions.append(Decision("curve", code, None, "strip", f"line_shape={shape}"))
    return res


# ═══════════════════════════════════════════════════════════════
# GATE: ความแข็งแรง / น้ำหนัก ก่อนรู้วัสดุ (rules_th ข้อ 3)
# ═══════════════════════════════════════════════════════════════

# "labeled" = ยังไม่รู้วัสดุ ให้เห็นได้เฉพาะค่าที่ผูกกับเกรดราย variant ห้ามค่ารวมระดับรุ่น
# "strict"  = ยังไม่รู้วัสดุ ตัดทุกค่า (ตามตัวอักษรของ rules_th ข้อ 3)
# ผู้ใช้เลือก labeled (2026-09-25) — strict ทำให้คำถามเชิงสำรวจ (ข้อ D) ตอบไม่ได้
STRENGTH_POLICY: Literal["labeled", "strict"] = "labeled"


@_gate
def strength_gate(products: list[dict], slots: Slots,
                  policy: Literal["labeled", "strict"] = None) -> GateResult:
    policy = policy or STRENGTH_POLICY
    res = GateResult(products=products)
    for p in products:
        der = p.get("derived", {})
        der.pop("strength_n_per_m", None)     # ค่ารวม min/max ไม่ผูกวัสดุ
        der.pop("weight_kg_per_m2", None)
        if slots.material or policy == "labeled":
            continue
        for v in _variants(p):
            for k in ("belt_strength", "belt_strength_curve", "belt_weight"):
                v.pop(k, None)
        res.ask("material")
        res.decisions.append(Decision("strength", p["belt_code"], None, "strip",
                                      "ยังไม่รู้วัสดุ (policy=strict)"))
    return res


# ═══════════════════════════════════════════════════════════════
# GATE: แหล่งตัวเลข (rules_th ข้อ 1, 2 + unit_conversion)
# ═══════════════════════════════════════════════════════════════

_NUM_WITH_UNIT = re.compile(
    r"\d[\d,]*(?:\.\d+)?\s*(?:มม\.?|มิลลิเมตร|mm|นิ้ว|inch|in\.|N/m|นิวตัน|kg|กก\.?|°\s*[CF]|องศา|%|เปอร์เซ็นต์)",
    re.IGNORECASE)
_ANGLE_OK = re.compile(r"90\s*(?:°|องศา)")   # "90° transfer" เป็นชื่องาน ไม่ใช่ค่าสเปก

_PRINTED_KEYS = ("catalogue_printed", "printed", "belt_code_printed", "surface_type_printed")


def _mask_numbers(text: str) -> str:
    def repl(m):
        return m.group(0) if _ANGLE_OK.fullmatch(m.group(0)) else "[ดูสเปก]"
    return _NUM_WITH_UNIT.sub(repl, text)


def _drop_keys(obj, keys: tuple[str, ...]):
    if isinstance(obj, dict):
        for k in keys:
            obj.pop(k, None)
        for v in obj.values():
            _drop_keys(v, keys)
    elif isinstance(obj, list):
        for v in obj:
            _drop_keys(v, keys)


INCH_BASIS_TH = "คำนวณจาก มม. (ค่านิ้วในเล่มมีที่พิมพ์ผิด ระบบไม่ใช้)"


def mm_to_inch(mm: Optional[float]) -> Optional[float]:
    return None if mm is None else round(mm / 25.4, 3)


def _recompute_inch(obj):
    """ค่าอิมพีเรียลในเล่มมีที่พิมพ์ผิด — แทนที่ด้วยค่าที่คำนวณจาก mm ไม่ใช่ตัดทิ้งเฉย ๆ

    เดิมตัดทิ้งอย่างเดียว (_drop_inch) ผลคือลูกค้าถามเป็นนิ้วแล้วบอตตอบไม่ได้
    แล้วลูกค้าต้องไปแปลงเอง ซึ่งเสี่ยงคลาดเคลื่อนกว่าให้โค้ดแปลงให้ (พบตอนเทส 2026-09-29)
    docstring เดิมเขียนไว้เองว่า "ถ้าต้องใช้ให้คำนวณจาก mm ในโค้ด" — นี่คือส่วนนั้น
    """
    if isinstance(obj, dict):
        mm = obj.get("mm")
        if isinstance(mm, (int, float)):
            obj["inch"] = mm_to_inch(mm)
            obj["inch_basis"] = INCH_BASIS_TH
        elif "mm" in obj:
            obj.pop("inch", None)
        for v in list(obj.values()):
            _recompute_inch(v)
    elif isinstance(obj, list):
        for v in obj:
            _recompute_inch(v)


@_gate
def source_gate(products: list[dict], slots: Slots) -> GateResult:
    """
    ตัวเลขต้องมาจาก specifications เท่านั้น
      · sales_knowledge / sales_knowledge_th → ตัวเลขที่มีหน่วยถูกแทนด้วย [ดูสเปก]
      · ตัดฟิลด์ catalogue_printed / printed และค่า inch ทุกระดับ
    """
    res = GateResult(products=products)
    for p in products:
        for key in ("sales_knowledge", "sales_knowledge_th"):
            sk = p.get(key)
            if isinstance(sk, dict):
                for k, v in sk.items():
                    if isinstance(v, str):
                        sk[k] = _mask_numbers(v)
        _drop_keys(p, _PRINTED_KEYS)
        _recompute_inch(p.get("specifications", {}))
    return res


# ═══════════════════════════════════════════════════════════════
# GATE: อุณหภูมิ (required_slots = wet_or_dry, blocking_for = wet)
# ═══════════════════════════════════════════════════════════════

_VARIANT_TEMP_KEYS = ("temperature", "temperature_effective", "temperature_note_th")
_PRODUCT_TEMP_PATHS = (
    ("derived", "temperature_envelope"),
    ("answering", "disqualifiers"),
    ("data_quality", "temperature_disputed_pairs"),
    ("data_quality", "temperature_f_typo_pairs"),
)


CONFLICT_NOTE_TH = ("เกรด {mat}: ตารางสเปกระบุอุณหภูมิสูงกว่าคุณสมบัติวัสดุในแคตตาล็อก ระบบใช้ค่าที่ต่ำกว่า "
                    "ต้องยืนยันกับฝ่ายเทคนิคก่อนรับปาก")


def real_temperature_conflict(v: dict) -> bool:
    """
    ข้อขัดแย้งจริง = ตารางสเปกสูงกว่าคุณสมบัติวัสดุ (pending_vendor_confirmation
    'spec-exceeds-material-limit') ตัดสินจากตัวเลขใน source_layers ไม่ใช่ข้อความ
    ตารางสเปกต่ำกว่า = ข้อจำกัดของพื้นผิว/พิน ไม่ใช่ข้อขัดแย้ง (detail_th ของข้อเดียวกัน)
    เจอ 5 variant ตรงกับรายการใน pending_vendor_confirmation ทุกตัว รวม HP508 RR
    ที่ disqualifiers ไม่ได้ติดธงไว้
    """
    layers = (v.get("temperature_effective") or {}).get("source_layers") or {}
    spec = (layers.get("spec_table") or {}).get("max_c")
    mat = (layers.get("belt_material") or {}).get("dry")
    return spec is not None and bool(mat) and spec > mat[1]


def c_to_f(c: Optional[float]) -> Optional[float]:
    """unit_conversion.factors.celsius_to_fahrenheit — ยึด °C เสมอ ไม่อ่าน °F จากเล่ม"""
    return None if c is None else round(c * 9 / 5 + 32, 1)


F_TYPO_TOLERANCE = 6.0   # เล่มปัด °F เป็นหลักสิบ (41→40, 199.4→200) ต่างเกินนี้ถือว่าพิมพ์ผิด


def audit_f_typos(catalogue: "Catalogue") -> list[dict]:
    """
    เทียบ °F ที่พิมพ์ในเล่ม (catalogue_printed.temperature_f) กับ °F ที่คำนวณจาก °C
    v30: 34 variant ผิด (ยืนยันด้วยตาที่ EC254 C หน้า 131 — คอลัมน์ PE/POM)
    ทั้งที่ data_quality.temperature_f_typo_pairs ว่างทุกรุ่น
    """
    out = []
    for p in catalogue.products:
        for v in _variants(p):
            printed = (v.get("catalogue_printed") or {}).get("temperature_f")
            t = v.get("temperature") or {}
            if not printed or t.get("min_c") is None:
                continue
            pf = [float(x) for x in re.findall(r"[-+]?\d+(?:\.\d+)?", printed)][:2]
            calc = [c_to_f(t["min_c"]), c_to_f(t["max_c"])]
            wrong = [side for side, a, b in zip(("min", "max"), pf, calc) if abs(a - b) > F_TYPO_TOLERANCE]
            if wrong:
                out.append({"belt_code": p["belt_code"], "variant_id": v["variant_id"],
                            "printed_c": (v.get("catalogue_printed") or {}).get("temperature_c"),
                            "printed_f": printed, "computed_f": {"min_f": calc[0], "max_f": calc[1]},
                            "wrong_side": wrong})
    return out


def _pop_path(obj: dict, path: tuple[str, ...]) -> bool:
    for key in path[:-1]:
        obj = obj.get(key) or {}
    return obj.pop(path[-1], None) is not None


@_gate
def temperature_gate(products: list[dict], slots: Slots) -> GateResult:
    """
    ① ตัดค่าดิบ + envelope + disqualifiers เสมอ (rules_th ข้อ 9)
    ② ยังไม่รู้เปียก/แห้ง และรุ่นต้องการ slot → ไม่เหลือตัวเลขอุณหภูมิเลย + ถามกลับ
    ③ เปียก: blocking_for ชี้ว่าต้องตรวจ → บล็อกเฉพาะ variant ใน wet_condition_unknown (ข้อ C)
       wet_status = not_recommended → บล็อก variant
       wet_status = partial_data   → ผ่านพร้อมคำเตือน (ผู้ใช้ยืนยัน 2026-09-25)
    ④ เหลือเฉพาะสภาวะที่ลูกค้าบอก จาก temperature_effective เก็บไว้ที่ variant.temperature_answer
       ผูกกับเกรดเสมอ (forbidden_th "ห้ามตอบช่วงอุณหภูมิเป็นค่าเดียวโดยไม่ระบุวัสดุ")
    """
    res = GateResult(products=products)
    for p in products:
        code = p.get("belt_code", "?")
        needs_slot = "wet_or_dry" in (p.get("answering", {}).get("required_slots") or [])
        warning = (p.get("derived", {}).get("temperature_envelope") or {}).get("warning_th")

        removed = [".".join(path) for path in _PRODUCT_TEMP_PATHS if _pop_path(p, path)]
        if warning:
            p.setdefault("derived", {})["temperature_warning_th"] = warning

        condition = slots.wet_or_dry
        if needs_slot and condition is None:
            for v in _variants(p):
                for k in _VARIANT_TEMP_KEYS:
                    if v.pop(k, None) is not None:
                        removed.append(f"variants[{v.get('variant_id')}].{k}")
            res.decisions.append(Decision("temperature", code, None, "strip",
                                          "ต้องรู้ wet_or_dry ก่อน"))
            res.ask("wet_or_dry")
            continue

        condition = condition or "dry"   # 9 รุ่นที่ไม่มี slot: PP ล้วน เปียก = แห้ง
        dq = p.get("data_quality", {})
        wet_unknown = set(dq.get("wet_condition_unknown") or [])
        check_wet = condition == "wet" and "wet" in (dq.get("blocking_for") or [])

        for v in _variants(p):
            vid, mat = v.get("variant_id"), v.get("belt_material")
            eff = v.pop("temperature_effective", None) or {}
            v.pop("temperature", None)
            v.pop("temperature_note_th", None)

            if check_wet and mat in wet_unknown:
                v["temperature_answer"] = {"condition": "wet", "material": mat,
                                           "status": "confirm_with_sales"}
                res.decisions.append(Decision("temperature", code, vid, "block_variant",
                                              f"{mat} อยู่ใน wet_condition_unknown"))
                continue
            if condition == "wet" and eff.get("wet_status") == "not_recommended":
                v["temperature_answer"] = {"condition": "wet", "material": mat,
                                           "status": "not_recommended_wet"}
                res.decisions.append(Decision("temperature", code, vid, "block_variant",
                                              "wet_status = not_recommended"))
                continue
            if v.get("temperature_trust") == "disputed":
                v["temperature_answer"] = {"condition": condition, "material": mat,
                                           "status": "confirm_with_tech"}
                res.decisions.append(Decision("temperature", code, vid, "block_variant",
                                              "temperature_trust = disputed"))
                continue

            rng = eff.get(condition) or {}
            status = eff.get("wet_status") if condition == "wet" else "ok"
            answer = {"condition": condition, "material": mat,
                      "pin_material": v.get("pin_material"),
                      "min_c": rng.get("min_c"), "max_c": rng.get("max_c"),
                      "min_f": c_to_f(rng.get("min_c")), "max_f": c_to_f(rng.get("max_c")),
                      "f_basis": "คำนวณจาก °C (°F ในเล่มมีที่พิมพ์ผิด)",
                      "status": status}
            action = "keep"
            if real_temperature_conflict({"temperature_effective": eff}):
                _note(p, CONFLICT_NOTE_TH.format(mat=mat))
            if condition == "wet" and status == "partial_data":
                answer["note_th"] = eff.get("wet_note_th", "")
                action = "pass_with_warning"
            # ค่าที่ใช้ตอบได้แต่ยังไม่ยืนยัน — ตอบตัวเลขที่เข้มกว่าไว้ พร้อมคำเตือนที่ผูกกับเกรด
            # ต่างจาก disputed ตรงที่ disputed ไม่ให้ตัวเลขเลย
            unconf = v.pop("temperature_unconfirmed_th", None)
            if v.get("temperature_trust") == "unconfirmed":
                answer["status"] = f"{status}_unconfirmed"
                action = "pass_with_warning"
                if unconf:
                    answer["note_th"] = unconf
                    _note(p, f"เกรด {mat}: {unconf}")
            v["temperature_answer"] = answer
            res.decisions.append(Decision("temperature", code, vid, action, f"status={status}"))
    return res


# ═══════════════════════════════════════════════════════════════
# GATE: สเตอร์ (blocking_for = sprocket_code / sprocket_dimension / sprocket_material)
# ═══════════════════════════════════════════════════════════════

def sprocket_gate(sprockets: list[dict], links: list[dict] = ()) -> GateResult:
    """
    sprocket_dimension บล็อกเสมอ (เล่มไม่นิยาม Di/Do/A/B และ & ในรูเพลายังตีความไม่ได้)
      → ตัด dimensions / bore / dimension_note_th ทิ้ง ส่งฝ่ายขาย (ตรงกับ escalate "สเปกเฟือง", "ขนาดเพลา")
    sprocket_code → คงรหัสไว้ แต่ติดคำว่าอนุมานจากซีรีส์ ต้องยืนยัน
    sprocket_material → เหลือเฉพาะรหัสที่มี suffix วัสดุพิมพ์ในเล่ม
    ไม่แก้ input
    """
    before = fingerprint([list(sprockets), list(links)])
    sp = copy.deepcopy(list(sprockets))
    ln = copy.deepcopy(list(links))
    res = GateResult(products=[])
    for s in sp:
        for k in ("dimensions", "bore", "dimension_note_th"):
            s.pop(k, None)
        _drop_keys(s, _PRINTED_KEYS)
        if s.get("material") is None:
            s["material"] = "เล่มไม่ระบุ ต้องยืนยันกับฝ่ายขาย"
        # บอกเหตุผลให้ชัด ไม่ใช่แค่บอกผล
        #
        # ข้อความเดิมคือ "ขนาดสเตอร์และรูเพลาต้องขอจากฝ่ายขาย" ซึ่งไม่ได้บอกว่าทำไม
        # LLM จึงเติมเหตุผลเอง แล้วตอบว่า "แคตตาล็อกไม่ได้ระบุไว้" (เจอใน graph_run2
        # เคส SPK-D01) — ซึ่งไม่จริง เล่มพิมพ์ Di 68,5 / Do 78,3 / รูเพลา 25-40 ไว้ที่หน้า 52
        # ถ้ากรรมการเปิดเล่มตามจะเจอตัวเลขอยู่ตรงนั้น
        #
        # เหตุผลจริงคือเล่มไม่ได้นิยามว่าแต่ละสัญลักษณ์เป็นระยะอะไร (di_do_abcex_undefined)
        # จึงยังเอาไปตอบไม่ได้ ไม่ใช่ว่าไม่มีข้อมูล
        _note(s, "เล่มพิมพ์ตัวเลขขนาดไว้ แต่ไม่ได้นิยามว่า Di Do A B คือระยะอะไร "
                 "ระบบจึงยังไม่ส่งค่าออกมา ต้องให้ฝ่ายขายยืนยันก่อน "
                 "ห้ามบอกว่าแคตตาล็อกไม่มีข้อมูลขนาด")
        res.decisions.append(Decision("sprocket", s["sprocket_code"], None, "strip", "dimensions,bore"))
    for l in ln:
        _note(l, l.get("answerable_th", {}).get("รหัสสเตอร์")
              or "รหัสสเตอร์อนุมานจากซีรีส์ ต้องยืนยันกับฝ่ายขาย")
    res.products = sp + ln
    if fingerprint([list(sprockets), list(links)]) != before:
        raise AssertionError("sprocket_gate แก้ input — ห้าม")
    return res


# ═══════════════════════════════════════════════════════════════
# ESCALATE TO SALES — ตอบจาก template ไม่ผ่าน LLM
# ═══════════════════════════════════════════════════════════════

# หัวข้อ = ค่าใน answering.escalate_to_sales (เหมือนกันทั้ง 68 รุ่น)
# คำพ้อง: ระวัง "เท่าไหร่" คำเดียว — "กว้างเท่าไหร่" ไม่ใช่คำถามราคา
ESCALATE_PATTERNS: dict[str, list[str]] = {
    "ราคา": [r"ราคา", r"กี่บาท", r"\bprice\b", r"\bcost\b", r"quotation", r"ใบเสนอราคา",
             r"ส่วนลด", r"discount"],
    "lead time": [r"lead\s*time", r"กี่วัน(?:ได้|ส่ง|ถึง)", r"ส่งของ", r"ระยะเวลา(?:ส่ง|ผลิต)",
                  r"(?:ได้|ส่ง)(?:ของ)?(?:เมื่อไ(?:หร่|ร))", r"delivery"],
    "MOQ": [r"\bmoq\b", r"สั่งขั้นต่ำ", r"ขั้นต่ำกี่", r"minimum\s*order"],
    "สต็อก": [r"สต็อก", r"สต๊อก", r"\bstock\b", r"ของพร้อมส่ง", r"มีของ(?:ไหม|มั้ย|หรือเปล่า)"],
    "สเปกเฟือง": [r"สเปก(?:เฟือง|สเตอร์)", r"ขนาด(?:เฟือง|สเตอร์)", r"pitch\s*diameter",
                  r"เส้นผ่าน(?:ศูนย์กลาง)?(?:เฟือง|สเตอร์)", r"\bP\.?D\.?\b", r"\bO\.?D\.?\b"],
    "ขนาดเพลา": [r"เพลา", r"\bshaft\b", r"\bbore\b", r"รูเพลา", r"รูสเตอร์"],
    "ความกว้างสั่งทำพิเศษ": [r"สั่งทำ(?:พิเศษ)?", r"ทำพิเศษ", r"custom\s*width", r"กว้างพิเศษ"],
    "รัศมีวงเลี้ยว": [r"รัศมี", r"radius", r"วงเลี้ยว", r"turning"],
    "อะไหล่": [r"อะไหล่", r"spare\s*part", r"\bspare\b"],
}
_ESC_RE = {k: re.compile("|".join(v), re.IGNORECASE) for k, v in ESCALATE_PATTERNS.items()}

# คำที่ติดมากับหัวข้อส่งฝ่ายขาย — ลบทิ้งด้วยก่อนดูว่ายังเหลือคำถามสเปกไหม
# ("สั่งทำความกว้าง 700" ไม่ใช่คำถามสเปกความกว้าง · "สเตอร์รูเพลาขนาดเท่าไหร่" ไม่ใช่คำถามสเปกสเตอร์)
_ESC_COMPANION = {
    "ความกว้างสั่งทำพิเศษ": r"(?:ความ)?กว้าง",
    "รัศมีวงเลี้ยว": r"เลี้ยว|โค้ง",
    "สเปกเฟือง": r"สเตอร์|เฟือง|ขนาด",
    "ขนาดเพลา": r"สเตอร์|เฟือง|ขนาด",
}
# คำถามสเปกที่บอทตอบเองได้ (ทั้งสองแบรนด์) — ใช้ตัดสินว่าเป็นคำถามผสมหรือไม่
SPEC_QUESTION_RX = re.compile(
    r"กว้าง|ยาว|หนา|น้ำหนัก|หนัก|แรงดึง|รับแรง|โหลด|\bload\b|workload|strength|"
    r"อุณหภูมิ|องศา|ร้อน|เย็น|แช่แข็ง|°|วัสดุ|ทำจาก|เกรด|สเปก|\bspec|pitch|พิทช์|"
    r"ช่องเปิด|open\s*area|เลี้ยว|โค้ง|ผิว|เปียก|แห้ง|\bFDA\b|ใช้กับ|เหมาะ|ผลิตยังไง|สเตอร์ตัวไหน|ฟัน",
    re.IGNORECASE)

# Movex: หัวข้อที่เป็นเรื่องการค้าล้วน (รัศมีวงเลี้ยว/สเตอร์ของ Movex มีในฐานข้อมูล ตอบเองได้)
MOVEX_ESCALATE_TOPICS = ("ราคา", "lead time", "MOQ", "สต็อก")

SALES_CONTACT_TH = "ฝ่ายขาย"   # ช่องทางติดต่อจริงอยู่ใน sales_contact.py


@dataclass
class Escalation:
    topics: list[str]
    reply_th: str               # ข้อความเต็ม — ใช้เมื่อคำถามมีแต่เรื่องฝ่ายขาย (ไม่เรียก LLM)
    handoff_th: str = ""        # ข้อความสั้น — ต่อท้ายคำตอบสเปกเมื่อเป็นคำถามผสม
    has_spec_question: bool = False


def _spec_remainder(user_text: str, topics: list[str]) -> str:
    rest = user_text or ""
    for t in topics:
        rest = _ESC_RE[t].sub(" ", rest)
        if t in _ESC_COMPANION:
            rest = re.sub(_ESC_COMPANION[t], " ", rest, flags=re.IGNORECASE)
    return rest


def escalation_gate(user_text: str, belt_code: Optional[str] = None,
                    allowed_topics: Optional[Iterable[str]] = None) -> Optional[Escalation]:
    """
    คืน Escalation ถ้าคำถามแตะหัวข้อต้องส่งฝ่ายขาย
      has_spec_question = False → chatbot ตอบ reply_th ตรง ไม่เรียก LLM
      has_spec_question = True  → คำถามผสม: ตอบสเปกตามปกติ แล้วโค้ดต่อท้ายด้วย handoff_th
    ข้อความฝ่ายขายสร้างในโค้ดเสมอ LLM ไม่ได้เขียนเอง
    """
    from sales_contact import sales_contact_line_th
    allowed = set(allowed_topics) if allowed_topics else set(ESCALATE_PATTERNS)
    topics = [t for t, rx in _ESC_RE.items() if t in allowed and rx.search(user_text or "")]
    if not topics:
        return None
    contact = sales_contact_line_th()
    subject = f"สายพาน {belt_code} " if belt_code else ""
    topic_txt = "/".join(topics)
    topic_txt = (" " + topic_txt + " ") if topic_txt[0].isascii() else topic_txt
    reply = (f"เรื่อง{topic_txt}ของ{subject}ต้องให้{SALES_CONTACT_TH}ยืนยันโดยตรงครับ "
             f"เพราะขึ้นกับรายละเอียดงานและเงื่อนไขการสั่งซื้อ "
             f"รบกวนแจ้งรุ่นสายพาน ความกว้าง ความยาว และจำนวนที่ต้องการ "
             f"ผมจะส่งเรื่องต่อให้ฝ่ายขายติดต่อกลับครับ")
    handoff = (f"ส่วนเรื่อง{topic_txt}{('ของ' + subject.strip()) if subject else ''} "
               f"ต้องให้{SALES_CONTACT_TH}ยืนยันโดยตรงครับ รบกวนแจ้งความกว้าง ความยาว และจำนวนที่ต้องการ "
               f"แล้วฝ่ายขายจะติดต่อกลับครับ")
    if contact:
        reply += "\n" + contact
        handoff += "\n" + contact
    spec = bool(SPEC_QUESTION_RX.search(_spec_remainder(user_text, topics)))
    return Escalation(topics=topics, reply_th=reply, handoff_th=handoff, has_spec_question=spec)


# ═══════════════════════════════════════════════════════════════
# PROJECTION — whitelist ฟิลด์ที่ส่งให้ LLM
# ═══════════════════════════════════════════════════════════════

_VARIANT_KEEP = ("variant_id", "belt_material", "pin_material", "material_th",
                 "belt_strength", "belt_strength_curve", "belt_weight", "temperature_answer")


def project_for_llm(p: dict) -> dict:
    """
    ชั้นสุดท้าย: ส่งเฉพาะฟิลด์ใน whitelist ฟิลด์ใหม่ที่เพิ่มใน v31+ จะไม่หลุดเข้า prompt
    จนกว่าจะเพิ่มชื่อที่นี่ — ตั้งใจให้ปิดไว้ก่อน
    """
    spec = p.get("specifications", {})
    rc = p.get("retrieval_card", {})
    we = spec.get("width_effective") or {}
    out = {
        "belt_code": p.get("belt_code"),
        "aliases": rc.get("aliases") or [],
        "belt_series": p.get("belt_series"),
        "surface_type": rc.get("surface_type"),
        "pitch_mm": (spec.get("pitch") or {}).get("mm"),
        "open_area_pct": spec.get("open_area_pct"),
        "open_area_note": spec.get("open_area_note"),
        "belt_thickness": spec.get("belt_thickness"),
        "approved": spec.get("approved"),
        "color": spec.get("color"),
        "cleanability": spec.get("cleanability"),
        "curve_capable": bool(spec.get("has_curve_rating")),
        "width_note_th": we.get("width_note_th"),
        "belt_widths_mm": spec.get("belt_widths_mm"),
        "belt_widths_inch": spec.get("belt_widths_inch"),
        "belt_widths_inch_basis": spec.get("belt_widths_inch_basis"),
        "materials": rc.get("materials") or [],
        "variants": [{k: v[k] for k in _VARIANT_KEEP if k in v} for v in _variants(p)],
        "industries": [
            {"industry_th": i.get("industry_th"),
             "surface_type": i.get("surface_type"),
             "highlights": (i.get("highlights") or {}).get("bullets"),
             "applications": [{"application_th": a.get("application_th"),
                               "materials": a.get("materials")}
                              for a in i.get("applications") or []]}
            for i in p.get("industries", [])
        ],
        "sales_knowledge_th": p.get("sales_knowledge_th"),
        "temperature_warning_th": (p.get("derived") or {}).get("temperature_warning_th"),
        "forbidden_th": (p.get("answering") or {}).get("forbidden_th") or [],
        "gate_notes_th": p.get("_gate_notes_th") or [],
    }
    for v in out["variants"]:
        for k in ("belt_strength", "belt_strength_curve", "belt_weight"):
            if isinstance(v.get(k), dict):
                v[k] = {kk: vv for kk, vv in v[k].items() if kk in ("n_per_m", "kg_per_m2", "unit", "note_th")}
    return out


# ═══════════════════════════════════════════════════════════════
# PIPELINE
# ═══════════════════════════════════════════════════════════════

GATE_ORDER = (disqualifier_gate, material_gate, width_gate, curve_gate,
              strength_gate, source_gate, temperature_gate)


def run_gates(products: list[dict], slots: Slots) -> GateResult:
    """ผ่านทุก gate ตามลำดับ แล้ว project — ผลลัพธ์พร้อม json.dumps เข้า prompt"""
    before = fingerprint(products)
    res = GateResult(products=products)
    for gate in GATE_ORDER:
        res.merge(gate(res.products, slots))
    res.products = [project_for_llm(p) for p in res.products]
    if fingerprint(products) != before:
        raise AssertionError("run_gates แก้ input — ห้าม")
    return res


# ═══════════════════════════════════════════════════════════════
# AUDIT — forbidden_th ทุกข้อต้องมี gate รองรับ
# ═══════════════════════════════════════════════════════════════

FORBIDDEN_COVERAGE: list[tuple[str, str]] = [
    (r"ห้ามตอบอุณหภูมิสูงสุดโดยไม่รู้ว่าไลน์เปียกหรือแห้ง", "temperature_gate ②"),
    (r"ห้ามตอบช่วงอุณหภูมิเป็นค่าเดียวโดยไม่ระบุวัสดุ", "temperature_gate ④ (ผูกเกรดราย variant)"),
    (r"ห้ามเสนอวัสดุ .+ กับรุ่นนี้", "material_gate"),
    (r"ห้ามเสนอเกรด .+ สำหรับไลน์เปียก", "temperature_gate ③ not_recommended"),
    (r"รุ่นนี้วิ่งโค้งได้ ต้องถามก่อนว่าเป็นไลน์ตรงหรือมีโค้ง", "curve_gate"),
    (r"ความกว้างแคบสุดตอบได้ที่ .+ ห้ามรับปาก", "width_gate"),
]


def audit_forbidden_coverage(catalogue: Catalogue) -> dict:
    """คืน {"covered": n, "uncovered": [(belt_code, text)]} — uncovered ต้องว่างเสมอ"""
    covered, uncovered = 0, []
    for p in catalogue.products:
        for f in p.get("answering", {}).get("forbidden_th") or []:
            if any(re.search(rx, f) for rx, _ in FORBIDDEN_COVERAGE):
                covered += 1
            else:
                uncovered.append((p["belt_code"], f))
    return {"covered": covered, "uncovered": uncovered}