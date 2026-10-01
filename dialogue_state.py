"""
dialogue_state.py
=================
ทาง จ — ตัดสินในโค้ดว่าจะ "ตอบ / ถามกลับ / ปฏิเสธ / ส่งฝ่ายขาย" ก่อนถึง LLM ทั้งสองแบรนด์

เดิม Movex ตัดสินผ่าน prompt ("ขั้นตอนที่ 1: ลูกค้าระบุรุ่นชัดแล้วหรือยัง") ย้ายมาไว้ที่นี่
หลักเดียวกับ gate ของ Modutech: สิ่งที่เป็นเงื่อนไขความถูกต้อง ต้องบังคับในโค้ด

ลำดับการตัดสินในหนึ่งเทิร์น
    รูปที่ระบบปฏิเสธ → reject_image
    ทักทาย/ขอบคุณ    → smalltalk
    ไม่เกี่ยวกับสินค้า  → refuse_offtopic
    แบรนด์ (slot 0)   → ask brand ถ้ายังไม่รู้
    Movex    : รหัสรุ่น / ซีรีส์+ความกว้าง / รูป (ViT)  → answer หรือ ask
    Modutech : ส่งฝ่ายขาย → ค้นทั้งฐาน → รหัสกำกวม → เปียก/แห้ง → ตรง/โค้ง → ความต้องการ → answer

ไม่ import torch / qdrant / gradio / genai — เทสต์ด้วย python ล้วน
action ที่เป็น template (ทุกอย่างยกเว้น answer และ search) ตอบได้โดยไม่เรียก LLM
"""
from __future__ import annotations

import copy
import re
from dataclasses import asdict, dataclass, field
from typing import Optional

from modutech_gates import Catalogue, escalation_gate, MOVEX_ESCALATE_TOPICS
from modutech_router import BrandRouter, CatalogueSearch, norm_code

# ═══════════════════════════════════════════════════════════════
# ชนิดข้อมูล
# ═══════════════════════════════════════════════════════════════


@dataclass
class VitInfo:
    """สรุปจาก ViTSignal ของ RetrieverV4 (ไม่ import ตัวจริง เพื่อให้เทสต์ได้)"""
    series: Optional[str] = None
    confidence: float = 0.0
    dominant: bool = False
    top_pid: str = ""


@dataclass
class DialogState:
    """เก็บใน gr.State เป็น dict (to_dict / from_dict)"""
    brand: Optional[str] = None
    movex_pids: list = field(default_factory=list)
    movex_series: Optional[str] = None
    modutech_codes: list = field(default_factory=list)
    modutech_slots: dict = field(default_factory=dict)   # wet_or_dry, material, line_shape, operating_temp_c
    modutech_sprocket_codes: list = field(default_factory=list)   # รหัสสเตอร์ของเทิร์นนี้ ไม่สะสมข้ามเทิร์น
    pending: Optional[str] = None                        # สิ่งที่ถามค้างไว้
    pending_question: str = ""                           # คำถามเดิมของลูกค้าก่อนถูกถามกลับ

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "DialogState":
        # กรองคีย์ที่คลาสไม่รู้จักออก — state ที่บันทึกจากโค้ดรุ่นอื่นจะไม่ทำให้พังทั้งสองแบรนด์
        return cls(**{k: v for k, v in d.items()
                     if k in cls.__dataclass_fields__}) if d else cls()


@dataclass
class Decision:
    action: str                # reject_image | smalltalk | refuse_offtopic | ask | escalate | cross_brand | search | answer
    brand: Optional[str]
    reply_th: str = ""         # มีค่าเมื่อเป็น template — ตอบได้โดยไม่เรียก LLM
    reason: str = ""
    topic: Optional[str] = None
    effective_text: str = ""   # คำถามที่รวมคำตอบของการถามกลับแล้ว
    movex: dict = field(default_factory=dict)      # pids, series, level
    modutech: dict = field(default_factory=dict)   # codes, slots, search_query
    state: DialogState = field(default_factory=DialogState)
    # คำถามผสม (สเปก + ราคา ฯลฯ): โค้ดต่อท้ายข้อความนี้หลังคำตอบ — LLM ไม่ได้เขียนส่วนนี้
    sales_handoff_th: str = ""
    sales_topics: list = field(default_factory=list)

    @property
    def needs_llm(self) -> bool:
        return self.action in ("answer", "search")


# ═══════════════════════════════════════════════════════════════
# คำศัพท์
# ═══════════════════════════════════════════════════════════════

# คำที่บอกว่าเป็นคำถามเรื่องสินค้า (ไม่มีคำพวกนี้ + ไม่มีรหัส = นอกเรื่อง)
DOMAIN_WORDS = [
    "สายพาน", "โซ่", "สเตอร์", "เฟือง", "เพลา", "พิน", "ก้าน", "ลำเลียง", "คอนเวเยอร์",
    "belt", "chain", "sprocket", "conveyor", "movex", "modutech", "โมดูล",
    "กว้าง", "ความยาว", "หนา", "workload", "โหลด", "รับน้ำหนัก", "แรงดึง", "น้ำหนัก", "แข็งแรง",
    "วัสดุ", "ทำจาก", "เลี้ยว", "โค้ง", "วิ่งตรง", "รัศมี", "ผลิต", "ทน", "อุณหภูมิ", "องศา",
    "ร้อน", "เย็น", "แช่แข็ง", "เปียก", "แห้ง", "drawing", "datasheet", "ดรออิ้ง", "แบบ", "สเปก", "spec",
    "ราคา", "สต็อก", "สต๊อก", "ส่งของ", "อะไหล่", "ผิว", "ไลน์", "รุ่น", "series", "ซีรีส์",
    "ฟัน", "pitch", "พิทช์", "เบอร์", "ขนาด",
    # หน่วยวัด — ลูกค้าถามเปลี่ยนหน่วยบ่อยมาก และเดิมถูกตีเป็นคำถามนอกเรื่อง
    # (พบตอนเทส 2026-09-29: "ขอให้ตอบเป็นหน่วยนิ้วได้ไหม" ถูกปฏิเสธ ทั้งที่คุยรุ่น HC127 C ค้างอยู่)
    "นิ้ว", "inch", "มิลลิเมตร", "เซนติเมตร", "มม", "ซม", "ฟุต", "feet", "หน่วย", "เมตร",
]

# คำถามต่อเนื่องสั้น ๆ หลังจากมีรุ่นอยู่ในโฟกัสแล้ว ให้ถือว่าเป็นเรื่องสินค้า
# เหตุผล: ลูกค้าที่เพิ่งถามสเปกของรุ่นหนึ่งแล้วพิมพ์สั้น ๆ ต่อ ("แล้วราคาล่ะ" "อันไหนดีกว่า"
# "ส่งของกี่วัน") กำลังถามถึงรุ่นนั้น การปฏิเสธว่านอกเรื่องทำให้เสียงานโดยไม่จำเป็น
# ไม่กระทบเคสนอกเรื่องจริง (CLARIF-002 "อากาศวันนี้เป็นยังไง") เพราะเป็นเทิร์นแรก ยังไม่มีรุ่นในโฟกัส
FOLLOWUP_MAX_CHARS = 40
SMALLTALK_WORDS = ["สวัสดี", "ขอบคุณ", "ขอบใจ", "hello", "hi ", "thank", "thanks", "ดีครับ", "ดีค่ะ"]
REFERENCE_WORDS = ["ตัวนั้น", "ตัวนี้", "อันนั้น", "อันนี้", "รุ่นนั้น", "รุ่นนี้", "ตัวที่", "อันที่",
                   "ตัวแรก", "ตัวหลัง", "เมื่อกี้", "ข้างบน", "ตัวเดิม"]

BRAND_ANSWERS = {
    "movex": [r"movex", r"มูฟเว็กซ์", r"โซ่", r"top\s*chain", r"ท็อปเชน", r"^\s*1\s*$"],
    "modutech": [r"modutech", r"โมดูเทค", r"โมดูลาร์", r"modular", r"สายพานโมดูล", r"^\s*2\s*$"],
}

# หัวข้อคำถาม Movex → ฟิลด์ที่ต้องใช้ตอบ (None = ตอบระดับซีรีส์ได้เสมอ)
MOVEX_TOPICS = [
    ("sprocket", r"สเตอร์|sprocket|เฟือง", None),
    ("curve", r"เลี้ยว|โค้ง|sideflex|curve|รัศมี|radius", "product_type"),
    ("material", r"วัสดุ|ทำจาก|material", "Material"),
    ("process", r"ผลิต|process|manufactur", None),
    ("workload", r"workload|โหลด|รับแรง|รับน้ำหนัก|load", "Max_Working_Load_N"),
    ("width", r"กว้าง|width", "Plate_Width_mm"),
    ("weight", r"น้ำหนัก|หนักเท่า|weight|kg", "Weight_kg_m"),
    ("drawing", r"drawing|datasheet|ดรออิ้ง|ขอรูป|ขอดูแบบ|แสดงแบบ", "product_id"),
]

STAINLESS_RX = r"สแตนเลส|stainless|(?<![A-Za-z])SS(?![A-Za-z])"
MOD_TEMP_TOPIC = r"อุณหภูมิ|องศา|°\s*[cC]|ร้อน|เย็น|แช่แข็ง|ฟรีซ|freez|temp|ทนความร้อน|ทนเย็น"
MOD_STRENGTH_TOPIC = r"แรงดึง|ความแข็งแรง|รับแรง|strength|N\s*/\s*m|รับน้ำหนัก|โหลด"
MOD_RECOMMEND = r"แนะนำ|ใช้รุ่นไหน|ใช้อะไรดี|ตัวไหนดี|รุ่นไหนดี|เหมาะกับ|มีอะไรบ้าง|สายพานอะไร"


def _has(pattern: str, text: str) -> bool:
    return bool(re.search(pattern, text or "", re.IGNORECASE))


def _any_word(words, text: str) -> Optional[str]:
    low = (text or "").lower()
    return next((w for w in words if w.lower() in low), None)


# ═══════════════════════════════════════════════════════════════
# ดึง slot ของ Modutech จากข้อความ
# ═══════════════════════════════════════════════════════════════

def extract_modutech_slots(text: str, materials: list[str], mask_codes: tuple = ()) -> dict:
    """ดึง slot จากข้อความ — mask_codes คือรหัสสินค้าที่ต้องลบออกก่อนหา slot

    ทำไมต้อง mask: รหัสสเตอร์มีวัสดุต่อท้าย เช่น EC127SQZ24*PA ตัว regex ของวัสดุ
    ใช้ (?<![A-Za-z])PA(?![A-Za-z]) ซึ่ง * ไม่ใช่ตัวอักษร จึงจับ PA ในรหัสได้
    ผลคือระบบคิดว่าลูกค้าต้องการเกรด PA ทั้งที่แค่พิมพ์รหัสสเตอร์มาถาม
    แล้วไปกรองสายพานผิดโดยไม่มีใครเห็น
    """
    t = text or ""
    for code in mask_codes:
        if code:
            t = re.sub(re.escape(code), " ", t, flags=re.IGNORECASE)
    out: dict = {}
    if _has(r"เปียก|ล้าง|ฉีดน้ำ|น้ำ|wet|washdown|ชื้น", t) and not _has(r"ไม่เปียก|ไม่โดนน้ำ", t):
        out["wet_or_dry"] = "wet"
    if _has(r"แห้ง|\bdry\b|ไม่เปียก|ไม่โดนน้ำ", t):
        out["wet_or_dry"] = "dry"
    if _has(r"ไลน์ตรง|วิ่งตรง|straight|ไม่มีโค้ง|ไม่เลี้ยว", t):
        out["line_shape"] = "straight"
    elif _has(r"มีโค้ง|วิ่งโค้ง|เลี้ยว|โค้ง|curve|radius", t):
        out["line_shape"] = "curve"
    for m in materials:                                 # ยาวก่อน: POM-NL ก่อน POM
        if re.search(rf"(?<![A-Za-z]){re.escape(m)}(?![A-Za-z])", t, re.IGNORECASE):
            out["material"] = m.upper()
            break
    m = re.search(r"(-?\d+(?:\.\d+)?)\s*(?:°\s*C|องศา|℃)", t)
    if m:
        out["operating_temp_c"] = float(m.group(1))
    return out


# ═══════════════════════════════════════════════════════════════
# DIALOGUE MANAGER
# ═══════════════════════════════════════════════════════════════


class DialogueManager:
    VIT_ANSWER = 0.90      # ค่าเดิมใน prompt ของ Movex
    VIT_SERIES = 0.50

    def __init__(self, movex_products: dict, modutech: Catalogue, movex_materials: Optional[dict] = None):
        self.mx = movex_products
        self.mx_materials = movex_materials or {}
        self.mod = modutech
        self.router = BrandRouter(modutech)
        self.search = CatalogueSearch(modutech)
        self.materials = self.search.materials
        self._index_movex()

    # ─────────────────────────── ดัชนี Movex ───────────────────────────

    def _index_movex(self) -> None:
        self.mx_ref: dict[str, str] = {}
        self.mx_art: dict[str, str] = {}
        self.mx_series_members: dict[str, list[str]] = {}
        for pid, p in self.mx.items():
            if p.get("Ref"):
                key = norm_code(p["Ref"])
                self.mx_ref[key] = pid
                self.mx_ref.setdefault(key.replace("TAB", ""), pid)      # "LF880 K450" ไม่มี TAB
            if p.get("Art_Nr"):
                self.mx_art[str(p["Art_Nr"])] = pid
            ser = self._series_of(p)
            if ser and "chain" in pid.lower():
                self.mx_series_members.setdefault(ser, []).append(pid)
        self._mx_keys = sorted(self.mx_ref, key=len, reverse=True)

    @staticmethod
    def _series_of(p: dict) -> Optional[str]:
        m = re.search(r"\d+", str(p.get("series", "")))
        return m.group() if m else None

    def movex_series_uniform(self, series: str, fieldname: Optional[str]) -> bool:
        """ทุกรุ่นในซีรีส์มีค่าฟิลด์นี้เท่ากัน → ตอบระดับซีรีส์ได้โดยไม่ต้องรู้ความกว้าง"""
        if fieldname is None:
            return True
        vals = {str(self.mx[p].get(fieldname)) for p in self.mx_series_members.get(series, [])}
        return len(vals) == 1

    def resolve_movex(self, text: str) -> dict:
        t = text or ""
        key = norm_code(t)
        pids: list[str] = []
        for k in self._mx_keys:
            if k in key and self.mx_ref[k] not in pids:
                pids.append(self.mx_ref[k])
                key = key.replace(k, " ")
        for art in re.findall(r"(?<!\d)5[4-9]\d{3}(?!\d)", t):
            if art in self.mx_art and self.mx_art[art] not in pids:
                pids.append(self.mx_art[art])
        series = None
        m = re.search(r"(?:LFN?\s*)?(?<!\d)(820|821|880|882)(?!\d)", t, re.IGNORECASE) \
            or re.search(r"LFN\s*(83|103)|(?<!\d)(83|103)\s*(?:MF[MH]|series|ซีรีส์)", t, re.IGNORECASE)
        if m:
            series = next(g for g in m.groups() if g)
        if not pids and series:
            pids = self.movex_pick_width(series, t)
        return {"pids": pids, "series": series}

    def movex_pick_width(self, series: str, text: str) -> list[str]:
        """เลือกรุ่นในซีรีส์จาก K-code หรือความกว้าง (ใช้ตอนลูกค้าตอบคำถาม "ความกว้างไหน")"""
        k = re.search(r"K\s?(\d{3,4})", text or "", re.IGNORECASE)
        nums = [float(n) for n in re.findall(r"(?<![\d.])(\d{2,3}(?:\.\d)?)(?![\d.])", text or "")]
        out = []
        for pid in self.mx_series_members.get(series, []):
            p = self.mx[pid]
            if k and pid.upper().endswith("K" + k.group(1)):
                out.append(pid)
            elif not k and any(abs(float(p.get("Plate_Width_mm", 0)) - n) < 0.05 for n in nums):
                out.append(pid)
        return out

    # ─────────────────────────── API หลัก ───────────────────────────

    def decide(self, text: str, has_image: bool = False, vit: Optional[VitInfo] = None,
               image_rejected: bool = False, state: Optional[DialogState] = None) -> Decision:
        st = copy.deepcopy(state) if state else DialogState()
        text = (text or "").strip()

        if image_rejected:
            return Decision("reject_image", "movex", REJECT_IMAGE_TH, "ViT/SigLIP ไม่พบสินค้า",
                            effective_text=text, state=st)

        # ── คำตอบของการถามกลับรอบก่อน: รวมกับคำถามเดิม ──
        effective = text
        if st.pending and st.pending_question:
            if st.pending == "brand":
                for brand, pats in BRAND_ANSWERS.items():
                    if any(_has(p, text) for p in pats):
                        st.brand = brand
                        break
            effective = f"{st.pending_question} {text}".strip()
        st.pending, pending_before, st.pending_question = None, st.pending, ""

        # ── ทักทาย / นอกเรื่อง ──
        route = self.router.route(effective, has_image=has_image, prev_brand=st.brand,
                                  no_signal_policy="ask")
        movex_codes = self.resolve_movex(effective)
        has_code = bool(route.codes.belt_codes or route.codes.sprocket_codes or route.codes.candidates
                        or movex_codes["pids"] or movex_codes["series"])
        # มีรุ่นค้างอยู่ในบทสนทนา + พิมพ์สั้น = ถามต่อเรื่องรุ่นนั้น ไม่ใช่เปลี่ยนเรื่อง
        in_focus = bool(st.brand) and bool(st.modutech_codes or st.movex_pids or st.movex_series)
        followup = in_focus and len(text) <= FOLLOWUP_MAX_CHARS
        domain = has_code or has_image or bool(_any_word(DOMAIN_WORDS, effective)) \
            or bool(_any_word(self.router.weak_terms, effective)) or pending_before is not None \
            or _has(STAINLESS_RX, effective) or followup
        if not domain:
            if _any_word(SMALLTALK_WORDS, text):
                return Decision("smalltalk", st.brand, SMALLTALK_TH, "ทักทาย", effective_text=effective, state=st)
            if not (st.brand and _any_word(REFERENCE_WORDS, text)):
                return Decision("refuse_offtopic", st.brand, OFFTOPIC_TH, "ไม่มีคำเกี่ยวกับสินค้า",
                                effective_text=effective, state=st)

        # ── slot 0: แบรนด์ ──
        if route.need_clarify and route.brand is None and route.reason == "สัญญาณทั้งสองแบรนด์":
            st.pending, st.pending_question = "brand", effective
            return Decision("ask", None, route.clarify_th, "เจอรหัสของทั้งสองแบรนด์",
                            topic="brand", effective_text=effective, state=st)
        brand = route.brand if route.brand else st.brand
        if route.brand and st.brand and route.brand != st.brand:
            # เปลี่ยนแบรนด์กลางบทสนทนา — ล้าง slot ของแบรนด์เดิมที่ไม่เกี่ยว
            st.movex_pids, st.movex_series = [], None
            st.modutech_codes, st.modutech_slots = [], {}
        if brand is None and _has(STAINLESS_RX, effective):
            # คำถามเรื่องวัสดุสแตนเลสโดยไม่บอกแบรนด์ → ตอบจากข้อมูลทั้งสองแบรนด์ (ผู้ใช้เลือก 2026-09-25)
            return Decision("cross_brand", None, self.stainless_both_brands_th(), "SS ไม่ระบุแบรนด์ → ตอบทั้งสองแบรนด์",
                            topic="material", effective_text=effective, state=st)
        if brand is None:
            st.pending, st.pending_question = "brand", effective
            return Decision("ask", None, ASK_BRAND_TH, "ยังไม่รู้แบรนด์", topic="brand",
                            effective_text=effective, state=st)
        st.brand = brand

        if brand == "movex":
            return self._decide_movex(effective, has_image, vit, st, movex_codes)
        return self._decide_modutech(effective, st, route.codes)

    # ─────────────────────────── ข้ามแบรนด์ ───────────────────────────

    def stainless_both_brands_th(self) -> str:
        """สร้างจากข้อมูลจริงของทั้งสองแบรนด์ทุกครั้ง ไม่ฮาร์ดโค้ดรายชื่อรุ่น"""
        mx_chain_mats = sorted({str(p.get("Material")) for pid, p in self.mx.items()
                                if "chain" in pid.lower() and p.get("Material")})
        mx_ss = [p.get("Ref") or pid for pid, p in self.mx.items()
                 if "SS" in {str(p.get("Material")), str(p.get("Pivot_material"))}]
        lines = ["สินค้าที่เกี่ยวกับสแตนเลส (SS) ของทั้งสองแบรนด์ครับ", "",
                 "**Movex (โซ่ท็อปเชน)**"]
        if mx_ss:
            lines.append("• รุ่นที่ใช้ SS: " + ", ".join(mx_ss))
        else:
            lines.append(f"• ไม่มีรุ่นที่ตัวโซ่ทำจากสแตนเลส — ตัวโซ่ผลิตจาก {', '.join(mx_chain_mats)} "
                         f"(LF = Low Friction Acetal)")
        ss_note = (self.mx_materials.get("SS") or {}).get("technical_specs", {}).get("application")
        if ss_note:
            lines.append(f"• ตารางวัสดุของ Movex ระบุ SS ไว้ว่า \"{ss_note}\" แต่ข้อมูลรายรุ่นไม่ได้ระบุวัสดุพิน "
                         f"ต้องยืนยันกับฝ่ายขาย")
        res = self.search.search(__import__("modutech_router").ExploratoryQuery(pin_material="SS"))
        lines += ["", "**Modutech (สายพานโมดูลาร์)**"]
        if res["matches"]:
            lines.append(f"• มี {res['matched']} รุ่นที่ใช้ก้าน (พิน) สแตนเลส — ตัวสายพานเป็นพลาสติก:")
            for m in res["matches"]:
                mats = ", ".join(f"{h['belt_material']}/{h['pin_material']}" for h in m["variants"])
                lines.append(f"  – {m['belt_code']} ({m['surface_type']}) เกรด {mats}")
        else:
            lines.append("• ไม่มีรุ่นที่ใช้ชิ้นส่วนสแตนเลส")
        lines += ["", "สนใจแบรนด์ไหนหรือรุ่นไหน บอกได้เลยครับ จะให้รายละเอียดสเปกเพิ่ม"]
        return "\n".join(lines)

    # ─────────────────────────── Movex ───────────────────────────

    def _movex_topic(self, text: str) -> tuple[Optional[str], Optional[str]]:
        for name, rx, fld in MOVEX_TOPICS:
            if _has(rx, text):
                return name, fld
        return None, None

    def _decide_movex(self, text: str, has_image: bool, vit: Optional[VitInfo],
                      st: DialogState, codes: dict) -> Decision:
        esc = escalation_gate(text, allowed_topics=MOVEX_ESCALATE_TOPICS)
        if esc and not esc.has_spec_question:
            ref = None
            pid = (codes["pids"] or st.movex_pids or [None])[0]
            if pid and pid in self.mx:
                ref = self.mx[pid].get("Ref") or pid
            esc = escalation_gate(text, belt_code=ref, allowed_topics=MOVEX_ESCALATE_TOPICS)
            if pid:
                st.movex_pids = list(codes["pids"] or st.movex_pids)
            return Decision("escalate", "movex", esc.reply_th.replace("สายพาน ", "โซ่ "),
                            f"หัวข้อ {esc.topics}", topic="escalate", effective_text=text, state=st)
        d = self._decide_movex_core(text, has_image, vit, st, codes)
        if esc:
            ref = (self.mx.get((d.state.movex_pids or [None])[0]) or {}).get("Ref")
            esc = escalation_gate(text, belt_code=ref, allowed_topics=MOVEX_ESCALATE_TOPICS)
            _attach_handoff(d, esc.handoff_th.replace("สายพาน ", "โซ่ "), esc.topics)
        return d

    def _decide_movex_core(self, text: str, has_image: bool, vit: Optional[VitInfo],
                           st: DialogState, codes: dict) -> Decision:
        topic, fld = self._movex_topic(text)
        pids, series, why = codes["pids"], codes["series"], "รหัสในข้อความ"

        if not pids and has_image and vit is not None:
            if vit.series and vit.confidence >= self.VIT_ANSWER and vit.dominant and vit.top_pid:
                pids, series, why = [vit.top_pid], vit.series, f"ViT {vit.confidence:.2f} dominant"
            elif vit.series and vit.confidence >= self.VIT_SERIES:
                series, why = vit.series, f"ViT {vit.confidence:.2f} รู้แค่ซีรีส์"
            else:
                # ViT < 0.5 = ไม่มั่นใจว่ารูปเป็นสินค้าเลย → ปฏิเสธรูป ไม่ยกรหัสตัวอย่าง
                # (IMG-003: รูปที่ไม่ใช่โซ่ ระบบค้นได้ผลแต่คะแนนต่ำ — ต้องไม่ตอบรหัสรุ่นใด ๆ)
                return Decision("reject_image", "movex", REJECT_IMAGE_TH,
                                f"ViT {vit.confidence:.2f} ต่ำกว่า {self.VIT_SERIES}",
                                topic=topic, effective_text=text, state=st)

        if not pids and not series and st.movex_series and not st.movex_pids:
            picked = self.movex_pick_width(st.movex_series, text)
            if picked:
                pids, why = picked, "ลูกค้าเลือกความกว้าง"
        if pids:
            st.movex_pids = pids
            st.movex_series = self._series_of(self.mx[pids[0]]) or series
        elif series:
            if series != st.movex_series:
                st.movex_pids = []
            st.movex_series = series
        else:
            why = "ต่อจากเทิร์นก่อน"

        pids, series = st.movex_pids, st.movex_series
        base = dict(pids=pids, series=series, topic=topic)
        if pids:
            return Decision("answer", "movex", reason=why, topic=topic, effective_text=text,
                            movex={**base, "level": "product"}, state=st)
        if series:
            if topic and self.movex_series_uniform(series, fld):
                return Decision("answer", "movex", reason=f"{why} · {topic} เท่ากันทั้งซีรีส์",
                                topic=topic, effective_text=text,
                                movex={**base, "level": "series"}, state=st)
            st.pending, st.pending_question = "movex_width", text
            return Decision("ask", "movex", self.ask_movex_width_th(series, from_image=has_image),
                            f"{why} · ต้องรู้ความกว้าง", topic=topic, effective_text=text,
                            movex={**base, "level": "series"}, state=st)
        st.pending, st.pending_question = "movex_model", text
        return Decision("ask", "movex", self.ask_movex_model_th(topic), "ยังไม่รู้รุ่น",
                        topic=topic, effective_text=text, state=st)

    def ask_movex_width_th(self, series: str, from_image: bool = False) -> str:
        members = self.mx_series_members.get(series, [])
        ptype = self.mx[members[0]].get("product_type", "") if members else ""
        lines = [f"{'จากรูปเป็น' if from_image else 'รุ่น'} {series} Series ({ptype}) ครับ มีความกว้างให้เลือก:"]
        for pid in sorted(members, key=lambda x: self.mx[x].get("Plate_Width_mm", 0)):
            p = self.mx[pid]
            lines.append(f"• {p.get('Plate_Width_mm'):g} มม. — {p.get('Ref')}")
        lines.append("ต้องการความกว้างไหนครับ")
        return "\n".join(lines)

    def ask_movex_model_th(self, topic: Optional[str]) -> str:
        by_type: dict[str, list[str]] = {}
        for ser, members in self.mx_series_members.items():
            by_type.setdefault(self.mx[members[0]].get("product_type", ""), []).append(ser)
        opts = " · ".join(f"{t}: {', '.join(sorted(s))}" for t, s in sorted(by_type.items()))
        return ("ขอทราบรุ่นก่อนนะครับ เพื่อให้ตอบได้ตรง\n"
                "• รหัสรุ่น เช่น LF 820 K325 หรือรหัสสเตอร์ 5 หลัก\n"
                "• หรือซีรีส์ + ความกว้าง เช่น 880 กว้าง 82.5 มม.\n"
                "• หรือส่งรูปโซ่มาให้ดูครับ\n"
                f"ซีรีส์ที่มี — {opts}")

    # ─────────────────────────── Modutech ───────────────────────────

    def _decide_modutech(self, text: str, st: DialogState, codes) -> Decision:
        # ลบรหัสที่ router จับได้ออกก่อนหา slot ไม่งั้นวัสดุที่ต่อท้ายรหัสสเตอร์
        # จะถูกอ่านเป็นความต้องการของลูกค้า (EC127SQZ24*PA → material=PA)
        st.modutech_slots.update(extract_modutech_slots(
            text, self.materials,
            mask_codes=tuple(codes.belt_codes) + tuple(codes.sprocket_codes)))
        slots = st.modutech_slots
        if codes.belt_codes:
            st.modutech_codes = list(codes.belt_codes)
        # รหัสสเตอร์ไม่เก็บลง state เพราะไม่ใช่ "รุ่นที่อยู่ในความสนใจ" แต่ต้องส่งต่อ
        # ให้ modutech_turn แปลงกลับเป็นสายพานด้วยกราฟ ไม่งั้นรหัสที่ลูกค้าพิมพ์หายทั้งตัว
        st.modutech_sprocket_codes = list(codes.sprocket_codes)

        esc = escalation_gate(text, belt_code=(st.modutech_codes or [None])[0])
        if esc and not esc.has_spec_question:
            return Decision("escalate", "modutech", esc.reply_th, f"หัวข้อ {esc.topics}",
                            topic="escalate", effective_text=text,
                            modutech=dict(codes=st.modutech_codes, slots=dict(slots)), state=st)
        d = self._decide_modutech_core(text, st, codes)
        if esc:
            _attach_handoff(d, esc.handoff_th, esc.topics)
        # แนบรหัสสเตอร์ที่จุดเดียว แทนการแก้ทุก return ใน _decide_modutech_core
        # ซึ่งมี 7 ทางออกและเพิ่มได้เรื่อย ๆ ถ้าแก้ทีละที่จะมีทางที่ลืม
        if codes.sprocket_codes and isinstance(d.modutech, dict):
            d.modutech["sprocket_codes"] = list(codes.sprocket_codes)
        return d

    def _decide_modutech_core(self, text: str, st: DialogState, codes) -> Decision:
        slots = st.modutech_slots

        q = self.search.parse(text)
        if q is not None and not codes.belt_codes:
            return Decision("search", "modutech", reason="คำถามเชิงสำรวจ → ค้นทั้งฐาน", topic="explore",
                            effective_text=text,
                            modutech=dict(codes=[], slots=dict(slots), search_query=q.__dict__), state=st)

        if codes.candidates and not codes.belt_codes:
            st.pending, st.pending_question = "modutech_code", text
            shown = codes.candidates[:10]
            reply = ("รหัสนี้มีหลายรุ่นครับ หมายถึงรุ่นไหนครับ\n"
                     + "\n".join(f"• {c} — {self.mod.by_code[c]['retrieval_card'].get('surface_type', '')}"
                                 for c in shown))
            return Decision("ask", "modutech", reply, "รหัสกำกวม", topic="code", effective_text=text,
                            modutech=dict(codes=[], candidates=codes.candidates, slots=dict(slots)), state=st)

        focus = self.mod.records(st.modutech_codes)
        temp_topic = _has(MOD_TEMP_TOPIC, text) or "operating_temp_c" in slots and not st.modutech_codes
        if temp_topic and not slots.get("wet_or_dry"):
            needs = (not focus) or any("wet_or_dry" in (p["answering"].get("required_slots") or []) for p in focus)
            if needs:
                st.pending, st.pending_question = "wet_or_dry", text
                return Decision("ask", "modutech", ASK_WET_DRY_TH, "ต้องรู้เปียก/แห้งก่อนตอบอุณหภูมิ",
                                topic="temperature", effective_text=text,
                                modutech=dict(codes=st.modutech_codes, slots=dict(slots)), state=st)

        if _has(MOD_STRENGTH_TOPIC, text) and not slots.get("line_shape") \
                and any(p["specifications"].get("has_curve_rating") for p in focus):
            st.pending, st.pending_question = "line_shape", text
            return Decision("ask", "modutech", ASK_LINE_SHAPE_TH, "รุ่นวิ่งโค้งได้ ต้องรู้ไลน์ตรง/โค้ง",
                            topic="strength", effective_text=text,
                            modutech=dict(codes=st.modutech_codes, slots=dict(slots)), state=st)

        # รหัสสเตอร์นับเป็น "รู้พอจะตอบ" แล้ว ไม่ต้องถามกลับว่าทำงานอะไร
        #
        # "EC127SQZ19*PA ใช้กับสายพานรุ่นไหน" มีคำว่า "สายพาน" จึงเข้าเงื่อนไขขอบล่าง
        # แล้วถูกตีเป็นคำถามแนะนำรุ่นแบบเปิด ทั้งที่ลูกค้าระบุของมาชัดเจนที่สุดแล้ว
        # ก่อน 1 ต.ค. มันรอดมาได้เพราะบั๊กอีกตัว — วัสดุต่อท้ายรหัส (*PA) ถูกอ่านเป็น
        # slot ของลูกค้า ทำให้ has_need เป็นจริง พอแก้บั๊กนั้นอาการนี้จึงโผล่
        # กล่าวคือมันเคย "ทำงานได้ด้วยเหตุผลที่ผิด" ไม่ใช่เคยถูกแล้วเพิ่งเสีย
        has_need = bool(_any_word(self.router.weak_terms, text)) or bool(slots) \
            or bool(codes.sprocket_codes)
        if not focus and not has_need and _has(MOD_RECOMMEND + r"|สายพาน", text):
            st.pending, st.pending_question = "modutech_need", text
            return Decision("ask", "modutech", ASK_MODUTECH_NEED_TH, "ยังไม่รู้งานของลูกค้า",
                            topic="recommend", effective_text=text,
                            modutech=dict(codes=[], slots=dict(slots)), state=st)

        return Decision("answer", "modutech", reason="รหัสชัด" if focus else "ค้นตามการใช้งาน",
                        topic="temperature" if temp_topic else None, effective_text=text,
                        modutech=dict(codes=st.modutech_codes, slots=dict(slots)), state=st)


def _attach_handoff(d: Decision, handoff_th: str, topics: list) -> None:
    """
    คำถามผสม: ส่วนสเปกเดินตามปกติ ส่วนการค้าส่งฝ่ายขาย
      ask            → ต่อข้อความฝ่ายขายท้ายคำถามกลับเลย (ลูกค้าเห็นทันที)
      answer/search  → เก็บไว้ใน sales_handoff_th ให้ chatbot ต่อท้ายหลังคำตอบของ LLM
    """
    d.sales_topics = list(topics)
    d.reason = f"{d.reason} · คำถามผสม: {'/'.join(topics)} → ฝ่ายขาย".strip(" ·")
    if d.needs_llm:
        d.sales_handoff_th = handoff_th
    elif d.action == "ask":
        d.reply_th = f"{d.reply_th}\n\n{handoff_th}"


# ═══════════════════════════════════════════════════════════════
# TEMPLATE — ตอบโดยไม่ผ่าน LLM
# ═══════════════════════════════════════════════════════════════

REJECT_IMAGE_TH = ("❌ ระบบตรวจพบว่ารูปภาพที่อัปโหลดไม่เกี่ยวข้องกับสินค้าในระบบ หรือยังไม่ชัดพอให้ระบุรุ่น "
                   "กรุณาอัปโหลดภาพโซ่หรือสเตอร์ที่ชัดเจน (ถ่ายจากด้านบนให้เห็นแผ่นโซ่) "
                   "หรือแจ้งรหัสที่ปั๊มบนตัวสินค้าครับ")
SMALLTALK_TH = "ยินดีครับ มีเรื่องโซ่ลำเลียงหรือสายพานให้ช่วยดูเพิ่มเติม บอกได้เลยครับ"
OFFTOPIC_TH = ("ขออภัยครับ ผมตอบได้เฉพาะเรื่องสินค้าโซ่ลำเลียง Movex และสายพานโมดูลาร์ Modutech "
               "เช่น สเปก การเลือกรุ่น หรืออะไหล่ที่ใช้คู่กัน มีเรื่องไหนให้ช่วยไหมครับ")
ASK_BRAND_TH = ("ขอทราบก่อนนะครับว่าถามถึงสินค้าแบบไหน\n"
                "1. โซ่ท็อปเชน Movex — ลำเลียงขวด กระป๋อง ลัง ในไลน์บรรจุเครื่องดื่ม\n"
                "2. สายพานโมดูลาร์ Modutech — งานอาหาร เช่น เนื้อสัตว์ สัตว์ปีก อาหารทะเล เบเกอรี่ ผักผลไม้\n"
                "ถ้ามีรหัสรุ่นหรือรูปสินค้า ส่งมาได้เลยครับ")
ASK_MOVEX_UNCLEAR_IMAGE_TH = ("จากรูปยังระบุรุ่นได้ไม่ชัดครับ รบกวนแจ้งรหัสรุ่นที่ปั๊มบนตัวโซ่ "
                              "หรือส่งรูปด้านบนของโซ่ที่เห็นแผ่นชัด ๆ อีกครั้งครับ")
ASK_WET_DRY_TH = ("ขอทราบก่อนครับว่าไลน์นี้เป็นสภาวะเปียก (มีน้ำ ล้างบ่อย) หรือแห้ง "
                  "เพราะค่าอุณหภูมิที่ทนได้ต่างกันมาก บางเกรดต่างกันถึง 33 °C")
ASK_LINE_SHAPE_TH = ("รุ่นนี้วิ่งโค้งได้ครับ ขอทราบก่อนว่าไลน์เป็นแบบตรงอย่างเดียว หรือมีช่วงโค้ง "
                     "เพราะค่าความแข็งแรงตอนวิ่งโค้งต่ำกว่าตอนวิ่งตรงมาก")
ASK_MODUTECH_NEED_TH = ("ขอข้อมูลเพิ่มนิดนะครับ เพื่อแนะนำรุ่นได้ตรง\n"
                        "• อุตสาหกรรม: เนื้อสัตว์ / สัตว์ปีก / อาหารทะเล / เบเกอรี่ / ผักผลไม้ / บรรจุภัณฑ์ / กล่องลูกฟูก\n"
                        "• จุดที่ใช้งาน: เช่น ไลน์ตัดแต่ง เข้าเตาอบ เครื่องตรวจจับโลหะ\n"
                        "• ไลน์เปียกหรือแห้ง และอุณหภูมิใช้งานโดยประมาณ")