"""
modutech_router.py
==================
แยกแบรนด์ก่อนค้น · resolve รหัสแบบ exact ก่อน vector search · ค้นทั้งฐานข้อมูลสำหรับคำถามเชิงสำรวจ

ไม่ import torch / qdrant / gradio — เทสต์ได้ด้วย python ล้วน
ฝั่ง Movex ไม่ถูกแตะ: router แค่บอกว่าคำถามนี้ของแบรนด์ไหน chatbot เรียก RetrieverV4 เดิมเหมือนเดิม
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal, Optional

from modutech_gates import Catalogue

Brand = Literal["movex", "modutech"]


# ═══════════════════════════════════════════════════════════════
# RESOLVE รหัสแบบ exact
# ═══════════════════════════════════════════════════════════════

def norm_code(s: str) -> str:
    """MP8 0 C / MP80-C / mp80c → MP80C  (& ในรหัสที่ OCR อ่านผิดก็ตัดทิ้ง)"""
    return re.sub(r"[^A-Z0-9%]", "", (s or "").upper())


# ช่วงข้อความที่อาจเป็นรหัส: ตัวอักษรละติน ตัวเลข ช่องว่าง - & % ต่อกัน
_CODE_RUN = re.compile(r"[A-Za-z0-9][A-Za-z0-9 \-&%*]*")


@dataclass
class CodeMatch:
    belt_codes: list[str] = field(default_factory=list)       # exact
    sprocket_codes: list[str] = field(default_factory=list)   # exact
    candidates: list[str] = field(default_factory=list)       # พิมพ์แค่ซีรีส์ เช่น "HC508" → หลายรุ่น


class CodeResolver:
    """
    ดัชนีจาก belt_code_alias_index (88) + belt_code + belt_code_key + retrieval_card.aliases
    จับแบบ longest-prefix ณ ต้นคำเท่านั้น — "EC254 R-GT" ไม่ถูกจับเป็น "EC254 R"
    """

    def __init__(self, cat: Catalogue):
        idx: dict[str, str] = {}
        for alias, code in cat.data.get("belt_code_alias_index", {}).items():
            idx[norm_code(alias)] = code
        for p in cat.products:
            code = p["belt_code"]
            for a in [code, p.get("belt_code_key", "")] + list(p.get("retrieval_card", {}).get("aliases") or []):
                if a:
                    idx.setdefault(norm_code(a), code)
        self.belt_idx = idx
        self.sprocket_idx = {norm_code(c): c for c in cat.sprockets}
        self.series_idx: dict[str, list[str]] = {}
        for p in cat.products:
            m = re.match(r"[A-Z]+\d+", norm_code(p["belt_code"]))
            if m:
                self.series_idx.setdefault(m.group(), []).append(p["belt_code"])
        self._keys = sorted(set(idx) | set(self.sprocket_idx), key=len, reverse=True)

    def resolve(self, text: str) -> CodeMatch:
        out = CodeMatch()
        for run in _CODE_RUN.finditer(text or ""):
            words = run.group().split()
            i = 0
            while i < len(words):
                hit = None
                # ลองรวมคำตั้งแต่ i ให้ยาวที่สุดที่ตรงดัชนี (รองรับ "MP8 0 C")
                for j in range(len(words), i, -1):
                    key = norm_code("".join(words[i:j]))
                    if key in self.belt_idx or key in self.sprocket_idx:
                        hit = (key, j)
                        break
                if hit:
                    key, j = hit
                    if key in self.belt_idx:
                        c = self.belt_idx[key]
                        if c not in out.belt_codes:
                            out.belt_codes.append(c)
                    else:
                        c = self.sprocket_idx[key]
                        if c not in out.sprocket_codes:
                            out.sprocket_codes.append(c)
                    i = j
                    continue
                key = norm_code(words[i])
                if key in self.series_idx:
                    for c in self.series_idx[key]:
                        if c not in out.candidates:
                            out.candidates.append(c)
                i += 1
        # ถ้าเจอ exact ในซีรีส์เดียวกันแล้ว ไม่ต้องถามกลับ
        out.candidates = [c for c in out.candidates if c not in out.belt_codes] if not out.belt_codes else []
        return out


# ═══════════════════════════════════════════════════════════════
# BRAND ROUTER
# ═══════════════════════════════════════════════════════════════

MOVEX_PATTERNS = [
    r"\bLF\s*N?\s*\d{2,3}", r"\bK\s?\d{3,4}\b",   # TAB ไม่นับ: Modutech มี EC254 TR-TAB
    r"(?<!\d)5[4-9]\d{3}(?!\d)",                 # Art.Nr สเตอร์ Movex
    r"(?<![A-Za-z\d])(?:820|821|880|882|103)(?!\d)",  # ซีรีส์ Movex
    r"movex", r"โซ่", r"\bchain\b", r"top\s*chain", r"workload",
]
_MOVEX_RE = re.compile("|".join(MOVEX_PATTERNS), re.IGNORECASE)

MODUTECH_WORDS = [r"modutech", r"โมดูเทค", r"modular", r"โมดูลาร์",
                  r"\bN\s*/\s*m\b", r"N/ม"]   # Movex ใช้ workload เป็น N ส่วน Modutech ใช้ N/m

# สัญญาณอ่อน — คำงานอาหาร/สถานีงานที่ Modutech ขาย (เสริมคำที่ดึงจาก v30 ซึ่งเป็นวลียาว)
MODUTECH_WEAK_EXTRA = [
    "เลาะกระดูก", "กระดูก", "ชำแหละ", "ไก่", "หมู", "เนื้อสัตว์", "เนื้อบด", "แล่ปลา", "กุ้ง", "อาหารทะเล",
    "เบเกอรี่", "ขนมปัง", "แป้ง", "โดว์", "ผัก", "ผลไม้", "เตาอบ", "สไปรัล", "แช่แข็ง", "ตู้แช่",
    "สะเด็ดน้ำ", "ล้างผัก", "กล่องลูกฟูก", "ลูกฟูก", "ตรวจจับโลหะ", "เครื่องชั่ง",
    "flush grid", "flat top", "ผิวตะแกรง", "ผิวเรียบ", "ผิวกันลื่น", "ผิวลูกกลิ้ง", "roller top",
]
# สัญญาณอ่อนของ Movex — งานขวด/กระป๋อง single file
MOVEX_WEAK = ["ขวด", "กระป๋อง", "PET", "bottle", "can ", "single file"]


# ไม่มีสัญญาณแบรนด์เลย และไม่มีเทิร์นก่อน
#   "legacy_movex" = ส่งเข้า pipeline Movex เดิม (ซึ่งถามกลับ/ปฏิเสธนอกเรื่องเองอยู่แล้ว)
#                    ทำให้ CLARIF-001..004 ใน testcases_v2.csv ได้คำตอบเหมือนเดิมทุกตัวอักษร
#   "ask"          = router ถามกลับเองว่าแบรนด์ไหน (คำถามนอกเรื่องจะได้คำถามกลับแทนคำปฏิเสธ)
NO_SIGNAL_POLICY: Literal["legacy_movex", "ask"] = "legacy_movex"


@dataclass
class Route:
    brand: Optional[Brand]
    reason: str
    need_clarify: bool = False
    clarify_th: str = ""
    codes: CodeMatch = field(default_factory=CodeMatch)


class BrandRouter:
    """
    สัญญาณ (เรียงจากแข็งไปอ่อน)
      แข็ง  Modutech: รหัสสายพาน/สเตอร์ exact · ชื่อแบรนด์
            Movex   : LF### / K### / Art.Nr 5xxxx / ซีรีส์ 820… / โซ่ / รูปภาพ (ViT ฝึกด้วย Movex เท่านั้น)
      อ่อน  Modutech: คำอุตสาหกรรม / สถานีงาน / ชื่อผิวสายพาน จาก v30
    ทั้งสองฝั่งมีสัญญาณแข็ง → ถามกลับ ห้ามเดา
    ไม่มีสัญญาณเลย → ใช้แบรนด์ของเทิร์นก่อน ถ้าไม่มีก็ถามกลับ
    """

    def __init__(self, cat: Catalogue):
        self.resolver = CodeResolver(cat)
        terms = set()
        for v in cat.data.get("application_vocabulary", []):
            terms.add(v.get("application_th", ""))
            terms.add(v.get("application", ""))
        for p in cat.products:
            for i in p.get("industries", []):
                terms.add(i.get("industry_th", ""))
        for s in cat.data.get("surface_type_vocabulary", []):
            terms.add(s.get("surface_type_th", ""))
        # คำทั่วไปที่ Movex ใช้ด้วย ไม่นับเป็นสัญญาณ
        generic = {"สายพาน", "บรรจุภัณฑ์", "Packaging", "Transfer Conveyors", "ลำเลียงจุดถ่ายโอน"}
        terms |= set(MODUTECH_WEAK_EXTRA)
        self.weak_terms = sorted({t for t in terms if t and len(t) >= 3 and t not in generic},
                                 key=len, reverse=True)
        self._mod_word = re.compile("|".join(MODUTECH_WORDS), re.IGNORECASE)

    def route(self, text: str, has_image: bool = False,
              prev_brand: Optional[Brand] = None,
              no_signal_policy: Optional[str] = None) -> Route:
        text = text or ""
        codes = self.resolver.resolve(text)
        mod_strong = bool(codes.belt_codes or codes.sprocket_codes or codes.candidates
                          or self._mod_word.search(text))
        mov_strong = bool(_MOVEX_RE.search(text)) or has_image
        mod_weak = next((t for t in self.weak_terms if t.lower() in text.lower()), None)
        mov_weak = next((t for t in MOVEX_WEAK if t.lower() in text.lower()), None)

        if mod_strong and mov_strong:
            return Route(None, "สัญญาณทั้งสองแบรนด์", True,
                         "ขอยืนยันก่อนนะครับ ถามถึงสินค้า Movex (โซ่ท็อปเชน) หรือ Modutech (สายพานโมดูลาร์) ครับ",
                         codes)
        if mod_strong:
            return Route("modutech", "รหัส/ชื่อแบรนด์ Modutech", codes=codes)
        if mov_strong:
            return Route("movex", "รูปภาพ" if has_image and not _MOVEX_RE.search(text)
                         else "รหัส/ซีรีส์ Movex", codes=codes)
        if mod_weak and mov_weak:
            return Route(None, f"สัญญาณอ่อนทั้งสองฝั่ง ({mod_weak} / {mov_weak})", True,
                         "งานนี้ลำเลียงอะไรเป็นหลักครับ และต้องการโซ่ท็อปเชน (Movex) "
                         "หรือสายพานโมดูลาร์ (Modutech) ครับ", codes)
        if mod_weak:
            return Route("modutech", f"คำอุตสาหกรรม/งาน: {mod_weak}", codes=codes)
        if mov_weak:
            return Route("movex", f"คำงานขวด/กระป๋อง: {mov_weak}", codes=codes)
        if prev_brand:
            return Route(prev_brand, "ต่อจากเทิร์นก่อน", codes=codes)
        if (no_signal_policy or NO_SIGNAL_POLICY) == "legacy_movex":
            return Route("movex", "ไม่มีสัญญาณแบรนด์ → pipeline เดิม", codes=codes)
        return Route(None, "ไม่มีสัญญาณแบรนด์", True,
                     "ขอทราบเพิ่มนิดนะครับ ต้องการโซ่ท็อปเชน (Movex) หรือสายพานโมดูลาร์ (Modutech) "
                     "และใช้กับงานแบบไหนครับ", codes)


def modutech_retriever_config_kwargs() -> dict:
    """
    ค่าที่ส่งเข้า RetrieverV4Config(**kwargs) สำหรับฝั่ง Modutech
    retriever_v4.py รับทุกค่านี้ทาง config อยู่แล้ว ไม่ต้องแก้ไฟล์นั้น
    ⚠️ image_collection ไม่มีในฝั่ง Modutech — ห้ามส่งรูปเข้า retriever ตัวนี้
       (router ส่งคำถามที่มีรูปไป Movex อยู่แล้ว)
    """
    return dict(
        qdrant_path="./qdrant_modutech_db",
        text_collection="modutech_belt_chunks",
        image_collection="modutech_images__not_built",
        payload_id_keys=("belt_code_id", "sprocket_code"),
        payload_series_key="belt_series",
        series_filter_fmt="{}",
        art_nr_regex=r"(?!x)x",          # ไม่มี Art.Nr — ปิด detector
        art_nr_field="sprocket_code",
        pid_strip_tokens=(),
        type_boost_rules=(),
        known_series=set(),
    )


# ═══════════════════════════════════════════════════════════════
# ค้นทั้งฐานข้อมูล (ข้อ D) — ห้ามสรุปว่า "ไม่มี" จาก top-5
# ═══════════════════════════════════════════════════════════════

@dataclass
class ExploratoryQuery:
    field: Optional[str] = None          # strength | weight | open_area | pitch
    op: Optional[str] = None             # >= | <=
    value: Optional[float] = None
    surface: Optional[str] = None        # surface_type ภาษาอังกฤษตาม vocabulary
    pin_material: Optional[str] = None
    belt_material: Optional[str] = None

    def is_empty(self) -> bool:
        return not any([self.field and self.value is not None, self.surface,
                        self.pin_material, self.belt_material])


_FIELD_WORDS = {
    "strength": r"แรงดึง|ความแข็งแรง|รับแรง|strength|N\s*/\s*m|นิวตัน",
    "weight": r"น้ำหนัก|weight|kg\s*/\s*m",
    "open_area": r"ช่องเปิด|พื้นที่เปิด|open\s*area",
    "pitch": r"พิทช์|pitch",
}
_GE = r"อย่างน้อย|ไม่ต่ำกว่า|ขึ้นไป|มากกว่า|เกินกว่า|สูงกว่า|>=|≥|>|at\s*least|minimum"
_LE = r"ไม่เกิน|ต่ำกว่า|น้อยกว่า|ไม่ถึง|<=|≤|<|at\s*most|maximum"
_EXPLORE = r"มี.*(?:ไหม|มั้ย|หรือเปล่า|บ้าง)|รุ่นไหน|ตัวไหน|อะไรบ้าง|ต้องการ|หา|แนะนำ|list|which"
_PIN_WORDS = {"SS": r"สแตนเลส|stainless|(?<![A-Za-z])SS(?![A-Za-z])"}


class CatalogueSearch:
    """ค้นทั้ง 68 รุ่นใน v30 คืนทุกรุ่นที่ผ่าน พร้อมค่าที่ผูกกับเกรด"""

    def __init__(self, cat: Catalogue):
        self.cat = cat
        self.surfaces = []
        for s in cat.data.get("surface_type_vocabulary", []):
            self.surfaces.append((s["surface_type_th"], s["surface_type"]))
            self.surfaces.append((s["surface_type"], s["surface_type"]))
        self.surfaces.sort(key=lambda x: len(x[0]), reverse=True)
        self.materials = sorted({v["belt_material"] for p in cat.products
                                 for v in p["specifications"]["variants"]}, key=len, reverse=True)

    def parse(self, text: str) -> Optional[ExploratoryQuery]:
        t = text or ""
        q = ExploratoryQuery()
        for name, rx in _FIELD_WORDS.items():
            if re.search(rx, t, re.IGNORECASE):
                q.field = name
                break
        m = re.search(r"(\d[\d,]*(?:\.\d+)?)\s*(?:k\b|พัน)?", t.replace(" ,", ","))
        if q.field and m:
            val = float(m.group(1).replace(",", ""))
            if re.search(r"\d\s*(?:k\b|พัน)", t, re.IGNORECASE):
                val *= 1000
            q.value = val
            q.op = "<=" if re.search(_LE, t, re.IGNORECASE) else ">="
        for th, en in self.surfaces:
            if th.lower() in t.lower():
                q.surface = en
                break
        for code, rx in _PIN_WORDS.items():
            if re.search(rx, t, re.IGNORECASE):
                q.pin_material = code
        if q.is_empty():
            return None
        if not (re.search(_EXPLORE, t, re.IGNORECASE) or q.value is not None):
            return None
        return q

    def match_application(self, text: str) -> Optional[ApplicationMatch]:
        """
        หางาน (application) จากคำในคำถาม แล้วคืนรายชื่อสายพานครบทุกรุ่นจาก application_material_index
        (เฉพาะ entry ที่ manufactured = true) — แทน vector top-5 ที่ทำให้ตอบไม่ครบ

        ยกเว้นระดับรุ่น (ผู้ใช้เลือก "ทางกลาง" 2026-09-26): entry ที่ manufactured = false
        แต่รุ่นนั้นผลิตเกรดที่ material_registry ระบุ base_polymer = วัสดุที่เล่มแนะนำ
        (SM254 FG50%: หน้างานเขียน PA · ตารางสเปกผลิต PA6) → นับรุ่นนั้นด้วยเกรดที่ผลิตจริง + grade_notes_th
        ไม่ได้ตั้ง PA = PA6 ทั้งระบบ — รุ่นอื่นที่เขียน PA ยังเป็น PA ตามเดิม
        """
        low = (text or "").lower()
        found = {tag for tag, kws in APPLICATION_KEYWORDS.items() if any(k.lower() in low for k in kws)}
        if not found:
            return None
        vocab = self.cat.data.get("application_vocabulary", [])
        scored = [(len(found & set(v.get("tags") or [])), v["application"]) for v in vocab]
        best = max((sc for sc, _ in scored), default=0)
        if best == 0:
            return None
        apps = sorted(a for sc, a in scored if sc == best)
        industry = next((ind for ind, kws in INDUSTRY_KEYWORDS.items()
                         if any(k.lower() in low for k in kws)), None)
        index = {a["application"]: a for a in self.cat.data.get("application_material_index", [])}

        registry = {m["material_code"]: m for m in self.cat.data.get("material_registry", [])}
        notes: dict[str, str] = {}

        def collect(use_industry: bool) -> dict:
            out: dict[str, set] = {}
            for app in apps:
                for e in (index.get(app) or {}).get("entries", []):
                    if use_industry and industry and industry not in (e.get("industries") or []):
                        continue
                    for c in e["belt_codes"]:
                        if e.get("manufactured", True):
                            out.setdefault(c, set()).add(e["material"])
                            continue
                        made = self._belt_grade_for(c, e["material"], registry)
                        if made:
                            out.setdefault(c, set()).update(made)
                            notes[c] = belt_grade_note_th(e["material"], made, registry)
            return out

        by_belt = collect(True)
        filtered = bool(industry) and bool(by_belt)
        if not by_belt:
            by_belt = collect(False)
        if not by_belt:
            return None
        return ApplicationMatch(apps, industry, sorted(by_belt),
                                {k: sorted(v) for k, v in sorted(by_belt.items())}, filtered,
                                {k: v for k, v in sorted(notes.items()) if k in by_belt})

    def _belt_grade_for(self, belt_code: str, recommended: str, registry: dict) -> list[str]:
        """เกรดที่รุ่นนี้ผลิตจริง ซึ่งเป็นชนิดย่อยของวัสดุที่เล่มแนะนำ (base_polymer ตรงกัน)"""
        p = self.cat.by_code.get(belt_code)
        if p is None:
            return []
        made = (p.get("derived") or {}).get("materials_manufactured") or []
        return [m for m in made if (registry.get(m) or {}).get("base_polymer") == recommended]

    def _variant_value(self, v: dict, fld: str) -> Optional[float]:
        if fld == "strength":
            return (v.get("belt_strength") or {}).get("n_per_m")
        if fld == "weight":
            return (v.get("belt_weight") or {}).get("kg_per_m2")
        return None

    def search(self, q: ExploratoryQuery) -> dict:
        matches = []
        for p in self.cat.products:
            spec, rc = p["specifications"], p["retrieval_card"]
            surfaces = {rc.get("surface_type")} | {i.get("surface_type") for i in p.get("industries", [])}
            if q.surface and q.surface not in surfaces:
                continue
            vs = spec["variants"]
            if q.pin_material:
                vs = [v for v in vs if v.get("pin_material") == q.pin_material]
            if q.belt_material:
                vs = [v for v in vs if v.get("belt_material") == q.belt_material]
            if not vs:
                continue
            if q.field in ("open_area", "pitch") and q.value is not None:
                val = spec.get("open_area_pct") if q.field == "open_area" else (spec.get("pitch") or {}).get("mm")
                if val is None or not (val >= q.value if q.op == ">=" else val <= q.value):
                    continue
                hits = [{"variant_id": v["variant_id"], "belt_material": v["belt_material"],
                         "pin_material": v["pin_material"], "value": val} for v in vs]
            elif q.field in ("strength", "weight") and q.value is not None:
                hits = []
                for v in vs:
                    val = self._variant_value(v, q.field)
                    if val is not None and (val >= q.value if q.op == ">=" else val <= q.value):
                        hits.append({"variant_id": v["variant_id"], "belt_material": v["belt_material"],
                                     "pin_material": v["pin_material"], "value": val})
                if not hits:
                    continue
            else:
                hits = [{"variant_id": v["variant_id"], "belt_material": v["belt_material"],
                         "pin_material": v["pin_material"]} for v in vs]
            matches.append({
                "belt_code": p["belt_code"],
                "surface_type": rc.get("surface_type"),
                "curve_capable": bool(spec.get("has_curve_rating")),
                "blocking_for": p["data_quality"].get("blocking_for") or [],
                "variants": hits,
            })
        for m in matches:                      # สถานะข้อมูลสภาวะเปียก — ไม่ให้ LLM สรุปเองว่าเหมาะ/ไม่เหมาะ
            p = self.cat.by_code[m["belt_code"]]
            for h in m["variants"]:
                h["wet_data"] = wet_data_status(p, h["variant_id"])
        if q.field in ("strength", "weight") and q.value is not None:
            matches.sort(key=lambda m: max(h["value"] for h in m["variants"]),
                         reverse=(q.op == ">="))
        return {
            "query": q.__dict__,
            "scanned": len(self.cat.products),
            "matched": len(matches),
            "matches": matches,
            "note_th": (f"ค้นครบทั้ง {len(self.cat.products)} รุ่นในฐานข้อมูล ผลนี้สมบูรณ์ "
                        + ("" if matches else "ไม่มีรุ่นที่ตรงเงื่อนไขจริง")
                        + (" ค่าความแข็งแรงของรุ่นที่วิ่งโค้งได้เป็นค่าวิ่งตรง"
                           if q.field == "strength" and any(m["curve_capable"] for m in matches) else "")),
        }


def wet_data_status(p: dict, variant_id: str) -> str:
    """
    ok            มีค่าสภาวะเปียกครบ
    partial_data  คำนวณจากบางชั้น (ขาดข้อมูลพินหรือวัสดุบางส่วน)
    no_data       แคตตาล็อกไม่มีข้อมูลสภาวะเปียกของเกรดสายพานนี้
    not_recommended  แคตตาล็อกระบุว่าไม่แนะนำสำหรับไลน์เปียก
    """
    v = next((x for x in p["specifications"]["variants"] if x["variant_id"] == variant_id), None)
    if v is None:
        return "no_data"
    eff = v.get("temperature_effective") or {}
    if eff.get("wet_status") == "not_recommended":
        return "not_recommended"
    mat = v.get("belt_material")
    if mat in (p.get("data_quality", {}).get("wet_condition_unknown") or []):
        return "no_data"
    disq = next((x for x in p.get("answering", {}).get("disqualifiers") or [] if x["exclude_material"] == mat), {})
    if (disq.get("wet") or {}).get("registry_text") or disq.get("wet_status") == "no_data":
        return "no_data"
    return eff.get("wet_status") or "ok"


# คำไทย/อังกฤษ → tag ใน application_vocabulary (คัดมือ ตรวจได้ทีละบรรทัด)
APPLICATION_KEYWORDS: dict[str, list[str]] = {
    "deboning": ["เลาะกระดูก", "deboning"], "cutting": ["ตัดแบ่ง", "cut-up", "cutting"],
    "trimming": ["ตัดแต่ง", "trim"], "bone": ["ลำเลียงกระดูก", "bone"],
    "oven": ["เตาอบ", "oven"], "spiral": ["สไปรัล", "spiral"], "freezing": ["แช่แข็ง", "ฟรีซ", "freez"],
    "cooling": ["ทำความเย็น", "พักเย็น", "cooling"], "chilling": ["แช่เย็น", "chiller"],
    "proofing": ["พักแป้ง", "หมักแป้ง", "proof"], "dough": ["แป้งดิบ", "แป้งโด", "ก้อนแป้ง", "รีดแป้ง", "dough"],
    "metal_detection": ["ตรวจจับโลหะ", "metal detect"], "inspection": ["ชั่งน้ำหนัก", "checkweigh"],
    "glazing": ["เคลือบน้ำแข็ง", "glazing"], "coating": ["ชุบเกล็ด", "เคลือบ", "breading", "coating"],
    "grading": ["คัดเกรด", "คัดขนาด", "grading"], "control_table": ["โต๊ะตรวจสอบ", "คัดแยก", "control table"],
    "box": ["กล่อง", "ลัง", "carton"], "labeling": ["ติดฉลาก", "label"], "packing": ["บรรจุและติดฉลาก"],
    "live_bird": ["ไก่เป็น", "live bird"], "slaughtering": ["เชือด", "slaughter"],
    "evisceration": ["ควักเครื่องใน", "eviscerat"], "byproduct": ["เครื่องใน", "ขนไก่", "ไขมัน", "offal"],
    "incline_decline": ["ลาดเอียง", "ขึ้นลง", "incline"], "washing": ["ล้าง", "wash"],
    "draining": ["สะเด็ดน้ำ", "drain"], "peeling": ["ปอกเปลือก", "peel"], "skinning": ["ลอกหนัง", "skinn"],
    "slicing": ["สไลซ์", "slic"], "minced": ["เนื้อบด", "minced"], "marination": ["ไลน์หมัก", "marinat"],
    "shrimp": ["กุ้ง", "shrimp"], "transfer": ["ถ่ายโอน", "transfer"], "accumulation": ["สะสม", "accumulat"],
    "palletizing": ["พาเลท", "pallet"], "stacking": ["เรียงกล่อง", "stack"], "pan": ["ถาดอบ"],
    "tray": ["บรรจุถาด", "tray"], "shrink": ["ฟิล์มหด", "shrink"], "sterilization": ["ฆ่าเชื้อ", "steril"],
    "elevator": ["ยกระดับ", "elevator"], "bulk": ["เทกอง", "bulk"], "impact": ["แรงกระแทก", "chute"],
    "dressing": ["ตกแต่งชิ้นเนื้อ"], "conditioning": ["ปรับสภาพ"], "filling": ["เข้าเครื่องบรรจุ", "filling"],
    "warehouse": ["คลังสินค้า", "warehouse"],
}
INDUSTRY_KEYWORDS: dict[str, list[str]] = {
    "Poultry": ["ไก่", "สัตว์ปีก", "เป็ด", "poultry", "chicken"],
    "Meat Processing": ["หมู", "เนื้อวัว", "เนื้อสัตว์", "meat", "pork", "beef"],
    "Fish & Seafood": ["ปลา", "อาหารทะเล", "กุ้ง", "seafood", "fish"],
    "Bakery Processing": ["ขนมปัง", "เบเกอรี่", "เค้ก", "bakery", "bread"],
    "Fruit & Vegetable": ["ผัก", "ผลไม้", "vegetable", "fruit"],
    "Packaging": ["บรรจุภัณฑ์", "packaging"],
    "Corrugated": ["ลูกฟูก", "corrugated"],
}


@dataclass
class ApplicationMatch:
    applications: list[str]
    industry: Optional[str]
    belt_codes: list[str]
    materials_by_belt: dict
    industry_filter_applied: bool
    grade_notes_th: dict = field(default_factory=dict)


def belt_grade_note_th(recommended: str, made: list[str], registry: dict) -> str:
    grades = ", ".join(made)
    note = (f"หน้างานในเล่มแนะนำเกรด {recommended} แต่ตารางสเปกของรุ่นนี้ผลิตเฉพาะ {grades} "
            f"({recommended} เป็นชื่อตระกูล {grades} เป็นชนิดย่อย) ให้ระบุเกรด {grades} เสมอ")
    wet = {(registry.get(m) or {}).get("temperature", {}).get("wet") for m in made}
    if "not recommended" in wet:
        note += f" · เกรด {grades} แคตตาล็อกไม่แนะนำให้ใช้ในสภาวะเปียก"
    return note


NO_ABSENCE_NOTE_TH = ("context นี้มีเพียง {n} รุ่นจาก {total} รุ่นที่ระบบค้นมาได้ "
                      "ห้ามสรุปว่า 'ไม่มี' หรือ 'สูงสุดคือ' จากรายการนี้ "
                      "ถ้าลูกค้าถามหาเงื่อนไข ให้เรียก search_catalogue")
