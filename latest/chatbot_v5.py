"""
chatbot_v5.py
=============
Movex + Modutech Sales AI Assistant v5

เปลี่ยนจาก v4 (v4 ไม่ถูกแก้ ยังรันได้เหมือนเดิม)
  ✦ ทาง จ — dialogue_state.DialogueManager ตัดสินในโค้ดก่อนถึง LLM:
      แบรนด์ (slot 0) · ถามกลับ · ปฏิเสธนอกเรื่อง · รูปที่ไม่ใช่สินค้า · ส่งฝ่ายขาย
    "ขั้นตอนที่ 1–2" ใน prompt ของ Movex ถูกย้ายมาเป็นโค้ด prompt เหลือแค่ผลตรวจจากระบบ
  ✦ Modutech เป็นสายที่สอง: collection modutech_belt_chunks · gate ของ modutech_gates · ค้นทั้งฐาน
  ✦ template (ถามกลับ/ปฏิเสธ/ฝ่ายขาย) ตอบโดยไม่เรียก Gemini
  ✦ API ของ Gradio เท่าเดิม (inputs/outputs 7 ตัว) — สถานะบทสนทนาเก็บฝั่ง server ตาม session_id
    frontend Svelte จึงไม่ต้องแก้
  ✦ ฝั่ง Movex: RetrieverV4, collection, tools, logger เหมือน v4 ทุกอย่าง

---- ข้อความเดิมของ v4 ----
Movex Sales AI Assistant v4 — Adaptive Multi-Signal Pipeline

เปลี่ยนจาก v3:
  ✦ ใช้ RetrieverV4 — confidence-weighted RRF + multi-hop graph + series filter
  ✦ InstrumentedRetriever ใหม่ — hook เข้า v4 internal states โดยตรง
  ✦ Reranker ถูกรวมเข้า RetrieverV4.rank() แล้ว — ไม่มี CE model
  ✦ filter_product_specs ใช้ Qdrant payload filter ผ่าน retriever.filter_by_spec()
  ✦ _build_graph_summary รับ GraphResult แทน GraphContext
  ✦ ทุกอย่างอื่น (UI, logger, tools, system prompt) คงเดิม
"""

import os, json, re, base64, csv, torch, time
from io import BytesIO
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
from dotenv import load_dotenv
load_dotenv()

import gradio as gr
import google.generativeai as genai

from transformers import (
    AutoTokenizer, AutoModel,
    ViTImageProcessor, ViTForImageClassification,
    SiglipModel, SiglipProcessor,
)
from peft import PeftModel

from knowledge_graph import KnowledgeGraph
from retriever_v4 import RetrieverV4, RetrieverV4Config, ViTSignal, GraphResult, RankedProduct, RankedResult
from pipeline_logger import PipelineLogger

from modutech_gates import Catalogue
from modutech_router import CatalogueSearch, modutech_retriever_config_kwargs
from dialogue_state import DialogueManager, DialogState, VitInfo
from modutech_turn import build_modutech_turn, make_search_tool

# ================================================================
# CONFIG
# ================================================================
GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("⚠️ ไม่พบ GEMINI_API_KEY")
QDRANT_PATH     = "./qdrant_db"
JSON_PATH       = "./extracted_data_v3_machine_ready.json"
KG_DB_PATH      = "./movex_kg.db"
VIT_MODEL_PATH  = "./movex-lora-tuned-3"
VIT_BASE_MODEL  = "google/vit-base-patch16-224-in21k"
VIT_NUM_LABELS  = 10
SIGLIP_MODEL_ID = "google/siglip-base-patch16-224"
JINA_MODEL_ID   = "jinaai/jina-embeddings-v3"
GEMINI_MODEL    = "gemini-flash-latest"

# ── Modutech (สายที่สอง — path และ collection แยกจาก Movex) ──
MODUTECH_JSON   = "./modutech_v30_1.json"
MODUTECH_ENABLED = os.path.exists(MODUTECH_JSON) and os.path.exists("./qdrant_modutech_db")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# cudnn autotuner — เลือก kernel ที่เร็วที่สุดสำหรับ input shape ที่ใช้ซ้ำๆ (ภาพขนาดคงที่)
# ช่วยเร่งความเร็วรอบ inference ถัดๆ ไปเมื่อรันบน GPU
if DEVICE == "cuda":
    torch.backends.cudnn.benchmark = True

# ================================================================
# STARTUP
# ================================================================
print("=" * 60)
print("  Movex + Modutech Chatbot v5 — dialogue gate + Modutech gates")
print(f"  Device: {DEVICE}")
print("=" * 60)

with open(JSON_PATH, "r", encoding="utf-8") as f:
    _data = json.load(f)

product_db  = {p["product_id"]: p for p in _data.get("products", []) if "product_id" in p}
material_db = {m["material_code"]: m for m in _data.get("Materials", [])}
process_db  = {p["manufacturing_process"]: p for p in _data.get("manufacturing_process", [])}
print(f"✅ JSON: {len(product_db)} products | {len(material_db)} materials | {len(process_db)} processes")

print("⏳ โหลด Jina v3...")
jina_tokenizer = AutoTokenizer.from_pretrained(JINA_MODEL_ID, trust_remote_code=True)
jina_model = AutoModel.from_pretrained(
    JINA_MODEL_ID, trust_remote_code=True,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    attn_implementation="eager"
).to(DEVICE)
jina_model.eval()
print(f"✅ Jina v3 ({'fp16' if DEVICE == 'cuda' else 'fp32'})")

print("⏳ โหลด ViT LoRA...")
vit_processor = ViTImageProcessor.from_pretrained(VIT_BASE_MODEL)
_base_vit     = ViTForImageClassification.from_pretrained(VIT_BASE_MODEL, num_labels=VIT_NUM_LABELS)
vit_model     = PeftModel.from_pretrained(_base_vit, VIT_MODEL_PATH)
vit_model.eval().to(DEVICE)
if DEVICE == "cuda":
    vit_model.half()
print(f"✅ ViT LoRA ({'fp16' if DEVICE == 'cuda' else 'fp32'})")

print("⏳ โหลด SigLIP...")
siglip_processor = SiglipProcessor.from_pretrained(SIGLIP_MODEL_ID)
siglip_model     = SiglipModel.from_pretrained(SIGLIP_MODEL_ID).to(DEVICE)
siglip_model.eval()
if DEVICE == "cuda":
    siglip_model.half()
print(f"✅ SigLIP ({'fp16' if DEVICE == 'cuda' else 'fp32'})")

kg = KnowledgeGraph.load_from_db(KG_DB_PATH)

_retriever_inner = RetrieverV4(
    cfg=RetrieverV4Config(qdrant_path=QDRANT_PATH, device=DEVICE),
    jina_tokenizer=jina_tokenizer,
    jina_model=jina_model,
    vit_processor=vit_processor,
    vit_model=vit_model,
    siglip_processor=siglip_processor,
    siglip_model=siglip_model,
    product_db=product_db,
    kg=kg,
)
print("✅ RetrieverV4 + KnowledgeGraph")

genai.configure(api_key=GEMINI_API_KEY)
print("✅ Gemini configured")

logger = PipelineLogger(log_dir="./logs")
print(f"✅ PipelineLogger → {logger.log_paths()}")


class _NullKG:
    """Modutech ยังไม่ต่อ KG — คืน context ว่างให้ RetrieverV4._graph_traverse"""
    def expand_products(self, **_):
        class _Ctx:
            compatible_products, same_series, material_details, process_details = [], [], {}, {}
        return _Ctx()


modutech_cat = Catalogue(MODUTECH_JSON)          # ตรวจ v30 + นับครบ ไม่ใช่ v30 = หยุด
modutech_search = CatalogueSearch(modutech_cat)
dialogue = DialogueManager(product_db, modutech_cat, movex_materials=material_db)
_dialog_states: dict[str, dict] = {}             # session_id → DialogState.to_dict()

modutech_retriever = None
if MODUTECH_ENABLED:
    _mod_product_db = {
        p["belt_code_id"]: {"product_id": p["belt_code_id"], "Ref": p["belt_code"],
                            "belt_series": p.get("belt_series", ""),
                            "product_type": p["retrieval_card"].get("surface_type", ""),
                            "Material": ",".join(p["retrieval_card"].get("materials") or [])}
        for p in modutech_cat.products
    }
    modutech_retriever = RetrieverV4(
        cfg=RetrieverV4Config(**modutech_retriever_config_kwargs(), device=DEVICE),
        jina_tokenizer=jina_tokenizer, jina_model=jina_model,
        vit_processor=vit_processor, vit_model=vit_model,
        siglip_processor=siglip_processor, siglip_model=siglip_model,
        product_db=_mod_product_db, kg=_NullKG(),
    )
    print(f"✅ Modutech: {len(_mod_product_db)} สายพาน · collection modutech_belt_chunks")
else:
    print("⚠️ Modutech retriever ปิดอยู่ (ไม่พบ ./qdrant_modutech_db) — ยังตอบจากรหัสและค้นทั้งฐานได้")
print("=" * 60)


# ================================================================
# HELPERS
# ================================================================

def image_to_base64_html(pil_image):
    if pil_image is None:
        return ""
    buf = BytesIO()
    pil_image.thumbnail((400, 400))
    pil_image.save(buf, format="JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f'<img src="data:image/jpeg;base64,{b64}" style="max-width:250px;border-radius:10px;margin-bottom:10px;" />'


def _enrich_product(p: dict) -> dict:
    p = p.copy()
    mat  = p.get("Material", "")
    proc = p.get("manufacturing_process", "")
    if mat in material_db:
        p["material_details"] = material_db[mat]
    if proc in process_db:
        p["manufacturing_details"] = process_db[proc]
    return p


def _build_graph_summary(graph_result) -> str:
    """
    สร้าง graph summary สำหรับ LLM
    รองรับทั้ง GraphResult (v4) และ GraphContext (v3 compat)
    """
    if graph_result is None:
        return "ไม่มี graph context"

    # v4 GraphResult
    if isinstance(graph_result, GraphResult):
        summary = graph_result.to_llm_context()
        return summary if summary else "ไม่มี graph context"

    # v3 GraphContext compat (fallback)
    parts = []
    if hasattr(graph_result, 'seed_products') and graph_result.seed_products:
        parts.append(f"[Seed products]: {', '.join(graph_result.seed_products)}")
    if hasattr(graph_result, 'compatible_products') and graph_result.compatible_products:
        parts.append(
            f"[Compatible parts สำหรับ {', '.join(graph_result.seed_products)}]: "
            f"{', '.join(graph_result.compatible_products[:5])}"
        )
    if hasattr(graph_result, 'same_series') and graph_result.same_series:
        parts.append(f"[สินค้า series เดียวกัน]: {', '.join(graph_result.same_series[:5])}")
    if hasattr(graph_result, 'material_details') and graph_result.material_details:
        for mat, info in graph_result.material_details.items():
            full = info.get("full_name", mat)
            desc = info.get("description", "")
            line = f"[วัสดุของ {', '.join(graph_result.seed_products)}]: รหัส={mat}, ชื่อ={full}"
            if desc:
                line += f", คุณสมบัติ={desc[:120]}"
            parts.append(line)
    if hasattr(graph_result, 'process_details') and graph_result.process_details:
        for proc, info in graph_result.process_details.items():
            full = info.get("full_name", proc)
            desc = info.get("description", "")
            adv  = info.get("advantages", "") or info.get("best_used_for", "")
            line = f"[กระบวนการผลิตของ {', '.join(graph_result.seed_products)}]: วิธี={full}"
            if desc:
                line += f", รายละเอียด={desc[:150]}"
            if adv:
                line += f", ข้อดี={adv[:100]}"
            parts.append(line)
    return "\n".join(parts) if parts else "ไม่มี graph context"


# ================================================================
# INSTRUMENTED RETRIEVER
# ================================================================

class InstrumentedRetriever:
    """
    Wrapper รอบ RetrieverV4 — expose internal states สำหรับ logger
    v4 ใช้ direct attribute แทน monkey-patch เพราะ v4 ออกแบบให้ inspect ได้
    """

    def __init__(self, inner: RetrieverV4):
        self.inner = inner
        self._reset()

    def _reset(self):
        self._vit_signal:        ViTSignal = ViTSignal()
        self._graph_result:      GraphResult = GraphResult()
        self._score_map_snapshot: dict = {}
        self._ranked_snapshot:   list = []
        self._graph_triggered_by: str = "none"

        # compat fields สำหรับ logger v3
        self._vit_hits_raw:     list = []
        self._siglip_hits_raw:  list = []
        self._vit_top_score:    float = 0.0
        self._vit_series:       str = ""
        self._vit_dominant:     bool = False
        self._text_hits_raw:    list = []

    def search(self, query, query_image=None, query_type="general"):
        self._reset()
        result = self.inner.search(
            query=query,
            query_image=query_image,
            query_type=query_type,
        )

        # capture v4 states
        if result.vit_signal:
            sig = result.vit_signal
            self._vit_signal     = sig
            self._vit_top_score  = sig.confidence
            self._vit_series     = sig.series or ""
            self._vit_dominant   = sig.dominant

        if result.graph_result:
            gr = result.graph_result
            self._graph_result = gr
            if gr.seed_products:
                self._graph_triggered_by = (
                    "vit" if result.vit_signal and result.vit_signal.series
                    else "keyword"
                )

        # capture score_map จาก result โดยตรง (แก้ bug seeds=[])
        self._score_map_snapshot = getattr(result, '_score_map', {})
        self._ranked_snapshot    = getattr(result, '_ranked', [])

        return result

    @property
    def cfg(self):
        return self.inner.cfg


retriever = InstrumentedRetriever(_retriever_inner)


# ================================================================
# TOOL CALL TRACKER
# ================================================================

_tool_call_log: list[dict] = []
_datasheet_cache: dict[str, str] = {}   # resolved_pid -> base64 PNG

def _tracked(fn):
    import inspect
    params = list(inspect.signature(fn).parameters.keys())
    def wrapper(*args, **kwargs):
        entry = {"tool_name": fn.__name__, "args": {}, "result": "", "success": True}
        try:
            entry["args"] = {**dict(zip(params, args)), **kwargs}
            result = fn(*args, **kwargs)
            entry["result"] = result
            
            # 👇 แก้ไขจาก return result เป็น return dict เพื่อให้ Gemini SDK ทำงานได้
            return {"result": result} 
            
        except Exception as e:
            entry["success"] = False
            entry["result"]  = str(e)
            raise
        finally:
            _tool_call_log.append(entry)
    wrapper.__name__ = fn.__name__
    wrapper.__doc__  = fn.__doc__
    return wrapper


# ================================================================
# AI TOOLS
# ================================================================

def analyze_product_database(
    category: str = "all",
    series: str = "",
) -> str:
    """
    วิเคราะห์ฐานข้อมูลสินค้า Movex ทั้งหมด
    ใช้เมื่อลูกค้าถามภาพรวม เช่น "มีสินค้ากี่รุ่น" "มีอะไรบ้าง"

    Parameters
    ----------
    category : "all" | "chain" | "sprocket"
    series   : กรอง series เช่น "820" "880" (ว่างหมายถึงทั้งหมด)
    """
    filtered = {}
    for pid, p in product_db.items():
        if category == "chain" and "chain" not in pid.lower():
            continue
        if category == "sprocket" and "sprocket" not in pid.lower():
            continue
        if series and series not in str(p.get("series", "")):
            continue
        filtered[pid] = p

    if not filtered:
        return f"ไม่พบสินค้าในหมวด {category} series {series}"

    chains    = [pid for pid in filtered if "chain"    in pid.lower()]
    sprockets = [pid for pid in filtered if "sprocket" in pid.lower()]

    result = f"พบสินค้าทั้งหมด {len(filtered)} รุ่น\n"
    result += f"  Chain: {len(chains)} รุ่น\n"
    result += f"  Sprocket: {len(sprockets)} รุ่น\n"
    if chains:
        result += f"  Chain list: {', '.join(sorted(chains)[:15])}\n"
    if sprockets:
        result += f"  Sprocket list: {', '.join(sorted(sprockets)[:10])}\n"
    return result


def compare_specific_products(product_ids: str) -> str:
    """
    เปรียบเทียบสินค้าหลายรุ่นแบบ side-by-side
    ใช้เมื่อลูกค้าต้องการเปรียบเทียบ 2 รุ่นขึ้นไป

    Parameters
    ----------
    product_ids : รหัสสินค้าคั่นด้วยจุลภาค เช่น "LF820_K325,LF820_K750"
    """
    ids = [x.strip() for x in product_ids.split(",")]
    results = []
    for pid in ids:
        # ค้นหาแบบ fuzzy ถ้าไม่เจอ exact match
        p = product_db.get(pid)
        if not p:
            for k in product_db:
                if pid.lower() in k.lower() or k.lower() in pid.lower():
                    p = product_db[k]
                    pid = k
                    break
        if not p:
            results.append(f"{pid}: ไม่พบในระบบ")
            continue
        specs = {
            "series":         p.get("series", "-"),
            "product_type":   p.get("product_type", "-"),
            "plate_width_mm": p.get("plate_width_mm", "-"),
            "pitch_mm":       p.get("pitch_mm", "-"),
            "max_load_n":     p.get("Max_Working_Load_N", p.get("max_load_n", "-")),
            "material":       p.get("Material", "-"),
            "min_curve_radius_mm": p.get("Min_curve_radius_mm", p.get("Radius_min_mm", "-")),
            "Ref":            p.get("Ref", "-"),
        }
        lines = [f"=== {pid} ==="]
        for k, v in specs.items():
            lines.append(f"  {k}: {v}")
        results.append("\n".join(lines))

    return "\n\n".join(results) if results else "ไม่พบสินค้าที่ระบุ"


def filter_product_specs(
    spec_type: str,
    condition: str,
    target_value: float,
    product_type: str = None,
    sort_by: str = None,    
    sort_order: str = "asc",     
) -> str:
    """
    กรองสินค้าตาม spec — ใช้ Qdrant payload filter (v4)
    ใช้เมื่อลูกค้าระบุเงื่อนไขตัวเลข เช่น "กว้างไม่เกิน 100mm" "รับโหลดมากกว่า 5000N"

    Parameters
    ----------
    spec_type    : "width" | "load" | "radius" | "pitch" | "weight" | "teeth" | "bore"
    condition    : "<" | "<=" | ">" | ">=" | "=="
    target_value : ค่าตัวเลข เช่น 100.0
    """
    # ลอง Qdrant filter ก่อน — เร็วกว่า JSON loop มาก
    try:
        hits = _retriever_inner.filter_by_spec(spec_type, condition, target_value)
        if hits:
            lines = [f"พบ {len(hits)} รุ่นที่ {spec_type} {condition} {target_value}:"]
            for pid, val in sorted(hits, key=lambda x: str(x[0])):
                lines.append(f"  {pid} ({val})")
            return "\n".join(lines)
    except Exception as e:
        print(f"⚠️ Qdrant filter fallback to JSON: {e}")

    # Fallback: JSON loop (compat กับ field ที่ยังไม่ได้ index ใน Qdrant)
    key_map = {
        "width":   ["plate_width_mm", "Width_mm"],
        "load":    ["Max_Working_Load_N", "max_load_n"],
        "radius":  ["Min_curve_radius_mm", "Radius_min_mm"],
        "pitch":   ["pitch_mm", "Pitch_mm"],
        "weight":  ["weight_kg_m"],
        "teeth":   ["z_teeth"],
        "bore":    ["bore_mm"],
    }
    target_keys = key_map.get(spec_type.lower(), [spec_type])
    matched = []
    for pid, p in product_db.items():
        val = None
        for key in target_keys:
            if key in p and p[key] is not None:
                try:
                    val = float(p[key])
                except (ValueError, TypeError):
                    nums = re.findall(r'\d+\.?\d*', str(p[key]))
                    val = float(nums[0]) if nums else None
                if val is not None:
                    break
        if val is None and spec_type.lower() == "width":
            km = re.search(r'K(\d{3,4})', pid)
            if km:
                val = (float(km.group(1)) / 100) * 25.4
        if val is not None:
            if product_type and product_type.lower() not in p.get("product_type", "").lower():
                continue
            ok = (condition == "<"  and val <  target_value or
                  condition == "<=" and val <= target_value or
                  condition == ">"  and val >  target_value or
                  condition == ">=" and val >= target_value or
                  condition == "==" and val == target_value)
            if ok:
                matched.append(f"{pid} ({val})")
    if sort_by and matched:
        matched.sort(key=lambda x: float(re.search(r'[\d.]+', x.split('(')[1]).group()), 
                     reverse=(sort_order == "desc"))            
    if not matched:
        return f"ไม่พบสินค้าที่มี {spec_type} {condition} {target_value}"
    return f"พบ {len(matched)} รุ่น ได้แก่: {', '.join(matched)}"


def generate_datasheet(product_id: str) -> str:
    """
    สร้าง datasheet PNG (drawing + spec table) สำหรับสินค้า Movex แล้วส่งให้ลูกค้า
    ใช้ทันทีเมื่อลูกค้าพูดว่า "ขอ drawing" / "ขอ datasheet" / "ขอรูป" / "แสดงแบบ"

    Parameters
    ----------
    product_id : รหัสสินค้า เช่น "movex_chain_LF820_K325", "LF820_K325", "54201"
                 รองรับ fuzzy match — ใส่แค่บางส่วนก็ได้
    """
    pid = product_id.strip()
    if pid not in product_db:
        for k in product_db:
            if pid.lower() in k.lower() or k.lower().endswith(pid.lower()):
                pid = k
                break
    if pid not in product_db:
        return f"ไม่พบสินค้า '{product_id}' ในระบบ กรุณาตรวจสอบรหัสสินค้า"

    if pid in _datasheet_cache:
        # ตอบกลับเพื่อให้ Gemini รู้ว่า "แสดงไปแล้ว" ไม่ต้องพยายามสร้างใหม่หรือเรียก Tool ซ้ำ
        return f"ข้อมูล Datasheet ของ {pid} ได้ถูกแสดงบนหน้าจอเรียบร้อยแล้วครับ ลูกค้าสามารถดูรูปและรายละเอียดด้านบนได้เลย"

    p = product_db[pid]

    # ── หา drawing image ─────────────────────────────────────────────
    drawing_img = None
    for img_entry in p.get("images", []):
        if img_entry.get("type") == "drawing":
            img_path = img_entry["image_path"]
            if os.path.exists(img_path):
                drawing_img = Image.open(img_path).convert("RGB")
                break

    # ── Build spec rows ──────────────────────────────────────────────
    # Ref: chain ใช้ "Ref", sprocket ใช้ "Art_Nr"
    ref_val = p.get("Ref") or p.get("Art_Nr") or pid
    spec_rows = [
        ("Ref / Art.Nr", ref_val),
        ("Series",       p.get("series", "-")),
        ("Type",         p.get("product_type", "-")),
    ]
    for field, label in [
        # --- Chain fields ---
        ("Plate_Width_mm",      "Width (mm)"),
        ("Weight_kg_m",         "Weight (kg/m)"),
        ("Max_Working_Load_N",  "Max Load (N)"),
        ("Material",            "Material"),
        ("Min_curve_radius_mm", "Min Curve R (mm)"),
        ("Pitch_mm",            "Pitch (mm)"),
        # --- Sprocket fields ---
        ("Sprocket_Type",       "Sprocket Type"),
        ("manufacturing_process", "Process"),
        ("Z_Teeth",             "Teeth (Z)"),
        ("Bore_mm",             "Bore (mm)"),
        ("PD_mm",               "PD (mm)"),
        ("OD_mm",               "OD (mm)"),
        ("S_mm",                "S (mm)"),
        ("compatible_chain",    "Compatible Chain"),
    ]:
        val = p.get(field)
        if val is not None:
            spec_rows.append((label, str(val)))

    # ── Canvas setup ─────────────────────────────────────────────────
    W, H      = 920, 520
    DW        = 460        # drawing section width
    HEADER_H  = 52
    FOOTER_H  = 24
    PAD       = 18
    ROW_H     = 26

    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    draw   = ImageDraw.Draw(canvas)

    def _font(size: int):
        for path in [
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/calibri.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]:
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
        return ImageFont.load_default()

    f_brand = _font(20)
    f_title = _font(16)
    f_label = _font(13)
    f_val   = _font(13)
    f_foot  = _font(11)

    # Header
    draw.rectangle([0, 0, W, HEADER_H], fill=(20, 50, 100))
    draw.text((PAD, 16),  f"Product Datasheet {p.get('Ref', pid)}",
              fill=(255, 255, 255), font=f_title)

    # Drawing area
    draw.rectangle([0, HEADER_H, DW, H - FOOTER_H], fill=(248, 248, 250))
    if drawing_img:
        max_w = DW - 2 * PAD
        max_h = H - HEADER_H - FOOTER_H - 2 * PAD
        drawing_img.thumbnail((max_w, max_h), Image.LANCZOS)
        dx = PAD + (max_w - drawing_img.width) // 2
        dy = HEADER_H + PAD + (max_h - drawing_img.height) // 2
        canvas.paste(drawing_img, (dx, dy))
    else:
        draw.text((PAD, HEADER_H + PAD), "No drawing available",
                  fill=(180, 180, 180), font=f_label)

    # Divider
    draw.line([(DW, HEADER_H), (DW, H - FOOTER_H)], fill=(180, 180, 180), width=1)

    # Spec table
    sx = DW + PAD
    y  = HEADER_H + PAD
    for i, (label, value) in enumerate(spec_rows):
        if y + ROW_H > H - FOOTER_H - PAD:
            break
        row_bg = (240, 245, 255) if i % 2 == 0 else (255, 255, 255)
        draw.rectangle([sx - 4, y - 2, W - 4, y + ROW_H - 6], fill=row_bg)
        draw.text((sx,       y), label + ":", fill=(50, 80, 130),  font=f_label)
        draw.text((sx + 155, y), value,       fill=(20, 20, 20),   font=f_val)
        y += ROW_H

    # Footer
    draw.rectangle([0, H - FOOTER_H, W, H], fill=(20, 50, 100))
    draw.text((PAD, H - FOOTER_H + 5),
              "Movex — Industrial Chain & Conveyor Solutions",
              fill=(180, 200, 230), font=f_foot)

    # Encode
    buf = BytesIO()
    canvas.save(buf, format="PNG", optimize=True)
    _datasheet_cache[pid] = base64.b64encode(buf.getvalue()).decode("utf-8")

    return f"สร้าง datasheet ของ {p.get('Ref', pid)} สำเร็จแล้ว"


tracked_analyze   = _tracked(analyze_product_database)
tracked_compare   = _tracked(compare_specific_products)
tracked_filter    = _tracked(filter_product_specs)
tracked_datasheet = _tracked(generate_datasheet)


# ================================================================
# SYSTEM PROMPT
# ================================================================

def _build_system_prompt(json_context: str, graph_summary: str,vit_signal=None, decision=None) -> str:
    decision_facts = "- ไม่มีข้อมูลจากระบบ"
    if decision is not None and decision.movex:
        mx = decision.movex
        refs = [product_db[p].get("Ref") or product_db[p].get("Art_Nr") or p
                for p in (mx.get("pids") or []) if p in product_db]
        decision_facts = (f"- รุ่นที่ลูกค้าหมายถึง: {', '.join(refs) or '-'}\n"
                          f"- ซีรีส์: {mx.get('series') or '-'}\n"
                          f"- ระดับ: {mx.get('level') or '-'}\n"
                          f"- หัวข้อคำถาม: {mx.get('topic') or '-'}\n"
                          f"- เหตุผล: {decision.reason}")
    if decision is not None and decision.sales_topics:
        decision_facts += (f"\n- เรื่อง {'/'.join(decision.sales_topics)}: ระบบจะแนบข้อความส่งต่อฝ่ายขายท้ายคำตอบเอง "
                           f"ให้ตอบเฉพาะส่วนสเปก ห้ามพูดถึงเรื่องนี้ และห้ามให้ตัวเลขราคา/ระยะเวลาส่ง/สต็อก")
    vit_context = ""
    if vit_signal and vit_signal.series:
        vit_context = f"""
    [ViT Signal — ผลการวิเคราะห์รูปภาพ]
    series     : {vit_signal.series}
    confidence : {vit_signal.confidence:.3f}
    dominant   : {"✅ มั่นใจสูง (≥0.85)" if vit_signal.dominant else "⚠️ ปานกลาง (0.50–0.85)"}
    top_pid    : {vit_signal.top_pid or "-"}
    """
    return f"""
คุณคือ 'Movex Assistant' วิศวกรฝ่ายขายมืออาชีพของบริษัท Movex
ตอบภาษาไทย เป็นธรรมชาติ มีหางเสียง (ครับ/ค่ะ)

{vit_context}
[RAG Context — สินค้าที่ระบบค้นหามาได้ เรียงตาม final_score สูงสุด]
{json_context}

[GraphRAG Context — ความสัมพันธ์ของสินค้า]
{graph_summary}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔴 ผลตรวจจากระบบ (ตัดสินในโค้ดแล้ว — ห้ามถามรุ่นซ้ำ)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{decision_facts}
- ระบบตรวจแล้วว่าข้อมูลพอสำหรับคำถามนี้ ให้ตอบเลย ไม่ต้องถามรุ่นหรือความกว้างซ้ำ
- ถ้าระดับ = series: ตอบเฉพาะค่าที่เหมือนกันทั้งซีรีส์ และบอกว่าเป็นค่าของทั้งซีรีส์
- ถ้าระดับ = product: ยึดรุ่นที่ระบบระบุ ถ้า rank 1 ใน RAG Context ไม่ตรงกับรุ่นนี้ ให้ใช้รุ่นที่ระบบระบุ

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🟢 ขั้นตอนที่ 3: ตอบ spec (เฉพาะเมื่อระบุรุ่นชัดแล้ว)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- ยึดข้อมูลจาก rank 1 เป็นหลัก
- ห้ามมั่วตัวเลขหรือรหัสสินค้าเด็ดขาด
- "เลี้ยวได้ไหม" → ดู product_type ก่อนเสมอ (ห้ามดู radius อย่างเดียว)
  product_type = "Straight Running Chain"  → เลี้ยวไม่ได้ ตอบทันทีโดยไม่ต้องดู radius
  product_type = "Sideflexing Chain"       → เลี้ยวได้ แล้วค่อยบอก Min_curve_radius_mm
- "workload" → ดู Max_Working_Load_N
- "ใช้สเตอร์ตัวไหน" → ตอบแค่รหัสสเตอร์ที่ compatible จาก GraphRAG Context
  ห้ามอธิบาย spec ยาว ตอบแค่ชื่อรุ่น + จำนวนฟัน (ถ้ามี) เท่านั้น
- "สเตอร์นั้นผลิตยังไง" → ถ้า GraphRAG มีสเตอร์ compatible หลายตัว ต้องตอบให้ครบทุกตัว
  ห้ามตอบแค่ตัวเดียวถ้า context มีมากกว่า 1
- ถ้า final_score < 0.2 ทุกรุ่น → แจ้งว่าข้อมูลไม่เพียงพอ
- ห้ามเรียก Tool เพื่อตอบ spec ที่อยู่ใน RAG Context แล้ว

🔴 กฎเด็ดขาดเรื่อง Compatible (ห้ามละเมิด):
- คำถามเรื่อง "ใช้คู่กับโซ่อะไร" / "compatible chain" / "โซ่ที่ใช้ได้" / "ใช้สเตอร์ตัวไหน"
  → ตอบจาก [GraphRAG Context] เท่านั้น ห้ามใช้ field "compatible_chain" หรือ "series" จาก RAG Context (JSON) โดยเด็ดขาด
  → ถ้า GraphRAG Context ว่างหรือไม่มี compatible → ตอบว่า "ไม่มีข้อมูล compatible ในระบบครับ" ห้ามคาดเดาหรืออ้างอิงจาก JSON field

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔧 กฎการใช้ Tool (เรียกเมื่อจำเป็นเท่านั้น)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. ถามจำนวน/ภาพรวมทั้งหมด → Tool: analyze_product_database
2. ถามหา Sprocket คู่ → ดูจาก GraphRAG Context [Compatible parts] ก่อนเสมอ
   ห้ามใช้ Tool เพื่อหา Sprocket ถ้า GraphRAG Context มีข้อมูลอยู่แล้ว
3. เปรียบเทียบ 2 รุ่นขึ้นไป → Tool: compare_specific_products
4. กรองตาม spec ตัวเลข เช่น "กว้างไม่เกิน 100mm" → Tool: filter_product_specs
5. query มีคำต่อไปนี้ → เรียก filter_product_specs ทันทีก่อนตอบ ห้ามตอบจาก RAG Context โดยตรง:
   - "เบาสุด / น้ำหนักน้อยสุด / lightweight"  → spec_type="weight", condition=">=", target_value=0, sort_by="weight", sort_order="asc"
   - "รับโหลดสูงสุด / แข็งแรงสุด"             → spec_type="load",   condition=">=", target_value=0, sort_by="load",   sort_order="desc"
   - "กว้างสุด / กว้างมากสุด"                  → spec_type="width",  condition=">=", target_value=0, sort_by="width",  sort_order="desc"
   - "width ≥ X / กว้างอย่างน้อย X mm"         → spec_type="width",  condition=">=", target_value=X, product_type="chain"
6. ลูกค้าพูดคำต่อไปนี้ → เรียก generate_datasheet ทันที ห้ามตอบข้อความล้วน:
   - "ขอ drawing" / "drawing หน่อย" / "แสดง drawing"
   - "ขอ datasheet" / "datasheet หน่อย"
   - "ขอรูป" / "ดูรูป" / "แสดงรูป" / "รูปสินค้า"
   - "แบบ" / "แสดงแบบ" / "ขอดูแบบ"
   → ใช้ product_id ของ rank 1 ใน RAG Context หรือ product ที่ลูกค้าพูดถึง
   → หลังจากแสดง datasheet แล้ว บอก spec สั้นๆ ด้วย
   → (กฎเพิ่มเติม): ห้ามเรียก generate_datasheet ซ้ำสำหรับสินค้าเดิม 
     หากในประวัติการสนทนา (Chat History) เพิ่งมีการแสดง datasheet ของสินค้านั้นไปแล้ว 
     ยกเว้นลูกค้าจะสั่ง "ขอรูปอีกครั้ง" หรือ "ขอดูรุ่นอื่น"
"""


# ================================================================
# MODUTECH TURN (v5)
# ================================================================

tracked_search_catalogue = make_search_tool(modutech_search, tracked=_tracked)


def _modutech_vector_search(query: str) -> list[str]:
    """คืน belt_code_id จาก collection modutech_belt_chunks (ถ้าเปิดใช้)"""
    if modutech_retriever is None:
        return []
    r = modutech_retriever.search(query=query, query_image=None, query_type="general")
    ranked = modutech_retriever.rank(r)
    return [p.product_id for p in ranked.products]


def _history_for_gemini(chat_history: list) -> list:
    out = []
    for msg in chat_history:
        clean = re.sub(r'<[^>]+>', '', msg.get("content", "")).strip()
        if clean:
            out.append({"role": "model" if msg["role"] == "assistant" else "user", "parts": [clean]})
    return out


def _modutech_reply(decision, chat_history: list, img_input) -> tuple[str, str]:
    turn = build_modutech_turn(decision, modutech_cat, modutech_search,
                               vector_search=_modutech_vector_search)
    refs = "**Modutech:** " + " · ".join(turn.debug[:12]) + "\n\n"
    model_chat = genai.GenerativeModel(
        model_name=GEMINI_MODEL,
        system_instruction=turn.system_prompt,
        tools=[tracked_search_catalogue],
    )
    chat_session = model_chat.start_chat(history=_history_for_gemini(chat_history),
                                         enable_automatic_function_calling=True)
    response = chat_session.send_message([turn.user_prompt])
    try:
        text = response.text
    except ValueError:
        text = ""
        try:
            text = next((p.text for p in response.candidates[0].content.parts
                         if getattr(p, "text", "")), "")
        except Exception:
            pass
    return (text.strip() or "ขออภัยครับ ระบบไม่สามารถสร้างคำตอบได้ในขณะนี้ กรุณาลองถามใหม่อีกครั้ง"), refs


# ================================================================
# MAIN CHAT FUNCTION
# ================================================================

def chat_interaction(
    user_text: str,
    img_input,
    chat_history: list,
    last_pids: list = None,
    session_id: str = "",
    last_vit_signal = None,
    last_graph_result = None,
):
    global _tool_call_log

    if last_pids is None:
        last_pids = []

    if not user_text.strip() and img_input is None:
        return (chat_history, chat_history, "⚠️ กรุณาพิมพ์ข้อความหรืออัปโหลดรูปภาพ",
                last_pids, session_id, last_vit_signal, last_graph_result)

    if not session_id:
        session_id = logger.new_session()

    _tool_call_log = []
    t_start = time.perf_counter()

    # ── Display content ────────────────────────────────────
    display_content = ""
    if img_input:
        display_content += image_to_base64_html(img_input) + "<br>"
    if user_text:
        display_content += user_text
    if not display_content:
        display_content = "[แนบรูปภาพ]"

    raw_refs_html = "### 🔍 ข้อมูลอ้างอิง (Debug v4)\n\n"
    ai_reply      = ""
    error_msg     = None

    # ── Detect follow-up ────────────────────────────────────
    # follow-up = ไม่มีรูปใหม่ + พิมพ์สั้น + มี context จาก turn ก่อน
    is_followup = (
        img_input is None and
        user_text.strip() and
        len(user_text.split()) <= 6 and
        bool(last_pids or last_vit_signal or last_graph_result)
    )

    # ── Effective text — carry-over last_pids ───────────────
    effective_text = user_text or "ลูกค้าส่งรูปภาพมา"
    if is_followup and last_pids:
        # ถ้าเป็นคำถามต่อเนื่อง ให้ส่ง PID เข้าไปในระบบค้นหา (Internal) 
        # แต่อย่าส่งเข้าไปใน prompt หลักของ Gemini ทั้งหมดถ้าไม่ได้ถามหา Drawing
        if not any(word in user_text for word in ["ขอรูป", "ขอดู", "drawing", "datasheet"]):
             # ถ้าถามคำถามทั่วไป ไม่ต้องเน้น PID ใน text มากเกินไป 
             # แค่ให้ Retriever รู้ก็พอ
             pass 
        else:
             pid_prefix = " ".join(last_pids[:1])
             effective_text = f"{pid_prefix} {user_text}"

    query_type = "image" if img_input is not None else "general"

    raw_refs_html += f"**follow-up:** {is_followup} | **last_pids:** {last_pids}\n"

    # ── Start turn log ──────────────────────────────────────
    turn_log = logger.start_turn(
        session_id=session_id, 
        user_text=user_text,
        has_image=img_input is not None,
        last_pid=" ".join(last_pids) if last_pids else "",
        effective_query=effective_text,
        query_type=query_type,
    )
    raw_refs_html += f"**Turn:** `{turn_log.turn_id}`\n"
    raw_refs_html += f"**Effective query:** {effective_text}\n\n"

    decision = None
    vit_sig = last_vit_signal      # v5: กัน NameError ถ้า error ก่อน retrieval (v4 มีช่องนี้อยู่)
    try:
        # ════════════════════════════════════════════════════
        # PHASE 2 (v5): ตัดสินในโค้ดก่อนถึง LLM — ทาง จ
        # รูปภาพ: ต้องรัน retrieval ก่อนเพื่อได้ ViT signal (รูปเป็นของ Movex เสมอ)
        # ════════════════════════════════════════════════════
        pre_result, vit_info, rejected = None, None, False
        if img_input is not None:
            pre_result = retriever.search(query=effective_text, query_image=img_input, query_type=query_type)
            _s = retriever._vit_signal
            vit_info = VitInfo(_s.series, _s.confidence, _s.dominant, _s.top_pid or "")
            rejected = (not pre_result.products) and pre_result.image_search_used
        dstate = DialogState.from_dict(_dialog_states.get(session_id))
        decision = dialogue.decide(user_text, has_image=img_input is not None, vit=vit_info,
                                   image_rejected=rejected, state=dstate)
        _dialog_states[session_id] = decision.state.to_dict()
        raw_refs_html += (f"**Decision:** {decision.action} · brand={decision.brand} · "
                          f"{decision.reason}\n\n")

        if not decision.needs_llm:
            ai_reply = decision.reply_th
            _log_finish(turn_log, ai_reply, [], t_start, error=None)
            chat_history += [
                {"role": "user",      "content": display_content},
                {"role": "assistant", "content": ai_reply},
            ]
            return (chat_history, chat_history, raw_refs_html,
                    last_pids, session_id, last_vit_signal, last_graph_result)

        if decision.brand == "modutech":
            ai_reply, extra_refs = _modutech_reply(decision, chat_history, img_input)
            if decision.sales_handoff_th:
                ai_reply = f"{ai_reply}\n\n{decision.sales_handoff_th}"
            raw_refs_html += extra_refs
            _log_finish(turn_log, ai_reply, _tool_call_log, t_start, error=None)
            chat_history += [
                {"role": "user",      "content": display_content},
                {"role": "assistant", "content": ai_reply},
            ]
            return (chat_history, chat_history, raw_refs_html,
                    last_pids, session_id, last_vit_signal, last_graph_result)

        # ── Movex: ต่อ flow เดิมของ v4 ─────────────────────
        if img_input is None and decision.effective_text and decision.effective_text != user_text.strip():
            effective_text = decision.effective_text      # ลูกค้าเพิ่งตอบคำถามกลับ → รวมกับคำถามเดิม

        # ════════════════════════════════════════════════════
        # PHASE 3: Retrieval
        # carry-over vit_signal จาก turn ก่อนถ้าเป็น follow-up ไม่มีรูปใหม่
        # ════════════════════════════════════════════════════
        retrieval_result = pre_result if pre_result is not None else retriever.search(
            query=effective_text,
            query_image=img_input if img_input else None,
            query_type=query_type,
        )

        # ── carry-over vit_signal เมื่อไม่มีรูปใหม่ ─────────
        vit_sig = retriever._vit_signal
        if is_followup and not vit_sig.series and last_vit_signal:
            # ViT ไม่ได้รันเพราะไม่มีรูป ใช้ signal จาก turn ก่อนแทน
            vit_sig = last_vit_signal
            raw_refs_html += f"**ViT (carry-over):** series={vit_sig.series} conf={vit_sig.confidence:.3f}\n"

        # ── carry-over graph_result เมื่อเป็น follow-up ─────
        if is_followup and last_graph_result:
            # inject graph ของ turn ก่อนเข้า retrieval result
            # เพื่อให้ LLM เห็น compatible/process ของ turn ก่อนโดยไม่ต้อง traverse ใหม่
            if not (retrieval_result.graph_result and
                    retrieval_result.graph_result.seed_products):
                retrieval_result.graph_result = last_graph_result
                retrieval_result.graph_context = retriever.inner._graph_result_to_context(last_graph_result)
                raw_refs_html += f"**Graph (carry-over):** seeds={last_graph_result.seed_products}\n"
        if vit_sig.series:
            raw_refs_html += (
                f"**ViT series:** {vit_sig.series} "
                f"(conf={vit_sig.confidence:.3f}, "
                f"dominant={'✅' if vit_sig.dominant else '❌'})\n"
            )
        if retrieval_result.graph_result and retrieval_result.graph_result.seed_products:
            gr = retrieval_result.graph_result
            raw_refs_html += (
                f"**Graph:** seeds={gr.seed_products} | "
                f"compatible={gr.compatible[:3]} | "
                f"hop_trace={gr.hop_trace}\n"
            )
        raw_refs_html += "\n"

        # ── LOG ──────────────────────────────────────────────
        logger.log_keyword(turn_log, retriever._score_map_snapshot)

        if img_input is not None:
            logger.log_image(
                turn_log,
                vit_hits_raw=retriever._vit_hits_raw,
                siglip_hits_raw=retriever._siglip_hits_raw,
                vit_series=retriever._vit_series or None,
                vit_dominant=retriever._vit_dominant,
                vit_top_score=retriever._vit_top_score,
                vit_confidence_threshold=_retriever_inner.cfg.vit_confidence_threshold,
            )

        logger.log_graph(
            turn_log,
            graph_context=retrieval_result.graph_context,
            triggered_by=retriever._graph_triggered_by,
        )

        logger.log_text(
            turn_log,
            hits_raw=retriever._text_hits_raw,
            skipped=False,   # v4 ไม่ skip text search แม้ ViT dominant
            skip_reason="",
        )

        logger.log_rrf(
            turn_log,
            score_map=retriever._score_map_snapshot,
            ranked=retriever._ranked_snapshot,
        )

        # ── Image not found ─────────────────────────────────
        if img_input and not retrieval_result.products and retrieval_result.image_search_used:
            ai_reply = (
                "❌ ระบบตรวจพบว่ารูปภาพที่อัปโหลดไม่เกี่ยวข้องกับสินค้าในระบบ "
                "กรุณาอัปโหลดภาพโซ่หรือสเตอร์ที่ชัดเจนครับ"
            )
            _log_finish(turn_log, ai_reply, [], t_start, error=None)
            chat_history += [
                {"role": "user",      "content": display_content},
                {"role": "assistant", "content": ai_reply},
            ]
            return (chat_history, chat_history, raw_refs_html,
                    last_pids, session_id, last_vit_signal, last_graph_result)

        # ════════════════════════════════════════════════════
        # PHASE 4: Rank — Keyword Boost + Final Sort
        # (ย้ายมาจาก reranker.py — ไม่มี CE model แล้ว)
        # ════════════════════════════════════════════════════
        retrieval_result.vit_signal = vit_sig
        ranked_result = _retriever_inner.rank(retrieval_result)
        raw_refs_html += f"**Mode:** RRF+KeywordBoost (no CE)\n"

        logger.log_reranker(turn_log, ranked_result)

        # ── Build LLM context ───────────────────────────────
        llm_context: list[dict] = []
        for rank, p in enumerate(ranked_result.products, 1):
            pid = p.product_id
            raw_refs_html += (
                f"**{rank}. {pid}** "
                f"(final:{p.final_score:.3f} | CE:{p.cross_encoder_score:.2f} | "
                f"RRF:{p.rrf_score:.4f})\n"
            )
            if pid in product_db:
                enriched = _enrich_product(product_db[pid])
                enriched.update({
                    "_rank": rank,
                    "_final_score": round(p.final_score, 4),
                    "_best_chunk": p.best_chunk_text,
                })
                llm_context.append(enriched)

        if not llm_context:
            raw_refs_html += "⚠️ ไม่พบข้อมูลสินค้าที่ตรงกัน"

        # ════════════════════════════════════════════════════
        # PHASE 5: Gemini Agent
        # ════════════════════════════════════════════════════
        json_context  = json.dumps(llm_context, ensure_ascii=False, indent=2)

        # ── Graph fallback: ถ้า graph ไม่มี seeds ให้ traverse จาก rank-1 โดยตรง ──
        _gr = retrieval_result.graph_result
        if (not _gr or not _gr.seed_products) and ranked_result.products:
            _top_pid = ranked_result.products[0].product_id
            try:
                _compatible   = kg.get_compatible_products(_top_pid)
                _same_series  = kg.get_same_series(_top_pid)
                _mat_details  = kg.get_material_details(_top_pid)
                _proc_details = kg.get_process_details(_top_pid)
                if _compatible or _same_series or _mat_details or _proc_details:
                    _fallback_gr = GraphResult()
                    _fallback_gr.seed_products    = [_top_pid]
                    _fallback_gr.compatible       = _compatible
                    _fallback_gr.same_series      = _same_series
                    _fallback_gr.material_details = _mat_details
                    _fallback_gr.process_details  = _proc_details
                    retrieval_result.graph_result = _fallback_gr
                    raw_refs_html += f"**Graph (fallback rank1):** seeds=[{_top_pid}] compatible={_compatible[:3]}\n"
            except Exception as _e:
                print(f"[WARN] graph fallback failed: {_e}")

        graph_summary = _build_graph_summary(retrieval_result.graph_result)
        system_instruction = _build_system_prompt(json_context, graph_summary, vit_signal=vit_sig,
                                                  decision=decision)

        # --- สร้าง gemini_history (mirror วิธีที่ Svelte ทำ) ---
        gemini_history = []
        for msg in chat_history:
            content = msg.get("content", "")

            # ลบ HTML tags ทั้งหมด (img base64, styles) — เหมือน Svelte ทำ replace(/<[^>]*>/g, "")
            clean = re.sub(r'<[^>]+>', '', content).strip()

            # skip message ที่เป็น <img> ล้วน ๆ (หลังลบ tag แล้วว่างเปล่า)
            if not clean:
                continue

            role = "model" if msg["role"] == "assistant" else "user"
            gemini_history.append({"role": role, "parts": [clean]})

        img_part = None
        if img_input:
            buf = BytesIO()
            img_input.convert("RGB").save(buf, format="JPEG")
            img_part = {"mime_type": "image/jpeg", "data": buf.getvalue()}

        model_chat = genai.GenerativeModel(
            model_name=GEMINI_MODEL,
            system_instruction=system_instruction,
            tools=[tracked_analyze, tracked_compare, tracked_filter, tracked_datasheet],
        )
        chat_session = model_chat.start_chat(
            history=gemini_history,
            enable_automatic_function_calling=True,
        )
        prompt   = (decision.effective_text if decision.effective_text != user_text.strip() else user_text.strip()) \
                   or "ลูกค้าส่งรูปภาพมา ช่วยวิเคราะห์สเปกจาก RAG Context"
        response = chat_session.send_message([img_part, prompt] if img_part else [prompt])

        # safe-access: response.text throws ValueError เมื่อ Gemini ส่งแค่ tool call
        # โดยไม่มี text part (finish_reason=STOP แต่ parts ว่าง)
        def _extract_text(resp):
            try:
                return resp.text
            except ValueError:
                pass
            try:
                for part in resp.candidates[0].content.parts:
                    if hasattr(part, "text") and part.text:
                        return part.text
            except Exception:
                pass
            return ""

        # --- 1. ดึงข้อความตอบกลับรอบแรก ---
        ai_reply = _extract_text(response).strip()

        # --- 2. Populate datasheet_b64_list ก่อน (ต้องมาก่อน check ทุกอย่าง) ---
        datasheet_b64_list = []
        for tc in _tool_call_log:
            if tc["tool_name"] == "generate_datasheet" and tc["success"]:
                req = tc["args"].get("product_id", "").strip()
                b64 = _datasheet_cache.get(req)
                if not b64:
                    for k, v in _datasheet_cache.items():
                        if req.lower() in k.lower() or k.lower().endswith(req.lower()):
                            b64 = v
                            break
                if b64:
                    datasheet_b64_list.append(b64)

        # --- 3. ตรวจสอบและแก้ไข ai_reply ---
        # กรณีที่ 1: datasheet สำเร็จ แต่ ai_reply ว่างเปล่า
        if datasheet_b64_list and not ai_reply:
            pid_label = _tool_call_log[-1]["args"].get("product_id", "สินค้า")
            ai_reply = f"นี่คือข้อมูล datasheet ของ **{pid_label}** ที่คุณขอครับ"
            print(f"[INFO] Auto-generated reply for datasheet: {pid_label}")

        # กรณีที่ 2: ไม่มี datasheet และ ai_reply ว่าง — retry
        if not ai_reply and not datasheet_b64_list:
            print("[WARN] Gemini returned empty response — retrying once")
            retry_response = chat_session.send_message("กรุณาสรุปคำตอบสั้นๆ ให้ลูกค้าทราบด้วยครับ")
            ai_reply = _extract_text(retry_response).strip()

        # กรณีที่ 3: ยังว่างอีก
        if not ai_reply:
            ai_reply = "ขออภัยครับ ระบบไม่สามารถสร้างคำตอบได้ในขณะนี้ กรุณาลองถามใหม่อีกครั้ง"

    except Exception as e:
        ai_reply  = f"❌ เกิดข้อผิดพลาด: {str(e)}"
        error_msg = str(e)
        datasheet_b64_list = []
        print(f"[ERROR] {e}")
        import traceback; traceback.print_exc()

    if decision is not None and decision.sales_handoff_th and not error_msg:
        ai_reply = f"{ai_reply}\n\n{decision.sales_handoff_th}"   # คำถามผสม: ส่วนฝ่ายขายมาจากโค้ด

    _log_finish(turn_log, ai_reply, _tool_call_log, t_start, error=error_msg)

    chat_history += [
        {"role": "user",      "content": display_content},
        {"role": "assistant", "content": ai_reply},
    ]
    # แยก datasheet image ออกเป็น message ต่างหาก (frontend รับ text/image แยกกันได้)
    for b64 in datasheet_b64_list:
        chat_history.append({
            "role": "assistant",
            "content": (
                '<img src="data:image/png;base64,' + b64 +
                '" style="max-width:100%;border-radius:8px;'
                'box-shadow:0 2px 8px rgba(0,0,0,.15);margin:8px 0;" />'
            ),
        })

    # ── Extract pids จาก reply + reranked products ──────────
    # collect จาก 2 แหล่ง: ai_reply + reranked top products
    new_last_pids = []
    try:
        # 1. จาก reranked products — ครบที่สุด
        if "ranked_result" in dir() and ranked_result:
            for p in ranked_result.products[:3]:
                if p.product_id not in new_last_pids:
                    new_last_pids.append(p.product_id)

        # 2. จาก ai_reply — pid ที่ LLM พูดถึงจริงๆ (เพิ่มเข้าถ้ายังไม่มี)
        for m in re.finditer(
            r'\b(LF\s*\d+\s*(?:TAB\s*)?[A-Z]?\d+|LFN\s*\d+\s*\w+|\d{5})\b',
            ai_reply, re.IGNORECASE
        ):
            pid = m.group().strip()
            if pid not in new_last_pids:
                new_last_pids.append(pid)
            if len(new_last_pids) >= 5:
                break
    except Exception:
        pass

    # ── carry-over vit_signal และ graph_result ──────────────
    new_vit_signal    = vit_sig if vit_sig and vit_sig.series else last_vit_signal
    new_graph_result  = None
    try:
        if retrieval_result.graph_result and retrieval_result.graph_result.seed_products:
            new_graph_result = retrieval_result.graph_result
        else:
            new_graph_result = last_graph_result
    except Exception:
        new_graph_result = last_graph_result

    # chatStateOut ที่ส่งกลับไปให้ Svelte ต้อง strip <img> datasheet ออก
    # เพื่อไม่ให้ Gemini เห็น base64 ใน history แล้วเรียก generate_datasheet ซ้ำ
    chat_state_out = [
        msg for msg in chat_history
        if not (msg["role"] == "assistant" and msg.get("content", "").startswith('<img'))
    ]

    return (chat_history, chat_state_out, raw_refs_html,
            new_last_pids, session_id, new_vit_signal, new_graph_result)


def _log_finish(turn_log, ai_reply: str, tool_calls: list, t_start: float, error=None):
    latency_ms = (time.perf_counter() - t_start) * 1000
    logger.log_llm(turn_log, ai_reply=ai_reply, tool_calls=tool_calls, latency_ms=latency_ms)
    logger.finish_turn(turn_log, error=error)
    print(
        f"📋 Logged {turn_log.turn_id}  "
        f"latency={latency_ms:.0f}ms  "
        f"top1={turn_log.summary_top1_pid}  "
        f"vit_conf={retriever._vit_top_score:.3f}  "
        f"tools={turn_log.summary_tool_calls}"
    )


# ================================================================
# GRADIO UI — คงเดิมทุกอย่าง
# ================================================================

css = ".gradio-container {max-width: 900px !important}"
js_code = """
function() {
    document.addEventListener('keydown', function(e) {
        const textbox = document.querySelector('#chat-textbox textarea');
        if (e.target === textbox && e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            document.querySelector('#submit-button').click();
        }
    });
}
"""

with gr.Blocks(css=css, js=js_code, title="Movex + Modutech Sales AI Assistant v5") as demo:
    chat_state          = gr.State([])
    last_pids_state     = gr.State([])        # list of pids จาก turn ก่อน
    session_state       = gr.State("")
    last_vit_state      = gr.State(None)      # ViTSignal carry-over
    last_graph_state    = gr.State(None)      # GraphResult carry-over

    gr.Markdown("<h1 style='text-align:center;'>🔗 Movex + Modutech Sales AI Assistant v5</h1>")
    gr.Markdown(
        "<p style='text-align:center;color:#666;'>"
        "Jina v3 + ViT LoRA + SigLIP + Multi-hop GraphRAG + "
        "Confidence-Weighted RRF + Enriched Reranker + Gemini"
        "</p>"
    )

    chatbot_ui = gr.Chatbot(height=600, type="messages", show_label=False)

    with gr.Row():
        # v5: กล่องเดิม 50×50 + container=False กดแล้วไม่เปิดหน้าต่างเลือกไฟล์ (เป็นทั้ง v4 และ v5)
        #     ขยายกล่องและเปิด container ให้มีพื้นที่คลิก/ลากวางจริง
        img_input_ui = gr.Image(
            type="pil", label="รูปสินค้า", show_label=True, container=True,
            scale=0, height=140, width=140, min_width=140,
            sources=["upload", "clipboard"], interactive=True,
        )
        txt_input = gr.Textbox(
            elem_id="chat-textbox", show_label=False,
            placeholder="พิมพ์ข้อความ · แนบรูปได้ที่กล่องซ้าย หรือปุ่ม 📷 ด้านล่าง",
            scale=10, container=False, lines=1, max_lines=5,
        )
        submit_btn = gr.Button("➤", elem_id="submit-button", variant="primary", scale=0)

    with gr.Row():
        # ปุ่มสำรอง: เลือกไฟล์ผ่านปุ่มปกติ แล้วใส่รูปลงกล่องรูปสินค้า (ส่งเข้า chat_interaction ทางเดิม)
        upload_btn = gr.UploadButton("📷 เลือกรูปจากเครื่อง", file_types=["image"],
                                     type="filepath", size="sm")
        clear_btn = gr.Button("🗑️ ล้างประวัติ", variant="secondary", size="sm")

    def _load_uploaded(path):
        if not path:
            return None
        path = path if isinstance(path, str) else getattr(path, "name", path)
        return Image.open(path).convert("RGB")

    upload_btn.upload(fn=_load_uploaded, inputs=upload_btn, outputs=img_input_ui)

    with gr.Accordion("🔍 ข้อมูลอ้างอิง + Log Info", open=False):
        raw_refs = gr.Markdown()
        gr.Markdown(
            f"📋 **Log files (rotate รายวัน):**\n"
            f"- `{logger.log_paths()['jsonl']}` — full structured log (JSONL)\n"
            f"- `{logger.log_paths()['csv']}` — summary spreadsheet (CSV)\n"
            f"- `{logger.log_paths()['txt']}` — human-readable debug (TXT)"
        )

    def reset_inputs():
        return None, ""

    _inputs  = [txt_input, img_input_ui, chat_state,
                last_pids_state, session_state, last_vit_state, last_graph_state]
    _outputs = [chatbot_ui, chat_state, raw_refs,
                last_pids_state, session_state, last_vit_state, last_graph_state]

    submit_btn.click(
        fn=chat_interaction,
        inputs=_inputs,
        outputs=_outputs,
    ).then(fn=reset_inputs, outputs=[img_input_ui, txt_input])

    txt_input.submit(
        fn=chat_interaction,
        inputs=_inputs,
        outputs=_outputs,
    ).then(fn=reset_inputs, outputs=[img_input_ui, txt_input])

    def _clear(old_session=""):
        _dialog_states.pop(old_session, None)
        return [], [], "", [], logger.new_session(), None, None

    clear_btn.click(
        fn=_clear,
        inputs=[session_state],
        outputs=[chatbot_ui, chat_state, raw_refs,
                 last_pids_state, session_state, last_vit_state, last_graph_state],
    )

if __name__ == "__main__":
    # v5: เปิดที่ 127.0.0.1 — Windows เปิด http://0.0.0.0:7860 ในเบราว์เซอร์ไม่ได้
    # ถ้าต้องให้เครื่องอื่นในวง LAN เข้าได้ ให้เปลี่ยนกลับเป็น "0.0.0.0" (แต่เปิดจากเครื่องนี้ด้วย 127.0.0.1)
    SERVER_HOST, SERVER_PORT = "127.0.0.1", 7860
    print(f"\n🌐 เปิดเบราว์เซอร์ที่: http://{SERVER_HOST}:{SERVER_PORT}\n")
    demo.launch(share=True, debug=True, server_name=SERVER_HOST, server_port=SERVER_PORT)