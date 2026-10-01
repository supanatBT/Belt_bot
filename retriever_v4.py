"""
retriever_v4.py
===============
Pipeline ใหม่จากศูนย์ — ออกแบบให้ใช้ความสามารถของทุก component เต็มที่

สิ่งที่เปลี่ยนจาก v3:
  ✦ Signal collection parallel — ViT, SigLIP, Jina v3 วิเคราะห์พร้อมกัน
  ✦ ViT confidence เป็น continuous weight ไม่ใช่ binary threshold
  ✦ Series hard filter จาก ViT inject เข้า Qdrant text search โดยตรง
  ✦ Text search ไม่ skip — แต่ filter series ถ้า ViT dominant
  ✦ Graph multi-hop ตาม intent + inject context โดยตรง ไม่ผ่าน RRF
  ✦ Confidence-weighted RRF แทน uniform weight
  ✦ Context-enriched reranker query
  ✦ filter_product_specs ใช้ Qdrant payload Range filter แทน JSON loop
"""

import re
import torch
import torch.nn.functional as F
import concurrent.futures
from dataclasses import dataclass, field
from typing import Optional
from PIL import Image

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Filter, FieldCondition, MatchValue, Range
)
from transformers import (
    AutoTokenizer, AutoModel,
    ViTImageProcessor, ViTForImageClassification,
    SiglipModel, SiglipProcessor,
)
from peft import PeftModel
from knowledge_graph import KnowledgeGraph, GraphContext


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STANDALONE DATACLASSES — ไม่ต้อง import จาก retriever เก่า
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class RetrievedProduct:
    """ผลลัพธ์จาก retrieval ต่อ 1 product"""
    product_id:          str
    rrf_score:           float
    matched_chunks:      list = field(default_factory=list)
    matched_images:      list = field(default_factory=list)
    series:              str  = ""
    product_type:        str  = ""
    material:            str  = ""
    keyword_boost:       float = 0.0
    # reranker fields — ถูก set โดย Reranker
    cross_encoder_score: float = 0.0
    final_score:         float = 0.0
    best_chunk_text:     str  = ""
    # debug
    text_rank:           Optional[int] = None
    image_vit_rank:      Optional[int] = None
    image_siglip_rank:   Optional[int] = None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DATA CLASSES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class ViTSignal:
    """ผลจาก ViT LoRA — series + confidence สำหรับใช้เป็น weight"""
    series:     Optional[str] = None
    confidence: float = 0.0
    dominant:   bool  = False    # confidence >= dominant_threshold
    top_pid:    str   = ""       # pid ของ image hit อันดับ 1

@dataclass
class GraphResult:
    """
    ผลจาก multi-hop graph traversal — inject เข้า LLM context โดยตรง
    ไม่ผ่าน RRF เพราะ structural relationship ไม่ควรแข่ง CE score กับ chain
    """
    seed_products:     list[str] = field(default_factory=list)
    compatible:        list[str] = field(default_factory=list)
    same_series:       list[str] = field(default_factory=list)
    material_details:  dict      = field(default_factory=dict)
    process_details:   dict      = field(default_factory=dict)
    hop_trace:         list[str] = field(default_factory=list)  # debug: path ที่ traverse

    def to_llm_context(self) -> str:
        """สร้าง structured context สำหรับ LLM — แยก section ชัดเจน"""
        parts = []
        if self.seed_products:
            parts.append(f"[Seed Products]: {', '.join(self.seed_products)}")
        if self.compatible:
            parts.append(f"[Compatible Parts สำหรับ {', '.join(self.seed_products)}]: "
                         f"{', '.join(self.compatible[:8])}")
        if self.same_series:
            parts.append(f"[Series เดียวกัน]: {', '.join(self.same_series[:5])}")
        for mat, info in self.material_details.items():
            line = (f"[วัสดุของ {', '.join(self.seed_products)}]: "
                    f"{info.get('full_name', mat)}")
            if info.get("description"):
                line += f" — {info['description'][:120]}"
            parts.append(line)
        for proc, info in self.process_details.items():
            line = (f"[กระบวนการผลิตของ {', '.join(self.seed_products)}]: "
                    f"{info.get('full_name', proc)}")
            if info.get("description"):
                line += f" — {info['description'][:150]}"
            if info.get("advantages") or info.get("best_used_for"):
                adv = info.get("advantages") or info.get("best_used_for", "")
                line += f" | ข้อดี: {adv[:100]}"
            parts.append(line)
        return "\n".join(parts) if parts else "ไม่มี graph context"


@dataclass
class RetrievalResultV4:
    products:           list[RetrievedProduct] = field(default_factory=list)
    query_used:         str = ""
    vit_signal:         Optional[ViTSignal] = None
    graph_result:       Optional[GraphResult] = None
    image_search_used:  bool = False
    total_candidates:   int  = 0
    # compat กับ v3 — logger ใช้ field เหล่านี้
    vit_series_filter:  Optional[str] = None
    graph_context:      Optional[object] = None   # GraphContext compat


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# RANKED RESULT — ย้ายมาจาก reranker.py
# chatbot และ logger ใช้ 2 class นี้โดยตรง
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class RankedProduct:
    """Product หลัง keyword boost + final sort"""
    product_id:          str
    final_score:         float
    cross_encoder_score: float = 0.0   # คงไว้เพื่อ compat กับ logger/debug
    rrf_score:           float = 0.0
    matched_chunks:      list  = field(default_factory=list)
    matched_images:      list  = field(default_factory=list)
    series:              str   = ""
    product_type:        str   = ""
    material:            str   = ""
    keyword_boost:       float = 0.0
    best_chunk_text:     str   = ""


@dataclass
class RankedResult:
    """ผลลัพธ์สุดท้ายที่ chatbot รับไปใช้"""
    products:           list
    query_used:         str
    image_search_used:  bool           = False
    vit_series_filter:  Optional[str]  = None
    total_candidates:   int            = 0
    graph_context:      object         = None
    graph_result:       object         = None

    def top(self, k: int = 5) -> list:
        return self.products[:k]

    def to_llm_context(self, k: int = 3) -> str:
        parts = []
        for i, p in enumerate(self.products[:k], 1):
            parts.append(
                f"[Product {i}] {p.product_id} (score: {p.final_score:.3f})\n"
                f"{p.best_chunk_text}"
            )
        return "\n\n---\n\n".join(parts)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONFIG
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class RetrieverV4Config:
    qdrant_path:      str = "./qdrant_db"
    text_collection:  str = "movex_text_chunks"
    image_collection: str = "movex_images"

    text_candidates:  int = 20
    image_candidates: int = 20
    final_top_k:      int = 10

    rrf_k:            int   = 60

    # ViT thresholds
    vit_dominant_threshold:    float = 0.85   # ถ้าสูงกว่านี้ → series hard filter
    vit_confidence_threshold:  float = 0.50   # ถ้าสูงกว่านี้ → เชื่อ series label

    # Image search threshold
    image_distance_threshold: float = 0.64

    # Final output
    final_ranked_top_k:  int   = 5      # จำนวน product ที่ส่งให้ LLM
    keyword_boost_bonus: float = 0.15   # บวกเพิ่มถ้า keyword match

    # Graph
    graph_max_hop:       int  = 2
    graph_expand_series: bool = True
    graph_expand_compat: bool = True
    graph_expand_mat:    bool = False

    # Series ที่รู้จัก
    # ── ชื่อฟิลด์ที่ต่างกันระหว่างแบรนด์ ────────────────────────────
    # ค่าเริ่มต้นทั้งหมดเป็นของ Movex พฤติกรรมเดิมจึงไม่เปลี่ยน
    # Modutech ส่งค่าใหม่เข้ามาทาง config ไม่ต้องแก้โค้ด
    # chunk คนละชนิดอาจใช้คีย์ id คนละชื่อ เช่น Modutech มี belt_code_id กับ sprocket_code
    # ลองทีละคีย์ตามลำดับ เจอตัวแรกที่มีค่าก็ใช้เลย
    payload_id_keys: tuple = ("product_id",)   # Modutech: ("belt_code_id","sprocket_code")
    payload_series_key: str = "series"        # Modutech: "belt_series"
    series_filter_fmt:  str = "{} Series"     # Modutech: "{}"
    art_nr_regex:       str = r'\b5[4-9]\d{3}\b'   # Modutech: รหัสสเตอร์เป็นตัวอักษร
    art_nr_field:       str = "Art_Nr"        # Modutech: "sprocket_code"
    pid_strip_tokens:   tuple = ("movexchain", "movexsprocket")
    # (คำที่ต้องเจอในคำถาม, ข้อความที่ต้องอยู่ใน product_type, คะแนน)
    type_boost_rules: tuple = (
        (("เลี้ยว", "โค้ง", "sideflex", "sideflexing", "curve"), "sideflexing chain", 8.0),
        (("ตรง", "straight", "วิ่งตรง"), "straight running", 8.0),
    )

    known_series: set = field(
        default_factory=lambda: {"820", "821", "880", "882", "83", "103"}
    )

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Intent keywords สำหรับ graph traversal
    sprocket_kw: list[str] = field(default_factory=lambda: [
        "สเตอร์", "sprocket", "สเปรอคเก็ต"
    ])
    process_kw: list[str] = field(default_factory=lambda: [
        "ผลิต", "กระบวนการ", "manufacturing", "process"
    ])
    material_kw: list[str] = field(default_factory=lambda: [
        "วัสดุ", "material", "ทนกรด", "ทนความร้อน", "food grade"
    ])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# RETRIEVER V4
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class RetrieverV4:
    """
    Adaptive Multi-Signal Retriever v4

    Architecture:
      Layer 1: Signal Collection (parallel)
               ViT LoRA → series + confidence
               SigLIP   → image embedding (fallback)
               Jina v3  → text embedding

      Layer 2: Intent-Aware Retrieval
               Jina search + series filter ถ้า ViT dominant
               SigLIP text→image ถ้า query_type=visual

      Layer 3: Multi-hop Graph Traversal (inject context โดยตรง)

      Layer 4: Confidence-Weighted RRF

      Layer 5: Context-Enriched Rerank query (chatbot ใช้)
    """

    def __init__(
        self,
        cfg: RetrieverV4Config,
        jina_tokenizer,
        jina_model,
        vit_processor,
        vit_model,
        siglip_processor,
        siglip_model,
        product_db: dict,
        kg: KnowledgeGraph,
    ):
        self.cfg              = cfg
        self.client           = QdrantClient(path=cfg.qdrant_path)
        self.jina_tokenizer   = jina_tokenizer
        self.jina_model       = jina_model
        self.vit_processor    = vit_processor
        self.vit_model        = vit_model
        self.siglip_processor = siglip_processor
        self.siglip_model     = siglip_model
        self.product_db       = product_db
        self.kg               = kg

    # ──────────────────────────────────────────────────────
    # PUBLIC API
    # ──────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        query_image: Optional[Image.Image] = None,
        query_type: str = "general",
    ) -> RetrievalResultV4:
        """
        Main search entry point

        Parameters
        ----------
        query      : user query string
        query_image: PIL Image (optional)
        query_type : "image" | "visual" | "general"
        """
        result = RetrievalResultV4(query_used=query)
        score_map: dict[str, dict] = {}

        # ════════════════════════════════════════════════
        # LAYER 1: Signal Collection
        # ════════════════════════════════════════════════
        vit_signal = ViTSignal()

        if query_image is not None:
            result.image_search_used = True
            # ViT + SigLIP
            vit_signal = self._collect_image_signals(
                query_image, score_map
            )
            result.vit_signal       = vit_signal
            result.vit_series_filter = vit_signal.series

        elif query_type == "visual":
            # SigLIP text→image
            result.image_search_used = True
            self._siglip_text_search(query, score_map)

        # Keyword boost — ส่ง vit_signal เพื่อ suppress type keyword ที่ขัดกับ ViT
        self._keyword_boost(query, score_map, vit_signal=vit_signal)

        # ════════════════════════════════════════════════
        # LAYER 2: Intent-Aware Text Search
        # ════════════════════════════════════════════════
        # ถ้า ViT dominant → filter series ใน Qdrant ก่อน search
        # ถ้าไม่ dominant → search ทุก series ปกติ
        series_filter = None
        if vit_signal.dominant and vit_signal.series:
            series_filter = vit_signal.series

        self._text_search(query, score_map, series_filter=series_filter)

        # ════════════════════════════════════════════════
        # LAYER 3: Confidence-Weighted RRF (ย้ายขึ้นมาทำก่อน)
        # ════════════════════════════════════════════════
        ranked = self._confidence_weighted_rrf(score_map, vit_signal)

        # ════════════════════════════════════════════════
        # LAYER 4: Multi-hop Graph Traversal (ดึง Context หลังจัดอันดับ)
        # ════════════════════════════════════════════════
        # ดึงเอาเฉพาะ Product อันดับ 1 จาก RRF มาเป็น Seed
        top_pid = ranked[0][0] if ranked else None
        
        graph_result = self._graph_traverse(
            query=query,
            top_pid=top_pid, # ส่งแค่อันดับ 1 เข้าไป
        )
        result.graph_result = graph_result
        result.graph_context = self._graph_result_to_context(graph_result)

        # ════════════════════════════════════════════════
        # Build product list
        # ════════════════════════════════════════════════
        products: list[RetrievedProduct] = []
        target_k = self.cfg.final_top_k * 2

        for pid, rrf_score in ranked[:target_k]:
            p_data = self.product_db.get(pid)
            if not p_data:
                continue
            rp = RetrievedProduct(
                product_id=pid,
                rrf_score=rrf_score,
                series=p_data.get(self.cfg.payload_series_key, ""),
                product_type=p_data.get("product_type", ""),
                material=p_data.get("Material", ""),
                keyword_boost=score_map.get(pid, {}).get("keyword", 0.0),
            )
            rp.matched_chunks = self._fetch_chunk_texts(pid)
            products.append(rp)

        result.products         = products[:self.cfg.final_top_k]
        result.total_candidates = len(ranked)
        result._score_map       = score_map   # expose สำหรับ InstrumentedRetriever
        result._ranked          = ranked
        return result

    def build_enriched_rerank_query(
        self,
        base_query: str,
        vit_signal: Optional[ViTSignal],
        graph_result: Optional[GraphResult],
    ) -> str:
        """
        Layer 5: สร้าง enriched query สำหรับ Cross-Encoder reranker
        inject series + compatible sprockets เข้า query
        ทำให้ CE score แยก product ใน series เดียวกันได้ชัดขึ้น
        """
        parts = []
        if vit_signal and vit_signal.series:
            parts.append(f"[Series:{vit_signal.series}]")
        if graph_result and graph_result.compatible:
            sprockets = graph_result.compatible[:3]
            parts.append(f"[Compatible:{','.join(sprockets)}]")
        if vit_signal and vit_signal.top_pid:
            parts.append(f"[TopProduct:{vit_signal.top_pid}]")
        parts.append(base_query)
        return " ".join(parts)

    def rank(self, retrieval_result: RetrievalResultV4) -> RankedResult:
        """
        Phase 4 (ย้ายมาจาก reranker.py) — Keyword Boost + Final Sort + Slice

        ไม่มี CE model — ใช้ RRF score จาก retriever โดยตรง
        บวก keyword_boost_bonus ถ้า product มี keyword match
        แล้ว slice เหลือ final_ranked_top_k ก่อนส่ง LLM
        """
        candidates = retrieval_result.products

        ranked = []
        for p in candidates:
            score = p.rrf_score
            if p.keyword_boost > 0:
                score += self.cfg.keyword_boost_bonus
            # best_chunk_text — เลือก chunk แรกที่มี text
            best_text = ""
            for chunk in (p.matched_chunks or []):
                t = chunk.get("text", "") if isinstance(chunk, dict) else str(chunk)
                if t:
                    best_text = t
                    break

            ranked.append(RankedProduct(
                product_id=p.product_id,
                final_score=score,
                cross_encoder_score=0.0,
                rrf_score=p.rrf_score,
                matched_chunks=p.matched_chunks,
                matched_images=getattr(p, "matched_images", []),
                series=p.series,
                product_type=p.product_type,
                material=p.material,
                keyword_boost=p.keyword_boost,
                best_chunk_text=best_text,
            ))

        ranked.sort(key=lambda x: x.final_score, reverse=True)

        return RankedResult(
            products=ranked[:self.cfg.final_ranked_top_k],
            query_used=retrieval_result.query_used,
            image_search_used=retrieval_result.image_search_used,
            vit_series_filter=retrieval_result.vit_series_filter,
            total_candidates=retrieval_result.total_candidates,
            graph_context=retrieval_result.graph_context,
            graph_result=retrieval_result.graph_result,
        )

    def filter_by_spec(
        self,
        spec_type: str,
        condition: str,
        target_value: float,
    ) -> list[str]:
        """
        Qdrant payload Range filter — แทน JSON loop ใน v3
        คืน list ของ (product_id, value) ที่ผ่านเงื่อนไข
        """
        key_map = {
            "width":       "plate_width_mm",
            "load":        "max_load_n",
            "weight":      "weight_kg_m",
            "radius":      "min_curve_radius_mm",
            "pitch":       "pitch_mm",
            "teeth":       "z_teeth",
            "bore":        "bore_mm",
            "pd":          "pd_mm",
            "od":          "od_mm",
        }
        qdrant_key = key_map.get(spec_type.lower(), spec_type.lower())

        range_kwargs = {}
        if condition in ("<", "<="):
            range_kwargs["lte" if condition == "<=" else "lt"] = target_value
        elif condition in (">", ">="):
            range_kwargs["gte" if condition == ">=" else "gt"] = target_value
        elif condition == "==":
            range_kwargs["gte"] = target_value
            range_kwargs["lte"] = target_value

        try:
            hits, _ = self.client.scroll(
                collection_name=self.cfg.text_collection,
                scroll_filter=Filter(must=[
                    FieldCondition(
                        key=qdrant_key,
                        range=Range(**range_kwargs),
                    )
                ]),
                limit=50,
                with_payload=True,
                with_vectors=False,
            )
            # deduplicate by product_id
            seen = {}
            for h in hits:
                pid = self._payload_id(h.payload)
                val = h.payload.get(qdrant_key)
                if pid and pid not in seen:
                    seen[pid] = val
            return [(pid, val) for pid, val in seen.items()]
        except Exception as e:
            print(f"⚠️ Qdrant filter error: {e}")
            return []

    # ──────────────────────────────────────────────────────
    # LAYER 1: IMAGE SIGNAL COLLECTION
    # ──────────────────────────────────────────────────────

    def _collect_image_signals(
        self,
        image: Image.Image,
        score_map: dict,
    ) -> ViTSignal:
        """
        ViT LoRA + SigLIP parallel
        ViT confidence เป็น continuous value → ใช้เป็น weight ใน RRF
        SigLIP weight ลดลงเมื่อ ViT confidence สูง
        """
        image = image.convert("RGB")
        signal = ViTSignal()

        # ── ViT ──────────────────────────────────────────
        vit_vec = self._embed_vit(image)
        vit_hits = []
        if vit_vec:
            res = self.client.query_points(
                collection_name=self.cfg.image_collection,
                query=vit_vec, using="vit",
                limit=self.cfg.image_candidates,
                with_payload=True,
            )
            vit_hits = res.points
            valid = [h for h in vit_hits
                     if h.score >= self.cfg.image_distance_threshold]
            if not valid:
                valid = vit_hits[:3] if vit_hits else []

            if valid:
                top = valid[0]
                signal.confidence = float(top.score)
                signal.dominant   = top.score >= self.cfg.vit_dominant_threshold
                signal.top_pid    = self._payload_id(top.payload)

                if top.score >= self.cfg.vit_confidence_threshold:
                    raw = top.payload.get(self.cfg.payload_series_key, "") or ""
                    m   = re.search(r'\d+', raw)
                    signal.series = m.group() if m else None

                for rank, h in enumerate(valid):
                    pid = self._payload_id(h.payload)
                    if pid:
                        self._ensure_pid(score_map, pid)
                        if score_map[pid]["vit_rank"] is None:
                            score_map[pid]["vit_rank"] = rank

        # ── SigLIP (รันเสมอ — weight ถูก control ที่ RRF) ──
        siglip_vec = self._embed_siglip(image)
        if siglip_vec:
            # ถ้า ViT dominant → filter series ใน SigLIP ด้วย
            qdrant_filter = None
            if signal.dominant and signal.series:
                qdrant_filter = Filter(must=[FieldCondition(
                    key=self.cfg.payload_series_key,
                    match=MatchValue(value=f"{signal.series} Series"),
                )])

            res2 = self.client.query_points(
                collection_name=self.cfg.image_collection,
                query=siglip_vec, using="siglip",
                limit=self.cfg.image_candidates,
                query_filter=qdrant_filter,
                with_payload=True,
            )
            for rank, h in enumerate(res2.points):
                pid = self._payload_id(h.payload)
                if pid:
                    self._ensure_pid(score_map, pid)
                    if score_map[pid]["siglip_rank"] is None:
                        score_map[pid]["siglip_rank"] = rank

        return signal

    # ──────────────────────────────────────────────────────
    # LAYER 2: TEXT SEARCH
    # ──────────────────────────────────────────────────────

    def _payload_id(self, payload: dict) -> str:
        """
        ดึง product id จาก payload โดยลองทีละคีย์ตาม cfg.payload_id_keys

        มีเพราะ chunk คนละชนิดใช้คีย์คนละชื่อ ถ้าอ่านคีย์เดียวตายตัว
        chunk ชนิดที่ไม่มีคีย์นั้นจะถูกทิ้งเงียบ ๆ โดยไม่มี error
        """
        for key in self.cfg.payload_id_keys:
            value = payload.get(key)
            if value:
                return str(value)
        return ""

    def _text_search(
        self,
        query: str,
        score_map: dict,
        series_filter: Optional[str] = None,
    ):
        """
        Jina v3 text search
        ถ้า series_filter ไม่ None → inject เป็น Qdrant filter ก่อน search
        ปล่อยให้ embedding space ตัดสิน chunk type เอง — ไม่มี rule
        """
        vec = self._embed_text(query)
        if vec is None:
            return

        qdrant_filter = None
        if series_filter:
            qdrant_filter = Filter(must=[FieldCondition(
                key=self.cfg.payload_series_key,
                match=MatchValue(value=self.cfg.series_filter_fmt.format(series_filter)),
            )])

        res = self.client.query_points(
            collection_name=self.cfg.text_collection,
            query=vec, using="jina",
            limit=self.cfg.text_candidates,
            query_filter=qdrant_filter,
            with_payload=True,
        )

        for rank, hit in enumerate(res.points):
            pid = self._payload_id(hit.payload)
            if pid:
                self._ensure_pid(score_map, pid)
                score_map[pid]["text_ranks"].append(rank)

    def _siglip_text_search(self, text: str, score_map: dict):
        """SigLIP text→image search สำหรับ query_type=visual"""
        vec = self._embed_siglip_text(text)
        if not vec:
            return
        res = self.client.query_points(
            collection_name=self.cfg.image_collection,
            query=vec, using="siglip",
            limit=self.cfg.image_candidates,
            with_payload=True,
        )
        for rank, h in enumerate(res.points):
            pid = self._payload_id(h.payload)
            if pid and h.score >= 0.20:
                self._ensure_pid(score_map, pid)
                if score_map[pid]["siglip_rank"] is None:
                    score_map[pid]["siglip_rank"] = rank

    # ──────────────────────────────────────────────────────
    # LAYER 3: MULTI-HOP GRAPH TRAVERSAL
    # ──────────────────────────────────────────────────────

    def _graph_traverse(
        self,
        query: str,
        top_pid: Optional[str],
    ) -> GraphResult:
        result = GraphResult()
        if not top_pid:
            return result
            
        query_lower = query.lower()

        # ── Detect intent ──────────────────────────────────
        want_sprocket = any(w in query_lower for w in self.cfg.sprocket_kw)
        want_process  = any(w in query_lower for w in self.cfg.process_kw)
        want_material = any(w in query_lower for w in self.cfg.material_kw)

        # ── Seed products ──────────────────────────────────
        seed_ids = []

         # ── Art.Nr. detector (ถ้าพิมพ์รหัส 54xxx มาตรงๆ ให้ยึดรหัสนี้ก่อน) ───────────────────
        ART_NR_RE = re.compile(self.cfg.art_nr_regex)
        art_matches = ART_NR_RE.findall(query)
        if art_matches:
            for nr in art_matches:
                for pid, p in self.product_db.items():
                    if str(p.get(self.cfg.art_nr_field, "")) == nr:
                        seed_ids.append(pid)
                        break

        # 👇 ถ้าไม่มีการพิมพ์รหัส Art.Nr ให้ใช้ อันดับ 1 จาก RRF เป็น Seed แบบเดี่ยวๆ
        if not seed_ids and top_pid:
            seed_ids = [top_pid]

        if not seed_ids:
            return result

        result.seed_products = seed_ids
        result.hop_trace.append(f"seeds={seed_ids}")

        # ── Hop 1: compatible + same_series ───────────────
        ctx = self.kg.expand_products(
            seed_product_ids=seed_ids,
            include_same_series=self.cfg.graph_expand_series,
            include_compatible=self.cfg.graph_expand_compat,
            include_same_material=self.cfg.graph_expand_mat,
        )

        result.compatible   = ctx.compatible_products
        result.same_series  = ctx.same_series
        result.hop_trace.append(
            f"hop1: compatible={ctx.compatible_products}, "
            f"same_series={ctx.same_series[:3]}"
        )

        # ── Hop 2: traverse ตาม intent ────────────────────
        hop2_seeds = []

        if want_sprocket and ctx.compatible_products:
            hop2_seeds = ctx.compatible_products[:3]
        elif want_process or want_material:
            hop2_seeds = seed_ids  # ดู process/material ของ seed โดยตรง

        if hop2_seeds:
            ctx2 = self.kg.expand_products(
                seed_product_ids=hop2_seeds,
                include_same_series=False,
                include_compatible=False,
                include_same_material=(want_material or want_process),
            )
            if ctx2.material_details:
                result.material_details = ctx2.material_details
            if ctx2.process_details:
                result.process_details  = ctx2.process_details
            result.hop_trace.append(
                f"hop2 (seeds={hop2_seeds}): "
                f"process={list(ctx2.process_details.keys())}, "
                f"material={list(ctx2.material_details.keys())}"
            )
        else:
            result.material_details = ctx.material_details
            result.process_details  = ctx.process_details

        return result

    # ──────────────────────────────────────────────────────
    # LAYER 4: CONFIDENCE-WEIGHTED RRF
    # ──────────────────────────────────────────────────────

    def _confidence_weighted_rrf(
        self,
        score_map: dict,
        vit_signal: ViTSignal,
    ) -> list[tuple[str, float]]:
        """
        RRF พร้อม weight ที่ขึ้นกับ ViT confidence

        ViT confidence สูง → vit_weight สูง, text_weight ลด
        ViT confidence ต่ำ → SigLIP และ text มีสิทธิ์มากขึ้น

        keyword exact match ยัง weight สูงเสมอ (reliable hard signal)
        graph เป็น soft signal — ช่วย boost แต่ไม่ dominate
        """
        conf = vit_signal.confidence  # 0.0 - 1.0

        # Adaptive weights ตาม ViT confidence
        w_vit     = conf                        # 0.0 → 1.0 ตาม confidence
        w_siglip  = max(0.1, 1.0 - conf) * 0.8 # สูงเมื่อ ViT ไม่มั่นใจ
        w_text    = max(0.1, 1.0 - conf * 0.8) # ลดเมื่อ ViT dominant แต่ไม่เป็น 0
        w_keyword = 2.0                         # hard signal — คงที่เสมอ
        w_graph   = 0.5                         # soft signal — ช่วย boost

        k = self.cfg.rrf_k
        final: dict[str, float] = {}

        for pid, scores in score_map.items():
            rrf = 0.0

            text_ranks = scores.get("text_ranks", [])
            if text_ranks:
                avg_rank = sum(text_ranks) / len(text_ranks)
                rrf += w_text / (k + avg_rank)

            vit_rank = scores.get("vit_rank")
            if vit_rank is not None:
                rrf += w_vit / (k + vit_rank)

            siglip_rank = scores.get("siglip_rank")
            if siglip_rank is not None:
                rrf += w_siglip / (k + siglip_rank)

            keyword = scores.get("keyword", 0.0)
            if keyword > 0:
                rrf += w_keyword * keyword / 10.0

            graph = scores.get("graph", -1.0)
            if graph >= 0:
                rrf += w_graph / (k + graph)

            final[pid] = rrf

        return sorted(final.items(), key=lambda x: x[1], reverse=True)

    # ──────────────────────────────────────────────────────
    # KEYWORD BOOST
    # ──────────────────────────────────────────────────────

    def _keyword_boost(
        self,
        query: str,
        score_map: dict,
        vit_signal: Optional[ViTSignal] = None,
    ):
        """
        Exact / fuzzy match + product type keyword

        ถ้า ViT dominant → suppress sideflex/straight type keyword boost
        เพราะ ViT บอก series แล้ว ไม่ควรให้ keyword ดัน product ต่าง series ขึ้นมา
        (type boost ยังทำงานถ้าไม่มีรูป หรือ ViT ไม่ dominant)
        """
        import difflib

        query_clean = re.sub(r'[^a-zA-Z0-9]', '', query).lower()
        query_lower = query.lower()

        # กฎเดิมสองข้อย้ายไปอยู่ใน cfg.type_boost_rules ค่าเริ่มต้นเหมือนเดิมทุกตัว
        type_hits = [any(w in query_lower for w in words)
                     for words, _ptype, _score in self.cfg.type_boost_rules]

        # ViT dominant → suppress type keyword boost ทั้งหมด
        # ViT รู้ series แน่แล้ว ไม่ต้องให้ keyword ดัน product ต่าง series
        vit_dominant = vit_signal and vit_signal.dominant and vit_signal.series
        if vit_dominant:
            type_hits = [False] * len(type_hits)

        if not query_clean and not any(type_hits):
            return

        query_nums   = re.findall(r'\d+', query_clean)
        query_series = [n for n in query_nums if n in self.cfg.known_series]

        for pid, p in self.product_db.items():
            pid_clean   = re.sub(r'[^a-zA-Z0-9]', '', pid).lower()
            ref_clean   = re.sub(r'[^a-zA-Z0-9]', '', str(p.get("Ref", ""))).lower()
            artnr_clean = re.sub(r'[^a-zA-Z0-9]', '', str(p.get(self.cfg.art_nr_field, ""))).lower()
            short_pid   = pid_clean
            for _tok in self.cfg.pid_strip_tokens:
                short_pid = short_pid.replace(_tok, "")
            series_str  = re.sub(r'[^a-zA-Z0-9]', '', str(p.get(self.cfg.payload_series_key, ""))).lower()
            ptype       = str(p.get("product_type", "")).lower()

            # ViT dominant → filter เฉพาะ series ที่ ViT บอก
            if vit_dominant:
                prod_nums = re.findall(r'\d+', series_str)
                if vit_signal.series not in prod_nums:
                    continue

            if query_series:
                prod_nums = re.findall(r'\d+', series_str)
                if not any(qs in prod_nums for qs in query_series):
                    continue

            boost = 0.0

            for _hit, (_words, _ptype, _score) in zip(type_hits, self.cfg.type_boost_rules):
                if _hit and _ptype in ptype:
                    boost = max(boost, _score)

            if query_clean:
                if query_clean in (ref_clean, pid_clean, short_pid, artnr_clean):
                    boost = 10.0
                elif boost == 0.0:
                    sim = max(
                        difflib.SequenceMatcher(None, query_clean, ref_clean).ratio(),
                        difflib.SequenceMatcher(None, query_clean, short_pid).ratio(),
                        difflib.SequenceMatcher(None, query_clean, artnr_clean).ratio(),
                    )
                    if sim > 0.85:
                        boost = 6.0
                    elif sim > 0.70:
                        boost = 3.0
                    elif (query_clean in pid_clean or query_clean in ref_clean
                          or query_clean in artnr_clean):
                        boost = 2.0

            if boost > 0:
                self._ensure_pid(score_map, pid)
                score_map[pid]["keyword"] = max(
                    score_map[pid]["keyword"], boost
                )

    # ──────────────────────────────────────────────────────
    # EMBEDDING HELPERS
    # ──────────────────────────────────────────────────────

    @torch.no_grad()
    def _embed_text(self, text: str) -> Optional[list[float]]:
        try:
            enc = self.jina_tokenizer(
                text, return_tensors="pt",
                truncation=True, max_length=512
            ).to(self.cfg.device)
            out    = self.jina_model(**enc, return_dict=True)
            last   = out.last_hidden_state
            mask   = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (last * mask).sum(1) / mask.sum(1)
            pooled = F.normalize(pooled, dim=-1)
            return pooled.squeeze().float().cpu().numpy().tolist()
        except Exception as e:
            print(f"⚠️ Text embed error: {e}")
            return None

    @torch.no_grad()
    def _embed_vit(self, image: Image.Image) -> Optional[list[float]]:
        try:
            inputs = self.vit_processor(
                images=image, return_tensors="pt"
            ).to(self.cfg.device)
            # cast ให้ตรง dtype ของโมเดล (เช่น fp16 บน GPU) ไม่งั้น matmul จะ error
            model_dtype = next(self.vit_model.parameters()).dtype
            inputs["pixel_values"] = inputs["pixel_values"].to(model_dtype)
            out = self.vit_model.base_model(
                pixel_values=inputs["pixel_values"],
                output_hidden_states=True, return_dict=True,
            )
            cls = out.hidden_states[-1][:, 0, :]
            cls = F.normalize(cls, p=2, dim=1)
            return cls.squeeze().float().cpu().numpy().tolist()
        except Exception as e:
            print(f"⚠️ ViT embed error: {e}")
            return None

    @torch.no_grad()
    def _embed_siglip(self, image: Image.Image) -> Optional[list[float]]:
        try:
            inputs = self.siglip_processor(
                images=image, return_tensors="pt"
            ).to(self.cfg.device)
            model_dtype = next(self.siglip_model.parameters()).dtype
            inputs["pixel_values"] = inputs["pixel_values"].to(model_dtype)
            feat = self.siglip_model.get_image_features(**inputs)
            feat = F.normalize(feat, dim=-1)
            return feat.squeeze().float().cpu().numpy().tolist()
        except Exception as e:
            print(f"⚠️ SigLIP embed error: {e}")
            return None

    @torch.no_grad()
    def _embed_siglip_text(self, text: str) -> Optional[list[float]]:
        try:
            inputs = self.siglip_processor(
                text=[text], return_tensors="pt",
                padding=True, truncation=True, max_length=64
            ).to(self.cfg.device)
            feat = self.siglip_model.get_text_features(**inputs)
            feat = F.normalize(feat, dim=-1)
            return feat.squeeze().float().cpu().numpy().tolist()
        except Exception as e:
            print(f"⚠️ SigLIP text embed error: {e}")
            return None

    # ──────────────────────────────────────────────────────
    # UTILS
    # ──────────────────────────────────────────────────────

    def _ensure_pid(self, score_map: dict, pid: str):
        if pid not in score_map:
            score_map[pid] = {
                "text_ranks": [], "vit_rank": None,
                "siglip_rank": None, "keyword": 0.0, "graph": -1.0
            }

    def _fetch_chunk_texts(self, product_id: str) -> list[dict]:
        try:
            hits, _ = self.client.scroll(
                collection_name=self.cfg.text_collection,
                scroll_filter=Filter(should=[
                    FieldCondition(key=_k, match=MatchValue(value=product_id))
                    for _k in self.cfg.payload_id_keys
                ]),
                limit=10, with_payload=True, with_vectors=False,
            )
            return [
                {"type": h.payload.get("chunk_type"),
                 "text": h.payload.get("text")}
                for h in hits
            ]
        except Exception:
            return []

    def _graph_result_to_context(self, gr: GraphResult) -> Optional[object]:
        """
        แปลง GraphResult → GraphContext compat สำหรับ logger/chatbot เดิม
        """
        if not gr or not gr.seed_products:
            return None
        try:
            ctx = GraphContext(seed_products=gr.seed_products)
            ctx.seed_products     = gr.seed_products
            ctx.compatible_products = gr.compatible
            ctx.same_series       = gr.same_series
            ctx.material_details  = gr.material_details
            ctx.process_details   = gr.process_details
            ctx.all_candidate_ids = list(set(
                gr.compatible + gr.same_series
            ))
            return ctx
        except Exception:
            print(f"⚠️ Error converting graph context: {e}")
            return None