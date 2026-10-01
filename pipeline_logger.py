"""
pipeline_logger.py
==================
Structured Logger สำหรับ Movex Chatbot v3 — บันทึกทุก workflow step

บันทึก 3 ระดับ:
  1. JSONL  (logs/pipeline_YYYYMMDD.jsonl)  — 1 record ต่อ 1 turn, ครบทุก field
  2. CSV    (logs/summary_YYYYMMDD.csv)     — 1 row ต่อ 1 turn, ดู Excel ง่าย
  3. TXT    (logs/debug_YYYYMMDD.txt)       — human-readable ด้านงานที่ตรวจสอบ manually

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
สิ่งที่บันทึกต่อ workflow step
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 0 — Input
  • session_id, turn_id, timestamp
  • user_text, has_image, last_pid_context
  • effective_query (หลัง prepend last_pid ถ้า follow-up)

STEP 1 — Keyword Boost
  • keyword_hits: [{pid, boost_score, match_type}]
  • sideflex_kw_triggered, straight_kw_triggered
  • query_series_detected

STEP 2 — Image Search (ถ้ามีรูป)
  • vit_top_score, vit_series_detected, vit_dominant
  • vit_hits: [{pid, score, rank}]
  • siglip_hits: [{pid, score, rank}]

STEP 3 — GraphRAG Expansion
  • graph_triggered_by: "vit" | "keyword" | "none"
  • seed_products
  • same_series_expanded: [pid, ...]
  • compatible_expanded: [pid, ...]
  • material_details_found: {mat: full_name}
  • process_details_found: {proc: full_name}
  • total_candidates_after_graph

STEP 4 — Text Search
  • text_search_skipped (vit_dominant)
  • text_hits: [{pid, rank}]

STEP 5 — RRF Fusion
  • score_map_snapshot: [{pid, text_ranks, vit_rank, siglip_rank, keyword, graph, rrf_final}]
  • total_unique_candidates

STEP 6 — Reranker
  • reranked_products: [{pid, rank, final_score, ce_score, rrf_score, keyword_boost}]
  • top1_pid, top1_score

STEP 7 — LLM (Gemini)
  • clarification_triggered: bool (ระบบตรวจว่า Gemini ถามกลับหรือตอบตรงๆ)
  • tool_calls_made: [{tool_name, args, result_preview}]
  • ai_reply_length, ai_reply_preview (100 chars)
  • latency_ms (total pipeline time)

STEP 8 — Evaluation Flags (สำหรับตรวจสอบ)
  • top1_matches_query_series: bool
  • answer_contains_product_ref: bool
  • graph_contributed: bool (graph expanded ids ≠ ∅)
  • error: str | null
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import os
import json
import csv
import time
import re
import uuid
import traceback
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Any, Optional


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LOG DIRECTORY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

LOG_DIR = "./logs"
os.makedirs(LOG_DIR, exist_ok=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DATA CLASSES — 1 ชั้นต่อ 1 step
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class StepInput:
    user_text: str = ""
    has_image: bool = False
    last_pid_context: str = ""
    effective_query: str = ""
    query_type: str = "general"


@dataclass
class KeywordHit:
    pid: str
    boost_score: float
    match_type: str    # "exact" | "fuzzy_high" | "fuzzy_mid" | "substring" | "product_type"


@dataclass
class StepKeyword:
    sideflex_kw_triggered: bool = False
    straight_kw_triggered: bool = False
    query_series_detected: list[str] = field(default_factory=list)
    hits: list[dict] = field(default_factory=list)   # [KeywordHit asdict]
    total_hits: int = 0


@dataclass
class ImageHit:
    pid: str
    score: float
    rank: int
    source: str    # "vit" | "siglip"


@dataclass
class StepImage:
    triggered: bool = False
    vit_top_score: float = 0.0
    vit_series_detected: Optional[str] = None
    vit_dominant: bool = False
    vit_confidence_threshold: float = 0.50
    vit_hits: list[dict] = field(default_factory=list)   # [ImageHit asdict]
    siglip_hits: list[dict] = field(default_factory=list)
    total_vit_hits: int = 0
    total_siglip_hits: int = 0


@dataclass
class StepGraph:
    triggered_by: str = "none"   # "vit" | "keyword" | "none"
    seed_products: list[str] = field(default_factory=list)
    same_series_expanded: list[str] = field(default_factory=list)
    compatible_expanded: list[str] = field(default_factory=list)
    same_material_expanded: list[str] = field(default_factory=list)
    material_details_found: dict = field(default_factory=dict)  # {mat_code: full_name}
    process_details_found: dict = field(default_factory=dict)   # {proc: full_name}
    total_graph_candidates: int = 0
    graph_contributed: bool = False   # True ถ้า expanded ≠ ∅


@dataclass
class StepText:
    skipped: bool = False             # True ถ้า vit_dominant
    skip_reason: str = ""
    hits: list[dict] = field(default_factory=list)   # [{pid, rank}]
    total_hits: int = 0


@dataclass
class ScoreMapEntry:
    pid: str
    text_ranks: list[int] = field(default_factory=list)
    vit_rank: Optional[int] = None
    siglip_rank: Optional[int] = None
    keyword_score: float = 0.0
    graph_rank: Optional[float] = None
    rrf_final: float = 0.0


@dataclass
class StepRRF:
    total_unique_candidates: int = 0
    top10_scores: list[dict] = field(default_factory=list)  # [ScoreMapEntry asdict] top-10
    score_map_full: list[dict] = field(default_factory=list)


@dataclass
class RerankedEntry:
    pid: str
    rank: int
    final_score: float
    ce_score: float
    rrf_score: float
    keyword_boost: float
    series: str = ""
    product_type: str = ""


@dataclass
class StepReranker:
    products: list[dict] = field(default_factory=list)  # [RerankedEntry asdict]
    top1_pid: str = ""
    top1_final_score: float = 0.0
    top1_ce_score: float = 0.0
    top1_series: str = ""
    total_reranked: int = 0


@dataclass
class ToolCallEntry:
    tool_name: str
    args: dict = field(default_factory=dict)
    result_preview: str = ""     # first 200 chars ของผลลัพธ์
    success: bool = True


@dataclass
class StepLLM:
    clarification_triggered: bool = False   # Gemini ถามกลับ (ไม่ตอบรุ่นตรงๆ)
    clarification_signal: str = ""          # คำที่ detect ว่า Gemini ถาม
    tool_calls_made: list[dict] = field(default_factory=list)
    total_tool_calls: int = 0
    ai_reply_length: int = 0
    ai_reply_preview: str = ""    # 200 chars แรก
    latency_ms: float = 0.0


@dataclass
class StepEvalFlags:
    """
    Auto-evaluation flags สำหรับตรวจสอบว่าระบบทำงานถูกต้องไหม
    ไม่ใช่ ground-truth — เป็น heuristic ที่คำนวณ auto
    """
    top1_matches_query_series: bool = False    # series ใน top1 ตรงกับ series ที่ query
    top1_has_sideflexing: bool = False         # ถ้า query ถาม sideflexing → top1 ควรเป็น sideflexing
    top1_has_straight: bool = False            # ถ้า query ถาม straight → top1 ควรเป็น straight
    answer_contains_product_ref: bool = False  # AI reply มี product ref หรือ clarification question
    graph_contributed: bool = False            # GraphRAG expand ได้ candidates เพิ่ม
    clarification_given_for_broad_query: bool = False  # broad query → Gemini ถามกลับ (ควรเป็น True)
    image_search_led_to_correct_series: bool = False   # ViT detect series ตรงกับ query
    error_occurred: bool = False
    error_message: str = ""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TURN LOG — รวมทุก step
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class TurnLog:
    """บันทึกทุก step ของ 1 turn"""
    # Identifiers
    session_id: str = ""
    turn_id: str = ""
    turn_number: int = 0
    timestamp: str = ""

    # Per-step data
    step_input: dict = field(default_factory=dict)
    step_keyword: dict = field(default_factory=dict)
    step_image: dict = field(default_factory=dict)
    step_graph: dict = field(default_factory=dict)
    step_text: dict = field(default_factory=dict)
    step_rrf: dict = field(default_factory=dict)
    step_reranker: dict = field(default_factory=dict)
    step_llm: dict = field(default_factory=dict)
    step_eval: dict = field(default_factory=dict)

    # Top-level summary (ดึงมาจาก steps เพื่อให้ CSV อ่านง่าย)
    summary_effective_query: str = ""
    summary_vit_series: str = ""
    summary_graph_triggered_by: str = "none"
    summary_graph_candidates_added: int = 0
    summary_top1_pid: str = ""
    summary_top1_score: float = 0.0
    summary_total_candidates: int = 0
    summary_clarification: bool = False
    summary_tool_calls: int = 0
    summary_ai_reply_preview: str = ""
    summary_latency_ms: float = 0.0
    summary_error: str = ""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PIPELINE LOGGER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class PipelineLogger:
    """
    Logger หลักสำหรับ Movex Chatbot v3

    ใช้งาน:
        logger = PipelineLogger()
        session_id = logger.new_session()

        # ต้น turn
        log = logger.start_turn(session_id, user_text, has_image, last_pid)

        # บันทึกแต่ละ step ทีละขั้น
        logger.log_keyword(log, score_map)
        logger.log_image(log, vit_hits, siglip_hits, vit_series, vit_dominant, vit_top_score)
        logger.log_graph(log, graph_context, triggered_by)
        logger.log_text(log, text_hits, skipped, skip_reason)
        logger.log_rrf(log, score_map, ranked)
        logger.log_reranker(log, reranked_result)
        logger.log_llm(log, ai_reply, tool_calls, latency_ms)

        # ปิด turn → flush ลง disk
        logger.finish_turn(log, error=None)
    """

    # Regex สำหรับ detect clarification (Gemini ถามกลับ ไม่ตอบรุ่น)
    CLARIFICATION_PATTERNS = [
        r'ขอถามเพิ่มเติม',
        r'ขอทราบ.{0,20}ก่อน',
        r'ช่วยระบุ',
        r'ความกว้างประมาณ',
        r'รูปแบบการวางไลน์',
        r'สินค้าที่ลำเลียง',
        r'วิ่งตรง.*โค้ง',
        r'กรุณาระบุ',
    ]

    def __init__(self, log_dir: str = LOG_DIR):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        # File paths (rotate รายวัน)
        today = datetime.now().strftime("%Y%m%d")
        self._jsonl_path = os.path.join(log_dir, f"pipeline_{today}.jsonl")
        self._csv_path   = os.path.join(log_dir, f"summary_{today}.csv")
        self._txt_path   = os.path.join(log_dir, f"debug_{today}.txt")

        # Init CSV header ถ้าไฟล์ยังไม่มี
        self._init_csv()

        # Session counter
        self._session_turn_counts: dict[str, int] = {}

        print(f"📋 PipelineLogger ready → {log_dir}/")

    # ──────────────────────────────────────────────────────
    # SESSION
    # ──────────────────────────────────────────────────────

    def new_session(self) -> str:
        """สร้าง session_id ใหม่ (เรียกตอน user กด clear หรือเริ่มใหม่)"""
        session_id = f"sess_{datetime.now().strftime('%H%M%S')}_{uuid.uuid4().hex[:6]}"
        self._session_turn_counts[session_id] = 0
        return session_id

    # ──────────────────────────────────────────────────────
    # TURN LIFECYCLE
    # ──────────────────────────────────────────────────────

    def start_turn(
        self,
        session_id: str,
        user_text: str,
        has_image: bool,
        last_pid: str,
        effective_query: str,
        query_type: str = "general",
    ) -> TurnLog:
        """
        เริ่ม turn ใหม่ — คืน TurnLog object ที่จะ pass ไปทุก step
        """
        self._session_turn_counts[session_id] = \
            self._session_turn_counts.get(session_id, 0) + 1

        log = TurnLog(
            session_id=session_id,
            turn_id=f"{session_id}_t{self._session_turn_counts[session_id]:03d}",
            turn_number=self._session_turn_counts[session_id],
            timestamp=datetime.now().isoformat(timespec="milliseconds"),
        )

        log.step_input = asdict(StepInput(
            user_text=user_text,
            has_image=has_image,
            last_pid_context=last_pid,
            effective_query=effective_query,
            query_type=query_type,
        ))
        log.summary_effective_query = effective_query
        return log

    def finish_turn(self, log: TurnLog, error: Optional[str] = None):
        """
        ปิด turn — ฝัง eval flags แล้ว flush ลง disk ทั้ง 3 format
        """
        if error:
            log.summary_error = error
            log.step_eval = asdict(StepEvalFlags(
                error_occurred=True,
                error_message=error,
            ))
        elif not log.step_eval:
            log.step_eval = asdict(self._compute_eval_flags(log))

        self._flush_jsonl(log)
        self._flush_csv(log)
        self._flush_txt(log)

    # ──────────────────────────────────────────────────────
    # STEP LOGGERS
    # ──────────────────────────────────────────────────────

    def log_keyword(self, log: TurnLog, score_map: dict):
        """
        บันทึก Keyword Boost step

        Parameters
        ----------
        score_map : dict  — score_map หลัง _keyword_boost() รัน
        """
        hits = []
        query_lower = log.step_input.get("effective_query", "").lower()

        sideflex = any(w in query_lower for w in ["เลี้ยว", "โค้ง", "sideflex", "curve"])
        straight = any(w in query_lower for w in ["ตรง", "straight", "วิ่งตรง"])

        # series detect
        series_nums = re.findall(r'\b(820|821|880|882|83|103)\b', query_lower)

        for pid, s in score_map.items():
            kw = s.get("keyword", 0.0)
            if kw > 0:
                if kw >= 10.0:
                    mtype = "exact"
                elif kw >= 6.0:
                    mtype = "fuzzy_high"
                elif kw >= 8.0:
                    mtype = "product_type"
                elif kw >= 3.0:
                    mtype = "fuzzy_mid"
                else:
                    mtype = "substring"
                hits.append(asdict(KeywordHit(pid=pid, boost_score=kw, match_type=mtype)))

        hits.sort(key=lambda x: x["boost_score"], reverse=True)

        log.step_keyword = asdict(StepKeyword(
            sideflex_kw_triggered=sideflex,
            straight_kw_triggered=straight,
            query_series_detected=series_nums,
            hits=hits,
            total_hits=len(hits),
        ))

    def log_image(
        self,
        log: TurnLog,
        vit_hits_raw: list,      # list of Qdrant ScoredPoint
        siglip_hits_raw: list,   # list of Qdrant ScoredPoint
        vit_series: Optional[str],
        vit_dominant: bool,
        vit_top_score: float,
        vit_confidence_threshold: float = 0.50,
    ):
        """บันทึก Image Search step (ViT + SigLIP)"""
        vit_hits = [
            asdict(ImageHit(
                pid=h.payload.get("product_id", ""),
                score=round(h.score, 4),
                rank=i,
                source="vit",
            ))
            for i, h in enumerate(vit_hits_raw[:10])
            if h.payload.get("product_id")
        ]
        siglip_hits = [
            asdict(ImageHit(
                pid=h.payload.get("product_id", ""),
                score=round(h.score, 4),
                rank=i,
                source="siglip",
            ))
            for i, h in enumerate(siglip_hits_raw[:10])
            if h.payload.get("product_id")
        ]

        log.step_image = asdict(StepImage(
            triggered=True,
            vit_top_score=round(vit_top_score, 4),
            vit_series_detected=vit_series,
            vit_dominant=vit_dominant,
            vit_confidence_threshold=vit_confidence_threshold,
            vit_hits=vit_hits,
            siglip_hits=siglip_hits,
            total_vit_hits=len(vit_hits),
            total_siglip_hits=len(siglip_hits),
        ))
        log.summary_vit_series = vit_series or ""

    def log_graph(
        self,
        log: TurnLog,
        graph_context,   # GraphContext dataclass
        triggered_by: str,   # "vit" | "keyword" | "none"
    ):
        """บันทึก GraphRAG Expansion step"""
        if graph_context is None:
            log.step_graph = asdict(StepGraph(triggered_by=triggered_by))
            return

        mat_found = {
            mat: info.get("full_name", mat)
            for mat, info in graph_context.material_details.items()
        }
        proc_found = {
            proc: info.get("full_name", proc)
            for proc, info in graph_context.process_details.items()
        }

        contributed = bool(
            graph_context.same_series
            or graph_context.compatible_products
            or graph_context.same_material
        )

        log.step_graph = asdict(StepGraph(
            triggered_by=triggered_by,
            seed_products=graph_context.seed_products,
            same_series_expanded=graph_context.same_series,
            compatible_expanded=graph_context.compatible_products,
            same_material_expanded=graph_context.same_material,
            material_details_found=mat_found,
            process_details_found=proc_found,
            total_graph_candidates=len(graph_context.all_candidate_ids),
            graph_contributed=contributed,
        ))
        log.summary_graph_triggered_by = triggered_by
        log.summary_graph_candidates_added = len(
            graph_context.same_series
            + graph_context.compatible_products
            + graph_context.same_material
        )

    def log_text(
        self,
        log: TurnLog,
        hits_raw: list,           # list of Qdrant ScoredPoint
        skipped: bool = False,
        skip_reason: str = "",
    ):
        """บันทึก Text Search step"""
        hits = [
            {"pid": h.payload.get("product_id", ""), "rank": i, "score": round(h.score, 4)}
            for i, h in enumerate(hits_raw[:20])
            if h.payload.get("product_id")
        ] if not skipped else []

        log.step_text = asdict(StepText(
            skipped=skipped,
            skip_reason=skip_reason,
            hits=hits,
            total_hits=len(hits),
        ))

    def log_rrf(
        self,
        log: TurnLog,
        score_map: dict,
        ranked: list[tuple[str, float]],
    ):
        """บันทึก RRF Fusion step"""
        # Full score map snapshot
        full = []
        for pid, s in score_map.items():
            # หา rrf_final จาก ranked
            rrf_val = next((v for p, v in ranked if p == pid), 0.0)
            full.append(asdict(ScoreMapEntry(
                pid=pid,
                text_ranks=s.get("text_ranks", []),
                vit_rank=s.get("vit_rank"),
                siglip_rank=s.get("siglip_rank"),
                keyword_score=s.get("keyword", 0.0),
                graph_rank=s.get("graph"),
                rrf_final=round(rrf_val, 6),
            )))

        full.sort(key=lambda x: x["rrf_final"], reverse=True)

        log.step_rrf = asdict(StepRRF(
            total_unique_candidates=len(ranked),
            top10_scores=full[:10],
            score_map_full=full,
        ))
        log.summary_total_candidates = len(ranked)

    def log_reranker(self, log: TurnLog, reranked_result):
        """บันทึก Reranker step"""
        products = [
            asdict(RerankedEntry(
                pid=p.product_id,
                rank=i + 1,
                final_score=round(p.final_score, 4),
                ce_score=round(p.cross_encoder_score, 4),
                rrf_score=round(p.rrf_score, 6),
                keyword_boost=p.keyword_boost,
                series=p.series,
                product_type=p.product_type,
            ))
            for i, p in enumerate(reranked_result.products)
        ]

        top1 = reranked_result.products[0] if reranked_result.products else None
        log.step_reranker = asdict(StepReranker(
            products=products,
            top1_pid=top1.product_id if top1 else "",
            top1_final_score=round(top1.final_score, 4) if top1 else 0.0,
            top1_ce_score=round(top1.cross_encoder_score, 4) if top1 else 0.0,
            top1_series=top1.series if top1 else "",
            total_reranked=len(products),
        ))
        log.summary_top1_pid   = top1.product_id if top1 else ""
        log.summary_top1_score = round(top1.final_score, 4) if top1 else 0.0

    def log_llm(
        self,
        log: TurnLog,
        ai_reply: str,
        tool_calls: list[dict],   # [{"tool_name": ..., "args": ..., "result": ...}]
        latency_ms: float,
    ):
        """บันทึก LLM (Gemini) step"""
        # ตรวจ clarification
        clarification = self._detect_clarification(ai_reply)
        clarification_signal = ""
        if clarification:
            for pat in self.CLARIFICATION_PATTERNS:
                if re.search(pat, ai_reply):
                    clarification_signal = pat
                    break

        # Tool calls
        tool_entries = [
            asdict(ToolCallEntry(
                tool_name=tc.get("tool_name", ""),
                args=tc.get("args", {}),
                result_preview=str(tc.get("result", ""))[:200],
                success=tc.get("success", True),
            ))
            for tc in tool_calls
        ]

        log.step_llm = asdict(StepLLM(
            clarification_triggered=clarification,
            clarification_signal=clarification_signal,
            tool_calls_made=tool_entries,
            total_tool_calls=len(tool_entries),
            ai_reply_length=len(ai_reply),
            ai_reply_preview=ai_reply[:200],
            latency_ms=round(latency_ms, 1),
        ))
        log.summary_clarification    = clarification
        log.summary_tool_calls       = len(tool_entries)
        log.summary_ai_reply_preview = ai_reply[:100]
        log.summary_latency_ms       = round(latency_ms, 1)

    # ──────────────────────────────────────────────────────
    # AUTO EVAL FLAGS
    # ──────────────────────────────────────────────────────

    def _compute_eval_flags(self, log: TurnLog) -> StepEvalFlags:
        """
        คำนวณ evaluation flags แบบ heuristic อัตโนมัติ
        ไม่ใช่ ground-truth แต่ใช้ spot-check ได้ดี
        """
        flags = StepEvalFlags()
        query = log.step_input.get("effective_query", "").lower()
        query_original = log.step_input.get("user_text", "").lower()

        # series ที่ query ระบุ
        series_in_query = re.findall(r'\b(820|821|880|882|83|103)\b', query)

        top1_pid    = log.summary_top1_pid
        top1_series = log.step_reranker.get("top1_series", "")

        # 1. top1 series ตรงกับที่ query ระบุ
        if series_in_query and top1_series:
            for s in series_in_query:
                if s in top1_series:
                    flags.top1_matches_query_series = True
                    break

        # 2. sideflexing query → top1 ควรเป็น sideflexing
        sideflex_q = any(w in query for w in ["เลี้ยว", "โค้ง", "sideflex"])
        if sideflex_q:
            top1_ptype = ""
            for p in log.step_reranker.get("products", []):
                if p.get("rank") == 1:
                    top1_ptype = p.get("product_type", "").lower()
                    break
            flags.top1_has_sideflexing = "sideflexing" in top1_ptype

        # 3. straight query → top1 ควรเป็น straight
        straight_q = any(w in query for w in ["ตรง", "straight"])
        if straight_q:
            top1_ptype = ""
            for p in log.step_reranker.get("products", []):
                if p.get("rank") == 1:
                    top1_ptype = p.get("product_type", "").lower()
                    break
            flags.top1_has_straight = "straight running" in top1_ptype

        # 4. AI reply มี product ref หรือเป็น clarification
        reply_preview = log.step_llm.get("ai_reply_preview", "")
        has_ref = bool(re.search(
            r'\b(LF|LFN|SS)\s*\d+|movex_chain|movex_sprocket|\d{5}', reply_preview
        ))
        flags.answer_contains_product_ref = has_ref or log.summary_clarification

        # 5. Graph contributed
        flags.graph_contributed = log.step_graph.get("graph_contributed", False)

        # 6. Broad query → clarification expected
        broad_signals = [
            "มีรุ่น", "แนะนำ", "มีอะไรบ้าง", "เลือกอะไรดี",
            "มีสายพานเลี้ยวได้", "อยากเปลี่ยน"
        ]
        is_broad = any(w in query_original for w in broad_signals)
        if is_broad:
            flags.clarification_given_for_broad_query = log.summary_clarification

        # 7. ViT series ตรงกับ query series
        vit_series = log.summary_vit_series
        if vit_series and series_in_query:
            flags.image_search_led_to_correct_series = vit_series in series_in_query

        return flags

    def _detect_clarification(self, ai_reply: str) -> bool:
        """ตรวจว่า Gemini ถามกลับ (clarification) หรือตอบรุ่นตรงๆ"""
        for pat in self.CLARIFICATION_PATTERNS:
            if re.search(pat, ai_reply):
                return True
        return False

    # ──────────────────────────────────────────────────────
    # FLUSH TO DISK
    # ──────────────────────────────────────────────────────

    def _flush_jsonl(self, log: TurnLog):
        """เขียน 1 JSON object ต่อ 1 บรรทัด (append)"""
        try:
            record = asdict(log)
            with open(self._jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"⚠️ Logger JSONL write error: {e}")

    def _flush_csv(self, log: TurnLog):
        """เขียน summary row ลง CSV (append)"""
        try:
            file_exists = os.path.isfile(self._csv_path)
            row = {
                "timestamp":            log.timestamp,
                "session_id":           log.session_id,
                "turn_id":              log.turn_id,
                "turn_number":          log.turn_number,
                # Input
                "user_text":            log.step_input.get("user_text", ""),
                "has_image":            log.step_input.get("has_image", False),
                "last_pid_context":     log.step_input.get("last_pid_context", ""),
                "effective_query":      log.summary_effective_query,
                "query_type":           log.step_input.get("query_type", ""),
                # Keyword
                "keyword_hits":         log.step_keyword.get("total_hits", 0),
                "sideflex_kw":          log.step_keyword.get("sideflex_kw_triggered", False),
                "straight_kw":          log.step_keyword.get("straight_kw_triggered", False),
                "query_series":         "|".join(log.step_keyword.get("query_series_detected", [])),
                # Image
                "image_search":         log.step_image.get("triggered", False),
                "vit_top_score":        log.step_image.get("vit_top_score", ""),
                "vit_series":           log.summary_vit_series,
                "vit_dominant":         log.step_image.get("vit_dominant", False),
                "vit_hits_count":       log.step_image.get("total_vit_hits", 0),
                "siglip_hits_count":    log.step_image.get("total_siglip_hits", 0),
                # Graph
                "graph_triggered_by":   log.summary_graph_triggered_by,
                "graph_candidates_added": log.summary_graph_candidates_added,
                "graph_contributed":    log.step_graph.get("graph_contributed", False),
                "graph_seed_products":  "|".join(log.step_graph.get("seed_products", [])),
                "graph_compatible":     "|".join(log.step_graph.get("compatible_expanded", [])),
                "graph_same_series":    "|".join(log.step_graph.get("same_series_expanded", [])),
                # Text
                "text_skipped":         log.step_text.get("skipped", False),
                "text_skip_reason":     log.step_text.get("skip_reason", ""),
                "text_hits_count":      log.step_text.get("total_hits", 0),
                # RRF
                "total_candidates":     log.summary_total_candidates,
                # Reranker
                "top1_pid":             log.summary_top1_pid,
                "top1_final_score":     log.summary_top1_score,
                "top1_ce_score":        log.step_reranker.get("top1_ce_score", ""),
                "top1_series":          log.step_reranker.get("top1_series", ""),
                "reranked_pids":        "|".join(
                    p["pid"] for p in log.step_reranker.get("products", [])
                ),
                # LLM
                "clarification":        log.summary_clarification,
                "tool_calls":           log.summary_tool_calls,
                "tool_names":           "|".join(
                    tc["tool_name"] for tc in log.step_llm.get("tool_calls_made", [])
                ),
                "ai_reply_length":      log.step_llm.get("ai_reply_length", 0),
                "ai_reply_preview":     log.summary_ai_reply_preview,
                "latency_ms":           log.summary_latency_ms,
                # Eval flags
                "eval_top1_series_ok":  log.step_eval.get("top1_matches_query_series", ""),
                "eval_sideflex_ok":     log.step_eval.get("top1_has_sideflexing", ""),
                "eval_straight_ok":     log.step_eval.get("top1_has_straight", ""),
                "eval_ref_in_reply":    log.step_eval.get("answer_contains_product_ref", ""),
                "eval_graph_contrib":   log.step_eval.get("graph_contributed", ""),
                "eval_clarif_ok":       log.step_eval.get("clarification_given_for_broad_query", ""),
                "eval_vit_series_ok":   log.step_eval.get("image_search_led_to_correct_series", ""),
                "error":                log.summary_error,
            }

            with open(self._csv_path, "a", encoding="utf-8-sig", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                if not file_exists:
                    writer.writeheader()
                writer.writerow(row)
        except Exception as e:
            print(f"⚠️ Logger CSV write error: {e}")

    def _flush_txt(self, log: TurnLog):
        """เขียน human-readable debug block"""
        try:
            lines = [
                "═" * 70,
                f"TURN  {log.turn_id}  [{log.timestamp}]",
                "═" * 70,
                f"  INPUT       : {log.step_input.get('user_text','')[:80]}",
                f"  HAS_IMAGE   : {log.step_input.get('has_image')}",
                f"  EFFECTIVE_Q : {log.summary_effective_query[:80]}",
                f"  QUERY_TYPE  : {log.step_input.get('query_type','')}",
                "",
                "── STEP 1: Keyword Boost ──────────────────────────────",
                f"  hits={log.step_keyword.get('total_hits',0)}  "
                f"sideflex={log.step_keyword.get('sideflex_kw_triggered')}  "
                f"straight={log.step_keyword.get('straight_kw_triggered')}  "
                f"series={log.step_keyword.get('query_series_detected',[])}",
            ]

            for h in log.step_keyword.get("hits", [])[:5]:
                lines.append(f"    [{h['match_type']:12s}] {h['pid']}  boost={h['boost_score']}")

            # Image
            lines += [
                "",
                "── STEP 2: Image Search ───────────────────────────────",
            ]
            if log.step_image.get("triggered"):
                lines += [
                    f"  vit_top_score={log.step_image.get('vit_top_score')}  "
                    f"series={log.step_image.get('vit_series_detected')}  "
                    f"dominant={log.step_image.get('vit_dominant')}",
                    f"  vit_hits={log.step_image.get('total_vit_hits')}  "
                    f"siglip_hits={log.step_image.get('total_siglip_hits')}",
                ]
                for h in log.step_image.get("vit_hits", [])[:3]:
                    lines.append(f"    [ViT  rank{h['rank']}] {h['pid']}  score={h['score']}")
                for h in log.step_image.get("siglip_hits", [])[:3]:
                    lines.append(f"    [SIGLIP rank{h['rank']}] {h['pid']}  score={h['score']}")
            else:
                lines.append("  (not triggered)")

            # Graph
            g = log.step_graph
            lines += [
                "",
                "── STEP 3: GraphRAG ───────────────────────────────────",
                f"  triggered_by={g.get('triggered_by','none')}  "
                f"seeds={g.get('seed_products',[])}",
                f"  same_series({len(g.get('same_series_expanded',[]))}) → "
                f"{g.get('same_series_expanded',[][:3])}",
                f"  compatible({len(g.get('compatible_expanded',[]))}) → "
                f"{g.get('compatible_expanded',[][:3])}",
                f"  materials={list(g.get('material_details_found',{}).keys())}",
                f"  processes={list(g.get('process_details_found',{}).keys())}",
                f"  total_graph_candidates={g.get('total_graph_candidates',0)}",
            ]

            # Text
            t = log.step_text
            lines += [
                "",
                "── STEP 4: Text Search ────────────────────────────────",
                f"  skipped={t.get('skipped')}  reason={t.get('skip_reason','')}  "
                f"hits={t.get('total_hits',0)}",
            ]
            for h in t.get("hits", [])[:5]:
                lines.append(f"    [rank{h['rank']}] {h['pid']}")

            # RRF
            lines += [
                "",
                "── STEP 5: RRF Fusion ─────────────────────────────────",
                f"  total_candidates={log.summary_total_candidates}",
                "  Top-5 after RRF:",
            ]
            for e in log.step_rrf.get("top10_scores", [])[:5]:
                lines.append(
                    f"    {e['pid']:40s}  rrf={e['rrf_final']:.5f}  "
                    f"kw={e['keyword_score']:.1f}  "
                    f"txt={e['text_ranks']}  "
                    f"vit={e['vit_rank']}  "
                    f"sig={e['siglip_rank']}  "
                    f"grph={e['graph_rank']}"
                )

            # Reranker
            lines += [
                "",
                "── STEP 6: Reranker ───────────────────────────────────",
            ]
            for p in log.step_reranker.get("products", []):
                lines.append(
                    f"  [{p['rank']}] {p['pid']:40s}  "
                    f"final={p['final_score']:.4f}  "
                    f"CE={p['ce_score']:.2f}  "
                    f"RRF={p['rrf_score']:.4f}  "
                    f"kw={p['keyword_boost']}"
                )

            # LLM
            llm = log.step_llm
            lines += [
                "",
                "── STEP 7: LLM (Gemini) ───────────────────────────────",
                f"  clarification={llm.get('clarification_triggered')}  "
                f"signal='{llm.get('clarification_signal','')}'",
                f"  tool_calls={llm.get('total_tool_calls')}  "
                f"latency={llm.get('latency_ms')}ms",
            ]
            for tc in llm.get("tool_calls_made", []):
                lines.append(f"    → {tc['tool_name']}({tc['args']})  "
                             f"result='{tc['result_preview'][:80]}'")
            lines.append(f"  reply_preview: {llm.get('ai_reply_preview','')[:120]}")

            # Eval
            ev = log.step_eval
            lines += [
                "",
                "── STEP 8: Eval Flags ─────────────────────────────────",
                f"  top1_series_ok     : {ev.get('top1_matches_query_series')}",
                f"  sideflex_ok        : {ev.get('top1_has_sideflexing')}",
                f"  straight_ok        : {ev.get('top1_has_straight')}",
                f"  ref_in_reply       : {ev.get('answer_contains_product_ref')}",
                f"  graph_contributed  : {ev.get('graph_contributed')}",
                f"  clarif_ok (broad)  : {ev.get('clarification_given_for_broad_query')}",
                f"  vit_series_ok      : {ev.get('image_search_led_to_correct_series')}",
                f"  error              : {ev.get('error_message','')}",
                "",
            ]

            with open(self._txt_path, "a", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
        except Exception as e:
            print(f"⚠️ Logger TXT write error: {e}")

    # ──────────────────────────────────────────────────────
    # CSV INIT
    # ──────────────────────────────────────────────────────

    def _init_csv(self):
        """สร้าง CSV header ถ้าไฟล์ยังไม่มี"""
        if not os.path.isfile(self._csv_path):
            headers = [
                "timestamp", "session_id", "turn_id", "turn_number",
                "user_text", "has_image", "last_pid_context", "effective_query", "query_type",
                "keyword_hits", "sideflex_kw", "straight_kw", "query_series",
                "image_search", "vit_top_score", "vit_series", "vit_dominant",
                "vit_hits_count", "siglip_hits_count",
                "graph_triggered_by", "graph_candidates_added", "graph_contributed",
                "graph_seed_products", "graph_compatible", "graph_same_series",
                "text_skipped", "text_skip_reason", "text_hits_count",
                "total_candidates",
                "top1_pid", "top1_final_score", "top1_ce_score", "top1_series",
                "reranked_pids",
                "clarification", "tool_calls", "tool_names",
                "ai_reply_length", "ai_reply_preview", "latency_ms",
                "eval_top1_series_ok", "eval_sideflex_ok", "eval_straight_ok",
                "eval_ref_in_reply", "eval_graph_contrib", "eval_clarif_ok",
                "eval_vit_series_ok", "error",
            ]
            with open(self._csv_path, "w", encoding="utf-8-sig", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()

    # ──────────────────────────────────────────────────────
    # UTILITY
    # ──────────────────────────────────────────────────────

    def log_paths(self) -> dict[str, str]:
        """คืน dict ของ path ทั้ง 3 ไฟล์"""
        return {
            "jsonl": self._jsonl_path,
            "csv":   self._csv_path,
            "txt":   self._txt_path,
        }

    def rotate_daily(self):
        """เปลี่ยน path เป็นวันใหม่ (เรียกตอน startup หรือ midnight)"""
        today = datetime.now().strftime("%Y%m%d")
        self._jsonl_path = os.path.join(self.log_dir, f"pipeline_{today}.jsonl")
        self._csv_path   = os.path.join(self.log_dir, f"summary_{today}.csv")
        self._txt_path   = os.path.join(self.log_dir, f"debug_{today}.txt")
        self._init_csv()
