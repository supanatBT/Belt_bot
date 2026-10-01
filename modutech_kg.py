"""กราฟความสัมพันธ์ของสินค้า Modutech — สร้างจาก modutech_v30_1.json ตอนสตาร์ต

ทำไมสร้างใหม่ทุกครั้ง ไม่โหลดไฟล์ .db
  ไฟล์ .db ที่สร้างไว้ล่วงหน้าคือสำเนาที่สองของข้อมูลเดียวกัน วันที่แก้ JSON แล้วลืมสร้าง .db ใหม่
  จะไม่มี error ให้เห็น — ระบบเดินต่อด้วยความสัมพันธ์รุ่นเก่า เป็นความพลาดชนิดเดียวกับที่เคยเกิด
  ตอนเอา zip เก่าทับไฟล์ที่แก้แล้ว การสร้างใหม่ใช้เวลา 1.2 ms จาก dict ที่แอปโหลดไว้อยู่แล้ว
  (from_catalogue) จึงไม่มีเหตุผลต้องเก็บสำเนาไว้
  (ตรงกับหลักการของโปรเจกต์: เก็บของดิบคู่ของที่ซ่อม)

ทำไมกราฟไม่เก็บตัวเลข
  กราฟตอบได้แค่ "อะไรคู่กับอะไร" ไม่เก็บ Di_mm · Do_mm · Bore_mm · teeth หรือค่าใด ๆ
  ค่าพวกนั้นอยู่ใน JSON หลัง sprocket_gate ซึ่งบล็อก sprocket_dimension ไว้ทั้งหมด
  เพราะเล่มไม่นิยาม Di/Do/A/B/E/X และเครื่องหมาย & ในช่องรูเพลายังตีความไม่ได้
  (บันทึกใน sprocket_open_questions: ampersand_vs_to หน้า 224/250)

  ไฟล์ modutech_kg.db รุ่นที่สร้างไว้ 7 ก.ย. เก็บ Di_mm/Do_mm/Bore_mm ไว้ใน node attrs
  ถ้าต่อกราฟแบบนั้นเข้า retriever ค่าที่ยังตีความไม่ได้จะเลี่ยงเกตออกไปถึงลูกค้าได้
  โมดูลนี้ตัดปัญหาที่ต้นทาง — ของที่ไม่มีอยู่ในกราฟ รั่วออกจากกราฟไม่ได้

ขอบเขต
  กราฟบอกความสัมพันธ์ ไม่ตัดสินว่าพูดได้หรือไม่ได้ — ธง blocking_for ยังเป็นหน้าที่ของเกต
  ตรงกับหลักการ: รายงานข้อขัดแย้ง ไม่ตัดสินแทน

รัน: python modutech_kg.py [modutech_v30_1.json]
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Optional

# ── ลายนิ้วมือของ "ส่วนที่กราฟใช้" ────────────────────────────────
# ใช้เทียบว่าไฟล์ .db ที่สร้างไว้ยังตรงกับ JSON ปัจจุบันหรือไม่
# แฮชแค่ส่วนที่มีผลต่อความสัมพันธ์ ไม่ใช่ทั้งไฟล์ เพราะถ้าแฮชทั้งไฟล์
# แก้หมายเหตุหนึ่งบรรทัด (เช่นตอนทำ v30_1) จะกลายเป็นว่า .db ล้าสมัยทั้งที่ความสัมพันธ์ไม่เปลี่ยน
# แล้วคนจะเลิกเชื่อคำเตือน ซึ่งอันตรายกว่าไม่มีคำเตือน
FINGERPRINT_KEY = "modutech_catalogue_fingerprint"


def catalogue_fingerprint(data: dict) -> str:
    """sha256 ของเฉพาะฟิลด์ที่กราฟใช้ — belt_code · series · วัสดุ · สเตอร์ · ลิงก์"""
    shape = {
        "belts": sorted(
            (p.get("belt_code", ""), p.get("belt_code_id", ""), p.get("belt_series", ""),
             tuple(sorted((p.get("retrieval_card") or {}).get("materials") or [])),
             tuple(sorted((p.get("derived") or {}).get("materials_manufactured") or [])))
            for p in data.get("products") or []),
        "sprockets": sorted((s.get("sprocket_code", ""), s.get("material") or "")
                            for s in data.get("sprockets") or []),
        "links": sorted((l.get("belt_code", ""), tuple(sorted(l.get("sprocket_codes") or [])))
                        for l in data.get("belt_sprocket_links") or []),
    }
    blob = json.dumps(shape, ensure_ascii=False, sort_keys=True, default=list)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def db_fingerprint(db_path: str) -> Optional[str]:
    """อ่านลายนิ้วมือที่ฝังไว้ในตาราง metadata — None ถ้าไฟล์เก่าที่ยังไม่มีการฝัง"""
    try:
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    except sqlite3.Error:
        return None
    try:
        row = con.execute("select value from metadata where key=?", (FINGERPRINT_KEY,)).fetchone()
        return row[0] if row else None
    except sqlite3.Error:
        return None
    finally:
        con.close()


PROTECTED_DB = ("movex_kg.db",)


def _refuse_if_movex(db_path: str, what: str) -> None:
    """ด่านกันเขียนทับกราฟของ Movex

    build_modutech_graph.py มี assert_not_movex อยู่แล้ว แต่ฟังก์ชันที่เขียนไฟล์
    ต้องกันตัวเองด้วย ไม่ใช่พึ่งว่าผู้เรียกกันให้ — ใครเรียกตรงจากที่อื่นก็ยังปลอดภัย
    ข้อห้ามของโปรเจกต์คือห้ามแก้ของ Movex จึงต้องกันที่ทุกจุดที่เขียนได้
    """
    import os
    name = os.path.basename(db_path).lower()
    if name in PROTECTED_DB or "movex" in name:
        raise PermissionError(f"ปฏิเสธ {what} กับ {db_path} — เป็นกราฟของ Movex ห้ามแตะ")


def stamp_db(db_path: str, data: dict) -> str:
    """ฝังลายนิ้วมือลงตาราง metadata หลังบันทึก .db เสร็จ"""
    _refuse_if_movex(db_path, "ฝังลายนิ้วมือ")
    fp = catalogue_fingerprint(data)
    con = sqlite3.connect(db_path)
    try:
        con.execute("create table if not exists metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        con.execute("insert or replace into metadata(key,value) values(?,?)", (FINGERPRINT_KEY, fp))
        con.commit()
    finally:
        con.close()
    return fp

# ชนิดปม — เก็บเป็นสตริงคงที่ เทียบกับ .db รุ่น 7 ก.ย. ได้ตรง ๆ
BELT, SPROCKET, SERIES, MATERIAL = "Belt", "Sprocket", "Series", "Material"
BELONGS_TO, MADE_OF, COMPATIBLE_WITH = "BELONGS_TO", "MADE_OF", "COMPATIBLE_WITH"


def _node_id(kind: str, key: str) -> str:
    return f"{kind.lower()}:{key}"


@dataclass(frozen=True)
class Node:
    """ปมหนึ่งปม — ถือได้แค่ชื่อกับชนิด ห้ามมีตัวเลขสเปก

    attrs รับได้เฉพาะสตริงหรือรายการสตริง ถ้าใครเผลอยัดตัวเลขเข้ามา __post_init__ จะฟ้อง
    """
    node_id: str
    node_type: str
    label: str
    attrs: tuple = ()

    def __post_init__(self) -> None:
        for k, v in self.attrs:
            if isinstance(v, bool) or isinstance(v, (int, float)):
                raise TypeError(
                    f"{self.node_id}: ห้ามเก็บค่าตัวเลขในกราฟ ({k}={v!r}) "
                    "ค่าสเปกต้องอ่านจาก JSON ผ่านเกตเท่านั้น")

    def as_dict(self) -> dict:
        return {"node_id": self.node_id, "node_type": self.node_type,
                "label": self.label, **dict(self.attrs)}


@dataclass
class GraphContext:
    """ผลการเดินกราฟ — ตั้งชื่อฟิลด์ซ้ำสองแบบเพื่อให้เข้ากับทั้ง RetrieverV4 และ chatbot_v5

    _NullKG เดิมคืน .compatible_products ส่วน GraphResult ใน chatbot_v5 ใช้ .compatible
    ยังไม่มี retriever_v4.py ให้อ่าน จึงคืนทั้งสองชื่อไว้ก่อน ไม่ใช่ความสวยงาม แต่กัน AttributeError
    """
    seed_products: list[str] = field(default_factory=list)
    compatible_products: list[str] = field(default_factory=list)
    same_series: list[str] = field(default_factory=list)
    material_details: dict = field(default_factory=dict)
    process_details: dict = field(default_factory=dict)
    hop_trace: list[str] = field(default_factory=list)

    @property
    def compatible(self) -> list[str]:
        return self.compatible_products


class ModutechGraph:
    """ความสัมพันธ์สินค้า Modutech เป็นกฎตายตัว — ไม่ผ่านการค้นเชิงความหมาย

    ปม     Belt 68 · Sprocket 142 · Series 7 · Material (จาก material_registry)
    เส้น   BELONGS_TO (สายพาน→ซีรีส์) · MADE_OF (สายพาน/สเตอร์→วัสดุ) ·
           COMPATIBLE_WITH (สายพาน↔สเตอร์ สองทาง)
    """

    # ธงบอกว่ากราฟตัวนี้รับและคืน "รหัสสายพาน" (HC508 C) ไม่ใช่ belt_code_id (hc508-c)
    #
    # ต้องเป็นธงชัด ๆ ไม่ใช่ hasattr เพราะ KnowledgeGraph ของ Movex มี get_same_series ·
    # get_compatible_products · get_material_details · get_process_details ครบเหมือนกัน
    # การเดาชนิดกราฟจากชื่อเมธอดจึงผิด และผิดแบบไม่มี error — คืน [] เงียบ ๆ
    # (เจอมาแล้วกับ same_series ที่คืน 0 รุ่นทั้งที่กราฟในไฟล์มีข้อมูลครบ 21 รุ่น)
    accepts_belt_codes = True

    def __init__(self, data: dict) -> None:
        self.nodes: dict[str, Node] = {}
        self._out: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
        self._belt_by_code: dict[str, str] = {}
        self._belt_by_id: dict[str, str] = {}
        self._build(data)
        self._verify(data)

    # ── สร้าง ────────────────────────────────────────────────
    def _add(self, node: Node) -> str:
        self.nodes.setdefault(node.node_id, node)
        return node.node_id

    def _edge(self, src: str, rel: str, dst: str, both: bool = False) -> None:
        if dst not in self._out[src][rel]:
            self._out[src][rel].append(dst)
        if both and src not in self._out[dst][rel]:
            self._out[dst][rel].append(src)

    def _build(self, data: dict) -> None:
        # วัสดุ — ชื่อกับคำอธิบายเป็นข้อความ ค่าอุณหภูมิใน registry ไม่เอาเข้ากราฟ
        for m in data.get("material_registry") or []:
            code = m.get("material_code")
            if not code:
                continue
            self._add(Node(_node_id(MATERIAL, code), MATERIAL, code,
                           (("name", m.get("name") or ""),
                            ("base_polymer", m.get("base_polymer") or ""))))

        for p in data.get("products") or []:
            code = p["belt_code"]
            bid = p.get("belt_code_id") or code
            nid = self._add(Node(_node_id(BELT, bid), BELT, code,
                                 (("belt_code", code),
                                  ("belt_code_id", bid),
                                  ("series", p.get("belt_series") or ""),
                                  ("surface_type",
                                   (p.get("retrieval_card") or {}).get("surface_type") or ""))))
            self._belt_by_code[code] = nid
            self._belt_by_id[bid] = nid

            series = p.get("belt_series")
            if series:
                self._edge(nid, BELONGS_TO,
                           self._add(Node(_node_id(SERIES, series), SERIES, series)))
            for mat in (p.get("retrieval_card") or {}).get("materials") or []:
                self._edge(nid, MADE_OF,
                           self._add(Node(_node_id(MATERIAL, mat), MATERIAL, mat)))

        for s in data.get("sprockets") or []:
            code = s["sprocket_code"]
            nid = self._add(Node(_node_id(SPROCKET, code), SPROCKET, code,
                                 (("sprocket_code", code),
                                  ("teeth_printed", s.get("teeth_printed") or ""),
                                  ("bore_type",
                                   (s.get("code_parts") or {}).get("bore_type_meaning") or ""))))
            mat = s.get("material")
            if mat:
                self._edge(nid, MADE_OF,
                           self._add(Node(_node_id(MATERIAL, mat), MATERIAL, mat)))

        # ความเข้ากันได้ — สองทาง เพื่อถามกลับได้ว่าสเตอร์ตัวนี้ใช้กับสายพานรุ่นไหน
        for l in data.get("belt_sprocket_links") or []:
            b = self._belt_by_code.get(l["belt_code"])
            if b is None:
                continue
            for sc in l.get("sprocket_codes") or []:
                sn = _node_id(SPROCKET, sc)
                if sn in self.nodes:
                    self._edge(b, COMPATIBLE_WITH, sn, both=True)

        # material_registry มี 22 เกรด แต่สินค้าอ้างถึงจริงแค่ 11 — ตัดปมที่ไม่มีเส้นออก
        # ปมลอยไม่ผิด แต่ทำให้ตัวเลขในเล่มอธิบายยากว่าทำไมนับวัสดุได้ 22 ทั้งที่ใช้ 11
        used = {d for rels in self._out.values() for d in rels.get(MADE_OF, ())}
        for nid in [n for n, v in self.nodes.items()
                    if v.node_type == MATERIAL and n not in used]:
            del self.nodes[nid]

    def _verify(self, data: dict) -> None:
        """นับให้ตรงกับไฟล์ ไม่ตรง = หยุด ดีกว่าเดินต่อด้วยกราฟที่ขาด"""
        want = {BELT: len(data.get("products") or []),
                SPROCKET: len(data.get("sprockets") or [])}
        got = {k: sum(1 for n in self.nodes.values() if n.node_type == k) for k in want}
        if got != want:
            raise AssertionError(f"จำนวนปมไม่ตรงไฟล์: ได้ {got} ต้องเป็น {want}")

        linked = {l["belt_code"] for l in data.get("belt_sprocket_links") or []
                  if l.get("sprocket_codes")}
        have = {self.nodes[n].label for n in self.nodes
                if self.nodes[n].node_type == BELT and self._out[n][COMPATIBLE_WITH]}
        if have != linked:
            raise AssertionError(
                f"สายพานที่มีคู่สเตอร์ไม่ตรงไฟล์: ขาด {sorted(linked - have)} "
                f"เกิน {sorted(have - linked)}")

        # การนับด้านบนเทียบกับไฟล์เดียวกัน จึงจับกรณี "สเตอร์หายไปจาก sprockets แต่ยังถูกอ้าง
        # ใน belt_sprocket_links" ไม่ได้ — ต้องตรวจความอ้างอิงข้ามหมวดแยกอีกชั้น
        # ถ้าปล่อยไว้ สายพานจะได้คู่สเตอร์น้อยกว่าที่เล่มระบุแบบไม่มีใครรู้
        declared = {sc for l in data.get("belt_sprocket_links") or []
                    for sc in l.get("sprocket_codes") or []}
        absent = sorted(sc for sc in declared if _node_id(SPROCKET, sc) not in self.nodes)
        if absent:
            raise AssertionError(
                f"belt_sprocket_links อ้างรหัสสเตอร์ {len(absent)} ตัวที่ไม่มีใน sprockets: "
                f"{absent[:5]}")

    # ── ค้นหา ────────────────────────────────────────────────
    def _resolve(self, key: str) -> Optional[str]:
        """รับได้ทั้ง belt_code, belt_code_id, sprocket_code หรือ node_id"""
        if key in self.nodes:
            return key
        for cand in (self._belt_by_code.get(key), self._belt_by_id.get(key),
                     _node_id(SPROCKET, key), _node_id(BELT, key)):
            if cand in self.nodes:
                return cand
        return None

    def sprockets_for_belt(self, belt: str) -> list[str]:
        """รหัสสเตอร์ที่คู่กับสายพานรุ่นนี้ — [] คือเล่มไม่ได้ผูกไว้ ไม่ใช่ไม่มีสเตอร์"""
        n = self._resolve(belt)
        return [self.nodes[x].label for x in self._out[n][COMPATIBLE_WITH]] if n else []

    def belts_for_sprocket(self, sprocket: str) -> list[str]:
        """ถามกลับ — สเตอร์ตัวนี้ใช้กับสายพานรุ่นไหน (dict เดิมทำไม่ได้ ต้องวนทั้งตาราง)"""
        n = self._resolve(sprocket)
        return [self.nodes[x].label for x in self._out[n][COMPATIBLE_WITH]] if n else []

    def get_compatible_products(self, key: str) -> list[str]:
        n = self._resolve(key)
        return [self.nodes[x].label for x in self._out[n][COMPATIBLE_WITH]] if n else []

    def get_same_series(self, key: str) -> list[str]:
        n = self._resolve(key)
        if not n:
            return []
        out: list[str] = []
        for s in self._out[n][BELONGS_TO]:
            for b, rels in self._out.items():
                if b != n and s in rels.get(BELONGS_TO, ()):
                    out.append(self.nodes[b].label)
        return sorted(set(out))

    def get_material_details(self, key: str) -> dict:
        """คืนชื่อวัสดุกับพอลิเมอร์ฐาน — ค่าอุณหภูมิไม่อยู่ในกราฟ ต้องไปเอาจากเกต

        คีย์ full_name กับ description ใส่ไว้เพราะ GraphResult.to_llm_context() ใน
        retriever_v4 อ่านสองชื่อนี้ ถ้าไม่มีมันจะพิมพ์แค่รหัสวัสดุเปล่า ๆ
        ข้อความมาจาก material_registry ตรง ๆ ไม่ได้แต่งเพิ่ม
        """
        n = self._resolve(key)
        if not n:
            return {}
        out: dict[str, dict] = {}
        for m in self._out[n][MADE_OF]:
            d = self.nodes[m].as_dict()
            code, name, base = d["label"], d.get("name") or "", d.get("base_polymer") or ""
            d["full_name"] = f"{code} ({name})" if name and name != code else code
            if base and base != code:
                d["description"] = f"พอลิเมอร์ฐาน {base}"
            out[code] = d
        return out

    def get_process_details(self, _key: str) -> dict:
        """Modutech ไม่มีข้อมูลกระบวนการผลิตในเล่ม — คืนว่างไว้ ไม่เดา"""
        return {}

    def expand_products(
        self,
        seed_product_ids: Iterable[str] = (),
        include_same_series: bool = True,
        include_compatible: bool = True,
        include_same_material: bool = False,
        **kwargs,
    ) -> GraphContext:
        """รูปแบบเดียวกับที่ RetrieverV4._graph_traverse เรียก (retriever_v4.py บรรทัด 712)

        ชื่อพารามิเตอร์ต้องเป็น seed_product_ids ตรงตัว ไม่ใช่ product_ids
        เพราะ retriever ส่งเป็นคีย์เวิร์ด ถ้าชื่อไม่ตรงจะได้ context ว่างเงียบ ๆ
        **kwargs รับชื่ออื่นไว้กันเหนียว เผื่อ retriever รุ่นถัดไปเปลี่ยนชื่อ

        include_same_material ควบคุมแค่ว่าจะดึง material_details มาหรือไม่
        process_details คืนว่างเสมอ เพราะเล่ม Modutech ไม่มีข้อมูลกระบวนการผลิต
        """
        ids = list(seed_product_ids)
        if not ids:
            for k in ("product_ids", "seed_products", "products", "pids", "belt_codes", "seeds"):
                v = kwargs.get(k)
                if v:
                    ids = list(v)
                    break

        ctx = GraphContext(seed_products=ids)
        for key in ids:
            n = self._resolve(key)
            if not n:
                ctx.hop_trace.append(f"{key}: ไม่พบในกราฟ")
                continue
            comp = self.get_compatible_products(n) if include_compatible else []
            same = self.get_same_series(n) if include_same_series else []
            ctx.compatible_products += [c for c in comp if c not in ctx.compatible_products]
            ctx.same_series += [s for s in same if s not in ctx.same_series]
            if include_same_material:
                ctx.material_details.update(self.get_material_details(n))
            ctx.hop_trace.append(
                f"{self.nodes[n].label}: เข้ากันได้ {len(comp)} · ซีรีส์เดียวกัน {len(same)}")
        return ctx

    def resolve_sprocket_to_belts(self, codes: Iterable[str]) -> list[str]:
        """รหัสสเตอร์ → รหัสสายพานที่ใช้ร่วมกันได้ เรียงตามลำดับที่พบ

        มีไว้เพราะลูกค้าพิมพ์รหัสสเตอร์มาเดี่ยว ๆ ได้ (เช่น "EC127SQZ24*PA ใช้กับรุ่นไหน")
        ซึ่ง router จับได้ว่าเป็นรหัสสเตอร์ แต่ไม่มีทางแปลงกลับเป็นสายพาน
        ตารางใน JSON ไล่จากสายพานไปสเตอร์ทางเดียว การถามกลับต้องวนทั้งตาราง
        """
        out: list[str] = []
        for c in codes:
            for b in self.belts_for_sprocket(c):
                if b not in out:
                    out.append(b)
        return out

    # ── รายงาน ───────────────────────────────────────────────
    def stats(self) -> dict:
        nt: dict[str, int] = defaultdict(int)
        for n in self.nodes.values():
            nt[n.node_type] += 1
        er: dict[str, int] = defaultdict(int)
        for rels in self._out.values():
            for rel, dsts in rels.items():
                er[rel] += len(dsts)
        return {"nodes": dict(nt), "edges": dict(er)}

    def belts_without_sprocket(self) -> list[str]:
        """สายพานที่เล่มไม่ได้ผูกสเตอร์ไว้ — ต้องตอบว่ายังยืนยันไม่ได้ ไม่ใช่เงียบ"""
        return sorted(n.label for nid, n in self.nodes.items()
                      if n.node_type == BELT and not self._out[nid][COMPATIBLE_WITH])

    # ── ส่งออกไว้ดู ──────────────────────────────────────────
    def to_sqlite(self, path: str) -> None:
        """เขียนไฟล์ .db ไว้ตรวจหรือแนบภาคผนวก — ระบบตอนรันไม่ได้อ่านไฟล์นี้"""
        _refuse_if_movex(path, "เขียนไฟล์กราฟ")
        con = sqlite3.connect(path)
        con.executescript("""
            drop table if exists nodes; drop table if exists edges;
            create table nodes(node_id text primary key, node_type text, label text, attrs text);
            create table edges(src text, dst text, relation text);
        """)
        con.executemany("insert into nodes values(?,?,?,?)",
                        [(n.node_id, n.node_type, n.label,
                          json.dumps(dict(n.attrs), ensure_ascii=False))
                         for n in self.nodes.values()])
        con.executemany("insert into edges values(?,?,?)",
                        [(s, d, rel) for s, rels in self._out.items()
                         for rel, ds in rels.items() for d in ds])
        con.commit()
        con.close()

    # ── ทางเข้า ──────────────────────────────────────────────
    @classmethod
    def from_json(cls, path: str) -> "ModutechGraph":
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f))

    @classmethod
    def from_catalogue(cls, cat) -> "ModutechGraph":
        """ใช้ Catalogue ที่โหลดไว้แล้ว ไม่ต้องอ่านไฟล์ซ้ำ — Catalogue ตรวจจำนวนมาแล้วชั้นหนึ่ง"""
        return cls(cat.data)


BLOCKED_NODE_FIELDS = ("Di_mm", "Do_mm", "Bore_mm", "Bore_inch",
                       "dimensions", "bore", "dimension_note_th")


def assert_no_blocked_fields(db_path: str) -> list[str]:
    """ตรวจว่าไฟล์ .db ไม่มีฟิลด์ที่ sprocket_gate บล็อกไว้ คืนรายชื่อฟิลด์ที่เจอ

    ทำไมต้องตรวจที่ไฟล์ ไม่ใช่ที่โค้ด: กราฟเป็นทางที่เลี่ยงเกตได้
    ค่าที่อยู่ในโหนดไม่ผ่าน sprocket_gate และไม่ผ่าน project_for_llm
    ตอนนี้ยังไม่มีใครพิมพ์มันออกมา แต่ "ยังไม่มีใครพิมพ์" กับ "พิมพ์ไม่ได้" ไม่เหมือนกัน
    """
    try:
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    except sqlite3.Error:
        return []
    found: set = set()
    try:
        for (attrs,) in con.execute("select attrs from nodes"):
            try:
                d = json.loads(attrs)
            except (json.JSONDecodeError, TypeError):
                continue
            found |= {k for k in BLOCKED_NODE_FIELDS if d.get(k) not in (None, "", [], {})}
    except sqlite3.Error:
        return []
    finally:
        con.close()
    return sorted(found)


def load_modutech_kg(cat, db_path: str = "./modutech_kg.db", quiet: bool = False):
    """โหลดกราฟจากไฟล์ .db ถ้าใช้ได้ ไม่งั้นสร้างจาก JSON

    ลำดับการตัดสินใจ — ทุกทางจบด้วยกราฟที่ถูกต้อง ไม่มีทางที่เดินต่อด้วยของผิด
      1. ไม่มีไฟล์            → สร้างจาก JSON บอกวิธีสร้างไฟล์
      2. ลายนิ้วมือไม่ตรง      → เตือนดัง ๆ แล้วสร้างจาก JSON (ของในไฟล์ล้าสมัย)
      3. ไม่มีลายนิ้วมือ       → เตือน แล้วใช้ไฟล์ (ไฟล์เก่าที่สร้างก่อนมีการฝัง)
      4. มีฟิลด์ที่เกตบล็อก    → เตือน แล้วใช้ไฟล์ ต้องไปสร้างใหม่ด้วย builder รุ่นใหม่
      5. ตรงหมด               → ใช้ไฟล์

    เลือก "เตือนแล้วสร้างใหม่" ไม่ใช่ "หยุดทำงาน" เพราะถ้าเจอตอนสอบ
    ระบบที่ขึ้นช้ากว่าหนึ่งมิลลิวินาทีดีกว่าระบบที่ไม่ขึ้นเลย
    """
    def say(msg: str) -> None:
        if not quiet:
            print(msg)

    import os

    if not os.path.exists(db_path):
        say(f"ℹ️ ไม่พบ {db_path} — สร้างกราฟจาก JSON (สร้างไฟล์ด้วย python build_modutech_graph.py)")
        return ModutechGraph.from_catalogue(cat)

    want = catalogue_fingerprint(cat.data)
    got = db_fingerprint(db_path)
    if got is None:
        say(f"⚠️ {db_path} ไม่มีลายนิ้วมือ — สร้างก่อนมีการฝัง ตรวจไม่ได้ว่าตรงกับข้อมูลปัจจุบัน\n"
            f"   ควรสร้างใหม่: python build_modutech_graph.py")
    elif got != want:
        say(f"⚠️ {db_path} ล้าสมัย — ความสัมพันธ์ในไฟล์ไม่ตรงกับ JSON ปัจจุบัน\n"
            f"   ไฟล์ {got[:12]} · ข้อมูล {want[:12]}\n"
            f"   ใช้กราฟที่สร้างจาก JSON แทนรอบนี้ แล้วรัน python build_modutech_graph.py")
        return ModutechGraph.from_catalogue(cat)

    blocked = assert_no_blocked_fields(db_path)
    if blocked:
        say(f"⚠️ {db_path} มีฟิลด์ที่ sprocket_gate บล็อกไว้อยู่ในโหนด: {', '.join(blocked)}\n"
            f"   ยังไม่มีทางที่พิมพ์ออกไปถึงลูกค้า แต่กราฟเป็นทางที่เลี่ยงเกตได้\n"
            f"   สร้างใหม่ด้วย builder รุ่นที่ตัดฟิลด์พวกนี้ออก")

    from knowledge_graph import KnowledgeGraph
    kg = KnowledgeGraph.load_from_db(db_path)
    say(f"✅ โหลดกราฟจาก {db_path} · {kg.graph.number_of_nodes()} โหนด "
        f"{kg.graph.number_of_edges()} เส้น")
    return kg


if __name__ == "__main__":
    import time

    path = sys.argv[1] if len(sys.argv) > 1 else "modutech_v30_1.json"
    t = time.perf_counter()
    g = ModutechGraph.from_json(path)
    ms = (time.perf_counter() - t) * 1000

    st = g.stats()
    print(f"สร้างกราฟจาก {path} ใน {ms:.0f} ms")
    print(f"  ปม  {st['nodes']}")
    print(f"  เส้น {st['edges']}")

    demo = "HC508 C"
    print(f"\n{demo}")
    print(f"  สเตอร์ที่คู่กัน {len(g.sprockets_for_belt(demo))} ตัว: "
          f"{', '.join(g.sprockets_for_belt(demo)[:4])} ...")
    print(f"  ซีรีส์เดียวกัน: {', '.join(g.get_same_series(demo))}")
    print(f"  วัสดุ: {', '.join(g.get_material_details(demo))}")

    one = g.sprockets_for_belt(demo)[0]
    print(f"\nถามกลับ {one} ใช้กับ: {', '.join(g.belts_for_sprocket(one))}")

    miss = g.belts_without_sprocket()
    print(f"\nสายพานที่เล่มไม่ได้ผูกสเตอร์ไว้ {len(miss)} รุ่น")
    print("  " + ", ".join(miss))

    ctx = g.expand_products(["HC508 C"])
    print(f"\nexpand_products: สเตอร์ {len(ctx.compatible_products)} · "
          f"ซีรีส์เดียวกัน {len(ctx.same_series)} · trace {ctx.hop_trace}")