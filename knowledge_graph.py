"""
knowledge_graph.py
==================
GraphRAG Module สำหรับ Movex Product Chatbot v3

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Persistence: SQLite  (movex_kg.db)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DB มี 3 tables:

  TABLE nodes
    node_id   TEXT PRIMARY KEY    e.g. "product:movex_chain_LF880_TAB_K325"
    node_type TEXT                "Product" | "Series" | "Material" | "ManufacturingProcess"
    attrs     TEXT  (JSON)        attributes ทั้งหมดของ node

  TABLE edges
    src       TEXT
    dst       TEXT
    relation  TEXT                "BELONGS_TO" | "MADE_OF" | "COMPATIBLE_WITH" | "USES_PROCESS"
    PRIMARY KEY (src, dst, relation)

  TABLE metadata
    key       TEXT PRIMARY KEY    "built_at" | "node_count" | "schema_version" | ...
    value     TEXT

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Workflow
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Build (manual เท่านั้น):
  python build_graph.py
  → อ่าน JSON → build NetworkX DiGraph → save ลง movex_kg.db

Runtime (ทุก startup ของ chatbot):
  kg = KnowledgeGraph.load_from_db("./movex_kg.db")
  → SELECT nodes + edges → rebuild NetworkX DiGraph ใน memory
  → เร็วกว่า build จาก JSON ประมาณ 10x

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Graph Structure
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Nodes:
  product:{product_id}   Product nodes
  series:{series_num}    Series nodes (820 | 821 | 880 | 882 | 83 | 103)
  material:{code}        Material nodes  (LF | SS | LFN | ...)
  process:{name}         ManufacturingProcess nodes (Machined | Molded)

Edges:
  BELONGS_TO        Product → Series
  MADE_OF           Product → Material
  USES_PROCESS      Product → ManufacturingProcess
  COMPATIBLE_WITH   Product ↔ Product  (bidirectional, chain ↔ sprocket)
"""

import re
import json
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import networkx as nx


# เพิ่มตัวเลขนี้ทุกครั้งที่เปลี่ยน FIELDS_BY_TYPE หรือ edge logic
# → build_graph.py จะบันทึกไว้ใน metadata table
SCHEMA_VERSION = "1"

FIELDS_BY_TYPE: dict[str, list[str]] = {
    "Product": [
        "Ref", "Art_Nr", "series", "product_type",
        "usage", "key_features", "constraints",
        "Plate_Width_mm", "Max_Working_Load_N", "Weight_kg_m",
        "Min_curve_radius_mm", "Radius_min_mm", "Pitch_mm", "Links_m",
        "Z_Teeth", "Bore_mm", "PD_mm", "OD_mm", "S_mm",
        "compatible_chain", "Sprocket_Type", "manufacturing_process",
        "Material", "Pivot_material",
    ],
    "Series": ["name"],
    "Material": ["full_name", "description", "best_used_for"],
    "ManufacturingProcess": ["full_name", "description", "best_used_for"],
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GraphContext dataclass
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class GraphContext:
    """Context ที่ graph expand ออกมา สำหรับส่งต่อให้ retriever / LLM"""
    seed_products: list[str]
    same_series: list[str] = field(default_factory=list)
    compatible_products: list[str] = field(default_factory=list)
    same_material: list[str] = field(default_factory=list)
    material_details: dict = field(default_factory=dict)
    process_details: dict = field(default_factory=dict)
    all_candidate_ids: list[str] = field(default_factory=list)

    def to_summary_str(self) -> str:
        return (
            f"seed={self.seed_products} | "
            f"same_series({len(self.same_series)}) | "
            f"compatible({len(self.compatible_products)}) | "
            f"same_material({len(self.same_material)})"
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# KnowledgeGraph  — Runtime class
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class KnowledgeGraph:
    """
    NetworkX Knowledge Graph สำหรับ Movex product catalog

    ไม่ instantiate โดยตรง — ใช้ classmethod:
      KnowledgeGraph.load_from_db(db_path)     ← chatbot runtime
      KnowledgeGraph.build_and_save(...)        ← build_graph.py
    """

    SERIES_RE = re.compile(r'\b(820|821|880|882|83|103)\b')

    def __init__(self, graph: nx.DiGraph, db_path: str = ""):
        self.graph   = graph
        self.db_path = db_path

    # ──────────────────────────────────────────────────────
    # ENTRY POINTS
    # ──────────────────────────────────────────────────────

    @classmethod
    def load_from_db(cls, db_path: str) -> "KnowledgeGraph":
        """
        โหลด graph จาก SQLite database

        Parameters
        ----------
        db_path : str
            path ไปยัง movex_kg.db

        Raises
        ------
        FileNotFoundError  ถ้า db_path ไม่มี
        RuntimeError       ถ้า DB ว่างเปล่า (ยังไม่ได้ build)
        """
        import os
        if not os.path.isfile(db_path):
            raise FileNotFoundError(
                f"ไม่พบ {db_path}\n"
                f"กรุณารัน:  python build_graph.py  ก่อนครับ"
            )

        t0  = time.perf_counter()
        con = sqlite3.connect(db_path)
        con.row_factory = sqlite3.Row

        # โหลด nodes
        g = nx.DiGraph()
        node_rows = con.execute(
            "SELECT node_id, node_type, attrs FROM nodes"
        ).fetchall()

        if not node_rows:
            con.close()
            raise RuntimeError(
                f"DB ว่างเปล่า: {db_path}\n"
                f"กรุณารัน:  python build_graph.py  ก่อนครับ"
            )

        for row in node_rows:
            attrs = json.loads(row["attrs"])
            attrs["node_type"] = row["node_type"]
            g.add_node(row["node_id"], **attrs)

        # โหลด edges
        for row in con.execute("SELECT src, dst, relation FROM edges"):
            g.add_edge(row["src"], row["dst"], relation=row["relation"])

        # metadata สำหรับแสดง info
        meta = dict(con.execute("SELECT key, value FROM metadata").fetchall())
        con.close()

        elapsed = (time.perf_counter() - t0) * 1000
        print(
            f"⚡ KnowledgeGraph loaded from SQLite in {elapsed:.0f}ms\n"
            f"   📁 {db_path}\n"
            f"   nodes={g.number_of_nodes()}  "
            f"edges={g.number_of_edges()}  "
            f"built={meta.get('built_at', 'unknown')}"
        )
        return cls(graph=g, db_path=db_path)

    @classmethod
    def build_and_save(
        cls,
        product_db: dict,
        material_db: dict,
        process_db: dict,
        db_path: str,
    ) -> "KnowledgeGraph":
        """
        Build graph จาก JSON dicts แล้ว save ลง SQLite

        ใช้โดย build_graph.py เท่านั้น
        """
        builder  = _GraphBuilder(product_db, material_db, process_db)
        graph    = builder.build()
        instance = cls(graph=graph, db_path=db_path)
        instance._save_to_db(db_path)
        return instance

    # ──────────────────────────────────────────────────────
    # SQLITE SAVE
    # ──────────────────────────────────────────────────────

    def _save_to_db(self, db_path: str):
        """บันทึก NetworkX graph ลง SQLite (เรียกใน build_and_save)"""
        import os
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)

        t0  = time.perf_counter()
        con = sqlite3.connect(db_path)
        cur = con.cursor()

        # สร้าง / ล้าง tables
        cur.executescript("""
            DROP TABLE IF EXISTS nodes;
            DROP TABLE IF EXISTS edges;
            DROP TABLE IF EXISTS metadata;

            CREATE TABLE nodes (
                node_id   TEXT PRIMARY KEY,
                node_type TEXT NOT NULL,
                attrs     TEXT NOT NULL
            );

            CREATE TABLE edges (
                src      TEXT NOT NULL,
                dst      TEXT NOT NULL,
                relation TEXT NOT NULL,
                PRIMARY KEY (src, dst, relation)
            );

            CREATE INDEX IF NOT EXISTS idx_edges_src  ON edges(src);
            CREATE INDEX IF NOT EXISTS idx_edges_dst  ON edges(dst);
            CREATE INDEX IF NOT EXISTS idx_nodes_type ON nodes(node_type);

            CREATE TABLE metadata (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
        """)

        # Insert nodes
        cur.executemany(
            "INSERT INTO nodes (node_id, node_type, attrs) VALUES (?,?,?)",
            [
                (
                    nid,
                    data.get("node_type", "Unknown"),
                    json.dumps(
                        {k: v for k, v in data.items() if k != "node_type"},
                        ensure_ascii=False,
                    ),
                )
                for nid, data in self.graph.nodes(data=True)
            ],
        )

        # Insert edges
        cur.executemany(
            "INSERT OR IGNORE INTO edges (src, dst, relation) VALUES (?,?,?)",
            [
                (src, dst, data.get("relation", ""))
                for src, dst, data in self.graph.edges(data=True)
            ],
        )

        # นับ stats สำหรับ metadata
        type_counts: dict[str, int] = {}
        for _, data in self.graph.nodes(data=True):
            t = data.get("node_type", "Unknown")
            type_counts[t] = type_counts.get(t, 0) + 1

        rel_counts: dict[str, int] = {}
        for _, _, data in self.graph.edges(data=True):
            r = data.get("relation", "")
            rel_counts[r] = rel_counts.get(r, 0) + 1

        # Insert metadata
        cur.executemany(
            "INSERT OR REPLACE INTO metadata (key, value) VALUES (?,?)",
            [
                ("built_at",       datetime.now().isoformat(timespec="seconds")),
                ("schema_version", SCHEMA_VERSION),
                ("node_count",     str(self.graph.number_of_nodes())),
                ("edge_count",     str(self.graph.number_of_edges())),
                ("node_types",     json.dumps(type_counts)),
                ("edge_relations", json.dumps(rel_counts)),
            ],
        )

        con.commit()
        con.close()

        elapsed = (time.perf_counter() - t0) * 1000
        print(
            f"💾 Saved to SQLite in {elapsed:.0f}ms → {db_path}\n"
            f"   node_types  : {type_counts}\n"
            f"   edge_relations: {rel_counts}"
        )

    # ──────────────────────────────────────────────────────
    # DB INFO
    # ──────────────────────────────────────────────────────

    def db_info(self) -> dict:
        """อ่าน metadata จาก DB (built_at, schema_version, counts, ...)"""
        if not self.db_path:
            return {"status": "no_db_path"}
        try:
            con  = sqlite3.connect(self.db_path)
            info = dict(con.execute("SELECT key, value FROM metadata").fetchall())
            con.close()
            info["status"] = "ok"
            for k in ("node_types", "edge_relations"):
                if k in info:
                    info[k] = json.loads(info[k])
            return info
        except Exception as e:
            return {"status": "error", "error": str(e)}

    # ──────────────────────────────────────────────────────
    # GRAPH QUERY API  (ไม่เปลี่ยนจาก v เดิม)
    # ──────────────────────────────────────────────────────

    def expand_products(
        self,
        seed_product_ids: list[str],
        include_same_series: bool = True,
        include_compatible: bool = True,
        include_same_material: bool = False,
    ) -> GraphContext:
        """Main entry สำหรับ retriever — expand seed products ผ่าน graph"""
        ctx = GraphContext(seed_products=list(seed_product_ids))
        visited_series:   set[str] = set()
        visited_material: set[str] = set()

        for pid in seed_product_ids:
            pnode = f"product:{pid}"
            if not self.graph.has_node(pnode):
                continue

            for nb in self.graph.successors(pnode):
                edata    = self.graph.edges[pnode, nb]
                relation = edata.get("relation", "")
                ndata    = self.graph.nodes[nb]
                ntype    = ndata.get("node_type", "")

                if relation == "BELONGS_TO" and ntype == "Series":
                    if include_same_series and nb not in visited_series:
                        visited_series.add(nb)
                        for pred in self.graph.predecessors(nb):
                            if self.graph.edges.get((pred, nb), {}).get("relation") == "BELONGS_TO":
                                pp = self.graph.nodes[pred].get("product_id", "")
                                if pp and pp not in seed_product_ids and pp not in ctx.same_series:
                                    ctx.same_series.append(pp)

                elif relation == "COMPATIBLE_WITH" and ntype == "Product":
                    if include_compatible:
                        np_ = ndata.get("product_id", "")
                        if np_ and np_ not in seed_product_ids and np_ not in ctx.compatible_products:
                            ctx.compatible_products.append(np_)

                elif relation == "MADE_OF" and ntype == "Material":
                    mat = ndata.get("material_code", "")
                    if mat and mat not in ctx.material_details:
                        ctx.material_details[mat] = self._node_fields(nb, "Material")
                    if include_same_material and nb not in visited_material:
                        visited_material.add(nb)
                        for pred in self.graph.predecessors(nb):
                            if self.graph.edges.get((pred, nb), {}).get("relation") == "MADE_OF":
                                pp = self.graph.nodes[pred].get("product_id", "")
                                if pp and pp not in seed_product_ids and pp not in ctx.same_material:
                                    ctx.same_material.append(pp)

                elif relation == "USES_PROCESS" and ntype == "ManufacturingProcess":
                    proc = ndata.get("process_name", "")
                    if proc and proc not in ctx.process_details:
                        ctx.process_details[proc] = self._node_fields(nb, "ManufacturingProcess")

        seen: set[str] = set(seed_product_ids)
        ctx.all_candidate_ids = list(seed_product_ids)
        for p in ctx.compatible_products + ctx.same_series + ctx.same_material:
            if p not in seen:
                seen.add(p)
                ctx.all_candidate_ids.append(p)
        return ctx

    def get_compatible_products(self, product_id: str) -> list[str]:
        pnode = f"product:{product_id}"
        if not self.graph.has_node(pnode):
            return []
        return [
            self.graph.nodes[nb].get("product_id", "")
            for nb in self.graph.successors(pnode)
            if self.graph.edges[pnode, nb].get("relation") == "COMPATIBLE_WITH"
            and self.graph.nodes[nb].get("product_id")
        ]

    def get_same_series(self, product_id: str) -> list[str]:
        pnode = f"product:{product_id}"
        if not self.graph.has_node(pnode):
            return []
        result: list[str] = []
        for nb in self.graph.successors(pnode):
            if self.graph.edges[pnode, nb].get("relation") == "BELONGS_TO":
                for pred in self.graph.predecessors(nb):
                    if self.graph.edges.get((pred, nb), {}).get("relation") == "BELONGS_TO":
                        pid = self.graph.nodes[pred].get("product_id", "")
                        if pid and pid != product_id and pid not in result:
                            result.append(pid)
        return result

    def get_series_num(self, product_id: str) -> Optional[str]:
        pnode = f"product:{product_id}"
        if not self.graph.has_node(pnode):
            return None
        for nb in self.graph.successors(pnode):
            if self.graph.edges[pnode, nb].get("relation") == "BELONGS_TO":
                return self.graph.nodes[nb].get("series_num")
        return None

    def get_material_details(self, product_id: str) -> Optional[dict]:
        pnode = f"product:{product_id}"
        if not self.graph.has_node(pnode):
            return None
        for nb in self.graph.successors(pnode):
            if self.graph.edges[pnode, nb].get("relation") == "MADE_OF":
                return self._node_fields(nb, "Material")
        return None

    def get_process_details(self, product_id: str) -> Optional[dict]:
        pnode = f"product:{product_id}"
        if not self.graph.has_node(pnode):
            return None
        for nb in self.graph.successors(pnode):
            if self.graph.edges[pnode, nb].get("relation") == "USES_PROCESS":
                return self._node_fields(nb, "ManufacturingProcess")
        return None

    def get_products_by_series(self, series_num: str) -> list[str]:
        snode = f"series:{series_num}"
        if not self.graph.has_node(snode):
            return []
        return [
            self.graph.nodes[pred].get("product_id", "")
            for pred in self.graph.predecessors(snode)
            if self.graph.edges.get((pred, snode), {}).get("relation") == "BELONGS_TO"
            and self.graph.nodes[pred].get("product_id")
        ]

    def get_products_by_material(self, material_code: str) -> list[str]:
        mnode = f"material:{material_code}"
        if not self.graph.has_node(mnode):
            return []
        return [
            self.graph.nodes[pred].get("product_id", "")
            for pred in self.graph.predecessors(mnode)
            if self.graph.edges.get((pred, mnode), {}).get("relation") == "MADE_OF"
            and self.graph.nodes[pred].get("product_id")
        ]

    def node_count(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for _, data in self.graph.nodes(data=True):
            t = data.get("node_type", "Unknown")
            counts[t] = counts.get(t, 0) + 1
        return counts

    def _extract_series_num(self, text: str) -> Optional[str]:
        m = self.SERIES_RE.search(text or "")
        return m.group(1) if m else None

    def _node_fields(self, node_id: str, node_type: str) -> dict:
        data = self.graph.nodes.get(node_id, {})
        return {k: data[k] for k in FIELDS_BY_TYPE.get(node_type, []) if k in data}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# _GraphBuilder  (internal — ใช้โดย build_and_save เท่านั้น)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class _GraphBuilder:
    """Build NetworkX DiGraph จาก product/material/process dicts"""

    SERIES_RE = re.compile(r'\b(820|821|880|882|83|103)\b')

    def __init__(self, product_db: dict, material_db: dict, process_db: dict):
        self.product_db  = product_db
        self.material_db = material_db
        self.process_db  = process_db

    def build(self) -> nx.DiGraph:
        t0 = time.perf_counter()
        print("⏳ Building Knowledge Graph from JSON...")
        g = nx.DiGraph()
        self._add_product_nodes(g)
        self._add_attribute_nodes_and_edges(g)
        compat_count = self._add_compatibility_edges(g)

        type_counts: dict[str, int] = {}
        for _, d in g.nodes(data=True):
            t = d.get("node_type", "?")
            type_counts[t] = type_counts.get(t, 0) + 1

        elapsed = (time.perf_counter() - t0) * 1000
        print(
            f"✅ Built in {elapsed:.0f}ms: "
            f"{g.number_of_nodes()} nodes {type_counts} | "
            f"{g.number_of_edges()} edges | "
            f"COMPATIBLE_WITH={compat_count}"
        )
        return g

    def _add_product_nodes(self, g: nx.DiGraph):
        for pid, p in self.product_db.items():
            attrs = {"node_type": "Product", "product_id": pid}
            for k in FIELDS_BY_TYPE["Product"]:
                v = p.get(k)
                if v is not None:
                    attrs[k] = v
            g.add_node(f"product:{pid}", **attrs)

    def _add_attribute_nodes_and_edges(self, g: nx.DiGraph):
        for pid, p in self.product_db.items():
            pnode = f"product:{pid}"

            # Series
            series_raw = str(p.get("series", "")).strip()
            snum = self._snum(series_raw) or self._snum(pid)
            if snum:
                snode = f"series:{snum}"
                if not g.has_node(snode):
                    g.add_node(snode, node_type="Series",
                               name=series_raw or f"{snum} Series",
                               series_num=snum)
                g.add_edge(pnode, snode, relation="BELONGS_TO")

            # Material
            mat = str(p.get("Material", "")).strip()
            if mat and mat.lower() not in ("nan", "none", ""):
                mnode = f"material:{mat}"
                if not g.has_node(mnode):
                    mi = self.material_db.get(mat, {})
                    attrs = {"node_type": "Material", "material_code": mat}
                    for k in FIELDS_BY_TYPE["Material"]:
                        if mi.get(k) is not None:
                            attrs[k] = mi[k]
                    g.add_node(mnode, **attrs)
                g.add_edge(pnode, mnode, relation="MADE_OF")

            # ManufacturingProcess
            proc = str(p.get("manufacturing_process", "")).strip()
            if proc and proc.lower() not in ("nan", "none", ""):
                prnode = f"process:{proc}"
                if not g.has_node(prnode):
                    pi = self.process_db.get(proc, {})
                    attrs = {"node_type": "ManufacturingProcess", "process_name": proc}
                    for k in FIELDS_BY_TYPE["ManufacturingProcess"]:
                        if pi.get(k) is not None:
                            attrs[k] = pi[k]
                    g.add_node(prnode, **attrs)
                g.add_edge(pnode, prnode, relation="USES_PROCESS")

    def _add_compatibility_edges(self, g: nx.DiGraph) -> int:
        """COMPATIBLE_WITH (chain ↔ sprocket, bidirectional)"""
        chains_by_series: dict[str, list[str]] = {}
        for pid, p in self.product_db.items():
            ptype = str(p.get("product_type", "")).lower()
            if "sprocket" not in ptype and "sprocket" not in pid.lower():
                snum = self._snum(str(p.get("series", ""))) or self._snum(pid)
                if snum:
                    chains_by_series.setdefault(snum, []).append(pid)

        count = 0
        for pid, p in self.product_db.items():
            ptype = str(p.get("product_type", "")).lower()
            if "sprocket" not in ptype and "sprocket" not in pid.lower():
                continue
            spnode = f"product:{pid}"

            compat: set[str] = set()
            for s in self.SERIES_RE.findall(str(p.get("compatible_chain", ""))):
                compat.add(s)
            own = self._snum(str(p.get("series", ""))) or self._snum(pid)
            if own:
                compat.add(own)

            for cs in compat:
                for chain_pid in chains_by_series.get(cs, []):
                    cnode = f"product:{chain_pid}"
                    if not g.has_edge(spnode, cnode):
                        g.add_edge(spnode, cnode, relation="COMPATIBLE_WITH")
                        count += 1
                    if not g.has_edge(cnode, spnode):
                        g.add_edge(cnode, spnode, relation="COMPATIBLE_WITH")
                        count += 1
        return count

    def _snum(self, text: str) -> Optional[str]:
        m = self.SERIES_RE.search(text or "")
        return m.group(1) if m else None