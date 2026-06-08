"""
Step 6: Export Neo4j Import Format
"""

import hashlib
import logging
import re

import pandas as pd

import config as C

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def entity_id(name: str) -> str:
    return "e_" + hashlib.md5(name.encode("utf-8")).hexdigest()[:12]


def clean_csv_value(val):
    if pd.isna(val):
        return ""
    val = str(val).replace("\n", " ").replace("\r", " ")
    if '"' in val:
        val = val.replace('"', '""')
    return val


def run_step6():
    log.info("=" * 70)
    log.info("  Step 6: Export Neo4j Import Format - Start")
    log.info("=" * 70)

    completed_triples = pd.read_csv(C.STEP4_COMPLETED_TRIPLES)
    merged_entities = pd.read_csv(C.STEP3_MERGED_ENTITIES)

    # 6.1 Nodes file
    log.info("Step 6.1: Generate nodes.csv")
    triple_entities = {}
    for _, row in completed_triples.iterrows():
        h, t = row["head"], row["tail"]
        if h not in triple_entities:
            triple_entities[h] = {"type": row.get("head_type", "其他治疗"), "source": set()}
        if t not in triple_entities:
            triple_entities[t] = {"type": row.get("tail_type", "其他治疗"), "source": set()}
        for s in str(row.get("source", "")).split(","):
            s = s.strip()
            if s:
                triple_entities[h]["source"].add(s)
                triple_entities[t]["source"].add(s)

    entity_info = {}
    for _, row in merged_entities.iterrows():
        entity_info[row["name"]] = {
            "description": str(row.get("description", "")),
            "source": str(row.get("source", "")),
        }

    nodes_rows = []
    for name, info in triple_entities.items():
        eid = entity_id(name)
        etype = info["type"] if pd.notna(info.get("type")) else "其他治疗"
        if etype not in C.UNIFIED_ENTITY_TYPES:
            etype = C.MEDICAL_KG_TYPE_MAP.get(etype, "其他治疗")
        desc, source = "", ",".join(sorted(info["source"])) if info["source"] else ""
        if name in entity_info:
            desc = entity_info[name]["description"]
            extra = entity_info[name]["source"]
            if extra:
                all_s = info["source"] | set(extra.split(","))
                source = ",".join(sorted(s.strip() for s in all_s if s.strip()))
        nodes_rows.append({
            ":ID": eid, "name": clean_csv_value(name), "type": etype,
            ":LABEL": etype, "description": clean_csv_value(str(desc)[:500]),
            "source": source,
        })

    nodes_df = pd.DataFrame(nodes_rows)
    nodes_df.to_csv(C.STEP6_NODES_CSV, index=False, encoding="utf-8")
    log.info(f"Nodes: {len(nodes_df)} -> {C.STEP6_NODES_CSV}")

    # 6.2 Edges file
    log.info("Step 6.2: Generate edges.csv")
    edges_rows = []
    for _, row in completed_triples.iterrows():
        rel_type = str(row["relation"]).strip()
        rel_neo4j = re.sub(r'[^a-zA-Z0-9\u4e00-\u9fff_]', '_', rel_type)
        edges_rows.append({
            ":START_ID": entity_id(row["head"]),
            ":END_ID": entity_id(row["tail"]),
            ":TYPE": rel_neo4j,
            "relation_name": rel_type,
            "description": clean_csv_value(row.get("description", ""))[:300],
            "weight:float": row.get("weight", 1.0) if pd.notna(row.get("weight")) else 1.0,
            "source": clean_csv_value(row.get("source", "")),
            "confidence:float": row.get("confidence", 1.0) if pd.notna(row.get("confidence")) else 1.0,
        })

    edges_df = pd.DataFrame(edges_rows)
    edges_df.to_csv(C.STEP6_EDGES_CSV, index=False, encoding="utf-8")
    log.info(f"Edges: {len(edges_df)} -> {C.STEP6_EDGES_CSV}")

    log.info("=" * 70)
    log.info("  Step 6 Complete")
    log.info(f"  Nodes: {len(nodes_df)}, Edges: {len(edges_df)}")
    log.info(f"  Node types: {nodes_df[':LABEL'].nunique()}, Edge types: {edges_df[':TYPE'].nunique()}")
    log.info(f"  Neo4j import: neo4j-admin database import full --nodes={C.STEP6_NODES_CSV} --relationships={C.STEP6_EDGES_CSV} lungkg")
    log.info("=" * 70)
    return nodes_df, edges_df


def main():
    run_step6()

if __name__ == "__main__":
    main()
