"""
Step 3: Deduplication

Functions:
  3.1 Unify entity names based on entity mapping table
  3.2 Merge triples from two sources, exact + semantic deduplication
  3.3 Conflict detection and marking
"""

import asyncio
import json
import logging
import re
from collections import defaultdict

import pandas as pd
from openai import AsyncOpenAI
from tqdm import tqdm

import config as C

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def get_client() -> AsyncOpenAI:
    return AsyncOpenAI(api_key=C.OPENAI_API_KEY, base_url=C.OPENAI_BASE_URL)


async def llm_call(client: AsyncOpenAI, prompt: str, sem: asyncio.Semaphore) -> str:
    async with sem:
        for attempt in range(C.MAX_RETRIES):
            try:
                await asyncio.sleep(C.REQUEST_DELAY)
                resp = await client.chat.completions.create(
                    model=C.MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=4000,
                )
                return resp.choices[0].message.content.strip()
            except Exception as e:
                log.warning(f"LLM call failed (attempt {attempt+1}): {e}")
                if attempt < C.MAX_RETRIES - 1:
                    await asyncio.sleep(2 * (attempt + 1))
        return ""


# ============================== 3.1 Entity Name Unification ==============================

def unify_entity_names(
    medical_kg_df: pd.DataFrame,
    graphml_df: pd.DataFrame,
    mapping_df: pd.DataFrame,
    graphml_entities: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Unify entity names in GraphML based on entity mapping table"""
    log.info("=" * 60)
    log.info("Step 3.1: Unify entity names")
    log.info("=" * 60)

    # Build graphml -> medical_kg mapping dictionary
    name_map = dict(zip(mapping_df["graphml_entity"], mapping_df["medical_kg_entity"]))
    log.info(f"Entity mapping count: {len(name_map)}")

    # Replace head/tail in GraphML triples
    before_heads = graphml_df["head"].nunique()
    before_tails = graphml_df["tail"].nunique()

    graphml_df["head"] = graphml_df["head"].map(lambda x: name_map.get(x, x))
    graphml_df["tail"] = graphml_df["tail"].map(lambda x: name_map.get(x, x))

    after_heads = graphml_df["head"].nunique()
    after_tails = graphml_df["tail"].nunique()

    log.info(f"GraphML head entities: {before_heads} -> {after_heads}")
    log.info(f"GraphML tail entities: {before_tails} -> {after_tails}")

    # Merge entity descriptions
    # For mapped entities, merge GraphML description into the unified entity
    entity_descriptions = {}
    for _, row in graphml_entities.iterrows():
        name = row["name"]
        unified_name = name_map.get(name, name)
        desc = row.get("description", "")
        if unified_name not in entity_descriptions:
            entity_descriptions[unified_name] = {
                "name": unified_name,
                "type": row["entity_type_unified"],
                "descriptions": [],
                "source": set(),
            }
        if desc and isinstance(desc, str) and len(desc) > 5:
            entity_descriptions[unified_name]["descriptions"].append(desc[:300])
        entity_descriptions[unified_name]["source"].add("guideline")

    # Add Medical KG entities (no description, but have type)
    for _, row in medical_kg_df.iterrows():
        for col, type_col in [("head", "head_type"), ("tail", "tail_type")]:
            name = row[col]
            if name not in entity_descriptions:
                entity_descriptions[name] = {
                    "name": name,
                    "type": row[type_col] if pd.notna(row[type_col]) else "其他治疗",
                    "descriptions": [],
                    "source": set(),
                }
            entity_descriptions[name]["source"].add("medical_kg")

    # Convert to DataFrame
    entities_rows = []
    for name, info in entity_descriptions.items():
        desc = " | ".join(info["descriptions"][:3]) if info["descriptions"] else ""
        sources = ",".join(sorted(info["source"]))
        entities_rows.append({
            "name": name,
            "type": info["type"],
            "description": desc[:800],
            "source": sources,
        })

    merged_entities = pd.DataFrame(entities_rows)
    log.info(f"Total entities after merge: {len(merged_entities)}")

    return graphml_df, merged_entities


# ============================== 3.2 Triple Deduplication ==============================

async def deduplicate_triples(
    medical_kg_df: pd.DataFrame,
    graphml_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge triples from two sources and deduplicate"""
    log.info("=" * 60)
    log.info("Step 3.2: Triple deduplication")
    log.info("=" * 60)

    # Standardize columns
    medical_kg_cols = medical_kg_df[["head", "head_type", "relation", "tail", "tail_type", "source"]].copy()
    medical_kg_cols["description"] = ""
    medical_kg_cols["weight"] = 1.0
    medical_kg_cols["confidence"] = 1.0

    graphml_cols = graphml_df[["head", "head_type", "relation", "tail", "tail_type", "source"]].copy()
    graphml_cols["description"] = graphml_df.get("description", "")
    graphml_cols["weight"] = graphml_df.get("weight", 1.0)
    graphml_cols["confidence"] = 1.0

    # Merge
    merged = pd.concat([medical_kg_cols, graphml_cols], ignore_index=True)
    before_count = len(merged)
    log.info(f"Total triples before merge: {before_count} (Medical KG: {len(medical_kg_cols)}, GraphML: {len(graphml_cols)})")

    # Exact deduplication: (head, relation, tail) are completely identical
    merged["triple_key"] = merged["head"] + "||" + merged["relation"] + "||" + merged["tail"]

    # For duplicate triples, keep the one with richer source info (prefer both)
    def merge_sources(group):
        sources = set()
        for s in group["source"]:
            sources.update(s.split(","))
        row = group.iloc[0].copy()
        row["source"] = ",".join(sorted(sources))
        if "medical_kg" in sources and "guideline" in sources:
            row["confidence"] = 1.0  # Both sources present, highest confidence
        # Keep non-empty description
        for _, r in group.iterrows():
            if r.get("description") and isinstance(r["description"], str) and len(str(r["description"])) > 5:
                row["description"] = r["description"]
                break
        return row

    # Group and deduplicate
    grouped = merged.groupby("triple_key", sort=False)
    deduped_rows = []
    dup_count = 0

    for key, group in tqdm(grouped, desc="Exact dedup", total=len(grouped)):
        if len(group) > 1:
            dup_count += len(group) - 1
        deduped_rows.append(merge_sources(group))

    deduped = pd.DataFrame(deduped_rows)
    deduped = deduped.drop(columns=["triple_key"])

    log.info(f"Exact dedup: removed {dup_count} duplicates, {len(deduped)} remaining")

    # Semantic dedup: for same (head, tail) pair with multiple relations, use LLM to check redundancy
    ht_groups = deduped.groupby(["head", "tail"]).filter(lambda x: len(x) > 1)

    if len(ht_groups) > 0:
        log.info(f"Detected {len(ht_groups)} triples needing semantic dedup")

        cache = C.load_cache("step3_semantic_dedup")
        client = get_client()
        sem = asyncio.Semaphore(C.MAX_CONCURRENT)

        to_remove = set()
        ht_multi = deduped.groupby(["head", "tail"]).filter(lambda x: len(x) > 1)
        ht_multi_groups = ht_multi.groupby(["head", "tail"])

        # Only process first 2000 groups (control API call volume)
        groups_list = list(ht_multi_groups)[:2000]
        batches = [groups_list[i:i+20] for i in range(0, len(groups_list), 20)]

        for batch in tqdm(batches, desc="Semantic dedup"):
            items_text = ""
            batch_indices = []
            for (h, t), group in batch:
                rels = group["relation"].tolist()
                items_text += f"\n头实体: {h}, 尾实体: {t}\n关系列表: {', '.join(rels)}\n"
                batch_indices.append(((h, t), group.index.tolist(), rels))

            prompt = f"""你是医学知识图谱专家。以下是同一对头尾实体之间的多条关系，请判断哪些关系是冗余的（表达相同含义）。

{items_text}

对于每组头尾实体对，如果存在冗余关系，请指出应该保留哪个关系，删除哪些。
输出JSON数组，每个元素含:
- head: 头实体
- tail: 尾实体 
- keep: 应保留的关系名
- remove: 应删除的关系名列表

如果没有冗余，该组不用输出。格式: [{{"head":"x","tail":"y","keep":"r1","remove":["r2"]}}]
如果全部无冗余输出 []。只输出JSON。"""

            ck = C.cache_key(prompt)
            if ck in cache:
                result = cache[ck]
            else:
                result = await llm_call(client, prompt, sem)
                cache[ck] = result

            try:
                json_match = re.search(r'\[[\s\S]*\]', result)
                if json_match:
                    dedup_items = json.loads(json_match.group(0))
                    for item in dedup_items:
                        h = item.get("head", "").lower()
                        t = item.get("tail", "").lower()
                        remove_rels = item.get("remove", [])
                        for (bh, bt), indices, rels in batch_indices:
                            if bh == h and bt == t:
                                for idx, rel in zip(indices, rels):
                                    if rel in remove_rels:
                                        to_remove.add(idx)
            except (json.JSONDecodeError, Exception) as e:
                log.warning(f"Failed to parse semantic dedup result: {e}")

        C.save_cache("step3_semantic_dedup", cache)
        await client.close()

        if to_remove:
            deduped = deduped.drop(index=list(to_remove), errors="ignore")
            log.info(f"Semantic dedup: removed {len(to_remove)}, {len(deduped)} remaining")
    else:
        log.info("No semantic dedup needed")

    log.info(f"Dedup complete: {before_count} -> {len(deduped)} (dedup rate: {(1-len(deduped)/before_count)*100:.1f}%)")

    return deduped


# ============================== Main Function ==============================

async def run_step3():
    """Run the full Step 3 pipeline"""
    log.info("=" * 70)
    log.info("  Step 3: Deduplication - Start")
    log.info("=" * 70)

    # Load data
    medical_kg_df = pd.read_csv(C.STEP1_MEDICAL_KG_STANDARDIZED)
    graphml_df = pd.read_csv(C.STEP1_GRAPHML_TRIPLES)
    graphml_entities = pd.read_csv(C.STEP1_GRAPHML_ENTITIES)
    mapping_df = pd.read_csv(C.STEP2_ENTITY_MAPPING)

    # 3.1 Unify entity names
    graphml_df, merged_entities = unify_entity_names(
        medical_kg_df, graphml_df, mapping_df, graphml_entities
    )

    # 3.2 Triple deduplication
    merged_triples = await deduplicate_triples(medical_kg_df, graphml_df)

    # Save results
    merged_triples.to_csv(C.STEP3_MERGED_TRIPLES, index=False, encoding="utf-8")
    merged_entities.to_csv(C.STEP3_MERGED_ENTITIES, index=False, encoding="utf-8")

    log.info("=" * 70)
    log.info("  Step 3: Deduplication - Complete")
    log.info(f"  Merged triples: {len(merged_triples)}")
    log.info(f"  Merged entities: {len(merged_entities)}")
    log.info(f"  Saved to: {C.STEP3_MERGED_TRIPLES}")
    log.info("=" * 70)

    return merged_triples, merged_entities


def main():
    asyncio.run(run_step3())


if __name__ == "__main__":
    main()
