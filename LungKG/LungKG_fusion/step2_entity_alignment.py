"""
Step 2: Entity Alignment

Functions:
  2.1 Basic alignment: exact match, substring match
  2.2 LLM-assisted alignment: grouped by entity type, batch semantic matching
  2.3 Generate entity mapping table entity_mapping.csv
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


# ============================== 2.1 Basic Alignment ==============================

def exact_match(medical_kg_entities: set, graphml_entities: set) -> dict:
    """Exact match (after case normalization)"""
    matched = medical_kg_entities & graphml_entities
    mapping = {e: e for e in matched}
    log.info(f"Exact match: {len(mapping)} pairs")
    return mapping


def fuzzy_match(medical_kg_entities: set, graphml_entities: set, existing_map: dict) -> dict:
    """Fuzzy match: match after removing spaces/punctuation"""
    def normalize(s):
        if not isinstance(s, str):
            return ""
        return re.sub(r'[\s\-—_·・()（）\[\]【】]', '', s.lower())

    # Build normalized index (skip non-string values)
    medical_kg_norm = {normalize(e): e for e in medical_kg_entities if isinstance(e, str) and e not in existing_map}
    graphml_norm = {normalize(e): e for e in graphml_entities if isinstance(e, str) and e not in existing_map.values()}

    mapping = {}
    for norm_key, medical_kg_e in medical_kg_norm.items():
        if norm_key in graphml_norm:
            mapping[graphml_norm[norm_key]] = medical_kg_e

    log.info(f"Fuzzy match: {len(mapping)} pairs")
    return mapping


# ============================== 2.2 LLM-Assisted Alignment ==============================

async def llm_entity_alignment(
    medical_kg_entities_by_type: dict,
    graphml_entities_by_type: dict,
    existing_map: dict,
) -> dict:
    """Use LLM to align entities grouped by type"""
    log.info("=" * 60)
    log.info("Step 2.2: LLM-assisted entity alignment")
    log.info("=" * 60)

    cache = C.load_cache("step2_alignment")
    client = get_client()
    sem = asyncio.Semaphore(C.MAX_CONCURRENT)

    all_mapped_graphml = set(existing_map.keys())
    all_mapped_medical_kg = set(v["medical_kg_entity"] if isinstance(v, dict) else v for v in existing_map.values())
    all_mapped = all_mapped_graphml | all_mapped_medical_kg
    mapping = {}

    # Align each entity type separately
    for etype in C.UNIFIED_ENTITY_TYPES:
        medical_kg_ents = [e for e in medical_kg_entities_by_type.get(etype, []) if e not in all_mapped]
        graphml_ents = [e for e in graphml_entities_by_type.get(etype, []) if e not in all_mapped]

        if not medical_kg_ents or not graphml_ents:
            continue

        log.info(f"Type [{etype}]: Medical KG {len(medical_kg_ents)}, GraphML {len(graphml_ents)}")

        # Batch GraphML entities, compare each batch against Medical KG list
        # To reduce API calls, take top 200 frequent entities from Medical KG
        medical_kg_sample = medical_kg_ents[:200]
        graphml_batches = [graphml_ents[i:i+40] for i in range(0, min(len(graphml_ents), 2000), 40)]

        for batch in tqdm(graphml_batches, desc=f"Aligning [{etype}]", leave=False):
            medical_kg_text = ", ".join(medical_kg_sample[:100])
            graphml_text = ", ".join(batch)

            prompt = f"""你是医学知识图谱实体对齐专家。请找出以下两组【{etype}】类型的医学实体中，指代同一概念的等价对。

Medical KG实体列表:
{medical_kg_text}

指南实体列表:
{graphml_text}

匹配规则:
- 完全相同含义的才算等价（如 "COPD" = "慢性阻塞性肺疾病"）
- 不要匹配仅部分相关的实体（如 "肺炎" ≠ "肺炎链球菌"）
- 同义词/缩写/英中对照算等价

请输出JSON数组，每个元素含:
- guideline: 指南实体名
- medical_kg: Medical KG实体名
- confidence: 置信度(0-1)

格式: [{{"guideline": "xxx", "medical_kg": "yyy", "confidence": 0.9}}]
如果没有匹配对，输出空数组 []。只输出JSON。"""

            ck = C.cache_key(prompt)
            if ck in cache:
                result = cache[ck]
            else:
                result = await llm_call(client, prompt, sem)
                cache[ck] = result

            try:
                json_match = re.search(r'\[[\s\S]*\]', result)
                if json_match:
                    pairs = json.loads(json_match.group(0))
                    for pair in pairs:
                        g_ent = pair.get("guideline", "").strip().lower()
                        c_ent = pair.get("medical_kg", "").strip().lower()
                        conf = pair.get("confidence", 0.5)
                        if g_ent and c_ent and conf >= 0.7:
                            if g_ent not in mapping:
                                mapping[g_ent] = {
                                    "medical_kg_entity": c_ent,
                                    "match_type": "llm",
                                    "confidence": conf,
                                }
            except (json.JSONDecodeError, Exception) as e:
                log.warning(f"Failed to parse alignment result: {e}")

    C.save_cache("step2_alignment", cache)
    await client.close()

    log.info(f"LLM alignment: {len(mapping)} pairs")
    return mapping


# ============================== Main Function ==============================

async def run_step2():
    """Run the full Step 2 pipeline"""
    log.info("=" * 70)
    log.info("  Step 2: Entity Alignment - Start")
    log.info("=" * 70)

    # Load Step 1 outputs
    medical_kg_df = pd.read_csv(C.STEP1_MEDICAL_KG_STANDARDIZED)
    graphml_triples = pd.read_csv(C.STEP1_GRAPHML_TRIPLES)
    graphml_entities = pd.read_csv(C.STEP1_GRAPHML_ENTITIES)

    # Collect entity sets (filter NaN)
    medical_kg_all_entities = set(medical_kg_df["head"].dropna().unique()) | set(medical_kg_df["tail"].dropna().unique())
    graphml_all_entities = set(graphml_entities["name"].dropna().unique())

    log.info(f"Medical KG entity count: {len(medical_kg_all_entities)}")
    log.info(f"GraphML entity count: {len(graphml_all_entities)}")

    # 2.1 Basic alignment
    exact_map = exact_match(medical_kg_all_entities, graphml_all_entities)
    fuzzy_map = fuzzy_match(medical_kg_all_entities, graphml_all_entities, exact_map)

    # Merge basic mappings: graphml_entity -> medical_kg_entity
    base_map = {}
    for e in exact_map:
        base_map[e] = {"medical_kg_entity": e, "match_type": "exact", "confidence": 1.0}
    for g_e, c_e in fuzzy_map.items():
        base_map[g_e] = {"medical_kg_entity": c_e, "match_type": "fuzzy", "confidence": 0.95}

    # 2.2 LLM alignment
    # Group entities by type
    medical_kg_by_type = defaultdict(list)
    for _, row in medical_kg_df.iterrows():
        if pd.notna(row["head"]) and pd.notna(row["head_type"]):
            medical_kg_by_type[row["head_type"]].append(str(row["head"]))
        if pd.notna(row["tail_type"]) and pd.notna(row["tail"]):
            medical_kg_by_type[row["tail_type"]].append(str(row["tail"]))
    medical_kg_by_type = {k: list(set(v)) for k, v in medical_kg_by_type.items()}

    graphml_by_type = defaultdict(list)
    for _, row in graphml_entities.iterrows():
        if pd.notna(row["name"]) and pd.notna(row["entity_type_unified"]):
            graphml_by_type[row["entity_type_unified"]].append(str(row["name"]))
    graphml_by_type = {k: list(set(v)) for k, v in graphml_by_type.items()}

    llm_map = await llm_entity_alignment(medical_kg_by_type, graphml_by_type, base_map)

    # Merge all mappings
    all_map = {**base_map, **llm_map}

    # Output mapping table
    mapping_rows = []
    for g_ent, info in all_map.items():
        mapping_rows.append({
            "graphml_entity": g_ent,
            "medical_kg_entity": info["medical_kg_entity"],
            "match_type": info["match_type"],
            "confidence": info["confidence"],
        })

    mapping_df = pd.DataFrame(mapping_rows)
    mapping_df.to_csv(C.STEP2_ENTITY_MAPPING, index=False, encoding="utf-8")

    log.info("=" * 70)
    log.info("  Step 2: Entity Alignment - Complete")
    log.info(f"  Total mapping count: {len(mapping_df)}")
    log.info(f"    Exact match: {len(exact_map)}")
    log.info(f"    Fuzzy match: {len(fuzzy_map)}")
    log.info(f"    LLM match: {len(llm_map)}")
    log.info(f"  Mapping table saved: {C.STEP2_ENTITY_MAPPING}")
    log.info("=" * 70)

    return mapping_df


def main():
    asyncio.run(run_step2())


if __name__ == "__main__":
    main()
