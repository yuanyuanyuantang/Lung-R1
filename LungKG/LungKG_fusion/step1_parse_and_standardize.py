"""
Step 1: Data Parsing and Standardization

Functions:
  1.1 Parse Medical KG CSV, map entity types and relations to unified schema
  1.2 Parse guideline GraphML, extract nodes/edges, normalize entity types
  1.3 Use LLM to fill missing entity2_type in Medical KG
  1.4 Use LLM to normalize GraphML's 290 entity types to 15 standard types
  1.5 Use LLM to map GraphML edge descriptions to structured relation types
"""

import asyncio
import json
import logging
import re
from collections import Counter

import networkx as nx
import pandas as pd
from openai import AsyncOpenAI
from tqdm import tqdm

import config as C

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ============================== LLM Utilities ==============================

def get_client() -> AsyncOpenAI:
    return AsyncOpenAI(api_key=C.OPENAI_API_KEY, base_url=C.OPENAI_BASE_URL)


async def llm_call(client: AsyncOpenAI, prompt: str, sem: asyncio.Semaphore) -> str:
    """LLM call with rate limiting and retry"""
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


# ============================== 1.1 Medical KG Parsing ==============================

def parse_medical_kg() -> pd.DataFrame:
    """Parse the custom Medical KG CSV and standardize it"""
    log.info("=" * 60)
    log.info("Step 1.1: Parse custom Medical KG CSV")
    log.info("=" * 60)

    df = pd.read_csv(C.MEDICAL_KG_CSV)
    log.info(f"Raw data: {len(df)} triples")

    # Standardize column names
    df = df.rename(columns={
        "entity1_norm": "head",
        "entity1_type": "head_type",
        "relation_canonical": "relation",
        "entity2_norm": "tail",
        "entity2_type": "tail_type",
    })

    # Map entity types to unified schema
    df["head_type"] = df["head_type"].map(C.MEDICAL_KG_TYPE_MAP).fillna("其他治疗")
    df["tail_type"] = df["tail_type"].map(C.MEDICAL_KG_TYPE_MAP)  # Keep NaN, LLM will fill later

    # Map relations to unified relations
    df["relation"] = df["relation"].map(C.RELATION_MERGE_MAP).fillna(df["relation"])

    # Basic cleaning
    df["head"] = df["head"].astype(str).str.strip().str.lower()
    df["tail"] = df["tail"].astype(str).str.strip().str.lower()
    df = df.dropna(subset=["head", "tail"])
    df = df[df["head"].str.len() > 0]
    df = df[df["tail"].str.len() > 0]

    # Mark source
    df["source"] = "medical_kg"

    missing_type = df["tail_type"].isna().sum()
    log.info(f"After standardization: {len(df)} triples")
    log.info(f"head_type distribution:\n{df['head_type'].value_counts().head(10)}")
    log.info(f"relation distribution:\n{df['relation'].value_counts().head(10)}")
    log.info(f"tail_type missing: {missing_type} ({missing_type/len(df)*100:.1f}%)")

    return df


# ============================== 1.2 GraphML Parsing ==============================

def parse_graphml() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parse guideline GraphML, return (entities_df, edges_df)"""
    log.info("=" * 60)
    log.info("Step 1.2: Parse guideline GraphML")
    log.info("=" * 60)

    G = nx.read_graphml(str(C.GRAPHML_FILE))
    log.info(f"Raw graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Parse nodes
    entities = []
    for node_id, data in G.nodes(data=True):
        # Clean node ID (remove quote wrapping)
        clean_id = node_id.strip('"').strip()
        entity_type = data.get("entity_type", "UNKNOWN").strip('"').strip()
        description = data.get("description", "")
        # Truncate description to first 500 characters
        if isinstance(description, str) and len(description) > 500:
            description = description[:500]
        source_id = data.get("source_id", "")
        entities.append({
            "name": clean_id.lower(),
            "name_raw": clean_id,
            "entity_type_raw": data.get("entity_type", "UNKNOWN"),
            "entity_type": entity_type,
            "description": description,
            "source_id": source_id,
        })

    entities_df = pd.DataFrame(entities)
    log.info(f"Parsed {len(entities_df)} entities")

    # Parse edges
    edges = []
    for u, v, data in G.edges(data=True):
        head = u.strip('"').strip().lower()
        tail = v.strip('"').strip().lower()
        description = data.get("description", "")
        weight = data.get("weight", 1.0)
        if isinstance(description, str) and len(description) > 300:
            description = description[:300]
        edges.append({
            "head": head,
            "tail": tail,
            "description": description,
            "weight": weight,
        })

    edges_df = pd.DataFrame(edges)
    log.info(f"Parsed {len(edges_df)} edges")

    # Statistics of raw type distribution
    type_counts = entities_df["entity_type"].value_counts()
    log.info(f"Number of raw entity types: {len(type_counts)}")
    log.info(f"Top 15 entity types:\n{type_counts.head(15)}")

    return entities_df, edges_df


# ============================== 1.3 LLM Fill Missing Types in Medical KG ==============================

async def fill_missing_types_llm(df: pd.DataFrame) -> pd.DataFrame:
    """Use LLM to batch-fill missing tail_type in Medical KG"""
    log.info("=" * 60)
    log.info("Step 1.3: LLM fill missing tail_type in Medical KG")
    log.info("=" * 60)

    missing_mask = df["tail_type"].isna()
    missing_df = df[missing_mask].copy()
    log.info(f"Need to fill: {len(missing_df)} rows")

    if len(missing_df) == 0:
        return df

    # Group by (relation, head_type), collect unique tail entities
    # Infer tail type from relation: e.g. "适应症" tail is usually "疾病"
    # First use rule-based inference to reduce LLM calls
    relation_to_type = {
        "不良反应": "症状",
        "临床表现": "症状",
        "适应症": "疾病",
        "成份": "药物",
        "治疗": "药物",
        "并发症": "疾病",
        "相关疾病": "疾病",
        "病因": "病原体",
        "检查": "检查",
        "所属科室": "科室",
        "发病部位": "部位",
        "高危因素": "流行病学",
        "预防": "其他治疗",
        "预后": "流行病学",
        "药物相互作用": "西药",
        "英文名称": "药物",
        "商品名": "西药",
        "禁忌": "疾病",
        "易感人群": "流行病学",
        "诊断依据": "检查",
        "注意事项": "其他治疗",
        "用法用量": "其他治疗",
        "分类": "其他治疗",
        "药品分类": "其他治疗",
        "药品监管分级": "其他治疗",
        "贮藏": "其他治疗",
        "传播途径": "流行病学",
        "流行病学": "流行病学",
        "同义词": None,  # Synonym: tail type should be same as head
    }

    filled_count = 0
    for rel, etype in relation_to_type.items():
        mask = missing_mask & (df["relation"] == rel)
        if mask.sum() > 0:
            if etype is not None:
                df.loc[mask, "tail_type"] = etype
            else:
                # Synonym: tail type = head type
                df.loc[mask, "tail_type"] = df.loc[mask, "head_type"]
            filled_count += mask.sum()

    # For remaining unmatched, use LLM to infer
    still_missing = df["tail_type"].isna()
    remaining = still_missing.sum()
    log.info(f"Rule-based fill completed: {filled_count} rows")
    log.info(f"Remaining needing LLM fill: {remaining} rows")

    if remaining > 0:
        # Collect unique (tail, relation, head_type) combinations
        unique_items = df[still_missing][["tail", "relation", "head_type"]].drop_duplicates()
        log.info(f"Unique combinations to infer: {len(unique_items)}")

        cache = C.load_cache("step1_type_fill")
        client = get_client()
        sem = asyncio.Semaphore(C.MAX_CONCURRENT)
        type_list_str = ", ".join(C.UNIFIED_ENTITY_TYPES)

        async def infer_batch(batch_items):
            items_text = "\n".join(
                f"- 实体: {row['tail']}, 关系: {row['relation']}, 头实体类型: {row['head_type']}"
                for _, row in batch_items.iterrows()
            )
            prompt = f"""你是医学知识图谱专家。请根据以下三元组信息，推断每个尾实体的类型。
可选类型: {type_list_str}

三元组列表:
{items_text}

请严格按以下JSON数组格式输出，每个元素包含entity和type:
[{{"entity": "实体名", "type": "类型"}}]

只输出JSON，不要其他内容。"""
            ck = C.cache_key(prompt)
            if ck in cache:
                return cache[ck]
            result = await llm_call(client, prompt, sem)
            cache[ck] = result
            return result

        # Batch processing
        batches = [unique_items.iloc[i:i+C.BATCH_SIZE] for i in range(0, len(unique_items), C.BATCH_SIZE)]
        results = []

        for batch in tqdm(batches, desc="LLM inferring tail_type"):
            result = await infer_batch(batch)
            results.append((batch, result))

        # Parse results
        type_map = {}
        for batch, result_text in results:
            try:
                # Extract JSON
                json_match = re.search(r'\[[\s\S]*\]', result_text)
                if json_match:
                    items = json.loads(json_match.group(0))
                    for item in items:
                        entity = item.get("entity", "").strip().lower()
                        etype = item.get("type", "").strip()
                        if entity and etype in C.UNIFIED_ENTITY_TYPES:
                            type_map[entity] = etype
            except (json.JSONDecodeError, Exception) as e:
                log.warning(f"Failed to parse LLM result: {e}")

        # Apply inferred results
        llm_filled = 0
        for idx in df[still_missing].index:
            tail = df.at[idx, "tail"]
            if tail in type_map:
                df.at[idx, "tail_type"] = type_map[tail]
                llm_filled += 1

        log.info(f"LLM inference fill: {llm_filled} rows")
        C.save_cache("step1_type_fill", cache)
        await client.close()

    # Final fallback: set remaining missing to "其他治疗"
    df["tail_type"] = df["tail_type"].fillna("其他治疗")
    log.info(f"Final tail_type missing count: {df['tail_type'].isna().sum()}")

    return df


# ============================== 1.4 GraphML Entity Type Normalization ==============================

async def normalize_graphml_types(entities_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize GraphML's 290 entity types to 15 standard types"""
    log.info("=" * 60)
    log.info("Step 1.4: Normalize GraphML entity types (290 -> 15)")
    log.info("=" * 60)

    # First apply rule-based mapping
    raw_types = entities_df["entity_type_raw"].unique()
    rule_mapped = {}
    unmapped = []

    for rt in raw_types:
        if rt in C.GRAPHML_TYPE_RULES:
            rule_mapped[rt] = C.GRAPHML_TYPE_RULES[rt]
        else:
            # Try matching after stripping quotes
            clean_rt = rt.strip('"').strip()
            if clean_rt in C.UNIFIED_ENTITY_TYPES:
                rule_mapped[rt] = clean_rt
            elif f'"{clean_rt}"' in C.GRAPHML_TYPE_RULES:
                rule_mapped[rt] = C.GRAPHML_TYPE_RULES[f'"{clean_rt}"']
            else:
                unmapped.append(rt)

    log.info(f"Rule-mapped: {len(rule_mapped)} types")
    log.info(f"Awaiting LLM mapping: {len(unmapped)} types")

    # LLM batch-maps remaining types
    if unmapped:
        cache = C.load_cache("step1_type_normalize")
        client = get_client()
        sem = asyncio.Semaphore(C.MAX_CONCURRENT)
        type_list_str = ", ".join(C.UNIFIED_ENTITY_TYPES)

        llm_map = {}
        batches = [unmapped[i:i+80] for i in range(0, len(unmapped), 80)]

        for batch in tqdm(batches, desc="LLM normalizing entity types"):
            types_text = "\n".join(f"- {t}" for t in batch)
            prompt = f"""你是医学知识图谱专家。请将以下GraphML中的实体类型映射到标准类型。

标准类型列表: {type_list_str}

需要映射的类型:
{types_text}

映射规则:
- 如果原类型是某个具体实体名（如药名、病名），根据其含义映射到对应标准类型
- 如果无法确定，映射到"其他治疗"

请严格按JSON格式输出:
{{"原类型1": "标准类型1", "原类型2": "标准类型2"}}

只输出JSON，不要其他内容。"""

            ck = C.cache_key(prompt)
            if ck in cache:
                result = cache[ck]
            else:
                result = await llm_call(client, prompt, sem)
                cache[ck] = result

            try:
                json_match = re.search(r'\{[\s\S]*\}', result)
                if json_match:
                    mapping = json.loads(json_match.group(0))
                    for k, v in mapping.items():
                        v = v.strip('"').strip()
                        if v in C.UNIFIED_ENTITY_TYPES:
                            llm_map[k] = v
                        else:
                            llm_map[k] = "其他治疗"
            except (json.JSONDecodeError, Exception) as e:
                log.warning(f"Failed to parse type mapping: {e}")
                for t in batch:
                    llm_map[t] = "其他治疗"

        C.save_cache("step1_type_normalize", cache)
        await client.close()

        # Merge mappings
        rule_mapped.update(llm_map)

    # Apply mappings
    entities_df["entity_type_unified"] = entities_df["entity_type_raw"].map(rule_mapped).fillna("其他治疗")

    log.info(f"Type distribution after normalization:\n{entities_df['entity_type_unified'].value_counts().head(15)}")

    return entities_df


# ============================== 1.5 GraphML Edge Relation Mapping ==============================

async def map_edge_relations(edges_df: pd.DataFrame) -> pd.DataFrame:
    """Map GraphML edge free-text descriptions to structured relation types"""
    log.info("=" * 60)
    log.info("Step 1.5: Map GraphML edge relations to structured types")
    log.info("=" * 60)

    unified_relations = sorted(set(C.RELATION_MERGE_MAP.values()))
    # Add guideline-specific relations
    extra_relations = [
        "推荐用药", "推荐剂量", "推荐疗程", "鉴别诊断", "发病机制",
        "危险因素", "保护因素", "预防措施", "监测指标", "给药途径",
        "耐药性", "敏感性", "联合用药", "替代用药", "一线治疗",
        "二线治疗", "经验性治疗", "相关指南", "证据等级",
    ]
    all_relations = unified_relations + extra_relations
    relation_str = ", ".join(all_relations)

    cache = C.load_cache("step1_edge_map")
    client = get_client()
    sem = asyncio.Semaphore(C.MAX_CONCURRENT)

    # Take unique edge descriptions (first 100 chars as key)
    edges_df["desc_short"] = edges_df["description"].astype(str).str[:100]
    unique_descs = edges_df["desc_short"].unique()
    log.info(f"Unique edge descriptions: {len(unique_descs)}")

    # Batch mapping
    desc_to_relation = {}
    batches = [unique_descs[i:i+C.BATCH_SIZE] for i in range(0, len(unique_descs), C.BATCH_SIZE)]

    for batch in tqdm(batches, desc="LLM mapping edge relations"):
        descs_text = "\n".join(f"{i+1}. {d}" for i, d in enumerate(batch))
        prompt = f"""你是医学知识图谱专家。请根据以下知识图谱边的描述，将每条边映射到最合适的标准关系类型。

标准关系类型: {relation_str}

边描述列表:
{descs_text}

请严格按JSON格式输出，key为编号，value为关系类型:
{{"1": "关系类型1", "2": "关系类型2"}}

如果描述不清或无法映射，使用"相关"。只输出JSON。"""

        ck = C.cache_key(prompt)
        if ck in cache:
            result = cache[ck]
        else:
            result = await llm_call(client, prompt, sem)
            cache[ck] = result

        try:
            json_match = re.search(r'\{[\s\S]*\}', result)
            if json_match:
                mapping = json.loads(json_match.group(0))
                for k, v in mapping.items():
                    idx = int(k) - 1
                    if 0 <= idx < len(batch):
                        desc_to_relation[batch[idx]] = v.strip()
        except (json.JSONDecodeError, ValueError, Exception) as e:
            log.warning(f"Failed to parse edge relation mapping: {e}")

    C.save_cache("step1_edge_map", cache)
    await client.close()

    # Apply mapping
    edges_df["relation"] = edges_df["desc_short"].map(desc_to_relation).fillna("相关")
    edges_df["source"] = "guideline"

    log.info(f"Edge relation mapping completed, relation type distribution:\n{edges_df['relation'].value_counts().head(15)}")

    return edges_df


# ============================== Main Function ==============================

async def run_step1():
    """Run the full Step 1 pipeline"""
    log.info("=" * 70)
    log.info("  Step 1: Data Parsing and Standardization - Start")
    log.info("=" * 70)

    # 1.1 Parse custom Medical KG
    medical_kg_df = parse_medical_kg()

    # 1.2 Parse GraphML
    entities_df, edges_df = parse_graphml()

    # 1.3 Fill missing types in Medical KG
    medical_kg_df = await fill_missing_types_llm(medical_kg_df)

    # 1.4 Normalize GraphML entity types
    entities_df = await normalize_graphml_types(entities_df)

    # 1.5 Map GraphML edge relations
    edges_df = await map_edge_relations(edges_df)

    # Supplement head_type and tail_type for GraphML edges
    entity_type_map = dict(zip(entities_df["name"], entities_df["entity_type_unified"]))
    edges_df["head_type"] = edges_df["head"].map(entity_type_map).fillna("其他治疗")
    edges_df["tail_type"] = edges_df["tail"].map(entity_type_map).fillna("其他治疗")

    # Save results
    medical_kg_out = medical_kg_df[["head", "head_type", "relation", "tail", "tail_type", "source"]]
    medical_kg_out.to_csv(C.STEP1_MEDICAL_KG_STANDARDIZED, index=False, encoding="utf-8")
    log.info(f"Medical KG standardized results saved: {C.STEP1_MEDICAL_KG_STANDARDIZED}")

    graphml_triples = edges_df[["head", "head_type", "relation", "tail", "tail_type", "source", "description", "weight"]]
    graphml_triples.to_csv(C.STEP1_GRAPHML_TRIPLES, index=False, encoding="utf-8")
    log.info(f"GraphML triples saved: {C.STEP1_GRAPHML_TRIPLES}")

    entities_out = entities_df[["name", "name_raw", "entity_type_unified", "description"]]
    entities_out.to_csv(C.STEP1_GRAPHML_ENTITIES, index=False, encoding="utf-8")
    log.info(f"GraphML entities saved: {C.STEP1_GRAPHML_ENTITIES}")

    log.info("=" * 70)
    log.info("  Step 1: Data Parsing and Standardization - Complete")
    log.info(f"  Medical KG: {len(medical_kg_out)} triples")
    log.info(f"  GraphML: {len(graphml_triples)} triples, {len(entities_out)} entities")
    log.info("=" * 70)

    return medical_kg_out, graphml_triples, entities_out


def main():
    asyncio.run(run_step1())


if __name__ == "__main__":
    main()
