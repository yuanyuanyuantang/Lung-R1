"""
Step 4: Relation Completion

Functions:
  4.1 Use GraphML entity descriptions to supplement missing triples
  4.2 LLM-assisted inference of missing relations for isolated entities
"""

import asyncio
import json
import logging
import re
from collections import Counter

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


async def extract_relations_from_descriptions(
    merged_triples: pd.DataFrame,
    merged_entities: pd.DataFrame,
) -> pd.DataFrame:
    """Extract missing triples from GraphML entity descriptions"""
    log.info("=" * 60)
    log.info("Step 4.1: Extract missing relations from entity descriptions")
    log.info("=" * 60)

    important_types = {"疾病", "症状", "西药", "药物", "病原体", "中成药"}
    rich_entities = merged_entities[
        (merged_entities["type"].isin(important_types)) &
        (merged_entities["description"].astype(str).str.len() > 50)
    ].copy()

    log.info(f"Important entities with rich descriptions: {len(rich_entities)}")
    if len(rich_entities) == 0:
        return pd.DataFrame()

    existing_triples = set()
    for _, row in merged_triples.iterrows():
        existing_triples.add((row["head"], row["relation"], row["tail"]))

    unified_relations = sorted(set(C.RELATION_MERGE_MAP.values()))
    relation_str = ", ".join(unified_relations)

    cache = C.load_cache("step4_extract")
    client = get_client()
    sem = asyncio.Semaphore(C.MAX_CONCURRENT)

    rich_entities = rich_entities.sort_values(
        "description", key=lambda x: x.astype(str).str.len(), ascending=False
    ).head(500)
    new_triples = []

    batches = [rich_entities.iloc[i:i+5] for i in range(0, len(rich_entities), 5)]

    for batch in tqdm(batches, desc="Extracting relations from descriptions"):
        entities_text = ""
        for _, row in batch.iterrows():
            existing = merged_triples[
                (merged_triples["head"] == row["name"]) | (merged_triples["tail"] == row["name"])
            ]
            existing_rels = []
            for _, er in existing.head(10).iterrows():
                existing_rels.append(f"{er['head']} --[{er['relation']}]--> {er['tail']}")
            existing_text = "\n".join(existing_rels) if existing_rels else "None"
            entities_text += f"""
Entity: {row['name']} (Type: {row['type']})
Description: {str(row['description'])[:400]}
Existing relations:
{existing_text}
---"""

        prompt = f"""你是肺部医学知识图谱专家。请根据以下实体的描述信息，提取描述中提到但当前图谱中**缺失**的重要医学关系三元组。

{entities_text}

可用关系类型: {relation_str}

提取规则:
- 只提取描述中明确提到的关系，不要臆断
- 只提取与肺部/呼吸科高度相关的三元组
- 不要重复已有关系
- 每个实体最多提取5条新关系

输出JSON数组:
[{{"head": "头实体", "head_type": "类型", "relation": "关系", "tail": "尾实体", "tail_type": "类型"}}]
如果没有新关系输出 []。只输出JSON。"""

        ck = C.cache_key(prompt)
        if ck in cache:
            result = cache[ck]
        else:
            result = await llm_call(client, prompt, sem)
            cache[ck] = result

        try:
            json_match = re.search(r'\[[\s\S]*\]', result)
            if json_match:
                items = json.loads(json_match.group(0))
                for item in items:
                    h = item.get("head", "").strip().lower()
                    t = item.get("tail", "").strip().lower()
                    r = item.get("relation", "").strip()
                    if h and t and r and (h, r, t) not in existing_triples:
                        new_triples.append({
                            "head": h,
                            "head_type": item.get("head_type", "其他治疗"),
                            "relation": r, "tail": t,
                            "tail_type": item.get("tail_type", "其他治疗"),
                            "source": "llm_inferred", "description": "",
                            "weight": 1.0, "confidence": 0.8,
                        })
                        existing_triples.add((h, r, t))
        except (json.JSONDecodeError, Exception) as e:
            log.warning(f"Failed to parse relation extraction result: {e}")

    C.save_cache("step4_extract", cache)
    await client.close()

    new_df = pd.DataFrame(new_triples)
    log.info(f"New triples extracted from descriptions: {len(new_df)}")
    return new_df


async def infer_isolated_entity_relations(
    merged_triples: pd.DataFrame,
    merged_entities: pd.DataFrame,
) -> pd.DataFrame:
    """Use LLM to infer missing relations for highly isolated important entities"""
    log.info("=" * 60)
    log.info("Step 4.2: Infer missing relations for isolated entities")
    log.info("=" * 60)

    entity_degree = Counter()
    for _, row in merged_triples.iterrows():
        entity_degree[row["head"]] += 1
        entity_degree[row["tail"]] += 1

    important_types = {"疾病", "西药", "症状", "病原体"}
    isolated = merged_entities[
        (merged_entities["type"].isin(important_types)) &
        (merged_entities["name"].map(lambda x: entity_degree.get(x, 0) <= 3))
    ]
    log.info(f"Isolated important entities: {len(isolated)}")
    if len(isolated) == 0:
        return pd.DataFrame()

    isolated = isolated.head(200)
    cache = C.load_cache("step4_isolated")
    client = get_client()
    sem = asyncio.Semaphore(C.MAX_CONCURRENT)

    unified_relations = sorted(set(C.RELATION_MERGE_MAP.values()))
    relation_str = ", ".join(unified_relations)

    existing_triples = set()
    for _, row in merged_triples.iterrows():
        existing_triples.add((row["head"], row["relation"], row["tail"]))

    new_triples = []
    batches = [isolated.iloc[i:i+10] for i in range(0, len(isolated), 10)]

    for batch in tqdm(batches, desc="Inferring isolated entity relations"):
        entities_text = "\n".join(
            f"- {row['name']} (Type: {row['type']}, Desc: {str(row.get('description', ''))[:200]})"
            for _, row in batch.iterrows()
        )
        prompt = f"""你是肺部医学知识图谱专家。以下是一些关系较少的肺部医学实体，请根据你的医学知识，为每个实体推断最重要的缺失关系。

实体列表:
{entities_text}

可用关系类型: {relation_str}

推断规则:
- 只推断你确信的医学事实
- 与肺部/呼吸科密切相关
- 每个实体推断2-5条关系

输出JSON数组:
[{{"head": "头实体", "head_type": "类型", "relation": "关系", "tail": "尾实体", "tail_type": "类型", "confidence": 0.8}}]
只输出JSON。"""

        ck = C.cache_key(prompt)
        if ck in cache:
            result = cache[ck]
        else:
            result = await llm_call(client, prompt, sem)
            cache[ck] = result

        try:
            json_match = re.search(r'\[[\s\S]*\]', result)
            if json_match:
                items = json.loads(json_match.group(0))
                for item in items:
                    h = item.get("head", "").strip().lower()
                    t = item.get("tail", "").strip().lower()
                    r = item.get("relation", "").strip()
                    conf = item.get("confidence", 0.7)
                    if h and t and r and (h, r, t) not in existing_triples:
                        new_triples.append({
                            "head": h, "head_type": item.get("head_type", "其他治疗"),
                            "relation": r, "tail": t,
                            "tail_type": item.get("tail_type", "其他治疗"),
                            "source": "llm_inferred", "description": "",
                            "weight": 1.0, "confidence": min(conf, 0.9),
                        })
                        existing_triples.add((h, r, t))
        except (json.JSONDecodeError, Exception) as e:
            log.warning(f"Failed to parse inference result: {e}")

    C.save_cache("step4_isolated", cache)
    await client.close()

    new_df = pd.DataFrame(new_triples)
    log.info(f"New triples inferred: {len(new_df)}")
    return new_df


async def run_step4():
    log.info("=" * 70)
    log.info("  Step 4: Relation Completion - Start")
    log.info("=" * 70)

    merged_triples = pd.read_csv(C.STEP3_MERGED_TRIPLES)
    merged_entities = pd.read_csv(C.STEP3_MERGED_ENTITIES)

    new_from_desc = await extract_relations_from_descriptions(merged_triples, merged_entities)
    new_from_isolated = await infer_isolated_entity_relations(merged_triples, merged_entities)

    all_new = pd.concat([new_from_desc, new_from_isolated], ignore_index=True)
    if len(all_new) > 0:
        completed = pd.concat([merged_triples, all_new], ignore_index=True)
    else:
        completed = merged_triples.copy()

    completed.to_csv(C.STEP4_COMPLETED_TRIPLES, index=False, encoding="utf-8")

    log.info("=" * 70)
    log.info("  Step 4: Relation Completion - Complete")
    log.info(f"  Original: {len(merged_triples)}, New: {len(all_new)}, Total: {len(completed)}")
    log.info("=" * 70)
    return completed


def main():
    asyncio.run(run_step4())

if __name__ == "__main__":
    main()
