"""
Step 5: Quality Check

Functions:
  5.1 Statistical report
  5.2 Sample validation (LLM evaluation)
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
                    temperature=0.0, max_tokens=4000,
                )
                return resp.choices[0].message.content.strip()
            except Exception as e:
                log.warning(f"LLM call failed (attempt {attempt+1}): {e}")
                if attempt < C.MAX_RETRIES - 1:
                    await asyncio.sleep(2 * (attempt + 1))
        return ""


def generate_statistics(completed_triples: pd.DataFrame, merged_entities: pd.DataFrame) -> dict:
    log.info("=" * 60)
    log.info("Step 5.1: Generate statistical report")
    log.info("=" * 60)

    stats = {
        "total_triples": len(completed_triples),
        "total_entities": len(merged_entities),
        "source_distribution": completed_triples["source"].value_counts().to_dict(),
        "entity_type_distribution": merged_entities["type"].value_counts().to_dict(),
        "relation_distribution": completed_triples["relation"].value_counts().to_dict(),
        "unique_relations": completed_triples["relation"].nunique(),
    }

    degree_counter = Counter()
    for _, row in completed_triples.iterrows():
        degree_counter[row["head"]] += 1
        degree_counter[row["tail"]] += 1
    degrees = list(degree_counter.values())
    if degrees:
        stats["degree_stats"] = {
            "max": max(degrees), "min": min(degrees),
            "mean": round(sum(degrees) / len(degrees), 2),
            "median": sorted(degrees)[len(degrees) // 2],
        }

    all_in_triples = set(completed_triples["head"].unique()) | set(completed_triples["tail"].unique())
    all_in_list = set(merged_entities["name"].unique())
    stats["isolated_nodes"] = len(all_in_list - all_in_triples)
    stats["entities_in_triples"] = len(all_in_triples)

    try:
        import networkx as nx
        G = nx.Graph()
        for _, row in completed_triples.iterrows():
            G.add_edge(row["head"], row["tail"])
        components = list(nx.connected_components(G))
        stats["connected_components"] = len(components)
        stats["largest_component_size"] = max(len(c) for c in components) if components else 0
    except Exception as e:
        log.warning(f"Connected component analysis failed: {e}")

    log.info(f"Total triples: {stats['total_triples']}, Total entities: {stats['total_entities']}")
    log.info(f"Unique relation types: {stats['unique_relations']}, Isolated nodes: {stats['isolated_nodes']}")
    return stats


async def sample_validation(completed_triples: pd.DataFrame, sample_size: int = 200) -> dict:
    log.info("=" * 60)
    log.info(f"Step 5.2: Sample validation (sample size: {sample_size})")
    log.info("=" * 60)

    sample = completed_triples.sample(n=min(sample_size, len(completed_triples)), random_state=42)

    cache = C.load_cache("step5_validation")
    client = get_client()
    sem = asyncio.Semaphore(C.MAX_CONCURRENT)

    scores = []
    batches = [sample.iloc[i:i+20] for i in range(0, len(sample), 20)]

    for batch in tqdm(batches, desc="LLM quality evaluation"):
        triples_text = "\n".join(
            f"{i+1}. ({row['head']})[{row['head_type']}] --[{row['relation']}]--> ({row['tail']})[{row['tail_type']}]  source:{row.get('source','')}"
            for i, (_, row) in enumerate(batch.iterrows())
        )
        prompt = f"""你是肺部医学知识图谱质量评估专家。请评估以下知识图谱三元组的正确性。

三元组列表:
{triples_text}

评估标准:
- 5分: 完全正确 - 4分: 基本正确 - 3分: 部分正确 - 2分: 明显错误 - 1分: 完全错误

请输出JSON数组: [{{"id": 1, "score": 5, "comment": "简短评语"}}]
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
                    scores.append(min(max(item.get("score", 3), 1), 5))
        except Exception as e:
            log.warning(f"Failed to parse evaluation result: {e}")

    C.save_cache("step5_validation", cache)
    await client.close()

    validation = {}
    if scores:
        validation = {
            "sample_size": len(scores),
            "avg_score": round(sum(scores) / len(scores), 2),
            "score_distribution": dict(Counter(scores)),
            "accuracy_4plus": round(sum(1 for s in scores if s >= 4) / len(scores) * 100, 1),
            "accuracy_3plus": round(sum(1 for s in scores if s >= 3) / len(scores) * 100, 1),
        }
        log.info(f"Avg score: {validation['avg_score']}, Accuracy (>=4): {validation['accuracy_4plus']}%")
    return validation


async def run_step5():
    log.info("=" * 70)
    log.info("  Step 5: Quality Check - Start")
    log.info("=" * 70)

    completed_triples = pd.read_csv(C.STEP4_COMPLETED_TRIPLES)
    merged_entities = pd.read_csv(C.STEP3_MERGED_ENTITIES)

    stats = generate_statistics(completed_triples, merged_entities)
    validation = await sample_validation(completed_triples)

    report = {"statistics": stats, "validation": validation}
    with open(C.STEP5_QUALITY_REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    log.info(f"  Report saved: {C.STEP5_QUALITY_REPORT}")
    log.info("=" * 70)
    return report


def main():
    asyncio.run(run_step5())

if __name__ == "__main__":
    main()
