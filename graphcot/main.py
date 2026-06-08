"""
graphcot: Main Execution Script
-----------------------------------
A standalone script to generate QA pairs using graphcot algorithms.
"""

import os
import argparse
import asyncio
import json
import logging
import time

# Local Modules (Self-contained)
from storage import SimpleGraphStorage
from llm_client import LLMClient, LLMClientError
from graphcot import LongTailPartitioner, CoTGenerator

logger = logging.getLogger("graphcot")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")

async def main():
    parser = argparse.ArgumentParser(description="graphcot QA Generator")
    parser.add_argument("--kg_dir", required=True, help="Directory containing nodes.csv and edges.csv")
    parser.add_argument("--output_dir", default="output", help="Output directory for generated QA")
    parser.add_argument("--llm_api", default="https://api.deepseek.com/v1", help="LLM API Base URL")
    parser.add_argument("--model", default="deepseek-chat", help="LLM Model Name")
    parser.add_argument("--max_communities", type=int, default=10, help="Number of communities to process")
    parser.add_argument("--tail_ratio", type=float, default=0.5, help="Ratio of long-tail entities")
    parser.add_argument("--api_key", default="***", help="API key for LLM service")
    parser.add_argument("--num_questions", type=int, default=2, help="Number of QA pairs per community")
    parser.add_argument("--concurrency", type=int, default=1, help="Number of concurrent communities to process (API: 1-2, Local: 10+)")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("graphcot: Starting QA Generation Pipeline")
    logger.info(f"  Target KG: {args.kg_dir}")
    logger.info(f"  Output Dir: {args.output_dir}")
    logger.info(f"  LLM Model: {args.model}")
    logger.info("=" * 60)
    
    # 1. Load Graph
    logger.info("[Step 1] Loading Graph from CSV...")
    storage = SimpleGraphStorage()
    nodes_csv = os.path.join(args.kg_dir, "nodes.csv")
    edges_csv = os.path.join(args.kg_dir, "edges.csv")
    
    if not os.path.exists(nodes_csv) or not os.path.exists(edges_csv):
        logger.error(f"Could not find nodes.csv or edges.csv in {args.kg_dir}")
        return
        
    storage.load_from_csv(nodes_csv, edges_csv)
    logger.info(f"  Loaded: {len(storage.get_all_nodes())} nodes, {len(storage.get_all_edges())} edges")
    
    # 2. Partition Graph (Long-Tail Strategy)
    logger.info("[Step 2] Partitioning Graph (LongTailPartitioner)...")
    partitioner = LongTailPartitioner()
    communities = partitioner.partition(
        storage,
        max_size=50,
        min_size=3,
        tail_ratio=args.tail_ratio,
        expansion_depth=2,
        max_communities=args.max_communities
    )
    logger.info(f"  Generated {len(communities)} communities.")
    if communities:
        sample = communities[0]
        logger.info(f"  Sample Community: Anchor={sample.metadata.get('anchor')}, Nodes={len(sample.nodes)}")

    # 3. Generate QA (CoT Strategy)
    logger.info("[Step 3] Generating QA Pairs (CoTGenerator)...")
    llm_client = LLMClient(base_url=args.llm_api, model=args.model, api_key=args.api_key)
    generator = CoTGenerator(llm_client, num_of_questions=args.num_questions)
    
    all_results = {}
    processed_count = 0
    
    start_time = time.time()
    
    # Pre-build node map (avoid rebuilding per community)
    all_nodes_map = {nid: data for nid, data in storage.get_all_nodes()}
    
    # Process communities with concurrency control
    semaphore = asyncio.Semaphore(args.concurrency)
    
    async def process_community(idx, comm):
        nonlocal processed_count
        async with semaphore:
            logger.info(f"  Processing community {idx+1}/{len(communities)}...")

            # Hydrate node data from pre-built map
            comm_nodes = [
                (nid, all_nodes_map[nid]) for nid in comm.nodes if nid in all_nodes_map
            ]

            # Use edges already collected by the partitioner
            comm_edges = comm.edges

            # Get community type from metadata
            comm_type = comm.metadata.get("type", "mixed")

            # Generate
            try:
                qa_batch = await generator.generate_from_data(comm_nodes, comm_edges, community_type=comm_type)

                # Intermittent saving (save every 100 successful batches)
                if qa_batch:
                    all_results.update(qa_batch)
                    processed_count += 1
                    
                    if processed_count % 100 == 0:
                        temp_file = os.path.join(args.output_dir, f"generated_qa_partial.json")
                        with open(temp_file, "w", encoding="utf-8") as f:
                            json.dump(all_results, f, ensure_ascii=False, indent=2)
                        logger.info(f"  [Checkpoint] Saved {len(all_results)} QA pairs so far.")
                        
                    return qa_batch
            except LLMClientError as e:
                logger.error(f"    LLM error for community {idx}: {e}")
            except Exception as e:
                logger.error(f"    Unexpected error for community {idx}: {e}")
            return {}
    
    # Create tasks for all communities
    tasks = [process_community(idx, comm) for idx, comm in enumerate(communities)]
    await asyncio.gather(*tasks)
            
    # 4. Save Results
    # Ensure output directory exists right before saving, in case it was deleted
    os.makedirs(args.output_dir, exist_ok=True)
    
    output_file = os.path.join(args.output_dir, "generated_qa.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
        
    logger.info("=" * 60)
    logger.info("[Done] Pipeline Finished.")
    logger.info(f"  Total QA Pairs: {len(all_results)}")
    logger.info(f"  Time Elapsed: {time.time() - start_time:.2f}s")
    logger.info(f"  Results saved to: {output_file}")
    logger.info("=" * 60)

    await llm_client.close()

if __name__ == "__main__":
    asyncio.run(main())
