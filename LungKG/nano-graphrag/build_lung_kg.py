"""
Lung medical knowledge graph construction script (API version)

Description:
    Extract lung-related entities and relationships from cleaned medical guideline
    documents to build a knowledge graph. Uses the nano-graphrag framework, with
    LLM via external API and local Embedding model.

Usage:
    python build_lung_kg.py --build              # Build knowledge graph
    python build_lung_kg.py --query              # Interactive query
    python build_lung_kg.py --example            # Run example queries
    python build_lung_kg.py --question "question" # Single query
"""

import os
import logging
from pathlib import Path
from functools import partial
from tqdm import tqdm
import asyncio
import numpy as np

# ==============================================================================
#                               Global configuration area
# ==============================================================================

# 1. Hardware configuration
# ------------------------------------------------------------------------------
# Specify GPU IDs to use (e.g., "0,2" means using the 1st and 3rd cards, skipping high-load ones)
# Note: After setting this env var, cuda:0 maps to physical GPU 0, cuda:1 maps to physical GPU 2
GPU_IDS = "1"

# 2. API configuration
# ------------------------------------------------------------------------------
# API Key
OPENAI_API_KEY = "***"  # Replace with your own API Key

# API Base URL
OPENAI_BASE_URL = "https://your-api-endpoint/v1"
# Model name
MODEL_NAME = "your-model-name"

# # API Base URL
# OPENAI_BASE_URL = "https://your-api-endpoint/v1"

# # Model name
# MODEL_NAME = "deepseek-v3.2"

# Embedding model name (ModelScope ID)
EMBEDDING_MODEL_ID = "BAAI/bge-m3"

# 3. Path configuration
# ------------------------------------------------------------------------------
# Input data directory (contains .txt guideline files)
DOCS_DIR = "./准备数据/4.ICD-指南清洗后数据混合"

# Output working directory (stores the generated knowledge graph) - API version uses separate directory to avoid conflicts
WORKING_DIR = "./lung_kg_cache_api"

# Model cache directory (ModelScope cache)
MODEL_CACHE_DIR = "./model_cache"  # or None to use default cache ~/.cache/modelscope


# ==============================================================================
#                               System initialization
# ==============================================================================

# Set visible GPUs
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_IDS

# Set API environment variables
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["OPENAI_BASE_URL"] = OPENAI_BASE_URL

from nano_graphrag import GraphRAG, QueryParam
from nano_graphrag._llm import openai_complete_if_cache
from nano_graphrag._utils import wrap_embedding_func_with_attrs
from nano_graphrag.prompt import PROMPTS  # Import PROMPTS to modify entity types

# Download model from ModelScope
from modelscope import snapshot_download

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("lung_kg_api")

# ============== Custom medical entity types ==============

# Entity types aligned with existing LungKG
LUNG_ENTITY_TYPES = [
    # Core entity types (aligned with existing graph)
    "疾病",                    # Pneumonia, tuberculosis, COPD, etc.
    "症状",                    # Cough, dyspnea, chest pain, etc.
    "西药",                    # Antibiotics, antivirals, chemical drugs, etc.
    "中成药",                  # Proprietary Chinese medicine
    "中草药",                  # Chinese herbal medicine
    "药物",                    # Generic drug category (when classification is unclear)
    "诊疗技术及设备（治疗）",   # CT, X-ray, bronchoscopy, surgery, etc.
    "检查",                    # Lab tests, imaging studies, etc.
    "部位",                    # Lung lobes, bronchi, alveoli, etc.
    "手术治疗",                # Surgical procedures
    "其他治疗",                # Other treatment methods
    # Extended entity types (guideline-specific)
    "病原体",                  # Bacteria, viruses, fungi, etc.
    "流行病学",                # Incidence, prevalence, risk factors, etc.
    "预后",                    # Prognosis information, survival rates, etc.
    "科室",                    # Clinical department
    "指南建议",                # Recommendation levels, evidence levels (A/B/C)
]

# Override default entity types with medical domain entities
PROMPTS["DEFAULT_ENTITY_TYPES"] = LUNG_ENTITY_TYPES


# ============== Embedding function ==============

# Local Embedding model (global singleton)
_LOCAL_EMBED_MODEL = None

def _detect_device():
    """Detect available computing device"""
    import torch
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

def preload_embedding_model():
    """Preload the embedding model (called at program startup) - API version uses GPU"""
    global _LOCAL_EMBED_MODEL
    if _LOCAL_EMBED_MODEL is not None:
        return _LOCAL_EMBED_MODEL

    device = _detect_device()
    logger.info(f"Preloading Embedding model: {EMBEDDING_MODEL_ID} (API version, using {device})")

    # Download model from ModelScope to local
    model_dir = snapshot_download(EMBEDDING_MODEL_ID, cache_dir=MODEL_CACHE_DIR)
    logger.info(f"Model path: {model_dir}")

    from sentence_transformers import SentenceTransformer

    # If CUDA, default load to the first visible card (cuda:0)
    load_device = "cuda:0" if device == "cuda" else device
    _LOCAL_EMBED_MODEL = SentenceTransformer(model_dir, device=load_device)

    # Multi-GPU parallel (if CUDA and multiple cards)
    if device == "cuda":
        import torch
        gpu_count = torch.cuda.device_count()
        if gpu_count > 1:
            try:
                # Use all visible GPUs
                # Since CUDA_VISIBLE_DEVICES is already set, cuda:0 and cuda:1 here
                # correspond to the 1st and 2nd cards in the physical GPU_IDS list
                target_devices = [f'cuda:{i}' for i in range(gpu_count)]

                if target_devices:
                    pool = _LOCAL_EMBED_MODEL.start_multi_process_pool(target_devices=target_devices)
                    _LOCAL_EMBED_MODEL._pool = pool
                    logger.info(f"Embedding model loaded (multi-GPU parallel: {target_devices})")
                else:
                    logger.warning("No multiple GPUs found, using default single card mode")
            except Exception as e:
                logger.warning(f"Multi-GPU parallel failed, using single card: {e}")

    return _LOCAL_EMBED_MODEL

def get_local_embed_model():
    """Get local embedding model (load if not preloaded yet)"""
    global _LOCAL_EMBED_MODEL
    if _LOCAL_EMBED_MODEL is None:
        return preload_embedding_model()
    return _LOCAL_EMBED_MODEL

@wrap_embedding_func_with_attrs(embedding_dim=1024, max_token_size=8192)
async def local_embedding(texts: list[str]) -> np.ndarray:
    """Generate embeddings using local SentenceTransformer model (multi-GPU parallel)"""
    model = get_local_embed_model()
    if hasattr(model, '_pool') and model._pool is not None:
        # Multi-GPU parallel encoding
        return model.encode_multi_process(texts, model._pool, normalize_embeddings=True, batch_size=64)
    else:
        return model.encode(texts, normalize_embeddings=True, batch_size=32)

@wrap_embedding_func_with_attrs(embedding_dim=1536, max_token_size=8192)
async def dummy_embedding(texts: list[str]) -> np.ndarray:
    """
    Fake Embedding function for testing without an Embedding API.
    Note: Using this function will disable vector retrieval; only for testing graph construction!
    """
    return np.random.rand(len(texts), 1536)


# ============== Main program ==============

# Add delayed wrapper to avoid API rate limiting
async def rate_limited_llm(model: str, *args, **kwargs):
    """LLM call with delay to avoid triggering API rate limiting"""
    await asyncio.sleep(0.5)  # 0.5 second delay per call
    return await openai_complete_if_cache(model, *args, **kwargs)

def create_lung_graphrag(embedding_type: str = "local"):
    """
    Create lung knowledge graph GraphRAG instance

    Args:
        embedding_type: embedding type (local/api/dummy)
    """
    # Bind LLM functions
    llm_func = partial(rate_limited_llm, MODEL_NAME)
    
    kwargs = {
        "working_dir": WORKING_DIR,
        "best_model_func": llm_func,
        "cheap_model_func": llm_func,
        "chunk_token_size": 1800,  # Increased to accommodate ICD-10 structured data
        "chunk_overlap_token_size": 150,
        "enable_llm_cache": True,
        "enable_local": True,
        # "best_model_max_async": 2,  # Limit concurrency
        # "cheap_model_max_async": 2, # Limit concurrency
    }

    if embedding_type == "dummy":
        kwargs["embedding_func"] = dummy_embedding
        logger.warning("Using fake Embedding function, vector retrieval will be disabled!")
    elif embedding_type == "local":
        kwargs["embedding_func"] = local_embedding
        logger.info("Using local SentenceTransformer Embedding")
    # API mode uses default OpenAI embedding
    
    return GraphRAG(**kwargs)


def load_documents(docs_dir: str) -> list[tuple[str, str]]:
    """
    Load guideline documents

    Returns:
        [(filename, content), ...]
    """
    docs = []
    docs_path = Path(docs_dir)

    if not docs_path.exists():
        logger.error(f"Document directory does not exist: {docs_dir}")
        return docs

    for doc_path in docs_path.glob("*.txt"):
        try:
            with open(doc_path, encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    docs.append((doc_path.name, content))
                    logger.info(f"Loaded document: {doc_path.name} ({len(content)} chars)")
        except Exception as e:
            logger.error(f"Failed to read document {doc_path.name}: {e}")

    logger.info(f"Loaded {len(docs)} documents total")
    return docs


def build_knowledge_graph(embedding_type: str = "local"):
    """
    Build lung knowledge graph (supports checkpoint resumption)

    Args:
        embedding_type: embedding type (local/api/dummy)
    """
    logger.info("=" * 50)
    logger.info("Starting lung medical knowledge graph construction (API version)")
    logger.info("=" * 50)

    # Create GraphRAG instance
    rag = create_lung_graphrag(embedding_type)

    # Load documents
    docs = load_documents(DOCS_DIR)
    if not docs:
        logger.error("No documents found, exiting")
        return None

    # Checkpoint resumption: load list of already processed documents
    progress_file = Path(WORKING_DIR) / "processed_docs_api.txt"
    processed_docs = set()
    if progress_file.exists():
        with open(progress_file, "r", encoding="utf-8") as f:
            processed_docs = set(line.strip() for line in f if line.strip())
        logger.info(f"Found {len(processed_docs)} already processed documents, skipping")

    # Filter already processed documents
    pending_docs = [(name, content) for name, content in docs if name not in processed_docs]
    logger.info(f"Pending documents: {len(pending_docs)}")

    if not pending_docs:
        logger.info("All documents already processed!")
        return rag

    # Insert documents one by one
    failed_docs = []
    for name, content in tqdm(pending_docs, desc="Building knowledge graph", unit="doc"):
        try:
            rag.insert(content)
            # Record processed document
            with open(progress_file, "a", encoding="utf-8") as f:
                f.write(f"{name}\n")
            logger.info(f"Document {name} processed successfully")
        except Exception as e:
            logger.error(f"Failed to process document {name}: {e}")
            failed_docs.append(name)
            continue

    if failed_docs:
        logger.warning(f"Total {len(failed_docs)} documents failed: {failed_docs}")

    logger.info("=" * 50)
    logger.info("Knowledge graph construction complete!")
    logger.info(f"Graph files saved at: {WORKING_DIR}")
    logger.info("=" * 50)

    return rag


def query_lung_kg(rag: GraphRAG, question: str, mode: str = "local"):
    """
    Query the lung knowledge graph

    Args:
        rag: GraphRAG instance
        question: Question
        mode: Query mode (local/global/naive)
    """
    logger.info(f"Query: {question}")
    logger.info(f"Mode: {mode}")

    result = rag.query(question, param=QueryParam(mode=mode))
    return result


def interactive_query(rag: GraphRAG):
    """
    Interactive query
    """
    print("\n" + "=" * 50)
    print("Lung Medical Knowledge Graph Query System (API version)")
    print("Enter a question to query, enter 'quit' to exit")
    print("Enter 'mode:local' or 'mode:global' to switch query mode")
    print("=" * 50 + "\n")

    mode = "local"

    while True:
        try:
            question = input(f"[{mode}] Please enter a question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting")
            break

        if not question:
            continue

        if question.lower() == 'quit':
            print("Exiting")
            break

        if question.startswith('mode:'):
            new_mode = question.split(':')[1].strip()
            if new_mode in ['local', 'global', 'naive']:
                mode = new_mode
                print(f"Switched to {mode} mode")
            else:
                print("Invalid mode, options: local, global, naive")
            continue

        try:
            result = query_lung_kg(rag, question, mode)
            print("\n" + "-" * 40)
            print("Answer:")
            print(result)
            print("-" * 40 + "\n")
        except Exception as e:
            print(f"Query failed: {e}")


# ============== Example queries ==============

EXAMPLE_QUERIES = [
    "肺部感染的常见病原体有哪些？",
    "肺炎的治疗方案是什么？",
    "侵袭性真菌感染如何诊断？",
    "耐药菌感染的治疗策略有哪些？",
    "肺结核的诊断方法有哪些？",
]


def run_example_queries(rag: GraphRAG):
    """
    Run example queries
    """
    print("\n" + "=" * 50)
    print("Running example queries (API version)")
    print("=" * 50 + "\n")

    for q in EXAMPLE_QUERIES:
        print(f"\nQuestion: {q}")
        print("-" * 40)
        try:
            result = query_lung_kg(rag, q, mode="local")
            print(f"Answer:\n{result}")
        except Exception as e:
            print(f"Query failed: {e}")
        print("-" * 40)


# ============== Entry point ==============

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Lung Medical Knowledge Graph Construction Tool (API version)")
    parser.add_argument("--build", action="store_true", help="Build knowledge graph")
    parser.add_argument("--query", action="store_true", help="Interactive query")
    parser.add_argument("--example", action="store_true", help="Run example queries")
    parser.add_argument("--embedding", type=str, default="local",
                        choices=["local", "dummy"],
                        help="Embedding type: local (local model), dummy (testing)")
    parser.add_argument("--question", type=str, help="Single query question")
    parser.add_argument("--mode", type=str, default="local",
                        choices=["local", "global", "naive"], help="Query mode")

    args = parser.parse_args()

    # Default behavior: build + example queries
    if not any([args.build, args.query, args.example, args.question]):
        args.build = True
        args.example = True

    # If using local embedding, preload the model first
    if args.embedding == "local":
        preload_embedding_model()

    rag = None

    if args.build:
        rag = build_knowledge_graph(args.embedding)

    if args.query or args.example or args.question:
        if rag is None:
            # Load existing graph
            logger.info("Loading existing knowledge graph...")
            rag = create_lung_graphrag(args.embedding)

        if args.question:
            result = query_lung_kg(rag, args.question, args.mode)
            print(f"\nAnswer:\n{result}")

        if args.example:
            run_example_queries(rag)

        if args.query:
            interactive_query(rag)
