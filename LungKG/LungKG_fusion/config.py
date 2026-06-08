"""
LungKG Knowledge Graph Fusion - Global Configuration
"""
import os
from pathlib import Path

# ==============================================================================
#                            Path Configuration
# ==============================================================================

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]
FUSION_ROOT = Path(__file__).resolve().parent

# Input data path (please replace with your actual path)
MEDICAL_KG_CSV = PROJECT_ROOT / "LungKG" / "your_kg_data" / "lung_kg_final_clean.csv"
GRAPHML_FILE = PROJECT_ROOT / "nano-graphrag" / "lung_kg_cache_api" / "graph_chunk_entity_relation.graphml"

# Intermediate results and output paths
CACHE_DIR = FUSION_ROOT / "cache"
OUTPUT_DIR = FUSION_ROOT / "output"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Step intermediate files
STEP1_MEDICAL_KG_STANDARDIZED = OUTPUT_DIR / "step1_medical_kg_standardized.csv"
STEP1_GRAPHML_TRIPLES = OUTPUT_DIR / "step1_graphml_triples.csv"
STEP1_GRAPHML_ENTITIES = OUTPUT_DIR / "step1_graphml_entities.csv"
STEP2_ENTITY_MAPPING = OUTPUT_DIR / "step2_entity_mapping.csv"
STEP3_MERGED_TRIPLES = OUTPUT_DIR / "step3_merged_triples.csv"
STEP3_MERGED_ENTITIES = OUTPUT_DIR / "step3_merged_entities.csv"
STEP4_COMPLETED_TRIPLES = OUTPUT_DIR / "step4_completed_triples.csv"
STEP5_QUALITY_REPORT = OUTPUT_DIR / "step5_quality_report.json"
STEP6_NODES_CSV = OUTPUT_DIR / "nodes.csv"
STEP6_EDGES_CSV = OUTPUT_DIR / "edges.csv"

# ==============================================================================
#                           LLM API Configuration
# ==============================================================================

OPENAI_API_KEY = "***"  # Please replace with your own API Key
OPENAI_BASE_URL = "https://your-api-endpoint/v1"
MODEL_NAME = "your-model-name"

# Concurrency and rate limiting
MAX_CONCURRENT = 4          # Maximum concurrency
REQUEST_DELAY = 0.5         # Interval between requests (seconds)
MAX_RETRIES = 3             # Maximum retries
BATCH_SIZE = 50             # Batch size per processing batch

# ==============================================================================
#                            Unified Schema
# ==============================================================================

# Unified entity types (15 types)
UNIFIED_ENTITY_TYPES = [
    "疾病", "症状", "西药", "中成药", "中草药", "药物",
    "病原体", "检查", "诊疗技术及设备", "手术治疗", "其他治疗",
    "部位", "科室", "指南建议", "流行病学",
]

# Medical KG entity type -> unified type mapping
MEDICAL_KG_TYPE_MAP = {
    "疾病": "疾病",
    "症状": "症状",
    "西药": "西药",
    "中成药": "中成药",
    "中草药": "中草药",
    "药物": "药物",
    "检查": "检查",
    "诊疗技术及设备（治疗）": "诊疗技术及设备",
    "手术治疗": "手术治疗",
    "其他治疗": "其他治疗",
    "其他": "其他治疗",
    "部位": "部位",
    "科室": "科室",
    "社会学": "流行病学",
    "预后": "流行病学",
    "流行病学": "流行病学",
    "病原体": "病原体",
    "指南建议": "指南建议",
}

# GraphML high-frequency entity type -> unified type rule mapping (covers top 15 frequent types)
GRAPHML_TYPE_RULES = {
    '"指南建议"': "指南建议",
    '"检查"': "检查",
    '"疾病"': "疾病",
    '"流行病学"': "流行病学",
    '"其他治疗"': "其他治疗",
    '"预后"': "流行病学",
    '"西药"': "西药",
    '"病原体"': "病原体",
    '"症状"': "症状",
    '"科室"': "科室",
    '"诊疗技术及设备（治疗）"': "诊疗技术及设备",
    '"部位"': "部位",
    '"药物"': "药物",
    '"手术治疗"': "手术治疗",
    '"中草药"': "中草药",
    '"中成药"': "中成药",
    "UNKNOWN": "其他治疗",
    # Common mislabeled types
    '"概念"': "其他治疗",
    '"其他"': "其他治疗",
    '"治疗"': "其他治疗",
    '"组织"': "科室",
    "PERSON": "流行病学",
    '"治疗方案"': "其他治疗",
    '"高危因素"': "流行病学",
    "RELATIONSHIP": "其他治疗",
    "LOCATION": "部位",
    '"药代动力学"': "西药",
    '"机构"': "科室",
    '"机制"': "流行病学",
}

# Medical KG relation merge mapping (72 types -> ~40 types)
RELATION_MERGE_MAP = {
    # Original relation -> unified relation
    "不良反应": "不良反应",
    "症状": "临床表现",
    "临床症状及体征": "临床表现",
    "相关症状": "临床表现",
    "适应症": "适应症",
    "成份": "成份",
    "药物相互作用": "药物相互作用",
    "同义词": "同义词",
    "治疗": "治疗",
    "治疗方式": "治疗",
    "药物治疗": "治疗",
    "分类": "分类",
    "一级分类": "分类",
    "二级分类": "分类",
    "英文名称": "英文名称",
    "注意事项": "注意事项",
    "并发症": "并发症",
    "检查": "检查",
    "辅助检查": "检查",
    "诊断依据": "诊断依据",
    "相关疾病": "相关疾病",
    "所属科室": "所属科室",
    "就诊科室": "所属科室",
    "用法用量": "用法用量",
    "病因": "病因",
    "禁忌": "禁忌",
    "禁忌症": "禁忌",
    "药品分类": "药品分类",
    "药品监管分级": "药品监管分级",
    "商品名": "商品名",
    "易感人群": "易感人群",
    "多发群体": "易感人群",
    "发病部位": "发病部位",
    "贮藏": "贮藏",
    "高危因素": "高危因素",
    "预防": "预防",
    "预后": "预后",
    "传播途径": "传播途径",
    "所属分类": "分类",
    "发病率": "流行病学",
    "患病率": "流行病学",
}

# ==============================================================================
#                           LLM Cache Utilities
# ==============================================================================

import json
import hashlib


def get_cache_path(step_name: str) -> Path:
    """Get the LLM cache file path for a step"""
    p = CACHE_DIR / f"{step_name}_llm_cache.json"
    return p


def load_cache(step_name: str) -> dict:
    """Load LLM call cache"""
    p = get_cache_path(step_name)
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cache(step_name: str, cache: dict):
    """Save LLM call cache"""
    p = get_cache_path(step_name)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def cache_key(text: str) -> str:
    """Generate cache key"""
    return hashlib.md5(text.encode("utf-8")).hexdigest()
