"""
Configuration for KG-Guided GRPO Reinforcement Learning.

All paths are relative to this config file's directory.
Copy the entire grpo/ folder to your server and run run.sh.
"""

import os

# ── Base directory (this file's directory) ──
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ═══════════════════════════════════════════════════════════════
# Paths — modify MODEL_PATH to point to your model
# ═══════════════════════════════════════════════════════════════

# Model path (HuggingFace model id or local path)
# For 0.5B test:   "Qwen/Qwen2.5-0.5B-Instruct"
# For 7B training: "Qwen/Qwen2.5-7B-Instruct"
# For 14B training: "Qwen/Qwen2.5-14B-Instruct"
MODEL_PATH = "Qwen/Qwen2.5-7B-Instruct"

# Training data (relative to this config file)
# EMR training data is available upon request to the authors
DATA_PATH = os.path.join(BASE_DIR, "data", "emr.json")

# Knowledge graph CSV files (relative to this config file)
KG_NODES_PATH = os.path.join(BASE_DIR, "kg_data", "nodes.csv")
KG_EDGES_PATH = os.path.join(BASE_DIR, "kg_data", "edges.csv")

# Output directory for checkpoints and final model
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# ═══════════════════════════════════════════════════════════════
# Data
# ═══════════════════════════════════════════════════════════════

NUM_SAMPLES = 3569  # Use all 3569 SFT training samples

# ═══════════════════════════════════════════════════════════════
# GRPO Training Hyperparameters
# ═══════════════════════════════════════════════════════════════

NUM_GENERATIONS = 4      # G: group size, candidates per prompt
LEARNING_RATE = 5e-6     # Learning rate
BETA = 0.04              # KL penalty coefficient
EPSILON_CLIP = 0.2       # PPO clipping epsilon

MAX_COMPLETION_LENGTH = 512   # Max tokens for generated response
MAX_PROMPT_LENGTH = 2560      # Max tokens for input prompt (covers 99%+ samples)

# Batch size settings (adjust based on GPU memory)
# 7B on 6×5090 (32GB): per_device=4, grad_accum=2 → effective batch=48
# 14B on 6×5090 (32GB): per_device=1, grad_accum=4 → effective batch=24
PER_DEVICE_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 2

NUM_TRAIN_EPOCHS = 1
LOGGING_STEPS = 5
SAVE_STEPS = 100

# ═══════════════════════════════════════════════════════════════
# Reward Weights (λ₁, λ₂, λ₃)
# ═══════════════════════════════════════════════════════════════

LAMBDA_DX = 0.50     # Diagnosis correctness
LAMBDA_GRAPH = 0.25  # Graph faithfulness
LAMBDA_PATH = 0.25   # Relation/path consistency

# ═══════════════════════════════════════════════════════════════
# NPU / Distributed Training
# ═══════════════════════════════════════════════════════════════

# Comma-separated NPU ids, e.g. "0,1,2,3,4,5"
# 7B: 4× Ascend with ZeRO-3
# 14B: 8× Ascend with ZeRO-3
NPU_DEVICE_IDS = "0,1,2,3"

# DeepSpeed config file (relative to BASE_DIR)
# Use deepspeed_zero3.json for 7B/14B models
DEEPSPEED_CONFIG = os.path.join(BASE_DIR, "deepspeed_zero3.json")

# ═══════════════════════════════════════════════════════════════
# System Prompt
# ═══════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """你是呼吸内科辅助诊断模型。请严格根据病例内容进行分析，先给出简洁、连贯的诊断推理，再输出仅与肺部相关的疾病诊断。不要输出治疗建议。最后一行使用"最终答案：..."格式。"""
