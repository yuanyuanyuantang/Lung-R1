#!/usr/bin/env python3
"""
KG-Guided GRPO Training Script for Lung-R1.

Implements GRPO (Group Relative Policy Optimization) with a KG-guided
reward function: R = λ₁·R_dx + λ₂·R_graph + λ₃·R_path

Usage:
    python grpo_train.py                                    # Use config.py defaults
    python grpo_train.py --model_name_or_path Qwen/Qwen2.5-7B-Instruct
    python grpo_train.py --deepspeed deepspeed_zero3.json
    python grpo_train.py --num_samples 100 --num_train_epochs 1
"""

import os
import re
import sys
import json
import argparse
import warnings
from datetime import datetime


def _truthy(value: str | None) -> bool:
    return value is not None and value.lower() in {"1", "true", "yes", "y", "on"}


def _cli_has_flag(flag: str) -> bool:
    return flag in sys.argv


def _cli_value(name: str) -> str | None:
    prefix = f"{name}="
    for idx, arg in enumerate(sys.argv):
        if arg == name and idx + 1 < len(sys.argv):
            return sys.argv[idx + 1]
        if arg.startswith(prefix):
            return arg[len(prefix):]
    return None


_USE_NPU = _truthy(os.environ.get("GRPO_USE_NPU")) or _cli_has_flag("--use_npu")
_NPU_DEVICE_IDS = _cli_value("--device_ids") or os.environ.get("GRPO_DEVICE_IDS")
if _USE_NPU and _NPU_DEVICE_IDS:
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = _NPU_DEVICE_IDS

import torch
if _USE_NPU:
    try:
        import torch_npu  # noqa: F401
        from torch_npu.contrib import transfer_to_npu  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "NPU mode was requested, but torch_npu is not installed in this Python environment."
        ) from exc
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

# Ensure local imports work from this directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    MODEL_PATH, DATA_PATH, KG_NODES_PATH, KG_EDGES_PATH,
    OUTPUT_DIR, NUM_SAMPLES, NUM_GENERATIONS, LEARNING_RATE,
    BETA, EPSILON_CLIP, MAX_COMPLETION_LENGTH, MAX_PROMPT_LENGTH,
    PER_DEVICE_BATCH_SIZE, GRADIENT_ACCUMULATION_STEPS,
    NUM_TRAIN_EPOCHS, LOGGING_STEPS, SAVE_STEPS,
    LAMBDA_DX, LAMBDA_GRAPH, LAMBDA_PATH,
    NPU_DEVICE_IDS, SYSTEM_PROMPT,
)
from kg_reward import KGRewardFunction
from metrics_logger import MetricsLoggerCallback, plot_training_curves

warnings.filterwarnings("ignore", category=FutureWarning)


# ═══════════════════════════════════════════════════════════════
# CLI Argument Parsing
# ═══════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="KG-Guided GRPO Training for Lung-R1"
    )
    # Model & Data
    parser.add_argument("--model_name_or_path", type=str, default=MODEL_PATH,
                        help="HuggingFace model id or local path")
    parser.add_argument("--data_path", type=str, default=DATA_PATH)
    parser.add_argument("--kg_nodes_path", type=str, default=KG_NODES_PATH)
    parser.add_argument("--kg_edges_path", type=str, default=KG_EDGES_PATH)
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--num_samples", type=int, default=NUM_SAMPLES)

    # GRPO hyperparameters
    parser.add_argument("--num_generations", type=int, default=NUM_GENERATIONS)
    parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--beta", type=float, default=BETA)
    parser.add_argument("--epsilon_clip", type=float, default=EPSILON_CLIP)
    parser.add_argument("--max_completion_length", type=int, default=MAX_COMPLETION_LENGTH)
    parser.add_argument("--max_prompt_length", type=int, default=MAX_PROMPT_LENGTH)
    parser.add_argument("--per_device_train_batch_size", type=int, default=PER_DEVICE_BATCH_SIZE)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=GRADIENT_ACCUMULATION_STEPS)
    parser.add_argument("--num_train_epochs", type=float, default=NUM_TRAIN_EPOCHS)
    parser.add_argument("--logging_steps", type=int, default=LOGGING_STEPS)
    parser.add_argument("--save_steps", type=int, default=SAVE_STEPS)

    # Reward weights
    parser.add_argument("--lambda_dx", type=float, default=LAMBDA_DX)
    parser.add_argument("--lambda_graph", type=float, default=LAMBDA_GRAPH)
    parser.add_argument("--lambda_path", type=float, default=LAMBDA_PATH)

    # Distributed training
    parser.add_argument("--deepspeed", type=str, default=None,
                        help="Path to DeepSpeed config JSON file")
    parser.add_argument("--device_ids", type=str, default=os.environ.get("GRPO_DEVICE_IDS"),
                        help="Comma-separated NPU ids for ASCEND_RT_VISIBLE_DEVICES")
    parser.add_argument("--use_npu", action="store_true", default=_USE_NPU,
                        help="Enable Ascend NPU compatibility mode")
    parser.add_argument("--bf16", action="store_true", default=True)

    # System prompt
    parser.add_argument("--system_prompt", type=str, default=SYSTEM_PROMPT)

    return parser.parse_args()


# ═══════════════════════════════════════════════════════════════
# Data Preparation
# ═══════════════════════════════════════════════════════════════

def _extract_diagnosis_from_output(output_text: str) -> str:
    """Extract diagnosis labels from model output text.

    Supports multiple formats:
      - \"最终答案：[疾病1],[疾病2]\"
      - \"肺部相关诊断：[疾病1],[疾病2]\"
      - Bare bracket format \"[疾病1],[疾病2]\"

    Returns the matched diagnosis string (e.g. \"[疾病1],[疾病2]\") or the
    original output_text if no pattern matches.
    """
    patterns = [
        r'最终答案[：:]\s*(.+)$',
        r'肺部相关诊断[：:]\s*(.+)$',
        r'诊断[：:]\s*(.+)$',
    ]
    for pat in patterns:
        m = re.search(pat, output_text)
        if m:
            return m.group(1).strip()
    # Fallback: look for bracket-wrapped content
    bracket_matches = re.findall(r'\[([^\]]+)\]', output_text)
    if bracket_matches:
        return "[" + "], [".join([m.strip() for m in bracket_matches]) + "]"
    return output_text.strip()


def prepare_dataset(data_path: str, num_samples: int, tokenizer, system_prompt: str):
    """Prepare GRPO training dataset from EMR data.

    Supports two data formats:
    1. GRPO format: has ``standard_output`` field with diagnosis labels
    2. SFT format: has ``output`` field with full reasoning + final answer
       (e.g. \"推理：...最终答案：[疾病1],[疾病2]\")
    """
    with open(data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    print(f"[Data] Loaded {len(raw_data)} samples from {data_path}")

    data = raw_data[:num_samples]
    dataset = []

    for item in data:
        instruction = item.get("instruction", "")
        input_text = item.get("input", "")

        # Extract ground truth: prefer standard_output, else parse from output
        if "standard_output" in item:
            ground_truth = item["standard_output"]
        elif "output" in item:
            ground_truth = _extract_diagnosis_from_output(item["output"])
        else:
            ground_truth = ""

        # Build prompt — if instruction already embeds the system prompt,
        # use it as-is to avoid duplication
        if instruction.startswith("你是呼吸内科") or instruction.startswith("你是一名"):
            # Instruction already contains system prompt, use directly as user message
            user_content = f"{instruction}\n{input_text}"
            messages = [{"role": "user", "content": user_content}]
        else:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"{instruction}\n{input_text}"},
            ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        dataset.append({"prompt": prompt, "ground_truth": ground_truth})

    print(f"[Data] Prepared {len(dataset)} training samples")
    print(f"[Data] Sample prompt length: {len(dataset[0]['prompt'])} chars")
    print(f"[Data] Sample ground_truth: {dataset[0]['ground_truth']}")
    return dataset


# ═══════════════════════════════════════════════════════════════
# Model Loading
# ═══════════════════════════════════════════════════════════════

def load_model_and_tokenizer(model_path: str):
    """Load model and tokenizer."""
    print(f"[Model] Loading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"[Model] Loading model from {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    total_p = sum(p.numel() for p in model.parameters())
    print(f"[Model] Total params: {total_p/1e9:.2f}B")
    return model, tokenizer


# ═══════════════════════════════════════════════════════════════
# Reward Wrapper
# ═══════════════════════════════════════════════════════════════

def create_reward_fn(kg_reward: KGRewardFunction):
    def reward_func(prompts, completions, ground_truth=None, **kwargs):
        if ground_truth is None:
            ground_truth = [""] * len(completions)
        return kg_reward(prompts, completions, ground_truth)
    return reward_func


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    # Setup NPU accelerator
    if args.device_ids:
        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = args.device_ids
    print(f"[Setup] ASCEND_RT_VISIBLE_DEVICES={os.environ.get('ASCEND_RT_VISIBLE_DEVICES', '<all visible>')}")
    print(f"[Setup] NPUs available: {torch.npu.device_count()}")

    # Load Model
    model, tokenizer = load_model_and_tokenizer(args.model_name_or_path)

    # Load KG Reward Function
    kg_reward = KGRewardFunction(
        nodes_path=args.kg_nodes_path,
        edges_path=args.kg_edges_path,
        lambda_weights=(args.lambda_dx, args.lambda_graph, args.lambda_path),
    )
    reward_fn = create_reward_fn(kg_reward)

    # Prepare Dataset
    train_dataset = prepare_dataset(
        args.data_path, args.num_samples, tokenizer, args.system_prompt,
    )

    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"grpo_run_{timestamp}")

    # GRPO Training Config
    training_args = GRPOConfig(
        output_dir=output_dir,
        overwrite_output_dir=True,
        # Training
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        # GRPO-specific
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        temperature=1.0,
        beta=args.beta,
        epsilon=args.epsilon_clip,
        scale_rewards=True,
        # DeepSpeed
        deepspeed=args.deepspeed,
        # Logging
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        logging_first_step=True,
        # Saving
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        # Precision
        bf16=args.bf16,
        fp16=False,
        # Misc
        seed=42,
        report_to="none",
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        max_grad_norm=1.0,
        # Data loading
        dataloader_drop_last=True,
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )

    print(f"\n[GRPO Config]")
    print(f"  Model: {args.model_name_or_path}")
    print(f"  Samples: {args.num_samples}")
    print(f"  num_generations (G): {args.num_generations}")
    print(f"  learning_rate: {args.learning_rate}")
    print(f"  beta (KL): {args.beta}")
    print(f"  epsilon (clip): {args.epsilon_clip}")
    print(f"  lambda_weights: ({args.lambda_dx}, {args.lambda_graph}, {args.lambda_path})")
    print(f"  batch_size × accum: {args.per_device_train_batch_size} × {args.gradient_accumulation_steps}")
    print(f"  epochs: {args.num_train_epochs}")
    print(f"  DeepSpeed: {args.deepspeed}")
    print(f"  output_dir: {output_dir}")

    # Metrics logger callback
    metrics_callback = MetricsLoggerCallback(output_dir)

    # Create Trainer
    print(f"\n[GRPO] Creating trainer...")
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[reward_fn],
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        callbacks=[metrics_callback],
    )

    # Train
    print(f"\n[GRPO] Starting training...\n")
    trainer.train()

    # Save
    final_dir = os.path.join(output_dir, "final_model")
    print(f"\n[GRPO] Saving final model to {final_dir}...")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)

    # Plot training curves
    metrics_file = os.path.join(output_dir, "training_metrics.jsonl")
    if os.path.exists(metrics_file):
        print(f"\n[GRPO] Generating training curves...")
        plot_training_curves(metrics_file, output_dir)

    print(f"\n[GRPO] Done! Model saved to {final_dir}")


if __name__ == "__main__":
    main()
