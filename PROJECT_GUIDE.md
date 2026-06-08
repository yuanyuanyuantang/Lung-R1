# Lung-R1 Project Guide

## 1. Overview

`Lung-R1` is an open-source research project for pulmonary disease diagnosis, covering the full pipeline from knowledge graph construction, CoT QA generation, SFT fine-tuning, to GRPO reinforcement learning.

Three core tasks:

1. Build pulmonary EMR diagnosis fine-tuning data
2. Generate medical knowledge QA (Graph-CoT) from fused knowledge graph
3. Evaluate and compare models on our self-designed LungBench pulmonary benchmark

See [README.md](README.md) for the English project overview.

## 2. Top-Level Directory Structure

```text
lung-R1/
├── README.md                  # English project overview
├── PROJECT_GUIDE.md            # This document (detailed guide)
├── LICENSE                    # MIT License
├── .gitignore
├── requirements.txt           # Graph-CoT dependencies
├── SFT/                       # EMR diagnosis SFT training data & configs
├── grpo/                      # GRPO reinforcement learning training
├── graphcot/                  # Long-tail KG CoT QA generation
├── LungKG/                    # Knowledge graph construction (KG + guidelines + fusion)
└── evaluation/                # Pulmonary benchmark & evaluation scripts
```

## 3. Directory Details

### 3.1 `SFT/`

LLaMA-Factory SFT training data and configuration files.

Key files:

- `with_cot/llama_factory_emr_kg_train.json` — KG+EMR mixed training data with CoT reasoning (available upon request)
- `without_cot/llama_factory_emr_kg_train_no_cot.json` — Version without CoT (available upon request)
- `configs/` — LLaMA-Factory environment configs:
  - `full_ft_qwen25_7b.env` — 7B full fine-tuning config
  - `full_ft_qwen25_14b.env` — 14B full fine-tuning config

### 3.2 `grpo/`

KG-guided GRPO reinforcement learning, optimizing diagnostic ability after SFT.

Key files:

- `grpo_train.py` — GRPO training main script (NPU-adapted)
- `kg_reward.py` — Three-component KG reward function (diagnosis correctness + graph faithfulness + relation consistency)
- `metrics_logger.py` — Training metrics logging and plotting
- `config.py` — Global configuration
- `run_npu.sh` — NPU launch script
- `data/emr.json` — Training data (available upon request)
- `kg_data/` — KG files (symlinks to LungKG output)

See `grpo/README.md` for details.

### 3.3 `graphcot/`

KG-based CoT QA generation engine. Core algorithm:

1. **Long-tail subgraph sampling** — Inverse-degree weighted sampling + BFS expansion, prioritizing sparse entities in the KG
2. **Multi-hop CoT generation** — Forces LLM to reason from graph evidence, outputs structured QA (evidence, thought, answer)

Key files:

- `main.py` — Entry point
- `graphcot.py` — `LongTailPartitioner` + `CoTGenerator`
- `llm_client.py` — Async LLM API client
- `storage.py` — NetworkX graph storage

See `graphcot/README.md` for details.

### 3.4 `LungKG/`

Knowledge graph construction pipeline, fusing two data sources:

- **Medical KG** (structured entities and relations) → `LungKGpart1/`
- **Clinical guidelines** (nano-graphrag extraction) → `nano-graphrag/`
- **Fusion result** → `LungKG_fusion/output/nodes.csv` + `edges.csv` (59,038 entities / 164,309 relations)

Fusion scripts under `LungKG_fusion/` (step1–step6), entry point `run_all.py`.

### 3.5 `evaluation/`

Self-designed pulmonary benchmark evaluation.

Key files:

- `choice_eval/` — 250-question self-constructed benchmark + 10+ model scores
- `emr_eval/` — EMR diagnosis evaluation
- `kgqa_eval/` — KGQA evaluation

### 3.6 Clinical Data Processing

Clinical EMR data processing artifacts (data cleaning → label extraction → CoT generation → fine-tuning format) are not included in this release due to privacy considerations.

## 4. Recommended Data Flows

### 4.1 SFT → GRPO Training Flow

1. SFT training (LLaMA-Factory): use env configs under `SFT/configs/`
2. GRPO training: `cd grpo && bash run_npu.sh --7b` (or `--14b`)

### 4.2 KG QA Data Flow

1. Prepare fused KG `LungKG/LungKG_fusion/output/nodes.csv` and `edges.csv`
2. Run `cd graphcot && python main.py --kg_dir ../LungKG/LungKG_fusion/output --output_dir output`
3. QA data output to `output/generated_qa.json`

### 4.3 Benchmark Evaluation Flow

```bash
cd evaluation/choice_eval
python benchmark_eval.py
```

## 5. Key Files

- `SFT/with_cot/llama_factory_emr_kg_train.json` — SFT training data (available upon request)
- `grpo/data/emr.json` — GRPO training data (available upon request)
- `LungKG/LungKG_fusion/output/nodes.csv` — KG entities
- `LungKG/LungKG_fusion/output/edges.csv` — KG relations
- `evaluation/choice_eval/questions.json` — Benchmark questions
- `evaluation/choice_eval/answers.json` — Benchmark answers

## 6. Current Status

- [x] Directory/file naming in English
- [x] API key sanitized
- [x] Absolute paths → relative paths
- [x] NPU-only training adaptation
- [x] .gitignore / LICENSE / README.md
- [x] KG data symlinks
- [x] Model weights uploaded to ModelScope (coming soon)
