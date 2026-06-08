# Lung-R1 KG-Guided GRPO Reinforcement Learning

KG-guided GRPO (Group Relative Policy Optimization) reinforcement learning training for optimizing the pulmonary diagnosis model Lung-R1.

## Method Overview

After the SFT phase, KG-guided reinforcement learning is used to further optimize the model, aligning its outputs with both **diagnostic correctness** and **reasoning processes supported by LungKG**.

### Reward Function

The reward function consists of a weighted combination of three components:

$$R = \lambda_1 \cdot R_{\text{dx}} + \lambda_2 \cdot R_{\text{graph}} + \lambda_3 \cdot R_{\text{path}}$$

| Component | Weight | Description |
|------|------|------|
| $R_{\text{dx}}$ | $\lambda_1=0.50$ | **Diagnostic Correctness Reward**: Extracts diagnostic labels from model output and computes F1 score against ground truth, supports partial credit for KG-neighboring entities |
| $R_{\text{graph}}$ | $\lambda_2=0.25$ | **Graph Faithfulness Reward**: Extracts medical entities from the response, checks whether entity pairs have connecting edges in the KG, and evaluates whether reasoning is grounded in KG evidence |
| $R_{\text{path}}$ | $\lambda_3=0.25$ | **Relation Consistency Reward**: Extracts entity-relation-entity triplets from the response and verifies whether relation types and entity pairs match valid relations in the KG |

### GRPO Algorithm

Using the TRL framework's `GRPOTrainer`, the core mechanisms are:

- **Group Sampling**: Generate $G=4$ candidate responses per prompt
- **Group-normalized advantages**: $\hat{A}_i = \frac{R_i - \mu_R}{\sigma_R + \epsilon}$
- **Clipped objective**: $\min(r_i \hat{A}_i, \operatorname{clip}(r_i, 1-\epsilon, 1+\epsilon)\hat{A}_i)$
- **KL Regularization**: Controls the divergence from the reference model via the $\beta$ parameter

## Directory Structure

```
grpo/
├── README.md                      # This document
├── config.py                      # Configuration file (paths, hyperparameters, reward weights)
├── grpo_train.py                  # GRPO training main script
├── kg_reward.py                   # KG-guided reward function implementation
├── metrics_logger.py              # Training metrics logging and curve plotting
├── run_npu.sh                     # NPU one-click launch script
├── requirements.txt               # Python dependencies
├── deepspeed_zero3.json           # DeepSpeed ZeRO-3 configuration
├── .gitignore
├── data/                           # Training data (available upon request)
└── kg_data/
    ├── nodes.csv                  # Knowledge graph entities (59,038)
    └── edges.csv                  # Knowledge graph relations (164,309)
```

## Quick Start

### Requirements

- Python 3.10+
- Ascend NPU + CANN environment
- NPU: 7B model requires 4x Ascend, 14B model requires 8x Ascend

### Installation

```bash
cd grpo/
pip install -r requirements.txt
```

### Test Run (Qwen2.5-0.5B)

Verify that the environment and pipeline are working correctly:

```bash
bash run_npu.sh --test
```

Configuration: 0.5B model, 8 samples, 1 epoch, single NPU, completes in approximately 10 minutes.

### Full Training

**Qwen2.5-7B (Recommended)**:

```bash
bash run_npu.sh --7b
```

Configuration: 7B model, 3,569 samples, 1 epoch, DeepSpeed ZeRO-3 (optional), 4x Ascend NPU.

**Qwen2.5-14B**:

```bash
bash run_npu.sh --14b
```

Configuration: 14B model, 3,569 samples, 1 epoch, DeepSpeed ZeRO-3 (optional), 8x Ascend NPU.

### Custom Training

```bash
python grpo_train.py \
    --model_name_or_path /path/to/model \
    --num_samples 3569 \
    --num_train_epochs 1 \
    --learning_rate 5e-6 \
    --beta 0.04 \
    --deepspeed deepspeed_zero3.json
```

### Model Evaluation

Use the LungBench evaluation script to assess the GRPO model:

```bash
cd ../evaluation/choice_eval

# Evaluate local model
python benchmark_eval.py --model_name /path/to/trained-model --base_url http://localhost:8080/v1

# Evaluate HuggingFace model
python benchmark_eval.py --model_name Qwen/Qwen2.5-7B-Instruct --api_key your-api-key

# Quick test with a small sample
python benchmark_eval.py --model_name /path/to/model --sample_size 50
```

See the evaluation code under the `../evaluation/` directory.

## Configuration Reference

Key parameters in `config.py`:

### Paths

| Parameter | Description | Default |
|------|------|--------|
| `MODEL_PATH` | HuggingFace model ID or local path | `Qwen/Qwen2.5-7B-Instruct` |
| `DATA_PATH` | Training data path | `data/emr.json` (available upon request) |
| `KG_NODES_PATH` | KG entity file | `kg_data/nodes.csv` |
| `KG_EDGES_PATH` | KG relation file | `kg_data/edges.csv` |

### GRPO Hyperparameters

| Parameter | Description | Default |
|------|------|--------|
| `NUM_GENERATIONS` | Number of candidates per prompt G | 4 |
| `LEARNING_RATE` | Learning rate | 5e-6 |
| `BETA` | KL penalty coefficient | 0.04 |
| `EPSILON_CLIP` | PPO clipping range | 0.2 |
| `MAX_COMPLETION_LENGTH` | Max tokens for generated response | 512 |
| `MAX_PROMPT_LENGTH` | Max tokens for input prompt | 2560 |
| `NUM_TRAIN_EPOCHS` | Number of training epochs | 1 |

### Reward Weights

| Parameter | Description | Default |
|------|------|--------|
| `LAMBDA_DX` | Diagnostic correctness weight | 0.50 |
| `LAMBDA_GRAPH` | Graph faithfulness weight | 0.25 |
| `LAMBDA_PATH` | Relation consistency weight | 0.25 |

### NPU Settings

| Parameter | Description | Default |
|------|------|--------|
| `NPU_DEVICE_IDS` | NPU device IDs used | `"0,1,2,3"` |
| `PER_DEVICE_BATCH_SIZE` | Batch size per NPU | 4 |
| `GRADIENT_ACCUMULATION_STEPS` | Gradient accumulation steps | 2 |

## Training Data Format

EMR training data is available upon request to the authors. Each sample contains:

```json
{
    "instruction": "You are a pulmonary medicine diagnostic assistant model. ...\nPlease read the following medical record and complete 'Diagnostic Reasoning -> Pulmonary Disease Diagnosis'.",
    "input": "Medical Record:\nChief Complaint: ...\nHistory of Present Illness: ...",
    "output": "Reasoning: ...Final Answer: [Pulmonary Infection]"
}
```

During training, `instruction + input` is used to construct the prompt, and diagnostic labels are extracted from the `"Final Answer: [...]"` in `output` for reward calculation.

## Training Monitoring and Visualization

Metrics are automatically logged to `output/grpo_run_xxx/training_metrics.jsonl` during training, and curve plots are automatically generated after training.

### Auto-Generated Charts

After training, the following are automatically generated in the output directory:

| File | Content |
|------|------|
| `training_curves.png` | 6-in-1 combined chart: Reward, Loss, KL Divergence, Generation Length, Reward Std+Grad Norm, Clipped Ratio |
| `reward_components.png` | Reward function component trends |
| `training_metrics.jsonl` | Complete metrics data per step (can be analyzed with other tools) |

### Manual Plotting

```bash
# Generate charts from existing metrics file
python metrics_logger.py --metrics output/grpo_run_xxx/training_metrics.jsonl

# Specify output directory
python metrics_logger.py --metrics output/grpo_run_xxx/training_metrics.jsonl --output ./my_plots
```

### Real-Time Monitoring (Optional)

To view training curves in real time, set `report_to` to `"wandb"` or `"tensorboard"` in `config.py`.

## Knowledge Graph

Fused from two sources:

- **Medical Knowledge Graph**: Structured medical knowledge including diseases, symptoms, medications, examinations, etc.
- **Clinical Guidelines**: Entities and relations extracted from pulmonary medicine guideline documents via nano-graphrag

Post-fusion KG scale:
- 59,038 entities, 15 unified types (disease, symptom, Western medicine, examination, etc.)
- 164,308 relations, 40+ relation types (clinical manifestation, treatment, diagnostic basis, complication, etc.)

## Output

After training completes, the output directory structure:

```
output/grpo_run_YYYYMMDD_HHMMSS/
├── checkpoint-100/            # Intermediate checkpoint
├── checkpoint-200/
├── final_model/               # Final model (ready for direct loading and inference)
│   ├── config.json
│   ├── model.safetensors
│   └── tokenizer files...
├── training_metrics.jsonl     # Training metrics log (one entry per step)
├── training_curves.png        # 6-in-1 training curve chart
├── reward_components.png      # Reward component trend chart
└── trainer_state.json         # Training state
```

## References

This implementation is based on the following methods:

- GRPO Algorithm: Guo et al., "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning", 2025
- KG-Guided Reward: Three-component reward function design based on the LungKG knowledge graph
- TRL Framework: HuggingFace TRL `GRPOTrainer`

## File Checklist

| File | Purpose | Required |
|------|------|----------|
| `config.py` | Global configuration | Required |
| `grpo_train.py` | Training entry point | Required |
| `kg_reward.py` | Reward function implementation | Required |
| `metrics_logger.py` | Metrics logging and curve plotting | Required |
| `run_npu.sh` | NPU one-click launch | Optional |
| `requirements.txt` | Dependency list | Required |
| `deepspeed_zero3.json` | Distributed training configuration | Required for multi-GPU |
| `data/*.json` | Training data | Required |
| `kg_data/*.csv` | Knowledge graph | Required |
