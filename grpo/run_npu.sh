#!/bin/bash
# Ascend NPU launcher for KG-Guided GRPO training.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODE="test"
INSTALL_DEPS="true"
USE_DEEPSPEED="false"
NPU_IDS="${GRPO_NPU:-}"
ATB_CXX_ABI="${GRPO_ATB_CXX_ABI:-1}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --test) MODE="test"; shift ;;
        --7b) MODE="7b"; shift ;;
        --14b) MODE="14b"; shift ;;
        --npu|--npu-ids)
            if [[ $# -lt 2 ]]; then
                echo "$1 requires a comma-separated NPU id list, e.g. 0,1,2,3"
                exit 1
            fi
            NPU_IDS="$2"
            shift 2
            ;;
        --install) INSTALL_DEPS="true"; shift ;;
        --no-install) INSTALL_DEPS="false"; shift ;;
        --deepspeed) USE_DEEPSPEED="true"; shift ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
done

if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
    # shellcheck disable=SC1091
    set +u
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    set -u
fi

if [ -f /usr/local/Ascend/nnal/atb/set_env.sh ]; then
    # shellcheck disable=SC1091
    set +u
    source /usr/local/Ascend/nnal/atb/set_env.sh --cxx_abi="$ATB_CXX_ABI"
    set -u
fi

if [ -z "$NPU_IDS" ]; then
    case "$MODE" in
        7b) NPU_IDS="0,1,2,3" ;;
        14b) NPU_IDS="0,1,2,3,4,5,6,7" ;;
    esac
fi

if [ -n "$NPU_IDS" ]; then
    export GRPO_NPU="$NPU_IDS"
    export GRPO_DEVICE_IDS="$NPU_IDS"
    export ASCEND_RT_VISIBLE_DEVICES="$NPU_IDS"
fi

export GRPO_USE_NPU=1
export TOKENIZERS_PARALLELISM=false

echo "========================================"
echo "  Lung-R1 KG-Guided GRPO Training (NPU)"
echo "  Directory: $SCRIPT_DIR"
echo "========================================"

NPU_COUNT=$(python -c "import torch, torch_npu; print(torch.npu.device_count())" 2>/dev/null || echo 0)
echo "Detected $NPU_COUNT NPU(s)"
echo "ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-<all visible>}"

if [ "$NPU_COUNT" -lt 1 ]; then
    echo "No visible NPU found. Check npu-smi and CANN environment variables."
    exit 1
fi

if [ "$INSTALL_DEPS" = "true" ] && ! python - <<'PY' 2>/dev/null
import importlib.metadata as metadata
import sys
from packaging.version import Version

required = {
    "transformers": "4.51.0",
    "trl": "0.18.2",
    "accelerate": "1.13.0",
}

for package, minimum in required.items():
    try:
        installed = Version(metadata.version(package))
    except metadata.PackageNotFoundError:
        sys.exit(1)
    if installed < Version(minimum):
        sys.exit(1)
PY
then
    echo "Installing NPU project dependencies..."
    python -m pip install -r requirements-npu.txt
fi

case "$MODE" in
    test)
        echo ""
        echo ">>> MODE: Quick NPU Test (Qwen2.5-0.5B, 8 samples)"
        echo ""
        GRPO_MODEL="${GRPO_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
        GRPO_SAMPLES="${GRPO_SAMPLES:-8}"
        GRPO_EPOCHS="${GRPO_EPOCHS:-1}"
        GRPO_LR="${GRPO_LR:-5e-6}"
        GRPO_BETA="${GRPO_BETA:-0.02}"
        GRPO_MAX_PROMPT="${GRPO_MAX_PROMPT:-1536}"
        GRPO_MAX_COMPLETION="${GRPO_MAX_COMPLETION:-128}"
        GRPO_NUM_GENERATIONS="${GRPO_NUM_GENERATIONS:-4}"
        GRPO_BATCH="${GRPO_BATCH:-1}"
        GRPO_ACCUM="${GRPO_ACCUM:-4}"
        ;;
    7b)
        echo ""
        echo ">>> MODE: Qwen2.5-7B NPU Training"
        echo ""
        GRPO_MODEL="${GRPO_MODEL:-<your-sft-model-path>}"
        GRPO_SAMPLES="${GRPO_SAMPLES:-3569}"
        GRPO_EPOCHS="${GRPO_EPOCHS:-1}"
        GRPO_LR="${GRPO_LR:-5e-6}"
        GRPO_BETA="${GRPO_BETA:-0.04}"
        GRPO_MAX_PROMPT="${GRPO_MAX_PROMPT:-2048}"
        GRPO_MAX_COMPLETION="${GRPO_MAX_COMPLETION:-512}"
        GRPO_NUM_GENERATIONS="${GRPO_NUM_GENERATIONS:-4}"
        GRPO_BATCH="${GRPO_BATCH:-4}"
        GRPO_ACCUM="${GRPO_ACCUM:-4}"
        ;;
    14b)
        echo ""
        echo ">>> MODE: Qwen2.5-14B NPU Training (memory heavy; multi-NPU recommended)"
        echo ""
        GRPO_MODEL="${GRPO_MODEL:-<your-sft-model-path>}"
        GRPO_SAMPLES="${GRPO_SAMPLES:-3569}"
        GRPO_EPOCHS="${GRPO_EPOCHS:-1}"
        GRPO_LR="${GRPO_LR:-5e-6}"
        GRPO_BETA="${GRPO_BETA:-0.04}"
        GRPO_MAX_PROMPT="${GRPO_MAX_PROMPT:-2048}"
        GRPO_MAX_COMPLETION="${GRPO_MAX_COMPLETION:-512}"
        GRPO_NUM_GENERATIONS="${GRPO_NUM_GENERATIONS:-4}"
        GRPO_BATCH="${GRPO_BATCH:-4}"
        GRPO_ACCUM="${GRPO_ACCUM:-4}"
        ;;
esac

TRAIN_CMD=(python grpo_train.py
    --use_npu
    --model_name_or_path "$GRPO_MODEL"
    --num_samples "$GRPO_SAMPLES"
    --num_train_epochs "$GRPO_EPOCHS"
    --learning_rate "$GRPO_LR"
    --beta "$GRPO_BETA"
    --num_generations "$GRPO_NUM_GENERATIONS"
    --max_prompt_length "$GRPO_MAX_PROMPT"
    --max_completion_length "$GRPO_MAX_COMPLETION"
    --per_device_train_batch_size "$GRPO_BATCH"
    --gradient_accumulation_steps "$GRPO_ACCUM"
)

if [ -n "${GRPO_DEVICE_IDS:-}" ]; then
    TRAIN_CMD+=(--device_ids "$GRPO_DEVICE_IDS")
fi

if [ "$USE_DEEPSPEED" = "true" ]; then
    if [ "$NPU_COUNT" -lt 2 ]; then
        echo "DeepSpeed was requested, but fewer than 2 NPUs are visible."
        exit 1
    fi
    TRAIN_CMD+=(--deepspeed deepspeed_zero3.json)
fi

echo ""
printf 'Command:'
printf ' %q' "${TRAIN_CMD[@]}"
echo ""
echo ""

"${TRAIN_CMD[@]}"
