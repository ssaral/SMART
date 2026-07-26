Once all four smoke tests passes we can start running actual training. The full and LoRA branches both produced finite training/evaluation losses, saved the expected artifacts, and exercised four-GPU distributed training successfully.

The memory measurements establish:

| Run type    | Peak GPU memory | Placement                 |
| ----------- | --------------: | ------------------------- |
| Llama2 full |         65.9 GB | H100, with limited margin |
| Llama2 LoRA |         17.1 GB | H100                      |
| Qwen2 full  |         81.6 GB | **H200 required**         |
| Qwen2 LoRA  |         27.8 GB | H100                      |

## Critical scheduler correction

There is one released-code issue that matters for production.

The original trainer constructs the scheduler **before** Accelerate shards the DataLoader across GPUs. It then recalculates `max_train_steps` after sharding but does not reconstruct the scheduler. If `--max_train_steps` is omitted on four GPUs, the cosine scheduler is configured for roughly four times too many steps.

We must explicitly provide:

```text
SMART-25K: ceil(25,000 / 64) = 391 optimizer steps
SMART-50K: ceil(50,000 / 64) = 782 optimizer steps
```

That gives:

```text
25K warmup: floor(391 × 0.01) = 3 steps
50K warmup: floor(782 × 0.01) = 7 steps
```

This implements the paper’s intended one epoch, global batch size 64, cosine schedule, and 1% warmup rather than reproducing an accidental scheduler bug in the released code. The other full-FT settings remain the paper settings: learning rate `2e-5`, weight decay `0.1`, and Flash Attention. 

# Step 18 — First production training run

Create a reusable single-experiment launcher. It supports all eight planned runs but we will first launch only **Llama2 full FT on SMART-25K**.

## 18.1 Create the launcher

```bash
cd /data/saral/wdir/smart || exit 1

cat > run_one_production_training.sh <<'BASH'
#!/usr/bin/env bash

set -euo pipefail

usage() {
    echo \
      "Usage: $0 MODEL_LABEL MODEL_PATH BUDGET MODE"
    echo
    echo \
      "  BUDGET: 25000 or 50000"
    echo \
      "  MODE:   full or lora"
    echo
    echo \
      "Example:"
    echo \
      "  $0 llama2_7b /data/saral/wdir/smart/llama2_7b 25000 full"
}

if [[ "$#" -ne 4 ]]; then
    usage
    exit 2
fi

MODEL_LABEL="$1"
MODEL_PATH="$2"
BUDGET="$3"
MODE="$4"

PROJECT=/data/saral/wdir/smart
ROOT=/mnt/warm_storage/saral/smart

DATA_ROOT="$ROOT/datasets/trainer_eval_safe"
OUTPUT_ROOT="$ROOT/models"
LOG_ROOT="$ROOT/logs/production_training"
CACHE_ROOT="$ROOT/cache/hf_training"

ACCELERATE_CONFIG="$PROJECT/accelerate_4gpu_bf16.yaml"
TRAINER="$PROJECT/instruction_tuner_local.py"

SEED=23
GLOBAL_BATCH_SIZE=64
PER_DEVICE_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=16
GPU_COUNT=4
LEARNING_RATE=2e-5

case "$BUDGET" in
    25000)
        DATASET="$DATA_ROOT/smart_25000"
        MAX_TRAIN_STEPS=391
        EXPECTED_WARMUP_STEPS=3
        ;;
    50000)
        DATASET="$DATA_ROOT/smart_50000"
        MAX_TRAIN_STEPS=782
        EXPECTED_WARMUP_STEPS=7
        ;;
    *)
        echo "Unsupported budget: $BUDGET" >&2
        usage
        exit 2
        ;;
esac

case "$MODE" in
    full|lora)
        ;;
    *)
        echo "Unsupported mode: $MODE" >&2
        usage
        exit 2
        ;;
esac

for path in \
    "$MODEL_PATH" \
    "$DATASET" \
    "$ACCELERATE_CONFIG" \
    "$TRAINER"
do
    if [[ ! -e "$path" ]]; then
        echo "Required path does not exist: $path" >&2
        exit 1
    fi
done

RUN_NAME="${MODEL_LABEL}_smart_${BUDGET}_${MODE}_seed${SEED}"
OUTPUT_DIR="$OUTPUT_ROOT/$RUN_NAME"
RUN_LOG_DIR="$LOG_ROOT/$RUN_NAME"

if [[ -e "$OUTPUT_DIR" ]]; then
    echo "Output already exists: $OUTPUT_DIR" >&2
    echo "Refusing to overwrite a production run." >&2
    exit 1
fi

if [[ -e "$RUN_LOG_DIR" ]]; then
    echo "Log directory already exists: $RUN_LOG_DIR" >&2
    echo "Refusing to overwrite a production run." >&2
    exit 1
fi

mkdir -p \
    "$OUTPUT_DIR" \
    "$RUN_LOG_DIR" \
    "$CACHE_ROOT"

export CUDA_VISIBLE_DEVICES=0,1,2,3

export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Running production preflight..."

python3 - "$DATASET" "$BUDGET" <<'PY'
import sys

import torch
from datasets import load_from_disk

dataset_path = sys.argv[1]
expected_train = int(sys.argv[2])

dataset = load_from_disk(dataset_path)

assert len(dataset["train"]) == expected_train
assert set(dataset) == {"train", "validation"}
assert dataset["train"].column_names == [
    "prompt",
    "response",
]
assert dataset["validation"].column_names == [
    "prompt",
    "response",
]

assert torch.cuda.is_available()
assert torch.cuda.device_count() == 4
assert torch.cuda.is_bf16_supported()

print("Dataset:", dataset_path)
print("Train rows:", len(dataset["train"]))
print(
    "Validation rows:",
    len(dataset["validation"]),
)

for index in range(torch.cuda.device_count()):
    free_bytes, total_bytes = torch.cuda.mem_get_info(index)

    print(
        f"GPU {index}: "
        f"{torch.cuda.get_device_name(index)}, "
        f"free={free_bytes / 2**30:.2f} GiB, "
        f"total={total_bytes / 2**30:.2f} GiB"
    )

try:
    import flash_attn
except Exception as exc:
    raise RuntimeError(
        "flash_attn is unavailable."
    ) from exc

print(
    "flash_attn:",
    getattr(
        flash_attn,
        "__version__",
        "installed",
    ),
)

print("Production preflight passed.")
PY

MIN_FREE_MIB=$(
    nvidia-smi \
      --query-gpu=memory.free \
      --format=csv,noheader,nounits |
    head -n 4 |
    sort -n |
    head -n 1
)

case "${MODEL_LABEL}:${MODE}" in
    qwen2_7b:full)
        REQUIRED_FREE_MIB=90000
        ;;
    llama2_7b:full)
        REQUIRED_FREE_MIB=68000
        ;;
    qwen2_7b:lora)
        REQUIRED_FREE_MIB=34000
        ;;
    llama2_7b:lora)
        REQUIRED_FREE_MIB=24000
        ;;
    *)
        # Unknown labels are allowed, but receive the
        # conservative full-training requirement.
        if [[ "$MODE" == "full" ]]; then
            REQUIRED_FREE_MIB=90000
        else
            REQUIRED_FREE_MIB=34000
        fi
        ;;
esac

echo "Minimum free GPU memory: ${MIN_FREE_MIB} MiB"
echo "Required free GPU memory: ${REQUIRED_FREE_MIB} MiB"

if (( MIN_FREE_MIB < REQUIRED_FREE_MIB )); then
    echo "Insufficient free GPU memory for this run." >&2
    exit 1
fi

AVAILABLE_DISK_KIB=$(
    df -Pk "$OUTPUT_ROOT" |
    awk 'NR==2 {print $4}'
)

if [[ "$MODE" == "full" ]]; then
    REQUIRED_DISK_KIB=$((40 * 1024 * 1024))
else
    REQUIRED_DISK_KIB=$((5 * 1024 * 1024))
fi

if (( AVAILABLE_DISK_KIB < REQUIRED_DISK_KIB )); then
    echo "Insufficient output disk space." >&2
    echo \
      "Available: $((AVAILABLE_DISK_KIB / 1024 / 1024)) GiB" >&2
    echo \
      "Required:  $((REQUIRED_DISK_KIB / 1024 / 1024)) GiB" >&2
    exit 1
fi

MODE_ARGS=()

if [[ "$MODE" == "lora" ]]; then
    MODE_ARGS=(
        --use_peft
        --peft_lora_r 64
        --peft_lora_alpha 16
        --peft_lora_dropout 0.05
        --peft_target_modules \
          q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,lm_head
    )
fi

CMD=(
    accelerate launch
    --config_file "$ACCELERATE_CONFIG"
    "$TRAINER"

    --load_data_from_disk
    --dataset_name_or_path "$DATASET"

    --model_name_or_path "$MODEL_PATH"
    --trust_remote_code
    --local_files_only
    --use_flash_attention_2
    --torch_dtype bfloat16
    --max_seq_length 4096

    --learning_rate "$LEARNING_RATE"
    --per_device_train_batch_size "$PER_DEVICE_BATCH_SIZE"
    --per_device_eval_batch_size 1
    --preprocessing_num_workers 16
    --dataloader_num_workers 8

    --seed "$SEED"
    --num_train_epochs 1
    --max_train_steps "$MAX_TRAIN_STEPS"
    --gradient_accumulation_steps \
      "$GRADIENT_ACCUMULATION_STEPS"
    --gradient_checkpointing

    --weight_decay 0.1
    --lr_scheduler_type cosine
    --lr_warmup_fraction 0.01

    --logging_steps 10
    --cache_dir "$CACHE_ROOT"
    --output_dir "$OUTPUT_DIR"

    "${MODE_ARGS[@]}"
)

{
    echo "run_name=$RUN_NAME"
    echo "model_label=$MODEL_LABEL"
    echo "model_path=$MODEL_PATH"
    echo "budget=$BUDGET"
    echo "mode=$MODE"
    echo "dataset=$DATASET"
    echo "seed=$SEED"
    echo "gpu_count=$GPU_COUNT"
    echo "per_device_batch_size=$PER_DEVICE_BATCH_SIZE"
    echo \
      "gradient_accumulation_steps=$GRADIENT_ACCUMULATION_STEPS"
    echo "global_batch_size=$GLOBAL_BATCH_SIZE"
    echo "max_train_steps=$MAX_TRAIN_STEPS"
    echo \
      "expected_warmup_steps=$EXPECTED_WARMUP_STEPS"
    echo "learning_rate=$LEARNING_RATE"
    echo
    printf 'command='
    printf '%q ' "${CMD[@]}"
    printf '\n'
} | tee "$RUN_LOG_DIR/run_config.txt"

python3 - <<PY > "$RUN_LOG_DIR/environment.json"
import json
import platform

import accelerate
import datasets
import peft
import torch
import transformers

payload = {
    "platform": platform.platform(),
    "python": platform.python_version(),
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "transformers": transformers.__version__,
    "accelerate": accelerate.__version__,
    "datasets": datasets.__version__,
    "peft": peft.__version__,
    "gpu_names": [
        torch.cuda.get_device_name(index)
        for index in range(
            torch.cuda.device_count()
        )
    ],
}

print(json.dumps(payload, indent=2))
PY

sha256sum \
    "$TRAINER" \
    "$ACCELERATE_CONFIG" \
    > "$RUN_LOG_DIR/code_sha256.txt"

/usr/bin/time -v \
    -o "$RUN_LOG_DIR/time.txt" \
    "${CMD[@]}" \
    2>&1 |
    tee "$RUN_LOG_DIR/training.log"

python3 - \
    "$OUTPUT_DIR" \
    "$MAX_TRAIN_STEPS" \
    "$MODE" <<'PY'
import json
import math
import sys
from pathlib import Path

output = Path(sys.argv[1])
expected_steps = int(sys.argv[2])
mode = sys.argv[3]

result_path = output / "all_results.json"

if not result_path.is_file():
    raise FileNotFoundError(result_path)

result = json.loads(
    result_path.read_text(
        encoding="utf-8"
    )
)

assert result["completed_steps"] == expected_steps
assert result["max_train_steps"] == expected_steps
assert math.isfinite(float(result["eval_loss"]))
assert math.isfinite(float(result["perplexity"]))

if mode == "lora":
    required = [
        output / "adapter_config.json",
    ]

    if not any(
        path.is_file()
        for path in [
            output / "adapter_model.safetensors",
            output / "adapter_model.bin",
        ]
    ):
        raise FileNotFoundError(
            "No saved LoRA adapter weights."
        )
else:
    required = [
        output / "config.json",
    ]

    model_files = (
        list(output.glob("model*.safetensors"))
        + list(output.glob("pytorch_model*.bin"))
    )

    if not model_files:
        raise FileNotFoundError(
            "No saved full-model weight files."
        )

for path in required:
    if not path.is_file():
        raise FileNotFoundError(path)

print("Production run verification passed.")
print("Output:", output)
print("Completed steps:", result["completed_steps"])
print("Evaluation loss:", result["eval_loss"])
print("Perplexity:", result["perplexity"])
PY

echo
echo "Production run completed successfully:"
echo "$RUN_NAME"
BASH

chmod +x run_one_production_training.sh

bash -n run_one_production_training.sh
```

## 18.2 Launch the primary reproduction run

Run this on the H100 system with no other GPU jobs:

```bash
cd /data/saral/wdir/smart || exit 1

./run_one_production_training.sh \
  llama2_7b \
  /data/saral/wdir/smart/llama2_7b \
  25000 \
  full
```

This run is the most important initial result:

```text
Model:          Llama2-7B
Mixture:        SMART-25K
Training:       full parameter
Epochs:         1
Global batch:   64
Optimizer steps: 391
Learning rate:  2e-5
Warmup steps:   3
Scheduler:      cosine
Weight decay:   0.1
Seed:           23
```

The expected final files are:

```text
/mnt/warm_storage/saral/smart/models/
└── llama2_7b_smart_25000_full_seed23/
    ├── config.json
    ├── model-*.safetensors
    ├── tokenizer*
    └── all_results.json
```

Logs and provenance are saved separately under:

```text
/mnt/warm_storage/saral/smart/logs/production_training/
└── llama2_7b_smart_25000_full_seed23/
```

## Remaining experiment placement

After this first production run passes:

**H100 or H200**

```text
Llama2 25K full
Llama2 50K full
Llama2 25K LoRA
Llama2 50K LoRA
Qwen2 25K LoRA
Qwen2 50K LoRA
```

**H200 only**

```text
Qwen2 25K full
Qwen2 50K full
```

We will not launch the remaining seven until the Llama2-25K full output passes the launcher’s final verification.
