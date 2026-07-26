All remaining seven runs can be launched sequentially on the same four H200s. Each experiment will use all four GPUs, so do not parallelize them.

The remaining workload is:

```text
Llama2-7B  SMART-50K  full
Qwen2-7B   SMART-25K  full
Qwen2-7B   SMART-50K  full
Llama2-7B  SMART-25K  LoRA
Llama2-7B  SMART-50K  LoRA
Qwen2-7B   SMART-25K  LoRA
Qwen2-7B   SMART-50K  LoRA
```

That is 4,301 optimizer steps in total, plus final validation for each run.

## Create the batch launcher

```bash
cd /data/saral/wdir/smart || exit 1

cat > run_remaining_production_trainings.sh <<'BASH'
#!/usr/bin/env bash

set -euo pipefail

PROJECT=/data/saral/wdir/smart
ROOT=/mnt/warm_storage/saral/smart

LAUNCHER="$PROJECT/run_one_production_training.sh"
MODEL_ROOT="$ROOT/models"
LOG_ROOT="$ROOT/logs/production_training"

LLAMA_MODEL="$PROJECT/llama2_7b"
QWEN_MODEL="$PROJECT/qwen2_7b"

SEED=23
TIMESTAMP=$(date -u +%Y%m%dT%H%M%SZ)

BATCH_DIR="$LOG_ROOT/batch_remaining_${TIMESTAMP}"
MASTER_LOG="$BATCH_DIR/master.log"
STATUS_FILE="$BATCH_DIR/run_status.tsv"

mkdir -p \
  "$MODEL_ROOT" \
  "$LOG_ROOT" \
  "$BATCH_DIR"

exec > >(tee -a "$MASTER_LOG") 2>&1

CURRENT_RUN="preflight"

on_error() {
    local exit_code=$?

    echo
    echo "=================================================="
    echo "BATCH FAILED"
    echo "Run:       $CURRENT_RUN"
    echo "Exit code: $exit_code"
    echo "Time:      $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "Master log: $MASTER_LOG"
    echo "=================================================="

    exit "$exit_code"
}

trap on_error ERR
trap 'echo "Interrupted while running: $CURRENT_RUN"' INT TERM

if [[ ! -x "$LAUNCHER" ]]; then
    echo "Production launcher is missing or not executable:"
    echo "$LAUNCHER"
    exit 1
fi

bash -n "$LAUNCHER"

for path in "$LLAMA_MODEL" "$QWEN_MODEL"; do
    if [[ ! -d "$path" ]]; then
        echo "Model path does not exist: $path"
        exit 1
    fi
done

GPU_COUNT=$(
    nvidia-smi \
      --query-gpu=index \
      --format=csv,noheader |
    wc -l
)

if [[ "$GPU_COUNT" -lt 4 ]]; then
    echo "At least four visible GPUs are required."
    echo "Visible GPU count: $GPU_COUNT"
    exit 1
fi

echo "=== GPU inventory ==="

nvidia-smi \
  --query-gpu=index,name,memory.total,memory.free,utilization.gpu \
  --format=csv

echo

python3 - <<'PY'
import torch

assert torch.cuda.is_available()
assert torch.cuda.device_count() >= 4
assert torch.cuda.is_bf16_supported()

for index in range(4):
    properties = torch.cuda.get_device_properties(index)

    print(
        f"GPU {index}: "
        f"{properties.name}, "
        f"{properties.total_memory / 2**30:.2f} GiB"
    )

    if properties.total_memory < 120 * 2**30:
        raise RuntimeError(
            f"GPU {index} does not appear to be an "
            "H200-class device with sufficient memory."
        )

print("Four-H200 preflight passed.")
PY

AVAILABLE_DISK_KIB=$(
    df -Pk "$MODEL_ROOT" |
    awk 'NR==2 {print $4}'
)

REQUIRED_DISK_KIB=$((80 * 1024 * 1024))

echo
echo "Available model disk: $((AVAILABLE_DISK_KIB / 1024 / 1024)) GiB"
echo "Required model disk:  $((REQUIRED_DISK_KIB / 1024 / 1024)) GiB"

if (( AVAILABLE_DISK_KIB < REQUIRED_DISK_KIB )); then
    echo "Insufficient free disk for the remaining checkpoints."
    exit 1
fi

printf "run_name\tstatus\tstarted_utc\tfinished_utc\n" \
  > "$STATUS_FILE"


run_is_complete() {
    local run_name="$1"
    local expected_steps="$2"
    local mode="$3"
    local output_dir="$MODEL_ROOT/$run_name"

    python3 - \
      "$output_dir" \
      "$expected_steps" \
      "$mode" <<'PY'
import json
import math
import sys
from pathlib import Path

output_dir = Path(sys.argv[1])
expected_steps = int(sys.argv[2])
mode = sys.argv[3]

result_path = output_dir / "all_results.json"

if not result_path.is_file():
    raise SystemExit(1)

try:
    result = json.loads(
        result_path.read_text(encoding="utf-8")
    )

    if int(result["completed_steps"]) != expected_steps:
        raise RuntimeError("completed_steps mismatch")

    if int(result["max_train_steps"]) != expected_steps:
        raise RuntimeError("max_train_steps mismatch")

    if not math.isfinite(float(result["eval_loss"])):
        raise RuntimeError("non-finite eval loss")

    if not math.isfinite(float(result["perplexity"])):
        raise RuntimeError("non-finite perplexity")

    if mode == "full":
        if not (output_dir / "config.json").is_file():
            raise RuntimeError("missing config.json")

        weight_files = (
            list(output_dir.glob("model*.safetensors"))
            + list(output_dir.glob("pytorch_model*.bin"))
        )

        if not weight_files:
            raise RuntimeError("missing full-model weights")

    elif mode == "lora":
        if not (
            output_dir / "adapter_config.json"
        ).is_file():
            raise RuntimeError("missing adapter_config.json")

        adapter_files = [
            output_dir / "adapter_model.safetensors",
            output_dir / "adapter_model.bin",
        ]

        if not any(path.is_file() for path in adapter_files):
            raise RuntimeError("missing adapter weights")

    else:
        raise RuntimeError(f"unknown mode: {mode}")

except Exception:
    raise SystemExit(1)

raise SystemExit(0)
PY
}


run_experiment() {
    local model_label="$1"
    local model_path="$2"
    local budget="$3"
    local mode="$4"

    local expected_steps
    local run_name
    local output_dir
    local run_log_dir
    local started
    local finished

    case "$budget" in
        25000)
            expected_steps=391
            ;;
        50000)
            expected_steps=782
            ;;
        *)
            echo "Unsupported budget: $budget"
            return 2
            ;;
    esac

    run_name="${model_label}_smart_${budget}_${mode}_seed${SEED}"
    output_dir="$MODEL_ROOT/$run_name"
    run_log_dir="$LOG_ROOT/$run_name"

    CURRENT_RUN="$run_name"

    if run_is_complete \
        "$run_name" \
        "$expected_steps" \
        "$mode"
    then
        echo
        echo "=================================================="
        echo "SKIPPING COMPLETED RUN"
        echo "$run_name"
        echo "=================================================="

        printf "%s\t%s\t%s\t%s\n" \
          "$run_name" \
          "already_complete" \
          "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
          "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
          >> "$STATUS_FILE"

        return 0
    fi

    if [[ -e "$output_dir" || -e "$run_log_dir" ]]; then
        echo
        echo "Incomplete or invalid prior run detected:"
        echo "  Output: $output_dir"
        echo "  Logs:   $run_log_dir"
        echo
        echo "The batch will not delete production artifacts automatically."
        echo "Inspect or move those paths, then relaunch."
        return 1
    fi

    started=$(date -u +%Y-%m-%dT%H:%M:%SZ)

    printf "%s\t%s\t%s\t%s\n" \
      "$run_name" \
      "started" \
      "$started" \
      "-" \
      >> "$STATUS_FILE"

    echo
    echo "##################################################"
    echo "STARTING RUN"
    echo "Name:   $run_name"
    echo "Model:  $model_path"
    echo "Budget: $budget"
    echo "Mode:   $mode"
    echo "Steps:  $expected_steps"
    echo "Time:   $started"
    echo "##################################################"
    echo

    "$LAUNCHER" \
      "$model_label" \
      "$model_path" \
      "$budget" \
      "$mode"

    if ! run_is_complete \
        "$run_name" \
        "$expected_steps" \
        "$mode"
    then
        echo "Run command exited but output verification failed:"
        echo "$run_name"
        return 1
    fi

    finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)

    printf "%s\t%s\t%s\t%s\n" \
      "$run_name" \
      "complete" \
      "$started" \
      "$finished" \
      >> "$STATUS_FILE"

    echo
    echo "##################################################"
    echo "COMPLETED RUN"
    echo "Name: $run_name"
    echo "Time: $finished"
    echo "##################################################"
}


echo
echo "=== Remaining SMART production runs ==="
echo "Batch directory: $BATCH_DIR"
echo "Master log:      $MASTER_LOG"
echo "Status file:     $STATUS_FILE"
echo

# Full fine-tuning runs first.
run_experiment \
  llama2_7b \
  "$LLAMA_MODEL" \
  50000 \
  full

run_experiment \
  qwen2_7b \
  "$QWEN_MODEL" \
  25000 \
  full

run_experiment \
  qwen2_7b \
  "$QWEN_MODEL" \
  50000 \
  full

# LoRA extensions.
run_experiment \
  llama2_7b \
  "$LLAMA_MODEL" \
  25000 \
  lora

run_experiment \
  llama2_7b \
  "$LLAMA_MODEL" \
  50000 \
  lora

run_experiment \
  qwen2_7b \
  "$QWEN_MODEL" \
  25000 \
  lora

run_experiment \
  qwen2_7b \
  "$QWEN_MODEL" \
  50000 \
  lora

CURRENT_RUN="final_verification"

echo
echo "=== Verifying all eight experiments ==="

python3 - "$MODEL_ROOT" "$BATCH_DIR" <<'PY'
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path


model_root = Path(sys.argv[1])
batch_dir = Path(sys.argv[2])

runs = [
    {
        "run": "llama2_7b_smart_25000_full_seed23",
        "model": "llama2_7b",
        "budget": 25000,
        "mode": "full",
        "steps": 391,
    },
    {
        "run": "llama2_7b_smart_50000_full_seed23",
        "model": "llama2_7b",
        "budget": 50000,
        "mode": "full",
        "steps": 782,
    },
    {
        "run": "qwen2_7b_smart_25000_full_seed23",
        "model": "qwen2_7b",
        "budget": 25000,
        "mode": "full",
        "steps": 391,
    },
    {
        "run": "qwen2_7b_smart_50000_full_seed23",
        "model": "qwen2_7b",
        "budget": 50000,
        "mode": "full",
        "steps": 782,
    },
    {
        "run": "llama2_7b_smart_25000_lora_seed23",
        "model": "llama2_7b",
        "budget": 25000,
        "mode": "lora",
        "steps": 391,
    },
    {
        "run": "llama2_7b_smart_50000_lora_seed23",
        "model": "llama2_7b",
        "budget": 50000,
        "mode": "lora",
        "steps": 782,
    },
    {
        "run": "qwen2_7b_smart_25000_lora_seed23",
        "model": "qwen2_7b",
        "budget": 25000,
        "mode": "lora",
        "steps": 391,
    },
    {
        "run": "qwen2_7b_smart_50000_lora_seed23",
        "model": "qwen2_7b",
        "budget": 50000,
        "mode": "lora",
        "steps": 782,
    },
]

summary_rows = []

for specification in runs:
    output_dir = model_root / specification["run"]
    result_path = output_dir / "all_results.json"

    if not result_path.is_file():
        raise FileNotFoundError(result_path)

    result = json.loads(
        result_path.read_text(encoding="utf-8")
    )

    completed_steps = int(result["completed_steps"])
    expected_steps = int(specification["steps"])
    eval_loss = float(result["eval_loss"])
    perplexity = float(result["perplexity"])

    if completed_steps != expected_steps:
        raise RuntimeError(
            f"{specification['run']}: "
            f"{completed_steps} steps, expected "
            f"{expected_steps}."
        )

    if not math.isfinite(eval_loss):
        raise RuntimeError(
            f"{specification['run']}: "
            "non-finite eval loss."
        )

    if not math.isfinite(perplexity):
        raise RuntimeError(
            f"{specification['run']}: "
            "non-finite perplexity."
        )

    if specification["mode"] == "full":
        weight_files = (
            list(output_dir.glob("model*.safetensors"))
            + list(output_dir.glob("pytorch_model*.bin"))
        )

        if not weight_files:
            raise FileNotFoundError(
                f"{specification['run']}: "
                "full weights are missing."
            )

        artifact_count = len(weight_files)

    else:
        adapter_files = [
            output_dir / "adapter_model.safetensors",
            output_dir / "adapter_model.bin",
        ]

        existing_adapters = [
            path
            for path in adapter_files
            if path.is_file()
        ]

        if not existing_adapters:
            raise FileNotFoundError(
                f"{specification['run']}: "
                "adapter weights are missing."
            )

        artifact_count = len(existing_adapters)

    row = {
        **specification,
        "completed_steps": completed_steps,
        "eval_loss": eval_loss,
        "perplexity": perplexity,
        "artifact_count": artifact_count,
        "output_dir": str(output_dir),
        "status": "complete",
    }

    summary_rows.append(row)

csv_path = batch_dir / "all_experiments_summary.csv"
json_path = batch_dir / "all_experiments_summary.json"

with csv_path.open(
    "w",
    encoding="utf-8",
    newline="",
) as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=list(summary_rows[0]),
    )
    writer.writeheader()
    writer.writerows(summary_rows)

json_path.write_text(
    json.dumps(
        {
            "status": "complete",
            "experiment_count": len(summary_rows),
            "experiments": summary_rows,
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)

print()
print(
    f"{'model':12s} "
    f"{'budget':>7s} "
    f"{'mode':>6s} "
    f"{'steps':>6s} "
    f"{'eval_loss':>12s} "
    f"{'perplexity':>12s}"
)

for row in summary_rows:
    print(
        f"{row['model']:12s} "
        f"{row['budget']:7,d} "
        f"{row['mode']:>6s} "
        f"{row['completed_steps']:6,d} "
        f"{row['eval_loss']:12.6f} "
        f"{row['perplexity']:12.6f}"
    )

print()
print("All eight production experiments passed.")
print("CSV:", csv_path)
print("JSON:", json_path)
PY

echo
echo "=================================================="
echo "ALL REMAINING TRAINING RUNS COMPLETED"
echo "Finished:   $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Batch logs: $BATCH_DIR"
echo "=================================================="
BASH

chmod +x run_remaining_production_trainings.sh

bash -n run_remaining_production_trainings.sh
```

## Launch the batch

Using a persistent shell session is preferable:

```bash
cd /data/saral/wdir/smart || exit 1

tmux new -s smart_remaining
```

Inside the `tmux` session:

```bash
./run_remaining_production_trainings.sh
```

Detach with `Ctrl-b`, then `d`.

Reattach later with:

```bash
tmux attach -t smart_remaining
```

If `tmux` is unavailable:

```bash
nohup \
  ./run_remaining_production_trainings.sh \
  > /mnt/warm_storage/saral/smart/logs/production_training/remaining_nohup.log \
  2>&1 &

echo $!
```

Monitor the active experiment with:

```bash
tail -f \
  /mnt/warm_storage/saral/smart/logs/production_training/batch_remaining_*/master.log
```

Monitor GPUs:

```bash
watch -n 5 \
  'nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu,power.draw --format=csv'
```

The script stops immediately on a failed experiment, refuses to overwrite partial outputs, skips any already-valid completed experiment, and verifies all eight model artifacts when the batch finishes.
