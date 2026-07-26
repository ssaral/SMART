You can reduce validation without repeating SMART Stage 1, Stage 2, dataset selection, or tokenization auditing.

The training weights are unaffected because this trainer evaluates only after the final optimizer step. There is no early stopping or best-checkpoint selection. Your log shows training finished after 22m 29s, while evaluation over 183,751 rows continued for roughly another 44 minutes. 

## Current situation

The current trainer has no `--max_eval_samples` command-line option, so the existing command cannot reduce validation by itself.

There are two clean solutions:

1. Create a smaller fixed validation DatasetDict once and point all training commands to it.
2. Patch the trainer with `--max_eval_samples`.

I recommend the first approach. It avoids further training-code changes and guarantees every run uses exactly the same validation subset.

## Recommended validation size

Use 10,000 fixed examples:

```text
Original validation: 183,751
Training validation:  10,000
Seed:                 23
```

Based on the observed 44-minute full evaluation, this should take roughly two to four minutes.

The complete validation set remains available for a separate final loss evaluation if needed. Leaderboard scores will come from lm-evaluation-harness, so training-time validation is primarily a sanity metric.

## Create the reduced validation datasets

```bash
cd /data/saral/wdir/smart || exit 1

cat > data_generation_scripts/build_reduced_validation_datasets.py <<'PY'
#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from datasets import Dataset, DatasetDict, load_from_disk


ROOT = Path("/mnt/warm_storage/saral/smart")

SOURCE_ROOT = (
    ROOT
    / "datasets"
    / "trainer_eval_safe"
)

OUTPUT_ROOT = (
    ROOT
    / "datasets"
    / "trainer_eval_safe_10k"
)

SEED = 23
VALIDATION_SIZE = 10_000


def dataset_hash(
    dataset: Dataset,
    batch_size: int = 1024,
) -> str:
    digest = hashlib.sha256()

    for start in range(
        0,
        len(dataset),
        batch_size,
    ):
        end = min(
            start + batch_size,
            len(dataset),
        )

        batch = dataset[start:end]

        for prompt, response in zip(
            batch["prompt"],
            batch["response"],
        ):
            digest.update(
                prompt.encode(
                    "utf-8",
                    errors="surrogatepass",
                )
            )
            digest.update(b"\0")
            digest.update(
                response.encode(
                    "utf-8",
                    errors="surrogatepass",
                )
            )
            digest.update(b"\n")

    return digest.hexdigest()


dataset_25 = load_from_disk(
    str(SOURCE_ROOT / "smart_25000")
)

dataset_50 = load_from_disk(
    str(SOURCE_ROOT / "smart_50000")
)

validation_25 = dataset_25["validation"]
validation_50 = dataset_50["validation"]

if len(validation_25) != len(validation_50):
    raise RuntimeError(
        "Validation counts differ between budgets."
    )

original_hash_25 = dataset_hash(validation_25)
original_hash_50 = dataset_hash(validation_50)

if original_hash_25 != original_hash_50:
    raise RuntimeError(
        "Validation content differs between budgets."
    )

if VALIDATION_SIZE > len(validation_50):
    raise RuntimeError(
        "Requested validation size exceeds dataset size."
    )

# Select indices once and reuse them for both budgets.
shuffled_indices = (
    list(range(len(validation_50)))
)

import random

generator = random.Random(SEED)
generator.shuffle(shuffled_indices)

selected_indices = shuffled_indices[
    :VALIDATION_SIZE
]

reduced_validation_25 = (
    validation_25.select(selected_indices)
)

reduced_validation_50 = (
    validation_50.select(selected_indices)
)

reduced_hash_25 = dataset_hash(
    reduced_validation_25
)
reduced_hash_50 = dataset_hash(
    reduced_validation_50
)

if reduced_hash_25 != reduced_hash_50:
    raise RuntimeError(
        "Reduced validation splits differ."
    )

outputs = {
    25_000: DatasetDict(
        {
            "train": dataset_25["train"],
            "validation": reduced_validation_25,
        }
    ),
    50_000: DatasetDict(
        {
            "train": dataset_50["train"],
            "validation": reduced_validation_50,
        }
    ),
}

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

for budget, dataset in outputs.items():
    destination = (
        OUTPUT_ROOT
        / f"smart_{budget}"
    )

    temporary = destination.with_name(
        destination.name + ".tmp"
    )

    if temporary.exists():
        shutil.rmtree(temporary)

    if destination.exists():
        raise FileExistsError(
            f"Refusing to overwrite {destination}"
        )

    dataset.save_to_disk(
        str(temporary)
    )

    temporary.rename(destination)

summary = {
    "status": "complete",
    "seed": SEED,
    "validation_policy": (
        "Fixed random subset selected before training "
        "and shared across all models and budgets."
    ),
    "counts": {
        "original_validation": len(
            validation_50
        ),
        "reduced_validation": (
            VALIDATION_SIZE
        ),
        "smart_25000_train": len(
            dataset_25["train"]
        ),
        "smart_50000_train": len(
            dataset_50["train"]
        ),
    },
    "hashes": {
        "original_validation_sha256": (
            original_hash_25
        ),
        "reduced_validation_sha256": (
            reduced_hash_25
        ),
    },
    "selected_indices": {
        "count": len(selected_indices),
        "first_20": selected_indices[:20],
    },
    "outputs": {
        "smart_25000": str(
            OUTPUT_ROOT / "smart_25000"
        ),
        "smart_50000": str(
            OUTPUT_ROOT / "smart_50000"
        ),
    },
}

summary_path = (
    OUTPUT_ROOT
    / "reduced_validation_summary.json"
)

summary_path.write_text(
    json.dumps(
        summary,
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)

print("=== Reduced validation datasets ===")
print(
    f"Original validation: "
    f"{len(validation_50):,}"
)
print(
    f"Reduced validation:  "
    f"{VALIDATION_SIZE:,}"
)
print(
    f"SMART-25K train:     "
    f"{len(dataset_25['train']):,}"
)
print(
    f"SMART-50K train:     "
    f"{len(dataset_50['train']):,}"
)
print(
    f"Validation hash:     "
    f"{reduced_hash_25}"
)
print(f"Output:              {OUTPUT_ROOT}")
print(f"Summary:             {summary_path}")
PY

chmod +x \
  data_generation_scripts/build_reduced_validation_datasets.py

python3 \
  data_generation_scripts/build_reduced_validation_datasets.py
```

## Verify it

```bash
python3 - <<'PY'
from datasets import load_from_disk

root = (
    "/mnt/warm_storage/saral/smart/"
    "datasets/trainer_eval_safe_10k"
)

for budget in (25000, 50000):
    dataset = load_from_disk(
        f"{root}/smart_{budget}"
    )

    print()
    print(f"SMART-{budget}")
    print("Train:", len(dataset["train"]))
    print(
        "Validation:",
        len(dataset["validation"]),
    )

    assert len(dataset["train"]) == budget
    assert len(dataset["validation"]) == 10_000

print()
print("Reduced validation verification passed.")
PY
```

## Point the production launcher to it

In the corrected V2 launcher, replace:

```bash
DATA_ROOT="$ROOT/datasets/trainer_eval_safe"
```

with:

```bash
DATA_ROOT=${DATA_ROOT:-"$ROOT/datasets/trainer_eval_safe_10k"}
```

Patch it:

```bash
cd /data/saral/wdir/smart || exit 1

python3 - <<'PY'
from pathlib import Path

path = Path(
    "run_one_production_training_v2.sh"
)

text = path.read_text(
    encoding="utf-8"
)

old = '''DATA_ROOT="$ROOT/datasets/trainer_eval_safe"
'''

new = '''DATA_ROOT=${DATA_ROOT:-"$ROOT/datasets/trainer_eval_safe_10k"}
'''

if old not in text:
    if new in text:
        print("Launcher already patched.")
    else:
        raise RuntimeError(
            "Could not find DATA_ROOT assignment."
        )
else:
    text = text.replace(
        old,
        new,
        1,
    )

    path.write_text(
        text,
        encoding="utf-8",
    )

    print("Launcher patched.")
PY

bash -n \
  run_one_production_training_v2.sh
```

The training command itself remains unchanged:

```bash
./run_one_production_training_v2.sh \
  llama2_7b \
  /data/saral/wdir/smart/llama2_7b \
  25000 \
  full
```

## Even faster alternative

You could completely skip training-time validation. That would require adding a `--skip_eval` option to the trainer. The trained weights would be identical, but there would be no final validation loss or perplexity in `all_results.json`.

For this study, the fixed 10,000-row subset is the better compromise:

```text
same training data
same optimizer steps
same trained weights
same validation subset for all eight runs
much shorter post-training evaluation
```



---


Use the corrected V2 launcher for **all eight runs**, sequentially. Each run will occupy all four H200s.

First verify the critical patches:

```bash
cd /data/saral/wdir/smart || exit 1

grep -nE \
  "step_scheduler_with_optimizer=False|lr_scheduler.step|Scheduler/optimizer step mismatch" \
  instruction_tuner_local.py

grep -nE \
  "peft_lora_r 64|peft_lora_alpha 32|peft_lora_dropout 0.05|peft_target_modules" \
  run_one_production_training_v2.sh
```

The LoRA target line should contain exactly:

```text
q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
```

Then create the batch launcher:

```bash
cd /data/saral/wdir/smart || exit 1

cat > run_all_finetuning_v2.sh <<'BASH'
#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT=/data/saral/wdir/smart
ROOT=/mnt/warm_storage/saral/smart

LAUNCHER="$PROJECT/run_one_production_training_v2.sh"

LLAMA="$PROJECT/llama2_7b"
QWEN="$PROJECT/qwen2_7b"

export DATA_ROOT="$ROOT/datasets/trainer_eval_safe_10k"
export CUDA_VISIBLE_DEVICES=0,1,2,3

BATCH_LOG_ROOT="$ROOT/logs/production_training_v2"
TIMESTAMP=$(date -u +%Y%m%dT%H%M%SZ)
BATCH_DIR="$BATCH_LOG_ROOT/batch_$TIMESTAMP"
MASTER_LOG="$BATCH_DIR/master.log"
STATUS_FILE="$BATCH_DIR/status.tsv"

mkdir -p "$BATCH_DIR"

exec > >(tee -a "$MASTER_LOG") 2>&1

printf \
  "run\tstatus\tstarted_utc\tfinished_utc\n" \
  > "$STATUS_FILE"

CURRENT_RUN="preflight"

on_error() {
    local rc=$?

    echo
    echo "Batch failed."
    echo "Run:       $CURRENT_RUN"
    echo "Exit code: $rc"
    echo "Master log: $MASTER_LOG"

    exit "$rc"
}

trap on_error ERR


run_experiment() {
    local model_label="$1"
    local model_path="$2"
    local budget="$3"
    local mode="$4"

    local run_tag
    local run_name
    local output_dir
    local started
    local finished

    if [[ "$mode" == "full" ]]; then
        run_tag="scheduler_fixed"
    else
        run_tag="scheduler_fixed_r64_a32_proj7"
    fi

    run_name="${model_label}_smart_${budget}_${mode}_${run_tag}_seed23"
    output_dir="$ROOT/models/$run_name"

    CURRENT_RUN="$run_name"

    if [[ -f "$output_dir/all_results.json" ]]; then
        echo
        echo "Skipping completed run: $run_name"

        printf \
          "%s\talready_complete\t%s\t%s\n" \
          "$run_name" \
          "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
          "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
          >> "$STATUS_FILE"

        return 0
    fi

    if [[ -e "$output_dir" ]]; then
        echo "Incomplete output directory exists:"
        echo "$output_dir"
        echo "Move or remove it before restarting."
        return 1
    fi

    started=$(date -u +%Y-%m-%dT%H:%M:%SZ)

    printf \
      "%s\tstarted\t%s\t-\n" \
      "$run_name" \
      "$started" \
      >> "$STATUS_FILE"

    echo
    echo "============================================================"
    echo "Starting:   $run_name"
    echo "Model:      $model_path"
    echo "Budget:     $budget"
    echo "Mode:       $mode"
    echo "Dataset:    $DATA_ROOT/smart_$budget"
    echo "Started:    $started"
    echo "============================================================"

    "$LAUNCHER" \
      "$model_label" \
      "$model_path" \
      "$budget" \
      "$mode"

    if [[ ! -f "$output_dir/all_results.json" ]]; then
        echo "Run finished without all_results.json:"
        echo "$run_name"
        return 1
    fi

    finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)

    printf \
      "%s\tcomplete\t%s\t%s\n" \
      "$run_name" \
      "$started" \
      "$finished" \
      >> "$STATUS_FILE"

    echo
    echo "Completed: $run_name"
    echo "Finished:  $finished"
}


echo "=== SMART corrected fine-tuning batch ==="
echo "Dataset root: $DATA_ROOT"
echo "Batch log:    $MASTER_LOG"
echo

# Full fine-tuning: author-comparable track.
run_experiment llama2_7b "$LLAMA" 25000 full
run_experiment llama2_7b "$LLAMA" 50000 full
run_experiment qwen2_7b  "$QWEN"  25000 full
run_experiment qwen2_7b  "$QWEN"  50000 full

# LoRA extension:
# r=64, alpha=32, dropout=0.05, seven projection targets.
run_experiment llama2_7b "$LLAMA" 25000 lora
run_experiment llama2_7b "$LLAMA" 50000 lora
run_experiment qwen2_7b  "$QWEN"  25000 lora
run_experiment qwen2_7b  "$QWEN"  50000 lora

CURRENT_RUN="final_verification"

python3 - <<'PY'
import json
import math
from pathlib import Path

root = Path("/mnt/warm_storage/saral/smart/models")

runs = [
    ("llama2_7b", 25000, "full", "scheduler_fixed", 391),
    ("llama2_7b", 50000, "full", "scheduler_fixed", 782),
    ("qwen2_7b", 25000, "full", "scheduler_fixed", 391),
    ("qwen2_7b", 50000, "full", "scheduler_fixed", 782),
    (
        "llama2_7b",
        25000,
        "lora",
        "scheduler_fixed_r64_a32_proj7",
        391,
    ),
    (
        "llama2_7b",
        50000,
        "lora",
        "scheduler_fixed_r64_a32_proj7",
        782,
    ),
    (
        "qwen2_7b",
        25000,
        "lora",
        "scheduler_fixed_r64_a32_proj7",
        391,
    ),
    (
        "qwen2_7b",
        50000,
        "lora",
        "scheduler_fixed_r64_a32_proj7",
        782,
    ),
]

for model, budget, mode, tag, expected_steps in runs:
    name = (
        f"{model}_smart_{budget}_{mode}_"
        f"{tag}_seed23"
    )

    directory = root / name
    result_path = directory / "all_results.json"

    if not result_path.is_file():
        raise FileNotFoundError(result_path)

    result = json.loads(
        result_path.read_text(encoding="utf-8")
    )

    assert int(result["completed_steps"]) == expected_steps
    assert int(result["max_train_steps"]) == expected_steps
    assert math.isfinite(float(result["eval_loss"]))
    assert math.isfinite(float(result["perplexity"]))

    if mode == "full":
        weights = (
            list(directory.glob("model*.safetensors"))
            + list(directory.glob("pytorch_model*.bin"))
        )

        if not weights:
            raise RuntimeError(
                f"Missing full weights: {name}"
            )

    else:
        adapter_config = json.loads(
            (
                directory / "adapter_config.json"
            ).read_text(encoding="utf-8")
        )

        assert int(adapter_config["r"]) == 64
        assert int(adapter_config["lora_alpha"]) == 32
        assert float(adapter_config["lora_dropout"]) == 0.05

        expected_targets = {
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        }

        actual_targets = set(
            adapter_config["target_modules"]
        )

        assert actual_targets == expected_targets, (
            name,
            actual_targets,
        )

    print(
        f"{name}: "
        f"steps={result['completed_steps']}, "
        f"eval_loss={float(result['eval_loss']):.6f}, "
        f"perplexity={float(result['perplexity']):.6f}"
    )

print()
print("All eight corrected fine-tuning runs passed.")
PY

echo
echo "============================================================"
echo "All corrected fine-tuning experiments completed."
echo "Master log: $MASTER_LOG"
echo "Status:     $STATUS_FILE"
echo "============================================================"
BASH

chmod +x run_all_finetuning_v2.sh

bash -n run_all_finetuning_v2.sh
```

## Run all experiments

Use a persistent terminal:

```bash
cd /data/saral/wdir/smart || exit 1

tmux new -s smart_finetuning_v2
```

Inside `tmux`:

```bash
./run_all_finetuning_v2.sh
```

Detach using `Ctrl-b`, then `d`.

Reattach with:

```bash
tmux attach -t smart_finetuning_v2
```

Monitor the current run:

```bash
tail -f \
  /mnt/warm_storage/saral/smart/logs/production_training_v2/batch_*/master.log
```

Monitor GPU utilization:

```bash
watch -n 5 \
  'nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu,power.draw --format=csv'
```

The final corrected checkpoints will have names such as:

```text
llama2_7b_smart_25000_full_scheduler_fixed_seed23
llama2_7b_smart_25000_lora_scheduler_fixed_r64_a32_proj7_seed23
qwen2_7b_smart_50000_full_scheduler_fixed_seed23
qwen2_7b_smart_50000_lora_scheduler_fixed_r64_a32_proj7_seed23
```
