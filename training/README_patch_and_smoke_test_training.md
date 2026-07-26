The data pipeline is once complete. The next gate is **training-system validation** before launching the eight experiments.

The paper’s closest-reproduction settings are one epoch, learning rate `2e-5` for Llama2, global batch size 64, weight decay `0.1`, cosine decay, 1% warmup, and Flash Attention. 

With four GPUs, preserve the authors’ global batch size as:

```text
per-device batch 1 × 4 GPUs × accumulation 16 = 64
```

Full fine-tuning is the author-comparable track. Qwen2 and all LoRA runs are extensions. The patch below starts from the released `instruction_tuner.py`; it does not replace the training algorithm. 

# Step 17 — Patch and smoke-test the authors’ trainer

## 17.1 Create a minimally patched local trainer

The changes are limited to:

* local-only model loading;
* four-GPU-compatible operational settings;
* typed `max_train_steps`;
* configurable DataLoader workers;
* non-finite loss checks;
* lightweight smoke-test output without saving a full 7B checkpoint.

The preprocessing, prompt masking, optimizer, scheduler, gradient accumulation, evaluation, and model training remain the authors’ implementation.

```bash
cd /data/saral/wdir/smart || exit 1

cat > data_generation_scripts/patch_instruction_tuner_local.py <<'PY'
#!/usr/bin/env python3

from pathlib import Path


SOURCE = Path("instruction_tuner.py")
DESTINATION = Path("instruction_tuner_local.py")


if not SOURCE.is_file():
    raise FileNotFoundError(SOURCE)

text = SOURCE.read_text(encoding="utf-8")


def replace_once(
    old: str,
    new: str,
    count: int = 1,
) -> None:
    global text

    occurrences = text.count(old)

    if occurrences < count:
        raise RuntimeError(
            f"Expected at least {count} occurrence(s), "
            f"found {occurrences}: {old[:120]!r}"
        )

    text = text.replace(
        old,
        new,
        count,
    )


replace_once(
    '''    parser.add_argument("--trust_remote_code", action="store_true", help="Whether to trust remote code")
''',
    '''    parser.add_argument("--trust_remote_code", action="store_true", help="Whether to trust remote code")
    parser.add_argument("--local_files_only", action="store_true", help="Only use local model/tokenizer files")
''',
)

replace_once(
    '''    parser.add_argument("--preprocessing_num_workers", type=int, default=12, help="The number of processes to use for the preprocessing.",)
''',
    '''    parser.add_argument("--preprocessing_num_workers", type=int, default=12, help="The number of processes to use for the preprocessing.",)
    parser.add_argument("--dataloader_num_workers", type=int, default=8, help="DataLoader workers per process")
''',
)

replace_once(
    '''    parser.add_argument("--max_train_steps", default=None, help="Max train steps")
''',
    '''    parser.add_argument("--max_train_steps", type=int, default=None, help="Max train steps")
''',
)

replace_once(
    '''    parser.add_argument("--checkpointing_steps", type=str, default=None, help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.")
''',
    '''    parser.add_argument("--checkpointing_steps", type=str, default=None, help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.")
    parser.add_argument("--logging_steps", type=int, default=10, help="Log a finite training loss every N optimizer steps")
    parser.add_argument("--skip_final_save", action="store_true", help="Skip model/tokenizer saving; useful for smoke tests")
''',
)

replace_once(
    '''    tokenizer=AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        token=args.hf_access_token,
    )
''',
    '''    tokenizer=AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        token=args.hf_access_token or None,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
''',
)

replace_once(
    '''            token=args.hf_access_token,
            use_flash_attention_2=args.use_flash_attention_2,
''',
    '''            token=args.hf_access_token or None,
            trust_remote_code=args.trust_remote_code,
            local_files_only=args.local_files_only,
            use_flash_attention_2=args.use_flash_attention_2,
''',
    count=2,
)

replace_once(
    '''        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size, pin_memory=True, num_workers=8
''',
    '''        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size, pin_memory=True, num_workers=args.dataloader_num_workers
''',
)

replace_once(
    '''        eval_dataset, shuffle=False, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size, pin_memory=True, num_workers=8
''',
    '''        eval_dataset, shuffle=False, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size, pin_memory=True, num_workers=args.dataloader_num_workers
''',
)

replace_once(
    '''                    loss=outputs.loss
                    # We keep track of loss at each epoch
''',
    '''                    loss=outputs.loss
                    if not torch.isfinite(loss):
                        raise FloatingPointError(
                            f"Non-finite training loss at "
                            f"epoch={epoch}, micro_step={step}: "
                            f"{loss.item()}"
                        )
                    # We keep track of loss at each epoch
''',
)

replace_once(
    '''                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    completed_steps += 1
''',
    '''                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    completed_steps += 1

                    if (
                        args.logging_steps > 0
                        and completed_steps
                        % args.logging_steps
                        == 0
                    ):
                        reduced_loss = accelerator.reduce(
                            loss.detach().float(),
                            reduction="mean",
                        )

                        logger.info(
                            f"optimizer_step={completed_steps} "
                            f"train_loss="
                            f"{reduced_loss.item():.8f} "
                            f"lr="
                            f"{optimizer.param_groups[0]['lr']:.12g}"
                        )
''',
)

replace_once(
    '''            eval_loss = torch.mean(losses)
            perplexity = math.exp(eval_loss)
''',
    '''            eval_loss = torch.mean(losses)

            if not torch.isfinite(eval_loss):
                raise FloatingPointError(
                    f"Non-finite evaluation loss: "
                    f"{eval_loss.item()}"
                )

            perplexity = math.exp(eval_loss)
''',
)

replace_once(
    '''    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
        )
        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir)
            if args.push_to_hub:
                repo.push_to_hub(commit_message="End of Training", auto_lfs_prune=True)
            with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
                json.dump({"perplexity": perplexity}, f)
    
    if args.use_peft and args.merge_weights:
''',
    '''    if args.output_dir is not None:
        accelerator.wait_for_everyone()

        if not args.skip_final_save:
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                args.output_dir,
                is_main_process=accelerator.is_main_process,
                save_function=accelerator.save,
            )

            if accelerator.is_main_process:
                tokenizer.save_pretrained(
                    args.output_dir
                )

                if args.push_to_hub:
                    repo.push_to_hub(
                        commit_message="End of Training",
                        auto_lfs_prune=True,
                    )

        if accelerator.is_main_process:
            with open(
                os.path.join(
                    args.output_dir,
                    "all_results.json",
                ),
                "w",
            ) as f:
                json.dump(
                    {
                        "perplexity": float(perplexity),
                        "eval_loss": float(
                            eval_loss.item()
                        ),
                        "completed_steps": int(
                            completed_steps
                        ),
                        "max_train_steps": int(
                            args.max_train_steps
                        ),
                        "skip_final_save": bool(
                            args.skip_final_save
                        ),
                    },
                    f,
                    indent=2,
                )
    
    if (
        args.use_peft
        and args.merge_weights
        and not args.skip_final_save
    ):
''',
)

DESTINATION.write_text(
    text,
    encoding="utf-8",
)

print(f"Source:      {SOURCE}")
print(f"Destination: {DESTINATION}")
print("Local trainer patch passed.")
PY

chmod +x \
  data_generation_scripts/patch_instruction_tuner_local.py

python3 \
  data_generation_scripts/patch_instruction_tuner_local.py

python3 -m py_compile \
  instruction_tuner_local.py
```

Preserve a diff for provenance:

```bash
mkdir -p \
  /mnt/warm_storage/saral/smart/artifacts/training_code

diff -u \
  instruction_tuner.py \
  instruction_tuner_local.py \
  > /mnt/warm_storage/saral/smart/artifacts/training_code/instruction_tuner_local.diff \
  || true

sha256sum \
  instruction_tuner.py \
  instruction_tuner_local.py \
  > /mnt/warm_storage/saral/smart/artifacts/training_code/trainer_sha256.txt

cat \
  /mnt/warm_storage/saral/smart/artifacts/training_code/trainer_sha256.txt
```

## 17.2 Create the four-GPU Accelerate configuration

```bash
cat > accelerate_4gpu_bf16.yaml <<'YAML'
compute_environment: LOCAL_MACHINE
debug: false
distributed_type: MULTI_GPU
downcast_bf16: 'no'
gpu_ids: 0,1,2,3
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 4
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
YAML
```

## 17.3 Create a length-stress smoke dataset

Rather than using arbitrary short rows, this selects the longest audited training and validation examples across both tokenizers.

```bash
cat > data_generation_scripts/build_training_smoke_dataset.py <<'PY'
#!/usr/bin/env python3

from __future__ import annotations

import json
import shutil
from pathlib import Path

from datasets import DatasetDict, load_from_disk


ROOT = Path("/mnt/warm_storage/saral/smart")

SOURCE = (
    ROOT
    / "datasets"
    / "trainer_eval_safe"
    / "smart_50000"
)

AUDIT_ROOT = (
    ROOT
    / "artifacts"
    / "tokenization_audit_eval_safe"
)

OUTPUT = (
    ROOT
    / "datasets"
    / "smoke"
    / "smart_train64_eval16"
)


dataset = load_from_disk(str(SOURCE))


def load_split_report(
    model_label: str,
    split_name: str,
) -> dict:
    report_path = (
        AUDIT_ROOT
        / f"{model_label}_tokenization_report.json"
    )

    report = json.loads(
        report_path.read_text(
            encoding="utf-8"
        )
    )

    matches = [
        split
        for split in report["splits"]
        if split["split"] == split_name
    ]

    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one report for "
            f"{model_label}/{split_name}; "
            f"found {len(matches)}."
        )

    return matches[0]


def longest_union(
    split_name: str,
    required_count: int,
    dataset_size: int,
) -> list[int]:
    scores: dict[int, int] = {}

    for model_label in (
        "llama2_7b",
        "qwen2_7b",
    ):
        split_report = load_split_report(
            model_label,
            split_name,
        )

        for row in split_report[
            "outliers"
        ]["longest_sequences"]:
            row_index = int(
                row["row_index"]
            )
            full_tokens = int(
                row["full_tokens"]
            )

            scores[row_index] = max(
                scores.get(row_index, -1),
                full_tokens,
            )

    ranked = [
        index
        for index, _ in sorted(
            scores.items(),
            key=lambda item: (
                -item[1],
                item[0],
            ),
        )
    ]

    used = set(ranked)

    if len(ranked) < required_count:
        for index in range(dataset_size):
            if index in used:
                continue

            ranked.append(index)

            if len(ranked) >= required_count:
                break

    result = ranked[:required_count]

    if len(result) != required_count:
        raise RuntimeError(
            f"Could only select {len(result)} rows; "
            f"required {required_count}."
        )

    return result


train_indices = longest_union(
    split_name="smart_50000_train",
    required_count=64,
    dataset_size=len(dataset["train"]),
)

validation_indices = longest_union(
    split_name="shared_validation",
    required_count=16,
    dataset_size=len(
        dataset["validation"]
    ),
)

smoke_dataset = DatasetDict(
    {
        "train": dataset["train"].select(
            train_indices
        ),
        "validation": (
            dataset["validation"].select(
                validation_indices
            )
        ),
    }
)

if OUTPUT.exists():
    shutil.rmtree(OUTPUT)

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

smoke_dataset.save_to_disk(
    str(OUTPUT)
)

summary = {
    "source": str(SOURCE),
    "output": str(OUTPUT),
    "train_rows": len(
        smoke_dataset["train"]
    ),
    "validation_rows": len(
        smoke_dataset["validation"]
    ),
    "train_indices": train_indices,
    "validation_indices": (
        validation_indices
    ),
}

summary_path = (
    OUTPUT.parent
    / "smoke_dataset_summary.json"
)

summary_path.write_text(
    json.dumps(
        summary,
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)

print(smoke_dataset)
print(f"Output:  {OUTPUT}")
print(f"Summary: {summary_path}")
print("Training smoke dataset passed.")
PY

chmod +x \
  data_generation_scripts/build_training_smoke_dataset.py

python3 \
  data_generation_scripts/build_training_smoke_dataset.py
```

Expected:

```text
train: 64
validation: 16
```

This gives exactly one global optimizer step:

```text
64 rows / global batch 64 = 1 step
```

## 17.4 Environment preflight

```bash
python3 - <<'PY'
import importlib

packages = [
    "torch",
    "transformers",
    "accelerate",
    "datasets",
    "peft",
]

for name in packages:
    module = importlib.import_module(name)
    print(
        f"{name:15s}",
        getattr(module, "__version__", "unknown"),
    )

import torch

print("CUDA available: ", torch.cuda.is_available())
print("CUDA version:   ", torch.version.cuda)
print("GPU count:      ", torch.cuda.device_count())

for index in range(
    torch.cuda.device_count()
):
    properties = (
        torch.cuda.get_device_properties(
            index
        )
    )

    print(
        f"GPU {index}: "
        f"{properties.name}, "
        f"{properties.total_memory / 2**30:.2f} GiB"
    )

try:
    import flash_attn
except Exception as exc:
    raise RuntimeError(
        "flash_attn is not importable. "
        "Do not run the smoke test with "
        "--use_flash_attention_2 until this is fixed."
    ) from exc

print(
    "flash_attn:      ",
    getattr(
        flash_attn,
        "__version__",
        "installed",
    ),
)

assert torch.cuda.device_count() >= 4
assert torch.cuda.is_bf16_supported()

print("Training environment preflight passed.")
PY
```

## 17.5 Run the four smoke tests

This runs:

1. Llama2 full fine-tuning
2. Llama2 LoRA
3. Qwen2 full fine-tuning
4. Qwen2 LoRA

The LoRA tests save their adapters, which also tests local checkpoint writing. Full-model smoke tests skip checkpoint saving to avoid writing roughly 14 GB twice.

```bash
cat > run_training_smokes.sh <<'BASH'
#!/usr/bin/env bash

set -euo pipefail

cd /data/saral/wdir/smart

ROOT=/mnt/warm_storage/saral/smart
SMOKE_DATA="$ROOT/datasets/smoke/smart_train64_eval16"
OUTPUT_ROOT="$ROOT/artifacts/training_smoke"
CACHE_ROOT="$ROOT/cache/hf_training"

LLAMA_MODEL=/data/saral/wdir/smart/llama2_7b
QWEN_MODEL=/data/saral/wdir/smart/qwen2_7b

mkdir -p \
  "$OUTPUT_ROOT" \
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


run_one() {
    local model_label="$1"
    local model_path="$2"
    local mode="$3"

    local run_name="${model_label}_${mode}"
    local output_dir="$OUTPUT_ROOT/$run_name"
    local log_path="$OUTPUT_ROOT/${run_name}.log"
    local time_path="$OUTPUT_ROOT/${run_name}.time.txt"

    local mode_args=()
    local save_args=()

    if [[ "$mode" == "lora" ]]; then
        mode_args=(
            --use_peft
            --peft_lora_r 64
            --peft_lora_alpha 16
            --peft_lora_dropout 0.05
            --peft_target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,lm_head
        )
    elif [[ "$mode" == "full" ]]; then
        # Avoid writing a full 7B checkpoint during smoke testing.
        save_args=(
            --skip_final_save
        )
    else
        echo "Unknown mode: $mode" >&2
        return 2
    fi

    rm -rf "$output_dir"
    mkdir -p "$output_dir"

    echo
    echo "=================================================="
    echo "Smoke run: $run_name"
    echo "Model:     $model_path"
    echo "Mode:      $mode"
    echo "=================================================="

    /usr/bin/time -v \
      -o "$time_path" \
      accelerate launch \
      --config_file accelerate_4gpu_bf16.yaml \
      instruction_tuner_local.py \
        --load_data_from_disk \
        --dataset_name_or_path "$SMOKE_DATA" \
        --model_name_or_path "$model_path" \
        --trust_remote_code \
        --local_files_only \
        --use_flash_attention_2 \
        --torch_dtype bfloat16 \
        --max_seq_length 4096 \
        --learning_rate 2e-5 \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 1 \
        --preprocessing_num_workers 2 \
        --dataloader_num_workers 2 \
        --seed 23 \
        --num_train_epochs 1 \
        --max_train_steps 1 \
        --gradient_accumulation_steps 16 \
        --gradient_checkpointing \
        --weight_decay 0.1 \
        --lr_scheduler_type cosine \
        --lr_warmup_fraction 0.01 \
        --logging_steps 1 \
        --cache_dir "$CACHE_ROOT" \
        --output_dir "$output_dir" \
        "${mode_args[@]}" \
        "${save_args[@]}" \
      2>&1 | tee "$log_path"
}


run_one \
  llama2_7b \
  "$LLAMA_MODEL" \
  full

run_one \
  llama2_7b \
  "$LLAMA_MODEL" \
  lora

run_one \
  qwen2_7b \
  "$QWEN_MODEL" \
  full

run_one \
  qwen2_7b \
  "$QWEN_MODEL" \
  lora

echo
echo "All four training smoke commands completed."
BASH

chmod +x run_training_smokes.sh

./run_training_smokes.sh
```

Run them sequentially, not concurrently.

## 17.6 Validate the smoke results

```bash
python3 - <<'PY'
from __future__ import annotations

import json
import math
from pathlib import Path


root = Path(
    "/mnt/warm_storage/saral/smart/"
    "artifacts/training_smoke"
)

runs = [
    ("llama2_7b_full", False),
    ("llama2_7b_lora", True),
    ("qwen2_7b_full", False),
    ("qwen2_7b_lora", True),
]

for run_name, is_lora in runs:
    output_dir = root / run_name
    result_path = (
        output_dir / "all_results.json"
    )

    if not result_path.is_file():
        raise FileNotFoundError(
            result_path
        )

    result = json.loads(
        result_path.read_text(
            encoding="utf-8"
        )
    )

    assert result["completed_steps"] == 1
    assert result["max_train_steps"] == 1
    assert math.isfinite(
        float(result["eval_loss"])
    )
    assert math.isfinite(
        float(result["perplexity"])
    )

    if is_lora:
        adapter_config = (
            output_dir
            / "adapter_config.json"
        )

        if not adapter_config.is_file():
            raise FileNotFoundError(
                adapter_config
            )

    print()
    print(run_name)
    print(
        "  eval loss:",
        result["eval_loss"],
    )
    print(
        "  perplexity:",
        result["perplexity"],
    )
    print(
        "  completed steps:",
        result["completed_steps"],
    )
    print(
        "  LoRA adapter saved:",
        is_lora,
    )

print()
print("All training smoke validations passed.")
PY
```

Inspect the essential log lines:

```bash
grep -hE \
  "Number of trainable parameters|optimizer_step=|perplexity:|GPU Total Peak Memory" \
  /mnt/warm_storage/saral/smart/artifacts/training_smoke/*.log
```

## Acceptance gate

All four runs must satisfy:

```text
exit status                 = 0
completed optimizer steps   = 1
training loss               = finite
evaluation loss             = finite
perplexity                  = finite
CUDA OOM                    = none
NCCL failure                = none
LoRA adapter files          = present
```

If full fine-tuning exceeds the approximately 70 GB available on the H100, do not reduce the 4,096 context length or effective batch size. Run the full-fine-tuning track on the H200. LoRA can remain on the H100. FSDP would be a larger departure from the released DDP training setup, so the H200 is the preferable author-faithful fallback.
