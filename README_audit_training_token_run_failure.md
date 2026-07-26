## Error
=== Tokenization audit summary === 

Status: failed 
Validation splits identical: True 
Fatal problems: 2 
  - llama2_7b/shared_validation: 119 rows have zero supervised tokens. 
  - qwen2_7b/shared_validation: 53 rows have zero supervised tokens. <br>
CSV:     /mnt/warm_storage/saral/smart/artifacts/tokenization_audit/tokenization_summary.csv <br>
Summary: /mnt/warm_storage/saral/smart/artifacts/tokenization_audit/tokenization_audit_summary.json <br>
Tokenization audit failed. Do not start training yet. 

| Model      | Split                | Rows    | Mean  | P99    | Max    | Resp | Trunc | Prompt >4k | Zero |
|------------|----------------------|--------:|------:|-------:|-------:|-----:|------:|-----------:|-----:|
| llama2_7b  | smart_25000_train    | 25,000  | 608.0 | 2288.1 | 5,681  | 25.1 |     3 |          0 |    0 |
| llama2_7b  | smart_50000_train    | 50,000  | 564.3 | 2270.0 | 12,373 | 25.3 |     5 |          0 |    0 |
| llama2_7b  | shared_validation    | 183,870 | 533.4 | 3090.0 | 14,117 | 36.3 |   139 |        119 |  119 |
| qwen2_7b   | smart_25000_train    | 25,000  | 535.3 | 2010.0 | 3,767  | 21.8 |     0 |          0 |    0 |
| qwen2_7b   | smart_50000_train    | 50,000  | 497.0 | 1989.0 | 7,842  | 21.9 |     1 |          0 |    0 |
| qwen2_7b   | shared_validation    | 183,870 | 462.4 | 2756.0 | 13,367 | 31.2 |    58 |         53 |   53 |




## Why the audit failed

This is **not a SMART selection failure** and it does not affect the 25K or 50K training mixtures. Both training splits passed with zero problematic rows.

The failure is limited to the shared validation set.

The released trainer preprocesses each example as follows:

```python
combined = tokenizer(
    prompt + " " + response,
    truncation=True,
    max_length=4096,
)

prompt_only = tokenizer(
    prompt,
    truncation=True,
    max_length=4096,
)

labels = combined["input_ids"].copy()
labels[:len(prompt_only["input_ids"])] = -100
```

Only tokens not set to `-100` contribute to the causal-language-model loss. This is the preprocessing implemented in the authors’ repository. 

For a sufficiently long prompt:

```text
tokenized prompt length          = 4096
tokenized prompt+response length = 4096
```

The combined sequence has already been right-truncated before the response appears. The trainer then masks all 4,096 labels:

```text
[-100, -100, ..., -100]
```

There are no target tokens left to predict.

Cross-entropy over a batch in which every target is ignored returns `NaN`. The released launcher uses:

```text
per_device_eval_batch_size = 1
```

so every affected row becomes an all-ignored evaluation batch. Once a `NaN` is included in the collected validation losses, the final average validation loss and perplexity become `NaN`.

## Interpreting your counts

For Llama2:

```text
139 validation rows exceed 4096 combined tokens
119 lose the complete response
20 retain at least one response token
```

For Qwen2:

```text
58 validation rows exceed 4096 combined tokens
53 lose the complete response
5 retain at least one response token
```

The counts differ because Llama and Qwen use different tokenizers. The same text may require more tokens under one tokenizer than the other.

The training data is safe:

```text
Llama 25K: zero rows = 0
Llama 50K: zero rows = 0
Qwen 25K:  zero rows = 0
Qwen 50K:  zero rows = 0
```

Therefore:

* SMART Stage 1 remains valid.
* SMART Stage 2 remains valid.
* The materialized training mixtures remain valid.
* No training examples need to be removed.
* Only validation preprocessing needs correction.

## Recommended resolution

Create a **common evaluation-safe validation set** by removing the union of rows that have zero supervised tokens for either tokenizer.

This is the best solution for the closest reproduction because it:

* leaves both SMART training mixtures unchanged;
* preserves the authors’ preprocessing exactly;
* preserves the 4,096-token context length;
* uses the same validation rows for Llama and Qwen;
* removes only unusable rows;
* avoids model-specific validation subsets.

Because the two sets can overlap, the number removed will be between 119 and 172. The script below calculates the exact union.

Do not overwrite the canonical materialized datasets. Save evaluation-safe copies separately.

# Step 16 — Build common evaluation-safe datasets

## 16.1 Create the filtering script

```bash
cd /data/saral/wdir/smart || exit 1

cat > data_generation_scripts/build_eval_safe_datasets.py <<'PY'
#!/usr/bin/env python3

"""Remove validation rows with no supervised tokens for any target model."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_from_disk
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset-25000",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--dataset-50000",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--audit-dataset",
        type=Path,
        required=True,
        help=(
            "Audit SMART dataset containing task_id, "
            "source_index and other provenance columns."
        ),
    )
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help="Model specification LABEL=PATH.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=4096,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


def parse_models(
    values: list[str],
) -> list[tuple[str, Path]]:
    result: list[tuple[str, Path]] = []
    labels: set[str] = set()

    for value in values:
        if "=" not in value:
            raise ValueError(
                f"Expected LABEL=PATH, received {value!r}."
            )

        label, raw_path = value.split("=", 1)
        label = label.strip()
        path = Path(raw_path.strip()).resolve()

        if not label:
            raise ValueError("Model label cannot be empty.")

        if label in labels:
            raise ValueError(
                f"Duplicate model label: {label}"
            )

        if not path.exists():
            raise FileNotFoundError(path)

        labels.add(label)
        result.append((label, path))

    return result


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


def find_zero_supervision_rows(
    tokenizer: Any,
    validation: Dataset,
    max_seq_length: int,
    batch_size: int,
    model_label: str,
) -> dict[int, dict[str, int]]:
    bad_rows: dict[int, dict[str, int]] = {}

    for start in range(
        0,
        len(validation),
        batch_size,
    ):
        end = min(
            start + batch_size,
            len(validation),
        )

        batch = validation[start:end]
        prompts = batch["prompt"]
        responses = batch["response"]

        combined_texts = [
            prompt + " " + response
            for prompt, response in zip(
                prompts,
                responses,
            )
        ]

        # These are the exact two tokenization calls used
        # by the released instruction_tuner.py.
        combined = tokenizer(
            combined_texts,
            add_special_tokens=True,
            truncation=True,
            max_length=max_seq_length,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )

        prompt_only = tokenizer(
            prompts,
            add_special_tokens=True,
            truncation=True,
            max_length=max_seq_length,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )

        for offset, (
            combined_ids,
            prompt_ids,
        ) in enumerate(
            zip(
                combined["input_ids"],
                prompt_only["input_ids"],
            )
        ):
            # instruction_tuner.py executes:
            # labels[:prompt_len] = -100
            #
            # Therefore no labels survive whenever
            # prompt_len >= combined sequence length.
            if len(prompt_ids) >= len(combined_ids):
                row_index = start + offset

                bad_rows[row_index] = {
                    "combined_retained_tokens": len(
                        combined_ids
                    ),
                    "prompt_retained_tokens": len(
                        prompt_ids
                    ),
                    "supervised_tokens": 0,
                }

        if end == len(validation) or end % 10_000 == 0:
            print(
                f"  {model_label}: "
                f"{end:,}/{len(validation):,}, "
                f"zero rows={len(bad_rows):,}",
                flush=True,
            )

    return bad_rows


def save_dataset_atomic(
    dataset: DatasetDict,
    destination: Path,
    overwrite: bool,
) -> None:
    temporary = destination.with_name(
        destination.name + ".tmp"
    )

    if temporary.exists():
        shutil.rmtree(temporary)

    if destination.exists():
        if not overwrite:
            raise FileExistsError(
                f"{destination} exists. "
                "Use --overwrite to replace it."
            )

        shutil.rmtree(destination)

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset.save_to_disk(
        str(temporary)
    )

    temporary.rename(destination)


def atomic_write_json(
    payload: dict[str, Any],
    path: Path,
) -> None:
    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    with temporary.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            payload,
            handle,
            indent=2,
            ensure_ascii=False,
        )
        handle.write("\n")

    temporary.replace(path)


def main() -> int:
    args = parse_args()
    models = parse_models(args.model)

    dataset_25_path = (
        args.dataset_25000.resolve()
    )
    dataset_50_path = (
        args.dataset_50000.resolve()
    )
    audit_path = (
        args.audit_dataset.resolve()
    )
    output_root = (
        args.output_root.resolve()
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset_25 = load_from_disk(
        str(dataset_25_path)
    )
    dataset_50 = load_from_disk(
        str(dataset_50_path)
    )
    audit_dataset = load_from_disk(
        str(audit_path)
    )

    if len(dataset_25["train"]) != 25_000:
        raise RuntimeError(
            "SMART-25K training count mismatch."
        )

    if len(dataset_50["train"]) != 50_000:
        raise RuntimeError(
            "SMART-50K training count mismatch."
        )

    validation_25 = dataset_25["validation"]
    validation_50 = dataset_50["validation"]
    audit_validation = audit_dataset[
        "validation"
    ]

    if len(validation_25) != len(validation_50):
        raise RuntimeError(
            "Validation split lengths differ."
        )

    if len(audit_validation) != len(
        validation_50
    ):
        raise RuntimeError(
            "Audit validation length differs from "
            "trainer validation length."
        )

    print("Hashing validation splits...")

    hash_25 = dataset_hash(
        validation_25
    )
    hash_50 = dataset_hash(
        validation_50
    )

    if hash_25 != hash_50:
        raise RuntimeError(
            "SMART-25K and SMART-50K validation "
            "splits differ."
        )

    original_count = len(
        validation_50
    )

    bad_by_model: dict[
        str,
        dict[int, dict[str, int]],
    ] = {}

    for model_label, model_path in models:
        print()
        print(f"Loading tokenizer: {model_label}")

        tokenizer = (
            AutoTokenizer.from_pretrained(
                str(model_path),
                use_fast=True,
                trust_remote_code=(
                    args.trust_remote_code
                ),
                local_files_only=(
                    args.local_files_only
                ),
            )
        )

        tokenizer.padding_side = "right"

        print(
            f"Scanning validation for "
            f"{model_label}..."
        )

        bad_by_model[model_label] = (
            find_zero_supervision_rows(
                tokenizer=tokenizer,
                validation=validation_50,
                max_seq_length=(
                    args.max_seq_length
                ),
                batch_size=args.batch_size,
                model_label=model_label,
            )
        )

        del tokenizer

    removed_union = sorted(
        set().union(
            *[
                set(rows)
                for rows in bad_by_model.values()
            ]
        )
    )

    removed_set = set(
        removed_union
    )

    keep_indices = [
        index
        for index in range(original_count)
        if index not in removed_set
    ]

    final_count = len(
        keep_indices
    )

    if final_count != (
        original_count - len(removed_union)
    ):
        raise RuntimeError(
            "Filtered validation count mismatch."
        )

    print()
    print("Building common validation subset...")
    print(
        f"Original validation: {original_count:,}"
    )

    for label, rows in bad_by_model.items():
        print(
            f"{label} zero rows:  {len(rows):,}"
        )

    print(
        f"Union removed:       {len(removed_union):,}"
    )
    print(
        f"Final validation:    {final_count:,}"
    )

    filtered_validation = (
        validation_50.select(
            keep_indices
        )
    )

    output_25 = DatasetDict(
        {
            "train": dataset_25["train"],
            "validation": filtered_validation,
        }
    )

    output_50 = DatasetDict(
        {
            "train": dataset_50["train"],
            "validation": filtered_validation,
        }
    )

    output_25_path = (
        output_root / "smart_25000"
    )
    output_50_path = (
        output_root / "smart_50000"
    )

    save_dataset_atomic(
        output_25,
        output_25_path,
        args.overwrite,
    )

    save_dataset_atomic(
        output_50,
        output_50_path,
        args.overwrite,
    )

    removed_jsonl = (
        output_root
        / "removed_validation_rows.jsonl"
    )

    temporary_jsonl = (
        removed_jsonl.with_suffix(
            ".jsonl.tmp"
        )
    )

    with temporary_jsonl.open(
        "w",
        encoding="utf-8",
    ) as handle:
        for row_index in removed_union:
            provenance = audit_validation[
                row_index
            ]

            model_details = {
                label: details.get(
                    row_index
                )
                for label, details
                in bad_by_model.items()
            }

            record = {
                "validation_row_index": (
                    row_index
                ),
                "task_id": provenance.get(
                    "task_id"
                ),
                "corpus": provenance.get(
                    "corpus"
                ),
                "task_name": provenance.get(
                    "task_name"
                ),
                "source_file": provenance.get(
                    "source_file"
                ),
                "source_index": provenance.get(
                    "source_index"
                ),
                "prompt_characters": len(
                    provenance["prompt"]
                ),
                "response_characters": len(
                    provenance["response"]
                ),
                "zero_supervision_by_model": {
                    label: (
                        model_details[label]
                        is not None
                    )
                    for label in model_details
                },
                "tokenization": (
                    model_details
                ),
            }

            handle.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
            )
            handle.write("\n")

    temporary_jsonl.replace(
        removed_jsonl
    )

    overlap_counts: dict[str, int] = {}

    labels = list(
        bad_by_model
    )

    for index, first in enumerate(labels):
        first_set = set(
            bad_by_model[first]
        )

        for second in labels[
            index + 1:
        ]:
            second_set = set(
                bad_by_model[second]
            )

            overlap_counts[
                f"{first}__{second}"
            ] = len(
                first_set & second_set
            )

    summary = {
        "format_version": 1,
        "stage": (
            "SMART_common_evaluation_filter"
        ),
        "status": "complete",
        "policy": {
            "max_seq_length": (
                args.max_seq_length
            ),
            "author_preprocessing_preserved": True,
            "training_splits_unchanged": True,
            "validation_policy": (
                "Remove union of rows having zero "
                "supervised tokens for any target "
                "tokenizer."
            ),
        },
        "counts": {
            "smart_25000_train": len(
                output_25["train"]
            ),
            "smart_50000_train": len(
                output_50["train"]
            ),
            "original_validation": (
                original_count
            ),
            "removed_union": len(
                removed_union
            ),
            "final_validation": (
                final_count
            ),
            "removed_by_model": {
                label: len(rows)
                for label, rows
                in bad_by_model.items()
            },
            "pairwise_overlap": (
                overlap_counts
            ),
        },
        "validation": {
            "original_sha256": hash_25,
            "shared_between_budgets": True,
        },
        "outputs": {
            "smart_25000": str(
                output_25_path
            ),
            "smart_50000": str(
                output_50_path
            ),
            "removed_rows": str(
                removed_jsonl
            ),
        },
    }

    summary_path = (
        output_root
        / "eval_safe_summary.json"
    )

    atomic_write_json(
        summary,
        summary_path,
    )

    print()
    print("=== Evaluation-safe datasets ===")
    print(
        f"SMART-25K train: {len(output_25['train']):,}"
    )
    print(
        f"SMART-50K train: {len(output_50['train']):,}"
    )
    print(
        f"Validation:      {final_count:,}"
    )
    print(
        f"Removed:         {len(removed_union):,}"
    )
    print(f"25K:             {output_25_path}")
    print(f"50K:             {output_50_path}")
    print(f"Summary:         {summary_path}")
    print(
        "Common evaluation-safe materialization passed."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
PY

chmod +x \
  data_generation_scripts/build_eval_safe_datasets.py

python3 -m py_compile \
  data_generation_scripts/build_eval_safe_datasets.py
```

## 16.2 Run it

```bash
cd /data/saral/wdir/smart || exit 1

ROOT=/mnt/warm_storage/saral/smart
SAFE="$ROOT/datasets/trainer_eval_safe"

mkdir -p "$SAFE"

export TOKENIZERS_PARALLELISM=true

python3 \
  data_generation_scripts/build_eval_safe_datasets.py \
  --dataset-25000 "$ROOT/datasets/trainer/smart_25000" \
  --dataset-50000 "$ROOT/datasets/trainer/smart_50000" \
  --audit-dataset "$ROOT/datasets/audit/smart_50000" \
  --model "llama2_7b=/data/saral/wdir/smart/llama2_7b" \
  --model "qwen2_7b=/data/saral/wdir/smart/qwen2_7b" \
  --output-root "$SAFE" \
  --max-seq-length 4096 \
  --batch-size 16 \
  --trust-remote-code \
  --local-files-only \
  2>&1 | tee \
  "$SAFE/eval_safe_build.log"
```

## 16.3 Update the audit script to accept the reduced count

The previous script hardcoded `183870`. Patch it once:

```bash
cd /data/saral/wdir/smart || exit 1

python3 - <<'PY'
from pathlib import Path

path = Path(
    "data_generation_scripts/"
    "audit_training_tokenization.py"
)

text = path.read_text(
    encoding="utf-8"
)

if "--expected-validation-count" not in text:
    marker = '''    parser.add_argument(
        "--trust-remote-code",
'''

    insertion = '''    parser.add_argument(
        "--expected-validation-count",
        type=int,
        default=183_870,
    )
'''

    if marker not in text:
        raise RuntimeError(
            "Could not find parser insertion point."
        )

    text = text.replace(
        marker,
        insertion + marker,
        1,
    )

old = '''    expected_validation_count = 183_870
'''

new = '''    expected_validation_count = (
        args.expected_validation_count
    )
'''

if old in text:
    text = text.replace(
        old,
        new,
        1,
    )
elif new not in text:
    raise RuntimeError(
        "Could not patch expected validation count."
    )

path.write_text(
    text,
    encoding="utf-8",
)

print("Audit script patched.")
PY

python3 -m py_compile \
  data_generation_scripts/audit_training_tokenization.py
```

## 16.4 Rerun the audit

```bash
cd /data/saral/wdir/smart || exit 1

ROOT=/mnt/warm_storage/saral/smart
SAFE="$ROOT/datasets/trainer_eval_safe"
AUDIT="$ROOT/artifacts/tokenization_audit_eval_safe"

mkdir -p "$AUDIT"

SAFE_COUNT=$(
  python3 - <<'PY'
import json

path = (
    "/mnt/warm_storage/saral/smart/"
    "datasets/trainer_eval_safe/"
    "eval_safe_summary.json"
)

with open(path, encoding="utf-8") as handle:
    summary = json.load(handle)

print(
    summary["counts"]["final_validation"]
)
PY
)

echo "Evaluation-safe validation count: $SAFE_COUNT"

python3 \
  data_generation_scripts/audit_training_tokenization.py \
  --dataset-25000 "$SAFE/smart_25000" \
  --dataset-50000 "$SAFE/smart_50000" \
  --model "llama2_7b=/data/saral/wdir/smart/llama2_7b" \
  --model "qwen2_7b=/data/saral/wdir/smart/qwen2_7b" \
  --output-root "$AUDIT" \
  --max-seq-length 4096 \
  --batch-size 16 \
  --top-outliers 100 \
  --expected-validation-count "$SAFE_COUNT" \
  --trust-remote-code \
  --local-files-only \
  2>&1 | tee \
  "$AUDIT/tokenization_audit.log"
```

The required result is:

```text
Status: complete
Fatal problems: 0

llama2_7b/shared_validation zero = 0
qwen2_7b/shared_validation  zero = 0
```

## Alternatives we should not use

**Increasing evaluation batch size** is not a valid fix. It can hide an all-ignored row by placing it beside a valid row, but that row still contributes no target tokens, and an all-invalid distributed batch can still produce `NaN`.

**Increasing the context window** would deviate from the authors’ 4,096-token configuration and is not supported natively by the Llama2-7B checkpoint.

**Response-preserving truncation**—truncating the prompt enough to reserve room for the response—is technically better for long-context data, but it changes the authors’ preprocessing. It is appropriate as a later ablation, not for the closest-reproduction run.

The common union filter is the smallest transparent correction: it changes no training data and removes only examples that the released loss function cannot evaluate.
