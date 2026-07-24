The provisional row-level classifier should **not** be used for the primary SMART baseline:

* only 24 of 309 tasks are ≥99% homogeneous;
* 925,798 rows are ambiguous;
* many tasks are close to a 50/50 split.

That is too unstable for a baseline. It would make template allocation depend on our regexes rather than the authors’ method.

## Frozen template policy

For the primary baseline:

```text
template_type = zs_noopt
```

for every valid row.

This is supported by the authors’ code: when only `zs_noopt` exists, the complete task budget is assigned to that pool. 

We are not claiming the prompts are literally all zero-shot/no-options. We are using `zs_noopt` as the **single available template bucket**, because the original four-way metadata is absent.

The provisional classifier remains available later as a separate ablation:

```text
SMART-Single-Template     primary baseline
SMART-Inferred-Templates optional sensitivity experiment
```

The large-pool issue remains, but exact blockwise Facility Location will handle it without an (n^2) matrix.

# Step 3A — Build one author-format corpus as a smoke test

We will first build only `cot`. After verifying its schema and counts, we will build all five corpora.

## 1. Freeze the policy

```bash
cd /data/saral/wdir/smart_v2 || exit 1

cat > configs/template_policy.json <<'JSON'
{
  "format_version": 1,
  "primary_baseline": "SMART-Single-Template",
  "template_type": "zs_noopt",
  "policy": "Every valid source row is assigned to one available SMART template bucket.",
  "reason": "The source files do not contain authoritative zs/fs or opt/noopt metadata. Provisional rendered-prompt inference produced substantial ambiguity and is not used for the primary baseline.",
  "synthetic_templates_created": false,
  "prompts_modified": false,
  "optional_ablation": "SMART-Inferred-Templates"
}
JSON
```

## 2. Create the author-format builder

```bash
cat > src/data/build_author_format.py <<'PY'
"""Build local Hugging Face datasets compatible with SMART's data model."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any, Iterator

import ijson
from datasets import (
    ClassLabel,
    Dataset,
    DatasetDict,
    Features,
    Value,
)


EXPECTED_CORPORA = (
    "cot",
    "flan2021",
    "sglue",
    "t0",
    "tulu",
)

TEMPLATE_TYPE = "zs_noopt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--corpora",
        nargs="+",
        choices=EXPECTED_CORPORA,
        required=True,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


def valid_row(row: Any) -> bool:
    if not isinstance(row, dict):
        return False

    inputs = row.get("inputs")
    targets = row.get("targets")

    return (
        isinstance(inputs, str)
        and bool(inputs.strip())
        and isinstance(targets, str)
        and bool(targets.strip())
    )


def load_manifest(
    path: Path,
) -> list[dict[str, Any]]:
    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        rows = list(csv.DictReader(handle))

    required = {
        "corpus",
        "task_id",
        "task_name",
        "source_file",
        "valid_train_count",
        "valid_validation_count",
    }

    if not rows:
        raise RuntimeError("Task manifest is empty.")

    missing = required - set(rows[0])

    if missing:
        raise ValueError(
            "Manifest is missing columns: "
            + ", ".join(sorted(missing))
        )

    parsed: list[dict[str, Any]] = []

    for row in rows:
        parsed.append(
            {
                "corpus": row["corpus"],
                "task_id": row["task_id"],
                "task_name": row["task_name"],
                "source_file": row["source_file"],
                "valid_train_count": int(
                    row["valid_train_count"]
                ),
                "valid_validation_count": int(
                    row["valid_validation_count"]
                ),
            }
        )

    return parsed


def generate_rows(
    task_specs: list[dict[str, Any]],
    split: str,
) -> Iterator[dict[str, Any]]:
    for task in task_specs:
        source_file = Path(task["source_file"])

        with source_file.open("rb") as handle:
            rows = ijson.items(
                handle,
                f"{split}.item",
            )

            for source_index, row in enumerate(rows):
                if not valid_row(row):
                    continue

                yield {
                    "inputs": row["inputs"],
                    "targets": row["targets"],
                    "task_source": task["corpus"],
                    "task_name": task["task_label"],
                    "template_type": TEMPLATE_TYPE,
                    "source_index": source_index,
                }


def atomic_write_json(
    payload: Any,
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

    manifest_path = args.manifest.resolve()
    output_root = args.output_root.resolve()
    cache_root = args.cache_root.resolve()

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )
    cache_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest = load_manifest(manifest_path)
    run_results: list[dict[str, Any]] = []

    for corpus in args.corpora:
        task_specs = sorted(
            (
                row
                for row in manifest
                if row["corpus"] == corpus
            ),
            key=lambda row: row["task_id"],
        )

        if not task_specs:
            raise RuntimeError(
                f"No manifest tasks found for {corpus}."
            )

        task_names = [
            row["task_id"]
            for row in task_specs
        ]

        if len(task_names) != len(set(task_names)):
            raise RuntimeError(
                f"Duplicate task IDs in corpus {corpus}."
            )

        task_label_lookup = {
            task_id: index
            for index, task_id in enumerate(task_names)
        }

        generator_specs: list[dict[str, Any]] = []

        for row in task_specs:
            generator_specs.append(
                {
                    **row,
                    "task_label": task_label_lookup[
                        row["task_id"]
                    ],
                }
            )

        features = Features(
            {
                "inputs": Value("string"),
                "targets": Value("string"),
                "task_source": Value("string"),
                "task_name": ClassLabel(
                    names=task_names
                ),
                "template_type": Value("string"),
                "source_index": Value("int64"),
            }
        )

        expected_train = sum(
            row["valid_train_count"]
            for row in task_specs
        )
        expected_validation = sum(
            row["valid_validation_count"]
            for row in task_specs
        )

        print()
        print(f"=== Building {corpus} ===")
        print(f"Tasks:               {len(task_specs)}")
        print(f"Expected train:      {expected_train:,}")
        print(
            f"Expected validation: {expected_validation:,}"
        )

        train_dataset = Dataset.from_generator(
            generate_rows,
            gen_kwargs={
                "task_specs": generator_specs,
                "split": "train",
            },
            features=features,
            cache_dir=str(cache_root / corpus / "train"),
            keep_in_memory=False,
        )

        validation_dataset = Dataset.from_generator(
            generate_rows,
            gen_kwargs={
                "task_specs": generator_specs,
                "split": "validation",
            },
            features=features,
            cache_dir=str(
                cache_root / corpus / "validation"
            ),
            keep_in_memory=False,
        )

        dataset = DatasetDict(
            {
                "train": train_dataset,
                "validation": validation_dataset,
            }
        )

        if len(train_dataset) != expected_train:
            raise RuntimeError(
                f"{corpus}: train count mismatch: "
                f"{len(train_dataset):,} != "
                f"{expected_train:,}"
            )

        if (
            len(validation_dataset)
            != expected_validation
        ):
            raise RuntimeError(
                f"{corpus}: validation count mismatch: "
                f"{len(validation_dataset):,} != "
                f"{expected_validation:,}"
            )

        train_template_types = set(
            train_dataset.unique(
                "template_type"
            )
        )

        validation_template_types = set(
            validation_dataset.unique(
                "template_type"
            )
        )

        if train_template_types != {TEMPLATE_TYPE}:
            raise RuntimeError(
                f"{corpus}: unexpected train template "
                f"types: {train_template_types}"
            )

        if (
            len(validation_dataset) > 0
            and validation_template_types
            != {TEMPLATE_TYPE}
        ):
            raise RuntimeError(
                f"{corpus}: unexpected validation template "
                f"types: {validation_template_types}"
            )

        if (
            train_dataset.features[
                "task_name"
            ].names
            != task_names
        ):
            raise RuntimeError(
                f"{corpus}: task ClassLabel names changed."
            )

        final_path = output_root / corpus
        temporary_path = (
            output_root / f".{corpus}.tmp"
        )

        if final_path.exists():
            if not args.overwrite:
                raise FileExistsError(
                    f"Output already exists: {final_path}"
                )

            shutil.rmtree(final_path)

        if temporary_path.exists():
            shutil.rmtree(temporary_path)

        dataset.save_to_disk(
            str(temporary_path)
        )
        temporary_path.replace(final_path)

        task_map_path = (
            output_root
            / f"{corpus}_task_labels.json"
        )

        atomic_write_json(
            {
                "corpus": corpus,
                "task_count": len(task_names),
                "task_label_names": task_names,
            },
            task_map_path,
        )

        result = {
            "corpus": corpus,
            "task_count": len(task_names),
            "train_rows": len(train_dataset),
            "validation_rows": len(
                validation_dataset
            ),
            "template_types": [TEMPLATE_TYPE],
            "dataset_path": str(final_path),
            "task_labels_path": str(
                task_map_path
            ),
        }

        run_results.append(result)

        print(f"Saved: {final_path}")
        print("Schema:")
        print(train_dataset.features)

    summary_path = (
        output_root / "build_summary.json"
    )

    existing_results: dict[str, Any] = {}

    if summary_path.is_file():
        with summary_path.open(
            "r",
            encoding="utf-8",
        ) as handle:
            existing = json.load(handle)

        existing_results = {
            item["corpus"]: item
            for item in existing.get(
                "corpora",
                []
            )
        }

    for result in run_results:
        existing_results[
            result["corpus"]
        ] = result

    atomic_write_json(
        {
            "format_version": 1,
            "status": "complete",
            "template_policy": (
                "All valid rows use the single "
                "zs_noopt template bucket."
            ),
            "corpora": [
                existing_results[key]
                for key in sorted(
                    existing_results
                )
            ],
        },
        summary_path,
    )

    print()
    print("Author-format build passed.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
PY

python3 -m py_compile \
  src/data/build_author_format.py
```

## 3. Build only `cot`

```bash
cd /data/saral/wdir/smart_v2 || exit 1
source ./smart_v2_env.sh

python3 -m src.data.build_author_format \
  --manifest /mnt/warm_storage/saral/smart_v2/manifests/task_manifest.csv \
  --output-root /mnt/warm_storage/saral/smart_v2/author_format \
  --cache-root /mnt/warm_storage/saral/smart_v2/cache/author_format_build \
  --corpora cot \
  2>&1 | tee \
  /mnt/warm_storage/saral/smart_v2/logs/build_author_format_cot.log
```

## 4. Verify the authors-compatible behavior

```bash
python3 - <<'PY'
from datasets import load_from_disk

path = (
    "/mnt/warm_storage/saral/smart_v2/"
    "author_format/cot"
)

dataset = load_from_disk(path)
train = dataset["train"]
validation = dataset["validation"]

print("Train rows:", f"{len(train):,}")
print("Validation rows:", f"{len(validation):,}")
print("Features:", train.features)
print("Task count:", len(train.features["task_name"].names))
print("Template types:", train.unique("template_type"))

print("\nFirst row:")
row = train[0]
print(row)

label = row["task_name"]
print(
    "\nResolved task name:",
    train.features["task_name"].int2str(label),
)
PY
```

Acceptance conditions:

```text
task_name feature  = ClassLabel
task count         = 18
template types     = ['zs_noopt']
inputs/targets     = strings
source_index       = original JSON row index
```

After this smoke test passes, build the remaining four corpora with the same script.
