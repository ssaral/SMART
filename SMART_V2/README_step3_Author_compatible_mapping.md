Once the five author-format corpora are built correctly. Follow these steps:

The prompt examples show that some rendered prompts contain demonstrations and options, but for the primary baseline we are intentionally treating each task as one available template pool, `zs_noopt`, rather than letting heuristic labels alter the authors’ allocation logic.

Next, build the authors-compatible mapping:

```python
task_indices[task_name][template_type] = [dataset_row_indices]
```

The authors create this mapping before computing task means and template budgets. 

# Step 4 — Build and verify task-index files

## 4.1 Create the script

```bash
cd /data/saral/wdir/smart_v2 || exit 1

cat > src/data/build_task_indices.py <<'PY'
"""Build authors-compatible SMART task/template index mappings."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any

from datasets import load_from_disk


EXPECTED_CORPORA = (
    "cot",
    "flan2021",
    "sglue",
    "t0",
    "tulu",
)

EXPECTED_TASK_COUNT = 309
EXPECTED_TRAIN_ROWS = 6_266_471
EXPECTED_TEMPLATE_TYPE = "zs_noopt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build one SMART-compatible task_indices pickle "
            "for each local author-format corpus."
        )
    )

    parser.add_argument(
        "--author-format-root",
        type=Path,
        required=True,
    )
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
        "--batch-size",
        type=int,
        default=100_000,
    )

    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


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


def atomic_write_pickle(
    payload: Any,
    path: Path,
) -> None:
    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    with temporary.open("wb") as handle:
        pickle.dump(
            payload,
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    temporary.replace(path)


def load_manifest_counts(
    path: Path,
) -> dict[str, dict[str, int]]:
    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise RuntimeError("Task manifest is empty.")

    required = {
        "corpus",
        "task_id",
        "valid_train_count",
    }

    missing = required - set(rows[0])

    if missing:
        raise ValueError(
            "Manifest is missing columns: "
            + ", ".join(sorted(missing))
        )

    result: dict[str, dict[str, int]] = {
        corpus: {}
        for corpus in EXPECTED_CORPORA
    }

    for row in rows:
        corpus = row["corpus"]
        task_id = row["task_id"]

        if corpus not in result:
            raise ValueError(
                f"Unexpected corpus in manifest: {corpus}"
            )

        if task_id in result[corpus]:
            raise RuntimeError(
                f"Duplicate task ID in manifest: {task_id}"
            )

        result[corpus][task_id] = int(
            row["valid_train_count"]
        )

    return result


def main() -> int:
    args = parse_args()

    if args.batch_size <= 0:
        raise ValueError(
            "--batch-size must be positive."
        )

    author_root = (
        args.author_format_root.resolve()
    )
    manifest_path = args.manifest.resolve()
    output_root = args.output_root.resolve()

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest_counts = load_manifest_counts(
        manifest_path
    )

    total_tasks = 0
    total_rows = 0
    corpus_summaries: list[dict[str, Any]] = []

    for corpus in EXPECTED_CORPORA:
        dataset_path = author_root / corpus

        if not dataset_path.is_dir():
            raise FileNotFoundError(
                f"Missing author-format dataset: "
                f"{dataset_path}"
            )

        dataset = load_from_disk(
            str(dataset_path)
        )
        train = dataset["train"]

        task_feature = train.features[
            "task_name"
        ]

        if not hasattr(task_feature, "names"):
            raise RuntimeError(
                f"{corpus}: task_name is not a ClassLabel."
            )

        task_names = list(task_feature.names)

        expected_task_names = set(
            manifest_counts[corpus]
        )

        if set(task_names) != expected_task_names:
            missing = sorted(
                expected_task_names - set(task_names)
            )
            unexpected = sorted(
                set(task_names) - expected_task_names
            )

            raise RuntimeError(
                f"{corpus}: task-name mismatch. "
                f"Missing={missing}, unexpected={unexpected}"
            )

        temporary_mapping: dict[
            str,
            dict[str, list[int]],
        ] = {
            task_name: defaultdict(list)
            for task_name in task_names
        }

        print()
        print(f"=== Building indices for {corpus} ===")
        print(f"Rows:  {len(train):,}")
        print(f"Tasks: {len(task_names)}")

        for batch_start in range(
            0,
            len(train),
            args.batch_size,
        ):
            batch_end = min(
                batch_start + args.batch_size,
                len(train),
            )

            batch = train[
                batch_start:batch_end
            ]

            task_labels = batch["task_name"]
            template_types = batch[
                "template_type"
            ]

            if len(task_labels) != len(
                template_types
            ):
                raise RuntimeError(
                    f"{corpus}: malformed batch at "
                    f"{batch_start}:{batch_end}"
                )

            for offset, (
                task_label,
                template_type,
            ) in enumerate(
                zip(
                    task_labels,
                    template_types,
                    strict=True,
                )
            ):
                dataset_index = (
                    batch_start + offset
                )

                task_name = task_feature.int2str(
                    int(task_label)
                )

                temporary_mapping[
                    task_name
                ][template_type].append(
                    dataset_index
                )

            print(
                f"  Indexed "
                f"{batch_end:,}/{len(train):,}",
                flush=True,
            )

        task_indices: dict[
            str,
            dict[str, list[int]],
        ] = {}

        corpus_index_count = 0
        per_task_summary: list[
            dict[str, Any]
        ] = []

        for task_name in task_names:
            template_mapping = dict(
                temporary_mapping[task_name]
            )

            actual_template_types = set(
                template_mapping
            )

            if actual_template_types != {
                EXPECTED_TEMPLATE_TYPE
            }:
                raise RuntimeError(
                    f"{corpus}/{task_name}: unexpected "
                    f"template types "
                    f"{sorted(actual_template_types)}"
                )

            indices = template_mapping[
                EXPECTED_TEMPLATE_TYPE
            ]

            if not indices:
                raise RuntimeError(
                    f"{corpus}/{task_name}: "
                    "empty task index list."
                )

            if any(
                later <= earlier
                for earlier, later in zip(
                    indices,
                    indices[1:],
                )
            ):
                raise RuntimeError(
                    f"{corpus}/{task_name}: "
                    "indices are not strictly increasing."
                )

            if indices[0] < 0:
                raise RuntimeError(
                    f"{corpus}/{task_name}: "
                    "negative dataset index."
                )

            if indices[-1] >= len(train):
                raise RuntimeError(
                    f"{corpus}/{task_name}: "
                    "dataset index exceeds train size."
                )

            actual_count = len(indices)
            expected_count = (
                manifest_counts[corpus][
                    task_name
                ]
            )

            if actual_count != expected_count:
                raise RuntimeError(
                    f"{corpus}/{task_name}: "
                    f"count mismatch: "
                    f"indices={actual_count:,}, "
                    f"manifest={expected_count:,}"
                )

            task_indices[task_name] = {
                EXPECTED_TEMPLATE_TYPE: indices
            }

            corpus_index_count += actual_count

            per_task_summary.append(
                {
                    "task_id": task_name,
                    "template_type": (
                        EXPECTED_TEMPLATE_TYPE
                    ),
                    "index_count": actual_count,
                    "first_dataset_index": (
                        indices[0]
                    ),
                    "last_dataset_index": (
                        indices[-1]
                    ),
                }
            )

        if corpus_index_count != len(train):
            raise RuntimeError(
                f"{corpus}: indexed "
                f"{corpus_index_count:,} rows but "
                f"dataset contains {len(train):,}."
            )

        pickle_path = (
            output_root / f"{corpus}.pkl"
        )
        task_summary_path = (
            output_root
            / f"{corpus}_task_indices.json"
        )

        atomic_write_pickle(
            task_indices,
            pickle_path,
        )

        atomic_write_json(
            {
                "corpus": corpus,
                "dataset_path": str(
                    dataset_path
                ),
                "task_count": len(task_names),
                "indexed_train_rows": (
                    corpus_index_count
                ),
                "template_types": [
                    EXPECTED_TEMPLATE_TYPE
                ],
                "tasks": per_task_summary,
            },
            task_summary_path,
        )

        # Reload the pickle to detect serialization
        # or filesystem corruption immediately.
        with pickle_path.open("rb") as handle:
            reloaded = pickle.load(handle)

        if reloaded != task_indices:
            raise RuntimeError(
                f"{corpus}: pickle reload mismatch."
            )

        corpus_summary = {
            "corpus": corpus,
            "task_count": len(task_names),
            "train_rows": len(train),
            "indexed_rows": corpus_index_count,
            "template_types": [
                EXPECTED_TEMPLATE_TYPE
            ],
            "pickle_path": str(
                pickle_path
            ),
            "pickle_sha256": sha256_file(
                pickle_path
            ),
            "task_summary_path": str(
                task_summary_path
            ),
        }

        corpus_summaries.append(
            corpus_summary
        )

        total_tasks += len(task_names)
        total_rows += corpus_index_count

        print(f"Saved: {pickle_path}")
        print(
            f"Indexed rows: {corpus_index_count:,}"
        )

    if total_tasks != EXPECTED_TASK_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_TASK_COUNT} tasks; "
            f"found {total_tasks}."
        )

    if total_rows != EXPECTED_TRAIN_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_TRAIN_ROWS:,} train "
            f"indices; found {total_rows:,}."
        )

    summary_path = (
        output_root
        / "task_indices_summary.json"
    )

    atomic_write_json(
        {
            "format_version": 1,
            "stage": (
                "smart_v2_author_compatible_task_indices"
            ),
            "status": "complete",
            "template_policy": (
                "single available zs_noopt bucket"
            ),
            "corpus_count": len(
                EXPECTED_CORPORA
            ),
            "task_count": total_tasks,
            "indexed_train_rows": total_rows,
            "expected_task_count": (
                EXPECTED_TASK_COUNT
            ),
            "expected_train_rows": (
                EXPECTED_TRAIN_ROWS
            ),
            "corpora": corpus_summaries,
        },
        summary_path,
    )

    print()
    print("=== Task-index summary ===")
    print(f"Corpora:      {len(EXPECTED_CORPORA)}")
    print(f"Tasks:        {total_tasks}")
    print(f"Train rows:   {total_rows:,}")
    print(
        f"Template:     {EXPECTED_TEMPLATE_TYPE}"
    )
    print(f"Summary:      {summary_path}")
    print()
    print("Task-index generation passed.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
PY

python3 -m py_compile \
  src/data/build_task_indices.py
```

## 4.2 Run it

```bash
cd /data/saral/wdir/smart_v2 || exit 1
source ./smart_v2_env.sh

mkdir -p \
  /mnt/warm_storage/saral/smart_v2/author_format/task_indices

python3 -m src.data.build_task_indices \
  --author-format-root /mnt/warm_storage/saral/smart_v2/author_format \
  --manifest /mnt/warm_storage/saral/smart_v2/manifests/task_manifest.csv \
  --output-root /mnt/warm_storage/saral/smart_v2/author_format/task_indices \
  --batch-size 100000 \
  2>&1 | tee \
  /mnt/warm_storage/saral/smart_v2/logs/build_task_indices.log
```

## 4.3 Inspect the result

```bash
cat \
  /mnt/warm_storage/saral/smart_v2/author_format/task_indices/task_indices_summary.json
```

Check one pickle:

```bash
python3 - <<'PY'
import pickle
from pathlib import Path

path = Path(
    "/mnt/warm_storage/saral/smart_v2/"
    "author_format/task_indices/cot.pkl"
)

with path.open("rb") as handle:
    mapping = pickle.load(handle)

print("Task count:", len(mapping))

for task_id in list(mapping)[:3]:
    templates = mapping[task_id]

    print()
    print("Task:", task_id)
    print("Templates:", list(templates))

    indices = templates["zs_noopt"]

    print("Count:", len(indices))
    print("First indices:", indices[:10])
    print("Last indices:", indices[-10:])
PY
```

Expected final summary:

```text
Corpora:      5
Tasks:        309
Train rows:   6,266,471
Template:     zs_noopt
Task-index generation passed.
```

Expected files:

```text
/mnt/warm_storage/saral/smart_v2/author_format/task_indices/
├── cot.pkl
├── flan2021.pkl
├── sglue.pkl
├── t0.pkl
├── tulu.pkl
├── cot_task_indices.json
├── flan2021_task_indices.json
├── sglue_task_indices.json
├── t0_task_indices.json
├── tulu_task_indices.json
└── task_indices_summary.json
```

Once this passes, the next implementation step is generating the reusable GTE-large prompt embeddings with mappings aligned exactly to these author-format dataset indices.
