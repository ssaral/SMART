Once the allocations passes. They match the earlier run exactly at the integer-budget level, and the largest required Stage 2 prefix is 404.

Before implementing Facility Location, we need a frozen manifest of every task/template pool. The authors’ mixture code consumes prefixes from precomputed instance orderings rather than recomputing Facility Location during mixture assembly. 

## Step 9 — Build the Stage 2 pool manifest

This step computes:

* complete candidate-pool size;
* 25K and 50K template budgets;
* required ordering prefix;
* embedding-memory size;
* raw dense-kernel memory;
* candidate pools suitable for dense/blockwise parity testing.

### 9.1 Create the script

```bash
cd /data/saral/wdir/smart_v2 || exit 1

cat > src/stage2/build_pool_manifest.py <<'PY'
"""Build the SMART-v2 Stage 2 task/template pool manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any


TEMPLATE_TYPES = (
    "zs_opt",
    "zs_noopt",
    "fs_opt",
    "fs_noopt",
)

EXPECTED_TEMPLATE_TYPE = "zs_noopt"
EXPECTED_TASK_COUNT = 309
EXPECTED_TOTAL_ROWS = 6_266_471


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--allocations",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--submixture-config",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--task-indices-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--embedding-dimension",
        type=int,
        default=1024,
    )
    parser.add_argument(
        "--budgets",
        type=int,
        nargs="+",
        default=[25_000, 50_000],
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


def load_submixtures(path: Path) -> list[str]:
    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        payload = json.load(handle)

    values = payload.get("submixtures")

    if not isinstance(values, list) or not values:
        raise ValueError(
            "Submixture configuration is invalid."
        )

    if len(values) != len(set(values)):
        raise ValueError(
            "Submixture list contains duplicates."
        )

    return values


def load_allocations(
    path: Path,
    budgets: list[int],
) -> dict[str, dict[str, Any]]:
    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        raw_rows = list(csv.DictReader(handle))

    if not raw_rows:
        raise RuntimeError(
            "Allocation CSV is empty."
        )

    required = {
        "graph_cut_rank",
        "task_index",
        "task_id",
        "corpus",
        "valid_train_count",
        "required_stage2_prefix",
    }

    for budget in budgets:
        required.add(
            f"final_task_budget_{budget}"
        )
        required.add(
            f"zs_noopt_budget_{budget}"
        )

    missing = required - set(raw_rows[0])

    if missing:
        raise ValueError(
            "Allocation CSV is missing columns: "
            + ", ".join(sorted(missing))
        )

    result: dict[str, dict[str, Any]] = {}

    for raw in raw_rows:
        task_id = raw["task_id"]

        if task_id in result:
            raise RuntimeError(
                f"Duplicate allocation task: {task_id}"
            )

        row: dict[str, Any] = {
            "graph_cut_rank": int(
                raw["graph_cut_rank"]
            ),
            "task_index": int(
                raw["task_index"]
            ),
            "task_id": task_id,
            "corpus": raw["corpus"],
            "valid_train_count": int(
                raw["valid_train_count"]
            ),
            "required_stage2_prefix": int(
                raw["required_stage2_prefix"]
            ),
        }

        for budget in budgets:
            row[
                f"task_budget_{budget}"
            ] = int(
                raw[
                    f"final_task_budget_{budget}"
                ]
            )

            row[
                f"zs_noopt_budget_{budget}"
            ] = int(
                raw[
                    f"zs_noopt_budget_{budget}"
                ]
            )

        result[task_id] = row

    if len(result) != EXPECTED_TASK_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_TASK_COUNT} allocation "
            f"rows; found {len(result)}."
        )

    return result


def gib(byte_count: int) -> float:
    return byte_count / (1024**3)


def size_class(pool_size: int) -> str:
    if pool_size <= 1_000:
        return "tiny"
    if pool_size <= 5_000:
        return "small"
    if pool_size <= 20_000:
        return "medium"
    if pool_size <= 100_000:
        return "large"
    return "very_large"


def main() -> int:
    args = parse_args()

    if args.embedding_dimension <= 0:
        raise ValueError(
            "--embedding-dimension must be positive."
        )

    budgets = list(
        dict.fromkeys(args.budgets)
    )

    if not budgets:
        raise ValueError(
            "At least one budget is required."
        )

    allocations_path = (
        args.allocations.resolve()
    )
    config_path = (
        args.submixture_config.resolve()
    )
    indices_root = (
        args.task_indices_root.resolve()
    )
    output_root = args.output_root.resolve()

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    submixtures = load_submixtures(
        config_path
    )

    allocations = load_allocations(
        allocations_path,
        budgets,
    )

    pool_rows: list[dict[str, Any]] = []
    total_pool_rows = 0
    seen_tasks: set[str] = set()

    for corpus_position, corpus in enumerate(
        submixtures,
        start=1,
    ):
        pickle_path = (
            indices_root / f"{corpus}.pkl"
        )

        if not pickle_path.is_file():
            raise FileNotFoundError(
                pickle_path
            )

        with pickle_path.open("rb") as handle:
            task_indices = pickle.load(handle)

        if not isinstance(task_indices, dict):
            raise TypeError(
                f"{pickle_path} is not a dictionary."
            )

        for corpus_task_position, (
            task_id,
            template_mapping,
        ) in enumerate(
            task_indices.items(),
            start=1,
        ):
            if task_id in seen_tasks:
                raise RuntimeError(
                    f"Duplicate task ID: {task_id}"
                )

            seen_tasks.add(task_id)

            if task_id not in allocations:
                raise RuntimeError(
                    f"Missing allocation for {task_id}"
                )

            allocation = allocations[task_id]

            if allocation["corpus"] != corpus:
                raise RuntimeError(
                    f"{task_id}: corpus mismatch."
                )

            available_templates = set(
                template_mapping
            )

            if available_templates != {
                EXPECTED_TEMPLATE_TYPE
            }:
                raise RuntimeError(
                    f"{task_id}: expected only "
                    f"{EXPECTED_TEMPLATE_TYPE}; found "
                    f"{sorted(available_templates)}."
                )

            indices = template_mapping[
                EXPECTED_TEMPLATE_TYPE
            ]

            pool_size = len(indices)

            if pool_size <= 0:
                raise RuntimeError(
                    f"{task_id}: empty Stage 2 pool."
                )

            if (
                pool_size
                != allocation[
                    "valid_train_count"
                ]
            ):
                raise RuntimeError(
                    f"{task_id}: pool size "
                    f"{pool_size:,} differs from "
                    f"allocation capacity "
                    f"{allocation['valid_train_count']:,}."
                )

            required_prefix = allocation[
                "required_stage2_prefix"
            ]

            if required_prefix > pool_size:
                raise RuntimeError(
                    f"{task_id}: required prefix "
                    f"{required_prefix:,} exceeds "
                    f"pool size {pool_size:,}."
                )

            embedding_bytes = (
                pool_size
                * args.embedding_dimension
                * 4
            )

            dense_kernel_elements = (
                pool_size * pool_size
            )
            dense_kernel_float32_bytes = (
                dense_kernel_elements * 4
            )
            dense_kernel_float64_bytes = (
                dense_kernel_elements * 8
            )

            row: dict[str, Any] = {
                "pool_rank": len(pool_rows) + 1,
                "corpus_position": (
                    corpus_position
                ),
                "corpus_task_position": (
                    corpus_task_position
                ),
                "graph_cut_rank": (
                    allocation[
                        "graph_cut_rank"
                    ]
                ),
                "task_index": (
                    allocation["task_index"]
                ),
                "corpus": corpus,
                "task_id": task_id,
                "template_type": (
                    EXPECTED_TEMPLATE_TYPE
                ),
                "pool_size": pool_size,
                "required_stage2_prefix": (
                    required_prefix
                ),
                "authors_full_ordering_budget": (
                    pool_size - 1
                ),
                "select_all_without_ordering": (
                    required_prefix == pool_size
                ),
                "size_class": size_class(
                    pool_size
                ),
                "embedding_dimension": (
                    args.embedding_dimension
                ),
                "embedding_float32_bytes": (
                    embedding_bytes
                ),
                "embedding_float32_gib": gib(
                    embedding_bytes
                ),
                "dense_kernel_elements": (
                    dense_kernel_elements
                ),
                "dense_kernel_float32_bytes": (
                    dense_kernel_float32_bytes
                ),
                "dense_kernel_float32_gib": gib(
                    dense_kernel_float32_bytes
                ),
                "dense_kernel_float64_gib": gib(
                    dense_kernel_float64_bytes
                ),
                "first_dataset_index": (
                    int(indices[0])
                ),
                "last_dataset_index": (
                    int(indices[-1])
                ),
                "indices_contiguous": all(
                    later == earlier + 1
                    for earlier, later in zip(
                        indices,
                        indices[1:],
                    )
                ),
            }

            for budget in budgets:
                task_budget = allocation[
                    f"task_budget_{budget}"
                ]
                template_budget = allocation[
                    f"zs_noopt_budget_{budget}"
                ]

                if task_budget != template_budget:
                    raise RuntimeError(
                        f"{task_id}: task/template "
                        f"budget mismatch for {budget}."
                    )

                if template_budget > pool_size:
                    raise RuntimeError(
                        f"{task_id}: budget "
                        f"{template_budget:,} exceeds "
                        f"pool size {pool_size:,}."
                    )

                row[
                    f"budget_{budget}"
                ] = template_budget

            pool_rows.append(row)
            total_pool_rows += pool_size

    if len(seen_tasks) != EXPECTED_TASK_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_TASK_COUNT} tasks; "
            f"found {len(seen_tasks)}."
        )

    if total_pool_rows != EXPECTED_TOTAL_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_TOTAL_ROWS:,} pool rows; "
            f"found {total_pool_rows:,}."
        )

    if set(allocations) != seen_tasks:
        missing = sorted(
            set(allocations) - seen_tasks
        )
        unexpected = sorted(
            seen_tasks - set(allocations)
        )

        raise RuntimeError(
            "Allocation and task-index sets differ. "
            f"Missing={missing}, unexpected={unexpected}"
        )

    if not all(
        row["indices_contiguous"]
        for row in pool_rows
    ):
        bad = [
            row["task_id"]
            for row in pool_rows
            if not row["indices_contiguous"]
        ]

        raise RuntimeError(
            f"Non-contiguous task pools: {bad}"
        )

    manifest_path = (
        output_root / "pool_manifest.csv"
    )

    with manifest_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                pool_rows[0].keys()
            ),
        )
        writer.writeheader()
        writer.writerows(pool_rows)

    largest_by_pool = sorted(
        pool_rows,
        key=lambda row: row["pool_size"],
        reverse=True,
    )

    largest_by_dense_memory = sorted(
        pool_rows,
        key=lambda row: (
            row[
                "dense_kernel_float32_bytes"
            ]
        ),
        reverse=True,
    )

    size_class_counts: dict[str, int] = {}

    for row in pool_rows:
        name = row["size_class"]
        size_class_counts[name] = (
            size_class_counts.get(name, 0)
            + 1
        )

    parity_candidates: dict[
        str,
        list[dict[str, Any]],
    ] = {}

    ranges = {
        "tiny": (100, 1_000),
        "small": (1_001, 5_000),
        "medium": (5_001, 20_000),
    }

    for label, (
        minimum,
        maximum,
    ) in ranges.items():
        candidates = [
            row
            for row in pool_rows
            if (
                minimum
                <= row["pool_size"]
                <= maximum
            )
        ]

        candidates.sort(
            key=lambda row: (
                abs(
                    row["pool_size"]
                    - (
                        minimum
                        + maximum
                    )
                    // 2
                ),
                row["task_id"],
            )
        )

        parity_candidates[label] = [
            {
                "task_id": row["task_id"],
                "corpus": row["corpus"],
                "pool_size": row["pool_size"],
                "required_stage2_prefix": (
                    row[
                        "required_stage2_prefix"
                    ]
                ),
                "dense_kernel_float32_gib": (
                    row[
                        "dense_kernel_float32_gib"
                    ]
                ),
            }
            for row in candidates[:5]
        ]

    summary = {
        "format_version": 1,
        "stage": (
            "smart_v2_stage2_pool_manifest"
        ),
        "status": "complete",
        "configuration": {
            "template_types": list(
                TEMPLATE_TYPES
            ),
            "available_template_type": (
                EXPECTED_TEMPLATE_TYPE
            ),
            "embedding_dimension": (
                args.embedding_dimension
            ),
            "budgets": budgets,
            "dense_kernel_estimate": (
                "n*n*sizeof(dtype), excluding "
                "temporary buffers and library copies"
            ),
        },
        "pool_count": len(pool_rows),
        "task_count": len(seen_tasks),
        "total_candidate_rows": (
            total_pool_rows
        ),
        "maximum_required_stage2_prefix": max(
            row["required_stage2_prefix"]
            for row in pool_rows
        ),
        "size_class_counts": (
            size_class_counts
        ),
        "largest_pools": [
            {
                "task_id": row["task_id"],
                "corpus": row["corpus"],
                "pool_size": row["pool_size"],
                "required_stage2_prefix": (
                    row[
                        "required_stage2_prefix"
                    ]
                ),
                "embedding_float32_gib": (
                    row[
                        "embedding_float32_gib"
                    ]
                ),
                "dense_kernel_float32_gib": (
                    row[
                        "dense_kernel_float32_gib"
                    ]
                ),
            }
            for row in largest_by_pool[:20]
        ],
        "largest_dense_kernels": [
            {
                "task_id": row["task_id"],
                "pool_size": row["pool_size"],
                "dense_kernel_float32_gib": (
                    row[
                        "dense_kernel_float32_gib"
                    ]
                ),
            }
            for row in largest_by_dense_memory[
                :20
            ]
        ],
        "raw_dense_kernel_threshold_counts": {
            "over_16_gib": sum(
                row[
                    "dense_kernel_float32_gib"
                ] > 16.0
                for row in pool_rows
            ),
            "over_32_gib": sum(
                row[
                    "dense_kernel_float32_gib"
                ] > 32.0
                for row in pool_rows
            ),
            "over_64_gib": sum(
                row[
                    "dense_kernel_float32_gib"
                ] > 64.0
                for row in pool_rows
            ),
            "over_128_gib": sum(
                row[
                    "dense_kernel_float32_gib"
                ] > 128.0
                for row in pool_rows
            ),
        },
        "parity_test_candidates": (
            parity_candidates
        ),
        "inputs": {
            "allocations": str(
                allocations_path
            ),
            "allocations_sha256": (
                sha256_file(
                    allocations_path
                )
            ),
            "submixture_config": str(
                config_path
            ),
            "submixture_config_sha256": (
                sha256_file(
                    config_path
                )
            ),
            "task_indices_root": str(
                indices_root
            ),
        },
        "outputs": {
            "pool_manifest": str(
                manifest_path
            ),
            "pool_manifest_sha256": (
                sha256_file(
                    manifest_path
                )
            ),
        },
    }

    summary_path = (
        output_root / "pool_summary.json"
    )

    atomic_write_json(
        summary,
        summary_path,
    )

    print("=== SMART-v2 Stage 2 pools ===")
    print(f"Pools:                 {len(pool_rows)}")
    print(f"Tasks:                 {len(seen_tasks)}")
    print(
        f"Candidate rows:        "
        f"{total_pool_rows:,}"
    )
    print(
        f"Maximum prefix:        "
        f"{summary['maximum_required_stage2_prefix']}"
    )

    print()
    print("Pool-size classes:")

    for name in (
        "tiny",
        "small",
        "medium",
        "large",
        "very_large",
    ):
        print(
            f"  {name:12s} "
            f"{size_class_counts.get(name, 0)}"
        )

    print()
    print("Largest 10 pools:")

    for rank, row in enumerate(
        largest_by_pool[:10],
        start=1,
    ):
        print(
            f"  {rank:2d}. "
            f"{row['task_id']} "
            f"n={row['pool_size']:,} "
            f"prefix={row['required_stage2_prefix']} "
            f"dense32="
            f"{row['dense_kernel_float32_gib']:.2f} GiB"
        )

    print()
    print("Raw float32 dense-kernel counts:")
    thresholds = summary[
        "raw_dense_kernel_threshold_counts"
    ]

    for key, value in thresholds.items():
        print(f"  {key:14s} {value}")

    print()
    print(f"Manifest: {manifest_path}")
    print(f"Summary:  {summary_path}")
    print()
    print(
        "Stage 2 pool-manifest construction passed."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
PY

python3 -m py_compile \
  src/stage2/build_pool_manifest.py
```

### 9.2 Run it

```bash
cd /data/saral/wdir/smart_v2 || exit 1
source ./smart_v2_env.sh

mkdir -p \
  /mnt/warm_storage/saral/smart_v2/stage2/pools

python3 -m src.stage2.build_pool_manifest \
  --allocations /mnt/warm_storage/saral/smart_v2/stage1/allocations/task_allocations.csv \
  --submixture-config /data/saral/wdir/smart_v2/configs/submixtures.json \
  --task-indices-root /mnt/warm_storage/saral/smart_v2/author_format/task_indices \
  --output-root /mnt/warm_storage/saral/smart_v2/stage2/pools \
  --embedding-dimension 1024 \
  --budgets 25000 50000 \
  2>&1 | tee \
  /mnt/warm_storage/saral/smart_v2/logs/stage2_pool_manifest.log
```

### 9.3 Inspect the compact summary

```bash
python3 - <<'PY'
import json
from pathlib import Path

path = Path(
    "/mnt/warm_storage/saral/smart_v2/"
    "stage2/pools/pool_summary.json"
)

with path.open(encoding="utf-8") as handle:
    summary = json.load(handle)

print("Status:", summary["status"])
print("Pools:", summary["pool_count"])
print(
    "Candidate rows:",
    f"{summary['total_candidate_rows']:,}",
)
print(
    "Maximum prefix:",
    summary["maximum_required_stage2_prefix"],
)

print("\nSize classes:")
for key, value in summary[
    "size_class_counts"
].items():
    print(f"  {key:12s} {value}")

print("\nDense thresholds:")
for key, value in summary[
    "raw_dense_kernel_threshold_counts"
].items():
    print(f"  {key:14s} {value}")

print("\nLargest 10 pools:")
for row in summary["largest_pools"][:10]:
    print(
        f"{row['pool_size']:9,d}  "
        f"{row['required_stage2_prefix']:4d}  "
        f"{row['dense_kernel_float32_gib']:9.2f} GiB  "
        f"{row['task_id']}"
    )

print("\nParity candidates:")
for group, rows in summary[
    "parity_test_candidates"
].items():
    print(f"\n{group}:")
    for row in rows:
        print(
            f"  n={row['pool_size']:6,d} "
            f"prefix={row['required_stage2_prefix']:3d} "
            f"{row['task_id']}"
        )
PY
```

Acceptance conditions:

```text
status                  = complete
pool count              = 309
task count              = 309
candidate rows          = 6,266,471
maximum prefix          = 404
all indices contiguous  = true
every budget            <= pool size
```

The following step will implement the dense authors-code oracle and the exact blockwise Facility Location engine, then require parity on the tiny, small, and medium pools identified here.
