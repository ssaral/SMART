After Stage 2 is completed successfully.

```text
309/309 pools verified
50,000 total ordering positions
19,811,849 exact gain evaluations
0 approximation
```

The large number of tied ranks is expected because many prompts or embeddings are identical or objective-equivalent. The frozen smallest-dataset-index policy makes those choices reproducible.

## Step 15 — Assemble authors-compatible ordering files

The authors’ mixture builder loads precomputed Facility Location orderings and slices each ordering to its task-template budget. 

We will create this structure:

```python
orderings[corpus][task_id][template_type] = [
    (dataset_index, marginal_gain),
    ...
]
```

Only `zs_noopt` contains entries under the frozen single-template policy.

### 15.1 Create the assembly script

```bash
cd /data/saral/wdir/smart_v2 || exit 1

cat > src/stage2/assemble_author_orderings.py <<'PY'
"""Assemble verified Stage 2 outputs into authors-compatible orderings.

Output structure:

    all_orderings[corpus][task_id][template_type]
        = [(dataset_index, marginal_gain), ...]

For the frozen SMART-Single-Template baseline, only zs_noopt contains
ordering entries. The other three template buckets are present but empty.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pickle
import re
from pathlib import Path
from typing import Any


EXPECTED_CORPORA = 5
EXPECTED_TASKS = 309
EXPECTED_TOTAL_ROWS = 6_266_471
EXPECTED_MAX_PREFIX_TOTAL = 50_000

TEMPLATE_TYPES = (
    "zs_opt",
    "zs_noopt",
    "fs_opt",
    "fs_noopt",
)

ACTIVE_TEMPLATE = "zs_noopt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--submixture-config",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--pool-manifest",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--task-indices-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--allocations-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--blockwise-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--batch-summary",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--budgets",
        type=int,
        nargs="+",
        default=[25_000, 50_000],
    )

    return parser.parse_args()


def task_slug(task_id: str) -> str:
    value = task_id.replace(
        "::",
        "__",
    )

    value = re.sub(
        r"[^A-Za-z0-9._-]+",
        "_",
        value,
    )

    return value.strip("_")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            block = handle.read(
                1024 * 1024
            )

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


def atomic_write_json(
    payload: Any,
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

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
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

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


def load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def load_json(path: Path) -> Any:
    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        return json.load(handle)


def load_submixtures(
    path: Path,
) -> list[str]:
    payload = load_json(path)

    submixtures = payload.get(
        "submixtures"
    )

    if not isinstance(
        submixtures,
        list,
    ):
        raise ValueError(
            "Submixture configuration must contain "
            "a submixtures list."
        )

    if len(submixtures) != EXPECTED_CORPORA:
        raise RuntimeError(
            f"Expected {EXPECTED_CORPORA} corpora; "
            f"found {len(submixtures)}."
        )

    if len(submixtures) != len(
        set(submixtures)
    ):
        raise RuntimeError(
            "Submixture list contains duplicates."
        )

    return submixtures


def load_manifest(
    path: Path,
) -> dict[str, dict[str, Any]]:
    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        raw_rows = list(
            csv.DictReader(handle)
        )

    if len(raw_rows) != EXPECTED_TASKS:
        raise RuntimeError(
            f"Expected {EXPECTED_TASKS} manifest rows; "
            f"found {len(raw_rows)}."
        )

    result: dict[
        str,
        dict[str, Any],
    ] = {}

    for raw in raw_rows:
        task_id = raw["task_id"]

        if task_id in result:
            raise RuntimeError(
                f"Duplicate task in manifest: {task_id}"
            )

        result[task_id] = {
            "pool_rank": int(
                raw["pool_rank"]
            ),
            "corpus": raw["corpus"],
            "task_id": task_id,
            "template_type": (
                raw["template_type"]
            ),
            "pool_size": int(
                raw["pool_size"]
            ),
            "required_stage2_prefix": int(
                raw[
                    "required_stage2_prefix"
                ]
            ),
            "first_dataset_index": int(
                raw[
                    "first_dataset_index"
                ]
            ),
            "last_dataset_index": int(
                raw[
                    "last_dataset_index"
                ]
            ),
        }

    return result


def validate_batch_summary(
    path: Path,
) -> dict[str, Any]:
    summary = load_json(path)

    if summary.get("status") != "verified":
        raise RuntimeError(
            "Stage 2 batch summary is not verified."
        )

    checks = {
        "pool_count": EXPECTED_TASKS,
        "verified_pool_count": (
            EXPECTED_TASKS
        ),
        "failed_pool_count": 0,
        "candidate_rows": (
            EXPECTED_TOTAL_ROWS
        ),
        "total_prefix_length": (
            EXPECTED_MAX_PREFIX_TOTAL
        ),
    }

    for field, expected in checks.items():
        actual = int(
            summary.get(field, -1)
        )

        if actual != expected:
            raise RuntimeError(
                f"Batch summary field {field!r}: "
                f"expected {expected:,}, found "
                f"{actual:,}."
            )

    configuration = summary.get(
        "configuration",
        {},
    )

    if (
        configuration.get(
            "approximation"
        )
        is not False
    ):
        raise RuntimeError(
            "Stage 2 batch is marked approximate."
        )

    if (
        configuration.get(
            "candidate_sampling"
        )
        is not False
    ):
        raise RuntimeError(
            "Stage 2 batch used candidate sampling."
        )

    if (
        configuration.get(
            "sparse_similarity"
        )
        is not False
    ):
        raise RuntimeError(
            "Stage 2 batch used sparse similarity."
        )

    return summary


def load_template_budgets(
    root: Path,
    budgets: list[int],
) -> dict[
    int,
    dict[str, dict[str, list[int]]],
]:
    result: dict[
        int,
        dict[
            str,
            dict[str, list[int]],
        ],
    ] = {}

    for budget in budgets:
        path = (
            root
            / (
                "task_template_budgets_"
                f"{budget}.pkl"
            )
        )

        if not path.is_file():
            raise FileNotFoundError(path)

        mapping = load_pickle(path)

        if not isinstance(
            mapping,
            dict,
        ):
            raise TypeError(
                f"{path}: expected dictionary."
            )

        result[budget] = mapping

    return result


def normalize_ordering(
    *,
    task_id: str,
    ordering: Any,
    expected_length: int,
    first_index: int,
    last_index: int,
) -> list[tuple[int, float]]:
    if not isinstance(
        ordering,
        list,
    ):
        raise TypeError(
            f"{task_id}: ordering is not a list."
        )

    if len(ordering) != expected_length:
        raise RuntimeError(
            f"{task_id}: ordering length "
            f"{len(ordering):,} differs from expected "
            f"{expected_length:,}."
        )

    normalized: list[
        tuple[int, float]
    ] = []

    seen: set[int] = set()

    for rank, item in enumerate(
        ordering,
        start=1,
    ):
        if (
            not isinstance(
                item,
                (tuple, list),
            )
            or len(item) < 2
        ):
            raise RuntimeError(
                f"{task_id}: invalid ordering item "
                f"at rank {rank}."
            )

        dataset_index = int(
            item[0]
        )
        gain = float(
            item[1]
        )

        if not (
            first_index
            <= dataset_index
            <= last_index
        ):
            raise RuntimeError(
                f"{task_id}: dataset index "
                f"{dataset_index:,} at rank {rank} "
                "lies outside the task pool."
            )

        if dataset_index in seen:
            raise RuntimeError(
                f"{task_id}: duplicate dataset index "
                f"{dataset_index:,}."
            )

        if not math.isfinite(gain):
            raise RuntimeError(
                f"{task_id}: non-finite gain at "
                f"rank {rank}."
            )

        seen.add(dataset_index)

        normalized.append(
            (
                dataset_index,
                gain,
            )
        )

    return normalized


def main() -> int:
    args = parse_args()

    budgets = list(
        dict.fromkeys(
            args.budgets
        )
    )

    if not budgets:
        raise ValueError(
            "At least one budget is required."
        )

    if any(
        budget <= 0
        for budget in budgets
    ):
        raise ValueError(
            "Budgets must be positive."
        )

    submixture_path = (
        args.submixture_config.resolve()
    )
    manifest_path = (
        args.pool_manifest.resolve()
    )
    task_indices_root = (
        args.task_indices_root.resolve()
    )
    allocations_root = (
        args.allocations_root.resolve()
    )
    blockwise_root = (
        args.blockwise_root.resolve()
    )
    batch_summary_path = (
        args.batch_summary.resolve()
    )
    output_root = (
        args.output_root.resolve()
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    submixtures = load_submixtures(
        submixture_path
    )

    manifest = load_manifest(
        manifest_path
    )

    validate_batch_summary(
        batch_summary_path
    )

    template_budgets = (
        load_template_budgets(
            allocations_root,
            budgets,
        )
    )

    all_orderings: dict[
        str,
        dict[
            str,
            dict[
                str,
                list[tuple[int, float]],
            ],
        ],
    ] = {}

    catalog_rows: list[
        dict[str, Any]
    ] = []

    corpus_summaries: list[
        dict[str, Any]
    ] = []

    seen_tasks: set[str] = set()
    total_candidate_rows = 0
    total_ordering_entries = 0

    selected_budget_totals = {
        budget: 0
        for budget in budgets
    }

    per_corpus_paths: dict[
        str,
        Path,
    ] = {}

    print(
        "=== Assemble authors-compatible "
        "Facility Location orderings ==="
    )

    for corpus_position, corpus in enumerate(
        submixtures,
        start=1,
    ):
        task_indices_path = (
            task_indices_root
            / f"{corpus}.pkl"
        )

        if not task_indices_path.is_file():
            raise FileNotFoundError(
                task_indices_path
            )

        task_indices = load_pickle(
            task_indices_path
        )

        if not isinstance(
            task_indices,
            dict,
        ):
            raise TypeError(
                f"{task_indices_path}: "
                "expected dictionary."
            )

        corpus_orderings: dict[
            str,
            dict[
                str,
                list[
                    tuple[int, float]
                ],
            ],
        ] = {}

        corpus_candidate_rows = 0
        corpus_ordering_entries = 0

        corpus_budget_totals = {
            budget: 0
            for budget in budgets
        }

        for corpus_task_position, (
            task_id,
            task_template_indices,
        ) in enumerate(
            task_indices.items(),
            start=1,
        ):
            if task_id in seen_tasks:
                raise RuntimeError(
                    f"Duplicate task ID: {task_id}"
                )

            seen_tasks.add(task_id)

            if task_id not in manifest:
                raise RuntimeError(
                    f"Missing manifest row for "
                    f"{task_id}."
                )

            pool = manifest[task_id]

            if pool["corpus"] != corpus:
                raise RuntimeError(
                    f"{task_id}: corpus mismatch."
                )

            if (
                pool["template_type"]
                != ACTIVE_TEMPLATE
            ):
                raise RuntimeError(
                    f"{task_id}: expected active "
                    f"template {ACTIVE_TEMPLATE!r}; "
                    f"found "
                    f"{pool['template_type']!r}."
                )

            if set(
                task_template_indices
            ) != {ACTIVE_TEMPLATE}:
                raise RuntimeError(
                    f"{task_id}: task-index mapping "
                    "does not match the frozen "
                    "single-template policy."
                )

            source_indices = (
                task_template_indices[
                    ACTIVE_TEMPLATE
                ]
            )

            if len(source_indices) != pool[
                "pool_size"
            ]:
                raise RuntimeError(
                    f"{task_id}: task-index pool size "
                    f"{len(source_indices):,} differs "
                    f"from manifest size "
                    f"{pool['pool_size']:,}."
                )

            if not source_indices:
                raise RuntimeError(
                    f"{task_id}: empty source pool."
                )

            if int(
                source_indices[0]
            ) != pool[
                "first_dataset_index"
            ]:
                raise RuntimeError(
                    f"{task_id}: first source index "
                    "mismatch."
                )

            if int(
                source_indices[-1]
            ) != pool[
                "last_dataset_index"
            ]:
                raise RuntimeError(
                    f"{task_id}: last source index "
                    "mismatch."
                )

            if any(
                later != earlier + 1
                for earlier, later in zip(
                    source_indices,
                    source_indices[1:],
                )
            ):
                raise RuntimeError(
                    f"{task_id}: task source indices "
                    "are not contiguous."
                )

            ordering_path = (
                blockwise_root
                / task_slug(task_id)
                / "ordering.pkl"
            )

            report_path = (
                blockwise_root
                / task_slug(task_id)
                / "run_report.json"
            )

            if not ordering_path.is_file():
                raise FileNotFoundError(
                    ordering_path
                )

            if not report_path.is_file():
                raise FileNotFoundError(
                    report_path
                )

            report = load_json(
                report_path
            )

            if report.get(
                "status"
            ) != "verified":
                raise RuntimeError(
                    f"{task_id}: Stage 2 report "
                    "is not verified."
                )

            if (
                report.get(
                    "configuration",
                    {},
                ).get(
                    "approximation"
                )
                is not False
            ):
                raise RuntimeError(
                    f"{task_id}: ordering is marked "
                    "approximate."
                )

            ordering = normalize_ordering(
                task_id=task_id,
                ordering=load_pickle(
                    ordering_path
                ),
                expected_length=pool[
                    "required_stage2_prefix"
                ],
                first_index=pool[
                    "first_dataset_index"
                ],
                last_index=pool[
                    "last_dataset_index"
                ],
            )

            task_budget_values: dict[
                int,
                int,
            ] = {}

            for budget in budgets:
                budget_mapping = (
                    template_budgets[
                        budget
                    ]
                )

                if corpus not in budget_mapping:
                    raise RuntimeError(
                        f"Budget {budget}: missing "
                        f"corpus {corpus}."
                    )

                if task_id not in (
                    budget_mapping[corpus]
                ):
                    raise RuntimeError(
                        f"Budget {budget}: missing "
                        f"task {task_id}."
                    )

                values = budget_mapping[
                    corpus
                ][task_id]

                if not isinstance(
                    values,
                    (list, tuple),
                ):
                    raise RuntimeError(
                        f"{task_id}: invalid template "
                        f"budget structure for {budget}."
                    )

                if len(values) != len(
                    TEMPLATE_TYPES
                ):
                    raise RuntimeError(
                        f"{task_id}: expected four "
                        f"template budgets for {budget}; "
                        f"found {len(values)}."
                    )

                values = [
                    int(value)
                    for value in values
                ]

                expected_values = [
                    0,
                    values[1],
                    0,
                    0,
                ]

                if values != expected_values:
                    raise RuntimeError(
                        f"{task_id}: non-zs_noopt "
                        f"budget for {budget}: {values}"
                    )

                task_budget = values[1]

                if task_budget > len(
                    ordering
                ):
                    raise RuntimeError(
                        f"{task_id}: budget "
                        f"{task_budget:,} exceeds "
                        f"ordering length "
                        f"{len(ordering):,}."
                    )

                task_budget_values[
                    budget
                ] = task_budget

                selected_budget_totals[
                    budget
                ] += task_budget

                corpus_budget_totals[
                    budget
                ] += task_budget

            expected_prefix = max(
                task_budget_values.values()
            )

            if expected_prefix != pool[
                "required_stage2_prefix"
            ]:
                raise RuntimeError(
                    f"{task_id}: maximum budget "
                    f"{expected_prefix:,} differs from "
                    f"required Stage 2 prefix "
                    f"{pool['required_stage2_prefix']:,}."
                )

            corpus_orderings[
                task_id
            ] = {
                "zs_opt": [],
                "zs_noopt": ordering,
                "fs_opt": [],
                "fs_noopt": [],
            }

            catalog_row: dict[
                str,
                Any,
            ] = {
                "corpus_position": (
                    corpus_position
                ),
                "corpus_task_position": (
                    corpus_task_position
                ),
                "corpus": corpus,
                "task_id": task_id,
                "template_type": (
                    ACTIVE_TEMPLATE
                ),
                "pool_size": pool[
                    "pool_size"
                ],
                "ordering_length": len(
                    ordering
                ),
                "first_pool_index": pool[
                    "first_dataset_index"
                ],
                "last_pool_index": pool[
                    "last_dataset_index"
                ],
                "first_selected_index": (
                    ordering[0][0]
                    if ordering
                    else None
                ),
                "last_selected_index": (
                    ordering[-1][0]
                    if ordering
                    else None
                ),
                "ordering_path": str(
                    ordering_path
                ),
                "ordering_sha256": (
                    sha256_file(
                        ordering_path
                    )
                ),
            }

            for budget in budgets:
                catalog_row[
                    f"budget_{budget}"
                ] = task_budget_values[
                    budget
                ]

            catalog_rows.append(
                catalog_row
            )

            corpus_candidate_rows += (
                pool["pool_size"]
            )
            corpus_ordering_entries += len(
                ordering
            )

            total_candidate_rows += (
                pool["pool_size"]
            )
            total_ordering_entries += len(
                ordering
            )

        all_orderings[
            corpus
        ] = corpus_orderings

        output_path = (
            output_root
            / (
                "facility_location_orderings_"
                f"{corpus}.pkl"
            )
        )

        atomic_write_pickle(
            corpus_orderings,
            output_path,
        )

        reloaded = load_pickle(
            output_path
        )

        if reloaded != corpus_orderings:
            raise RuntimeError(
                f"{corpus}: pickle reload "
                "verification failed."
            )

        per_corpus_paths[
            corpus
        ] = output_path

        corpus_summaries.append(
            {
                "corpus": corpus,
                "corpus_position": (
                    corpus_position
                ),
                "task_count": len(
                    corpus_orderings
                ),
                "candidate_rows": (
                    corpus_candidate_rows
                ),
                "ordering_entries": (
                    corpus_ordering_entries
                ),
                "budget_totals": {
                    str(budget): (
                        corpus_budget_totals[
                            budget
                        ]
                    )
                    for budget in budgets
                },
                "output_path": str(
                    output_path
                ),
                "output_sha256": (
                    sha256_file(
                        output_path
                    )
                ),
            }
        )

        print(
            f"{corpus:10s} "
            f"tasks={len(corpus_orderings):3d} "
            f"rows={corpus_candidate_rows:9,d} "
            f"ordering={corpus_ordering_entries:6,d}"
        )

    if len(seen_tasks) != EXPECTED_TASKS:
        raise RuntimeError(
            f"Expected {EXPECTED_TASKS} tasks; "
            f"assembled {len(seen_tasks)}."
        )

    if set(manifest) != seen_tasks:
        missing = sorted(
            set(manifest) - seen_tasks
        )
        unexpected = sorted(
            seen_tasks - set(manifest)
        )

        raise RuntimeError(
            "Manifest and ordering task sets differ. "
            f"Missing={missing}, "
            f"unexpected={unexpected}"
        )

    if (
        total_candidate_rows
        != EXPECTED_TOTAL_ROWS
    ):
        raise RuntimeError(
            f"Expected {EXPECTED_TOTAL_ROWS:,} "
            f"candidate rows; assembled "
            f"{total_candidate_rows:,}."
        )

    if (
        total_ordering_entries
        != EXPECTED_MAX_PREFIX_TOTAL
    ):
        raise RuntimeError(
            f"Expected "
            f"{EXPECTED_MAX_PREFIX_TOTAL:,} "
            f"ordering entries; assembled "
            f"{total_ordering_entries:,}."
        )

    for budget in budgets:
        if (
            selected_budget_totals[
                budget
            ]
            != budget
        ):
            raise RuntimeError(
                f"Budget {budget:,}: template "
                f"budgets sum to "
                f"{selected_budget_totals[budget]:,}."
            )

    combined_path = (
        output_root
        / (
            "facility_location_orderings_all.pkl"
        )
    )

    atomic_write_pickle(
        all_orderings,
        combined_path,
    )

    combined_reloaded = load_pickle(
        combined_path
    )

    if combined_reloaded != all_orderings:
        raise RuntimeError(
            "Combined ordering pickle reload "
            "verification failed."
        )

    catalog_path = (
        output_root
        / "ordering_catalog.csv"
    )

    with catalog_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                catalog_rows[0].keys()
            ),
        )
        writer.writeheader()
        writer.writerows(
            catalog_rows
        )

    summary_path = (
        output_root
        / "assembly_summary.json"
    )

    atomic_write_json(
        {
            "format_version": 1,
            "stage": (
                "smart_v2_author_compatible_"
                "facility_location_orderings"
            ),
            "status": "verified",
            "configuration": {
                "submixture_order": (
                    submixtures
                ),
                "template_order": list(
                    TEMPLATE_TYPES
                ),
                "active_template": (
                    ACTIVE_TEMPLATE
                ),
                "budgets": budgets,
                "ordering_contents": (
                    "author-format dataset index "
                    "and marginal gain"
                ),
                "candidate_sampling": False,
                "sparse_similarity": False,
                "approximation": False,
            },
            "corpus_count": len(
                submixtures
            ),
            "task_count": len(
                seen_tasks
            ),
            "candidate_rows": (
                total_candidate_rows
            ),
            "total_ordering_entries": (
                total_ordering_entries
            ),
            "selected_budget_totals": {
                str(budget): (
                    selected_budget_totals[
                        budget
                    ]
                )
                for budget in budgets
            },
            "corpora": (
                corpus_summaries
            ),
            "inputs": {
                "submixture_config": str(
                    submixture_path
                ),
                "pool_manifest": str(
                    manifest_path
                ),
                "task_indices_root": str(
                    task_indices_root
                ),
                "allocations_root": str(
                    allocations_root
                ),
                "blockwise_root": str(
                    blockwise_root
                ),
                "batch_summary": str(
                    batch_summary_path
                ),
            },
            "outputs": {
                "combined_orderings": str(
                    combined_path
                ),
                "combined_orderings_sha256": (
                    sha256_file(
                        combined_path
                    )
                ),
                "ordering_catalog": str(
                    catalog_path
                ),
                "ordering_catalog_sha256": (
                    sha256_file(
                        catalog_path
                    )
                ),
                "per_corpus_orderings": {
                    corpus: {
                        "path": str(path),
                        "sha256": (
                            sha256_file(path)
                        ),
                    }
                    for corpus, path
                    in per_corpus_paths.items()
                },
            },
        },
        summary_path,
    )

    print()
    print(
        "=== Authors-compatible ordering summary ==="
    )
    print("Status:                  verified")
    print(
        f"Corpora:                 "
        f"{len(submixtures)}"
    )
    print(
        f"Tasks:                   "
        f"{len(seen_tasks)}"
    )
    print(
        f"Candidate rows:          "
        f"{total_candidate_rows:,}"
    )
    print(
        f"Stored ordering entries: "
        f"{total_ordering_entries:,}"
    )

    for budget in budgets:
        print(
            f"Budget {budget:6,d}:          "
            f"{selected_budget_totals[budget]:,}"
        )

    print(
        f"Combined pickle:         "
        f"{combined_path}"
    )
    print(
        f"Catalog:                 "
        f"{catalog_path}"
    )
    print(
        f"Summary:                 "
        f"{summary_path}"
    )
    print()
    print(
        "Authors-compatible ordering assembly passed."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
PY

python3 -m py_compile \
  src/stage2/assemble_author_orderings.py
```

### 15.2 Run the assembly

```bash
cd /data/saral/wdir/smart_v2 || exit 1
source ./smart_v2_env.sh

mkdir -p \
  /mnt/warm_storage/saral/smart_v2/stage2/author_orderings

python3 -m src.stage2.assemble_author_orderings \
  --submixture-config /data/saral/wdir/smart_v2/configs/submixtures.json \
  --pool-manifest /mnt/warm_storage/saral/smart_v2/stage2/pools/pool_manifest.csv \
  --task-indices-root /mnt/warm_storage/saral/smart_v2/author_format/task_indices \
  --allocations-root /mnt/warm_storage/saral/smart_v2/stage1/allocations \
  --blockwise-root /mnt/warm_storage/saral/smart_v2/stage2/blockwise \
  --batch-summary /mnt/warm_storage/saral/smart_v2/stage2/blockwise/batch_summary.json \
  --output-root /mnt/warm_storage/saral/smart_v2/stage2/author_orderings \
  --budgets 25000 50000 \
  2>&1 | tee \
  /mnt/warm_storage/saral/smart_v2/logs/assemble_author_orderings.log
```

### 15.3 Inspect the assembled structure

```bash
python3 - <<'PY'
import json
import pickle
from pathlib import Path

root = Path(
    "/mnt/warm_storage/saral/smart_v2/"
    "stage2/author_orderings"
)

with (
    root / "assembly_summary.json"
).open(encoding="utf-8") as handle:
    summary = json.load(handle)

print("Status:", summary["status"])
print("Corpora:", summary["corpus_count"])
print("Tasks:", summary["task_count"])
print(
    "Candidate rows:",
    f"{summary['candidate_rows']:,}",
)
print(
    "Stored ordering entries:",
    f"{summary['total_ordering_entries']:,}",
)
print(
    "Budget totals:",
    summary["selected_budget_totals"],
)

with (
    root
    / "facility_location_orderings_all.pkl"
).open("rb") as handle:
    orderings = pickle.load(handle)

print("\nCorpora:", list(orderings))

example = (
    orderings["sglue"]
    ["sglue::qqp"]
)

print(
    "QQP template keys:",
    list(example),
)
print(
    "QQP zs_noopt ordering length:",
    len(example["zs_noopt"]),
)
print(
    "QQP first five entries:",
    example["zs_noopt"][:5],
)
PY
```

Expected outputs:

```text
/mnt/warm_storage/saral/smart_v2/stage2/author_orderings/
├── facility_location_orderings_flan2021.pkl
├── facility_location_orderings_t0.pkl
├── facility_location_orderings_sglue.pkl
├── facility_location_orderings_cot.pkl
├── facility_location_orderings_tulu.pkl
├── facility_location_orderings_all.pkl
├── ordering_catalog.csv
└── assembly_summary.json
```

Acceptance conditions:

```text
status                    = verified
corpora                   = 5
tasks                     = 309
candidate rows            = 6,266,471
stored ordering entries   = 50,000
25K budget total          = 25,000
50K budget total          = 50,000
QQP ordering length       = 187
active template           = zs_noopt
approximation             = false
```

The next step is materializing and independently verifying the final SMART 25K and 50K Hugging Face training datasets from these ordering prefixes.
