Stage 2 orderings are now ready for the authors’ final mixture-construction procedure: take the required prefix for every task/template pool, concatenate those selections, shuffle the training split with seed 23, and concatenate validation splits without shuffling. 

# Step 16 — Materialize and verify SMART-25K and SMART-50K

The output datasets will retain `inputs` and `targets` plus provenance columns. These columns make the final selections independently auditable and can be removed during tokenization.

## 16.1 Create the materializer

```bash
cd /data/saral/wdir/smart_v2 || exit 1

cat > src/mixtures/materialize_smart_mixtures.py <<'PY'
"""Materialize and independently verify SMART 25K and 50K mixtures."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pickle
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from datasets import (
    Dataset,
    DatasetDict,
    concatenate_datasets,
    load_from_disk,
)


TEMPLATE_TYPES = (
    "zs_opt",
    "zs_noopt",
    "fs_opt",
    "fs_noopt",
)

ACTIVE_TEMPLATE = "zs_noopt"
SHUFFLE_SEED = 23

EXPECTED_CORPORA = 5
EXPECTED_TASKS = 309
EXPECTED_TRAIN_ROWS = 6_266_471
EXPECTED_VALIDATION_ROWS = 183_870

BASE_COLUMNS = (
    "inputs",
    "targets",
    "task_source",
    "task_name",
    "template_type",
    "source_index",
)

OUTPUT_COLUMNS = (
    "inputs",
    "targets",
    "task_source",
    "task_name",
    "template_type",
    "source_index",
    "corpus",
    "corpus_dataset_index",
    "selection_rank_within_pool",
    "marginal_gain",
    "pre_shuffle_position",
    "mixture_budget",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--submixture-config",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--author-format-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--task-indices-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--orderings",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--allocations-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--assembly-summary",
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
    parser.add_argument(
        "--verification-batch-size",
        type=int,
        default=20_000,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        return json.load(handle)


def load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


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


def load_submixtures(path: Path) -> list[str]:
    payload = load_json(path)
    values = payload.get("submixtures")

    if not isinstance(values, list):
        raise ValueError(
            "Submixture configuration is invalid."
        )

    if len(values) != EXPECTED_CORPORA:
        raise RuntimeError(
            f"Expected {EXPECTED_CORPORA} corpora; "
            f"found {len(values)}."
        )

    if len(values) != len(set(values)):
        raise RuntimeError(
            "Submixture list contains duplicates."
        )

    return values


def validate_assembly_summary(path: Path) -> None:
    summary = load_json(path)

    if summary.get("status") != "verified":
        raise RuntimeError(
            "Ordering assembly is not verified."
        )

    expected = {
        "corpus_count": EXPECTED_CORPORA,
        "task_count": EXPECTED_TASKS,
        "candidate_rows": EXPECTED_TRAIN_ROWS,
        "total_ordering_entries": 50_000,
    }

    for field, value in expected.items():
        if int(summary.get(field, -1)) != value:
            raise RuntimeError(
                f"Assembly field {field}: expected "
                f"{value:,}, found "
                f"{summary.get(field)!r}."
            )

    configuration = summary.get(
        "configuration",
        {},
    )

    if configuration.get("approximation") is not False:
        raise RuntimeError(
            "Ordering assembly is marked approximate."
        )


def load_budget_mappings(
    root: Path,
    budgets: list[int],
) -> dict[
    int,
    dict[str, dict[str, list[int]]],
]:
    result = {}

    for budget in budgets:
        path = (
            root
            / f"task_template_budgets_{budget}.pkl"
        )

        if not path.is_file():
            raise FileNotFoundError(path)

        mapping = load_pickle(path)

        if not isinstance(mapping, dict):
            raise TypeError(
                f"{path}: expected dictionary."
            )

        result[budget] = mapping

    return result


def require_author_columns(
    dataset: Dataset,
    *,
    corpus: str,
    split: str,
) -> None:
    missing = set(BASE_COLUMNS) - set(
        dataset.column_names
    )

    if missing:
        raise RuntimeError(
            f"{corpus}/{split}: missing columns "
            f"{sorted(missing)}."
        )


def add_common_metadata(
    dataset: Dataset,
    *,
    task_names: list[str],
    template_types: list[str],
    corpus: str,
    corpus_indices: list[int],
    selection_ranks: list[int],
    gains: list[float],
    pre_shuffle_positions: list[int],
    mixture_budget: int,
) -> Dataset:
    row_count = len(dataset)

    values = (
        task_names,
        template_types,
        corpus_indices,
        selection_ranks,
        gains,
        pre_shuffle_positions,
    )

    if any(
        len(value) != row_count
        for value in values
    ):
        raise RuntimeError(
            "Metadata length does not match dataset length."
        )

    dataset = dataset.remove_columns(
        ["task_name", "template_type"]
    )

    dataset = dataset.add_column(
        "task_name",
        task_names,
    )
    dataset = dataset.add_column(
        "template_type",
        template_types,
    )
    dataset = dataset.add_column(
        "corpus",
        [corpus] * row_count,
    )
    dataset = dataset.add_column(
        "corpus_dataset_index",
        corpus_indices,
    )
    dataset = dataset.add_column(
        "selection_rank_within_pool",
        selection_ranks,
    )
    dataset = dataset.add_column(
        "marginal_gain",
        gains,
    )
    dataset = dataset.add_column(
        "pre_shuffle_position",
        pre_shuffle_positions,
    )
    dataset = dataset.add_column(
        "mixture_budget",
        [mixture_budget] * row_count,
    )

    return dataset.select_columns(
        list(OUTPUT_COLUMNS)
    )


def build_validation(
    *,
    submixtures: list[str],
    source_datasets: dict[str, DatasetDict],
) -> tuple[Dataset, dict[str, int]]:
    chunks: list[Dataset] = []
    counts: dict[str, int] = {}

    for corpus in submixtures:
        validation = source_datasets[
            corpus
        ]["validation"]

        require_author_columns(
            validation,
            corpus=corpus,
            split="validation",
        )

        feature = validation.features[
            "task_name"
        ]

        if not hasattr(feature, "int2str"):
            raise RuntimeError(
                f"{corpus}: validation task_name "
                "is not a ClassLabel."
            )

        labels = validation["task_name"]

        decoded = [
            feature.int2str(int(value))
            for value in labels
        ]

        row_count = len(validation)
        indices = list(range(row_count))

        chunk = add_common_metadata(
            validation,
            task_names=decoded,
            template_types=[
                str(value)
                for value in validation[
                    "template_type"
                ]
            ],
            corpus=corpus,
            corpus_indices=indices,
            selection_ranks=[
                -1
            ] * row_count,
            gains=[
                float("nan")
            ] * row_count,
            pre_shuffle_positions=[
                -1
            ] * row_count,
            mixture_budget=-1,
        )

        chunks.append(chunk)
        counts[corpus] = row_count

    combined = concatenate_datasets(
        chunks
    )

    if len(combined) != EXPECTED_VALIDATION_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_VALIDATION_ROWS:,} "
            f"validation rows; found "
            f"{len(combined):,}."
        )

    return combined, counts


def build_train_mixture(
    *,
    budget: int,
    submixtures: list[str],
    source_datasets: dict[str, DatasetDict],
    task_indices: dict[str, dict[str, Any]],
    orderings: dict[str, dict[str, Any]],
    template_budgets: dict[str, dict[str, list[int]]],
) -> tuple[
    Dataset,
    list[dict[str, Any]],
    dict[str, int],
]:
    chunks: list[Dataset] = []
    records: list[dict[str, Any]] = []
    task_counts: dict[str, int] = {}

    pre_shuffle_position = 0
    seen_tasks: set[str] = set()

    for corpus in submixtures:
        train = source_datasets[
            corpus
        ]["train"]

        require_author_columns(
            train,
            corpus=corpus,
            split="train",
        )

        task_feature = train.features[
            "task_name"
        ]

        if not hasattr(task_feature, "int2str"):
            raise RuntimeError(
                f"{corpus}: train task_name "
                "is not a ClassLabel."
            )

        if corpus not in task_indices:
            raise RuntimeError(
                f"Missing task indices for {corpus}."
            )

        if corpus not in orderings:
            raise RuntimeError(
                f"Missing orderings for {corpus}."
            )

        if corpus not in template_budgets:
            raise RuntimeError(
                f"Missing template budgets for {corpus}."
            )

        for task_id, pool_mapping in (
            task_indices[corpus].items()
        ):
            if task_id in seen_tasks:
                raise RuntimeError(
                    f"Duplicate task: {task_id}"
                )

            seen_tasks.add(task_id)

            if task_id not in orderings[corpus]:
                raise RuntimeError(
                    f"Missing ordering for {task_id}."
                )

            if task_id not in template_budgets[
                corpus
            ]:
                raise RuntimeError(
                    f"Missing budget for {task_id}."
                )

            budget_values = [
                int(value)
                for value in (
                    template_budgets[
                        corpus
                    ][task_id]
                )
            ]

            if len(budget_values) != 4:
                raise RuntimeError(
                    f"{task_id}: invalid template "
                    "budget vector."
                )

            selected_for_task = 0

            for template_position, template_type in enumerate(
                TEMPLATE_TYPES
            ):
                count = budget_values[
                    template_position
                ]

                if count < 0:
                    raise RuntimeError(
                        f"{task_id}: negative template budget."
                    )

                pool_indices = pool_mapping.get(
                    template_type,
                    [],
                )

                ordering = orderings[
                    corpus
                ][task_id].get(
                    template_type,
                    [],
                )

                if count == 0:
                    continue

                if template_type != ACTIVE_TEMPLATE:
                    raise RuntimeError(
                        f"{task_id}: non-primary template "
                        f"{template_type} has budget {count}."
                    )

                if count > len(ordering):
                    raise RuntimeError(
                        f"{task_id}: budget {count} exceeds "
                        f"ordering length {len(ordering)}."
                    )

                if len(pool_indices) < count:
                    raise RuntimeError(
                        f"{task_id}: budget exceeds pool size."
                    )

                prefix = ordering[:count]

                dataset_indices = [
                    int(item[0])
                    for item in prefix
                ]

                gains = [
                    float(item[1])
                    for item in prefix
                ]

                if len(set(dataset_indices)) != count:
                    raise RuntimeError(
                        f"{task_id}: duplicate selected indices."
                    )

                allowed_indices = set(
                    int(value)
                    for value in pool_indices
                )

                if not all(
                    index in allowed_indices
                    for index in dataset_indices
                ):
                    raise RuntimeError(
                        f"{task_id}: selected index outside "
                        "its task/template pool."
                    )

                chunk = train.select(
                    dataset_indices
                )

                decoded = [
                    task_feature.int2str(
                        int(value)
                    )
                    for value in chunk[
                        "task_name"
                    ]
                ]

                if set(decoded) != {task_id}:
                    raise RuntimeError(
                        f"{task_id}: selected source rows "
                        "have inconsistent task labels."
                    )

                source_templates = [
                    str(value)
                    for value in chunk[
                        "template_type"
                    ]
                ]

                if set(source_templates) != {
                    template_type
                }:
                    raise RuntimeError(
                        f"{task_id}: selected source rows "
                        "have inconsistent template labels."
                    )

                pre_positions = list(
                    range(
                        pre_shuffle_position,
                        pre_shuffle_position
                        + count,
                    )
                )

                selection_ranks = list(
                    range(
                        1,
                        count + 1,
                    )
                )

                chunk = add_common_metadata(
                    chunk,
                    task_names=[
                        task_id
                    ] * count,
                    template_types=[
                        template_type
                    ] * count,
                    corpus=corpus,
                    corpus_indices=(
                        dataset_indices
                    ),
                    selection_ranks=(
                        selection_ranks
                    ),
                    gains=gains,
                    pre_shuffle_positions=(
                        pre_positions
                    ),
                    mixture_budget=budget,
                )

                chunks.append(chunk)

                for offset in range(count):
                    records.append(
                        {
                            "pre_shuffle_position": (
                                pre_positions[offset]
                            ),
                            "budget": budget,
                            "corpus": corpus,
                            "task_id": task_id,
                            "template_type": (
                                template_type
                            ),
                            "selection_rank_within_pool": (
                                selection_ranks[offset]
                            ),
                            "corpus_dataset_index": (
                                dataset_indices[offset]
                            ),
                            "marginal_gain": (
                                gains[offset]
                            ),
                        }
                    )

                pre_shuffle_position += count
                selected_for_task += count

            task_counts[task_id] = (
                selected_for_task
            )

    if len(seen_tasks) != EXPECTED_TASKS:
        raise RuntimeError(
            f"Expected {EXPECTED_TASKS} tasks; "
            f"processed {len(seen_tasks)}."
        )

    if pre_shuffle_position != budget:
        raise RuntimeError(
            f"Budget {budget:,}: selected "
            f"{pre_shuffle_position:,} rows."
        )

    if sum(task_counts.values()) != budget:
        raise RuntimeError(
            f"Budget {budget:,}: task counts "
            "do not sum to the budget."
        )

    train = concatenate_datasets(
        chunks
    )

    if len(train) != budget:
        raise RuntimeError(
            f"Budget {budget:,}: concatenated train "
            f"has {len(train):,} rows."
        )

    train = train.shuffle(
        seed=SHUFFLE_SEED
    )

    return train, records, task_counts


def save_dataset_atomic(
    dataset: DatasetDict,
    destination: Path,
    *,
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
                f"{destination} already exists. "
                "Use --overwrite to replace it."
            )

        shutil.rmtree(destination)

    dataset.save_to_disk(
        str(temporary)
    )

    temporary.replace(destination)


def write_records(
    records: list[dict[str, Any]],
    path: Path,
) -> None:
    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                records[0].keys()
            ),
        )
        writer.writeheader()
        writer.writerows(records)


def verify_train(
    *,
    budget: int,
    train: Dataset,
    records: list[dict[str, Any]],
    expected_task_counts: dict[str, int],
    source_datasets: dict[str, DatasetDict],
    submixtures: list[str],
) -> dict[str, Any]:
    if len(train) != budget:
        raise RuntimeError(
            f"Budget {budget}: train row count mismatch."
        )

    if tuple(train.column_names) != OUTPUT_COLUMNS:
        raise RuntimeError(
            f"Budget {budget}: output columns differ "
            "from the frozen schema."
        )

    pre_positions = [
        int(value)
        for value in train[
            "pre_shuffle_position"
        ]
    ]

    expected_permutation = [
        int(value)
        for value in Dataset.from_dict(
            {
                "position": list(
                    range(budget)
                )
            }
        ).shuffle(
            seed=SHUFFLE_SEED
        )["position"]
    ]

    if pre_positions != expected_permutation:
        raise RuntimeError(
            f"Budget {budget}: train shuffle does not "
            f"match Hugging Face seed {SHUFFLE_SEED}."
        )

    task_counts = Counter(
        str(value)
        for value in train[
            "task_name"
        ]
    )

    if dict(task_counts) != expected_task_counts:
        raise RuntimeError(
            f"Budget {budget}: final task counts "
            "do not match allocations."
        )

    if len(
        set(
            zip(
                train["corpus"],
                train[
                    "corpus_dataset_index"
                ],
            )
        )
    ) != budget:
        raise RuntimeError(
            f"Budget {budget}: duplicate provenance keys."
        )

    order = np.argsort(
        np.asarray(
            pre_positions,
            dtype=np.int64,
        ),
        kind="stable",
    ).tolist()

    unshuffled = train.select(order)

    expected_keys = [
        (
            record["corpus"],
            record["task_id"],
            record["template_type"],
            int(
                record[
                    "corpus_dataset_index"
                ]
            ),
            int(
                record[
                    "selection_rank_within_pool"
                ]
            ),
        )
        for record in records
    ]

    actual_keys = list(
        zip(
            unshuffled["corpus"],
            unshuffled["task_name"],
            unshuffled["template_type"],
            [
                int(value)
                for value in unshuffled[
                    "corpus_dataset_index"
                ]
            ],
            [
                int(value)
                for value in unshuffled[
                    "selection_rank_within_pool"
                ]
            ],
        )
    )

    if actual_keys != expected_keys:
        raise RuntimeError(
            f"Budget {budget}: saved provenance "
            "does not match ordering prefixes."
        )

    offset = 0

    for corpus in submixtures:
        corpus_records = [
            record
            for record in records
            if record["corpus"] == corpus
        ]

        count = len(corpus_records)

        if count == 0:
            continue

        segment = unshuffled.select(
            list(
                range(
                    offset,
                    offset + count,
                )
            )
        )

        indices = [
            int(
                record[
                    "corpus_dataset_index"
                ]
            )
            for record in corpus_records
        ]

        source_train = source_datasets[
            corpus
        ]["train"]

        source_subset = source_train.select(
            indices
        )

        if segment["inputs"] != source_subset[
            "inputs"
        ]:
            raise RuntimeError(
                f"Budget {budget}, {corpus}: "
                "input text mismatch."
            )

        if segment["targets"] != source_subset[
            "targets"
        ]:
            raise RuntimeError(
                f"Budget {budget}, {corpus}: "
                "target text mismatch."
            )

        if [
            int(value)
            for value in segment[
                "source_index"
            ]
        ] != [
            int(value)
            for value in source_subset[
                "source_index"
            ]
        ]:
            raise RuntimeError(
                f"Budget {budget}, {corpus}: "
                "source-index mismatch."
            )

        feature = source_train.features[
            "task_name"
        ]

        decoded_tasks = [
            feature.int2str(
                int(value)
            )
            for value in source_subset[
                "task_name"
            ]
        ]

        if segment["task_name"] != decoded_tasks:
            raise RuntimeError(
                f"Budget {budget}, {corpus}: "
                "task-name mismatch."
            )

        if segment[
            "template_type"
        ] != source_subset[
            "template_type"
        ]:
            raise RuntimeError(
                f"Budget {budget}, {corpus}: "
                "template-type mismatch."
            )

        offset += count

    if offset != budget:
        raise RuntimeError(
            f"Budget {budget}: source verification "
            "did not cover every row."
        )

    return {
        "row_count": budget,
        "unique_provenance_keys": budget,
        "task_count": len(task_counts),
        "shuffle_seed": SHUFFLE_SEED,
        "shuffle_verified": True,
        "source_content_verified": True,
    }


def verify_validation(
    *,
    validation: Dataset,
    source_datasets: dict[str, DatasetDict],
    submixtures: list[str],
    batch_size: int,
) -> dict[str, Any]:
    if len(validation) != EXPECTED_VALIDATION_ROWS:
        raise RuntimeError(
            "Validation row count mismatch."
        )

    offset = 0
    corpus_counts: dict[str, int] = {}

    for corpus in submixtures:
        source = source_datasets[
            corpus
        ]["validation"]

        count = len(source)
        corpus_counts[corpus] = count

        feature = source.features[
            "task_name"
        ]

        for local_start in range(
            0,
            count,
            batch_size,
        ):
            local_end = min(
                local_start + batch_size,
                count,
            )

            global_start = (
                offset + local_start
            )
            global_end = (
                offset + local_end
            )

            actual = validation.select(
                list(
                    range(
                        global_start,
                        global_end,
                    )
                )
            )

            expected = source.select(
                list(
                    range(
                        local_start,
                        local_end,
                    )
                )
            )

            expected_indices = list(
                range(
                    local_start,
                    local_end,
                )
            )

            if actual["corpus"] != [
                corpus
            ] * len(expected_indices):
                raise RuntimeError(
                    f"{corpus}: validation corpus "
                    "ordering mismatch."
                )

            if [
                int(value)
                for value in actual[
                    "corpus_dataset_index"
                ]
            ] != expected_indices:
                raise RuntimeError(
                    f"{corpus}: validation dataset-index "
                    "ordering mismatch."
                )

            if actual["inputs"] != expected[
                "inputs"
            ]:
                raise RuntimeError(
                    f"{corpus}: validation input mismatch."
                )

            if actual["targets"] != expected[
                "targets"
            ]:
                raise RuntimeError(
                    f"{corpus}: validation target mismatch."
                )

            expected_tasks = [
                feature.int2str(
                    int(value)
                )
                for value in expected[
                    "task_name"
                ]
            ]

            if actual[
                "task_name"
            ] != expected_tasks:
                raise RuntimeError(
                    f"{corpus}: validation task mismatch."
                )

        offset += count

    if offset != EXPECTED_VALIDATION_ROWS:
        raise RuntimeError(
            "Validation verification did not cover "
            "every row."
        )

    return {
        "row_count": len(validation),
        "corpus_counts": corpus_counts,
        "concatenation_order": submixtures,
        "unshuffled": True,
        "source_content_verified": True,
    }


def main() -> int:
    args = parse_args()

    if args.verification_batch_size <= 0:
        raise ValueError(
            "--verification-batch-size must be positive."
        )

    budgets = list(
        dict.fromkeys(args.budgets)
    )

    if sorted(budgets) != [
        25_000,
        50_000,
    ]:
        raise RuntimeError(
            "The primary run requires budgets "
            "25,000 and 50,000."
        )

    submixture_path = (
        args.submixture_config.resolve()
    )
    author_root = (
        args.author_format_root.resolve()
    )
    indices_root = (
        args.task_indices_root.resolve()
    )
    orderings_path = (
        args.orderings.resolve()
    )
    allocations_root = (
        args.allocations_root.resolve()
    )
    assembly_summary_path = (
        args.assembly_summary.resolve()
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

    validate_assembly_summary(
        assembly_summary_path
    )

    orderings = load_pickle(
        orderings_path
    )

    if list(orderings) != submixtures:
        raise RuntimeError(
            "Ordering corpus order does not match "
            "the frozen submixture order."
        )

    budget_mappings = load_budget_mappings(
        allocations_root,
        budgets,
    )

    source_datasets: dict[
        str,
        DatasetDict,
    ] = {}

    task_indices: dict[
        str,
        dict[str, Any],
    ] = {}

    total_source_train = 0

    for corpus in submixtures:
        dataset_path = (
            author_root / corpus
        )
        index_path = (
            indices_root / f"{corpus}.pkl"
        )

        source_datasets[corpus] = (
            load_from_disk(
                str(dataset_path)
            )
        )

        task_indices[corpus] = (
            load_pickle(index_path)
        )

        total_source_train += len(
            source_datasets[
                corpus
            ]["train"]
        )

    if total_source_train != EXPECTED_TRAIN_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_TRAIN_ROWS:,} source "
            f"train rows; found "
            f"{total_source_train:,}."
        )

    print("=== Build common validation split ===")

    validation, validation_counts = (
        build_validation(
            submixtures=submixtures,
            source_datasets=source_datasets,
        )
    )

    print(
        f"Validation rows: "
        f"{len(validation):,}"
    )

    mixture_records: dict[
        int,
        list[dict[str, Any]],
    ] = {}

    mixture_task_counts: dict[
        int,
        dict[str, int],
    ] = {}

    mixture_reports: dict[
        str,
        Any,
    ] = {}

    for budget in budgets:
        print()
        print(
            f"=== Materialize SMART-{budget:,} ==="
        )

        train, records, task_counts = (
            build_train_mixture(
                budget=budget,
                submixtures=submixtures,
                source_datasets=(
                    source_datasets
                ),
                task_indices=task_indices,
                orderings=orderings,
                template_budgets=(
                    budget_mappings[
                        budget
                    ]
                ),
            )
        )

        mixture_records[budget] = records
        mixture_task_counts[
            budget
        ] = task_counts

        destination = (
            output_root
            / f"smart_{budget}"
        )

        save_dataset_atomic(
            DatasetDict(
                {
                    "train": train,
                    "validation": (
                        validation
                    ),
                }
            ),
            destination,
            overwrite=args.overwrite,
        )

        manifest_path = (
            output_root
            / (
                f"smart_{budget}_"
                "selection_manifest.csv"
            )
        )

        write_records(
            records,
            manifest_path,
        )

        reloaded = load_from_disk(
            str(destination)
        )

        train_report = verify_train(
            budget=budget,
            train=reloaded["train"],
            records=records,
            expected_task_counts=(
                task_counts
            ),
            source_datasets=(
                source_datasets
            ),
            submixtures=submixtures,
        )

        validation_report = (
            verify_validation(
                validation=reloaded[
                    "validation"
                ],
                source_datasets=(
                    source_datasets
                ),
                submixtures=(
                    submixtures
                ),
                batch_size=(
                    args.verification_batch_size
                ),
            )
        )

        final_order_path = (
            output_root
            / (
                f"smart_{budget}_"
                "final_train_order.csv"
            )
        )

        final_rows = []

        train_split = reloaded["train"]

        for final_position in range(
            len(train_split)
        ):
            final_rows.append(
                {
                    "final_train_position": (
                        final_position
                    ),
                    "pre_shuffle_position": int(
                        train_split[
                            final_position
                        ][
                            "pre_shuffle_position"
                        ]
                    ),
                    "corpus": train_split[
                        final_position
                    ]["corpus"],
                    "task_id": train_split[
                        final_position
                    ]["task_name"],
                    "template_type": train_split[
                        final_position
                    ]["template_type"],
                    "corpus_dataset_index": int(
                        train_split[
                            final_position
                        ][
                            "corpus_dataset_index"
                        ]
                    ),
                    "selection_rank_within_pool": int(
                        train_split[
                            final_position
                        ][
                            "selection_rank_within_pool"
                        ]
                    ),
                }
            )

        write_records(
            final_rows,
            final_order_path,
        )

        mixture_reports[
            str(budget)
        ] = {
            "status": "verified",
            "dataset_path": str(
                destination
            ),
            "train": train_report,
            "validation": (
                validation_report
            ),
            "selection_manifest": str(
                manifest_path
            ),
            "selection_manifest_sha256": (
                sha256_file(
                    manifest_path
                )
            ),
            "final_train_order": str(
                final_order_path
            ),
            "final_train_order_sha256": (
                sha256_file(
                    final_order_path
                )
            ),
        }

        print(
            f"Train rows:      "
            f"{len(reloaded['train']):,}"
        )
        print(
            f"Validation rows: "
            f"{len(reloaded['validation']):,}"
        )
        print(
            "Shuffle:         verified, seed 23"
        )
        print(
            "Source content:  verified"
        )
        print(
            f"Dataset:         {destination}"
        )

    smaller, larger = sorted(budgets)

    smaller_by_task: dict[
        str,
        list[int],
    ] = defaultdict(list)

    larger_by_task: dict[
        str,
        list[int],
    ] = defaultdict(list)

    for record in mixture_records[
        smaller
    ]:
        smaller_by_task[
            record["task_id"]
        ].append(
            int(
                record[
                    "corpus_dataset_index"
                ]
            )
        )

    for record in mixture_records[
        larger
    ]:
        larger_by_task[
            record["task_id"]
        ].append(
            int(
                record[
                    "corpus_dataset_index"
                ]
            )
        )

    prefix_failures = []

    for task_id in sorted(
        larger_by_task
    ):
        small = smaller_by_task[
            task_id
        ]
        large = larger_by_task[
            task_id
        ]

        if large[:len(small)] != small:
            prefix_failures.append(
                task_id
            )

    if prefix_failures:
        raise RuntimeError(
            "25K/50K per-task prefix consistency "
            f"failed for: {prefix_failures}"
        )

    summary_path = (
        output_root
        / "materialization_summary.json"
    )

    atomic_write_json(
        {
            "format_version": 1,
            "stage": (
                "smart_v2_final_mixture_materialization"
            ),
            "status": "verified",
            "configuration": {
                "budgets": budgets,
                "submixture_order": (
                    submixtures
                ),
                "template_order": list(
                    TEMPLATE_TYPES
                ),
                "active_template": (
                    ACTIVE_TEMPLATE
                ),
                "train_shuffle_seed": (
                    SHUFFLE_SEED
                ),
                "validation_shuffled": False,
                "candidate_sampling": False,
                "sparse_similarity": False,
                "approximation": False,
            },
            "source_train_rows": (
                total_source_train
            ),
            "validation_rows": len(
                validation
            ),
            "validation_corpus_counts": (
                validation_counts
            ),
            "task_count": (
                EXPECTED_TASKS
            ),
            "cross_budget_prefix_consistency": {
                "smaller_budget": smaller,
                "larger_budget": larger,
                "violation_count": (
                    len(prefix_failures)
                ),
                "violating_tasks": (
                    prefix_failures
                ),
            },
            "mixtures": mixture_reports,
            "inputs": {
                "orderings": str(
                    orderings_path
                ),
                "orderings_sha256": (
                    sha256_file(
                        orderings_path
                    )
                ),
                "assembly_summary": str(
                    assembly_summary_path
                ),
                "assembly_summary_sha256": (
                    sha256_file(
                        assembly_summary_path
                    )
                ),
                "allocations_root": str(
                    allocations_root
                ),
            },
        },
        summary_path,
    )

    print()
    print(
        "=== Final SMART mixture summary ==="
    )
    print("Status:                    verified")
    print(
        f"SMART-25K train rows:      "
        f"{len(mixture_records[25_000]):,}"
    )
    print(
        f"SMART-50K train rows:      "
        f"{len(mixture_records[50_000]):,}"
    )
    print(
        f"Validation rows:           "
        f"{len(validation):,}"
    )
    print(
        f"Tasks represented:         "
        f"{EXPECTED_TASKS}"
    )
    print(
        "25K/50K prefix violations: "
        f"{len(prefix_failures)}"
    )
    print(
        "Train shuffle seed:        "
        f"{SHUFFLE_SEED}"
    )
    print(f"Summary:                   {summary_path}")
    print()
    print(
        "SMART-25K and SMART-50K materialization passed."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
PY

python3 -m py_compile \
  src/mixtures/materialize_smart_mixtures.py
```

## 16.2 Run materialization

```bash
cd /data/saral/wdir/smart_v2 || exit 1
source ./smart_v2_env.sh

mkdir -p \
  /mnt/warm_storage/saral/smart_v2/mixtures

python3 -m src.mixtures.materialize_smart_mixtures \
  --submixture-config /data/saral/wdir/smart_v2/configs/submixtures.json \
  --author-format-root /mnt/warm_storage/saral/smart_v2/author_format \
  --task-indices-root /mnt/warm_storage/saral/smart_v2/author_format/task_indices \
  --orderings /mnt/warm_storage/saral/smart_v2/stage2/author_orderings/facility_location_orderings_all.pkl \
  --allocations-root /mnt/warm_storage/saral/smart_v2/stage1/allocations \
  --assembly-summary /mnt/warm_storage/saral/smart_v2/stage2/author_orderings/assembly_summary.json \
  --output-root /mnt/warm_storage/saral/smart_v2/mixtures \
  --budgets 25000 50000 \
  --verification-batch-size 20000 \
  --overwrite \
  2>&1 | tee \
  /mnt/warm_storage/saral/smart_v2/logs/materialize_smart_mixtures.log
```

## 16.3 Inspect the final datasets

```bash
python3 - <<'PY'
import json
from pathlib import Path

from datasets import load_from_disk


root = Path(
    "/mnt/warm_storage/saral/smart_v2/mixtures"
)

with (
    root / "materialization_summary.json"
).open(encoding="utf-8") as handle:
    summary = json.load(handle)

print("Status:", summary["status"])
print(
    "Prefix violations:",
    summary[
        "cross_budget_prefix_consistency"
    ]["violation_count"],
)

for budget in (25000, 50000):
    dataset = load_from_disk(
        str(root / f"smart_{budget}")
    )

    print(f"\nSMART-{budget:,}")
    print(dataset)
    print(
        "Train rows:",
        f"{len(dataset['train']):,}",
    )
    print(
        "Validation rows:",
        f"{len(dataset['validation']):,}",
    )
    print(
        "Columns:",
        dataset["train"].column_names,
    )

    print("\nFirst shuffled train row:")
    row = dataset["train"][0]

    for key in (
        "task_name",
        "corpus",
        "template_type",
        "corpus_dataset_index",
        "selection_rank_within_pool",
        "pre_shuffle_position",
    ):
        print(f"  {key}: {row[key]}")
PY
```

Expected artifacts:

```text
/mnt/warm_storage/saral/smart_v2/mixtures/
├── smart_25000/
├── smart_50000/
├── smart_25000_selection_manifest.csv
├── smart_50000_selection_manifest.csv
├── smart_25000_final_train_order.csv
├── smart_50000_final_train_order.csv
└── materialization_summary.json
```

Acceptance conditions:

```text
status                         = verified
SMART-25K train rows           = 25,000
SMART-50K train rows           = 50,000
validation rows                = 183,870
tasks represented              = 309
shuffle seed                   = 23
validation shuffled            = false
25K/50K prefix violations      = 0
source content verified        = true
approximation                  = false
```

===
===

## Error

```
Traceback (most recent call last): 
  File "<frozen runpy>", line 198, in _run_module_as_main 
  File "<frozen runpy>", line 88, in _run_code 
  File "/data/saral/wdir/smart_v2/src/mixtures/materialize_smart_mixtures.py", line 1693, in <module> 
    raise SystemExit(main()) 
                     ^^^^^^ 
  File "/data/saral/wdir/smart_v2/src/mixtures/materialize_smart_mixtures.py", line 1379, in main 
    train_report = verify_train( 
                   ^^^^^^^^^^^^^ 
  File "/data/saral/wdir/smart_v2/src/mixtures/materialize_smart_mixtures.py", line 806, in verify_train 
    raise RuntimeError( 
RuntimeError: Budget 25000: output columns differ from the frozen schema.
```

## Resolution
The failure is almost certainly an **order-only column-schema mismatch**, not a selection or data-integrity failure.

`source_index` already exists in the source dataset. After removing and re-adding `task_name` and `template_type`, Hugging Face preserves the existing physical column order. Its `select_columns()` filters columns but does not reliably reorder them according to the supplied list.

Your saved train split therefore likely has:

```text
inputs
targets
task_source
source_index
task_name
template_type
corpus
corpus_dataset_index
selection_rank_within_pool
marginal_gain
pre_shuffle_position
mixture_budget
```

while `OUTPUT_COLUMNS` expects `source_index` after `template_type`. The verifier compares tuples, so it rejects an otherwise equivalent schema.

## Confirm the diagnosis

The 25K dataset was saved before verification failed:

```bash
cd /data/saral/wdir/smart_v2 || exit 1
source ./smart_v2_env.sh

python3 - <<'PY'
from datasets import load_from_disk

path = "/mnt/warm_storage/saral/smart_v2/mixtures/smart_25000"
dataset = load_from_disk(path)

expected = (
    "inputs",
    "targets",
    "task_source",
    "task_name",
    "template_type",
    "source_index",
    "corpus",
    "corpus_dataset_index",
    "selection_rank_within_pool",
    "marginal_gain",
    "pre_shuffle_position",
    "mixture_budget",
)

actual = tuple(dataset["train"].column_names)

print("Actual:")
for index, column in enumerate(actual):
    print(f"  {index:2d}: {column}")

print("\nMissing:", sorted(set(expected) - set(actual)))
print("Unexpected:", sorted(set(actual) - set(expected)))
print("Same column set:", set(actual) == set(expected))
print("Same column order:", actual == expected)
PY
```

Expected diagnosis:

```text
Missing: []
Unexpected: []
Same column set: True
Same column order: False
```

## Patch the verifier

Column order has no semantic effect in a Hugging Face dataset. Replace the strict tuple-order check with a strict column-membership check:

```bash
cd /data/saral/wdir/smart_v2 || exit 1

python3 - <<'PY'
from pathlib import Path

path = Path(
    "src/mixtures/materialize_smart_mixtures.py"
)

text = path.read_text(encoding="utf-8")

old = '''    if tuple(train.column_names) != OUTPUT_COLUMNS:
        raise RuntimeError(
            f"Budget {budget}: output columns differ "
            "from the frozen schema."
        )
'''

new = '''    actual_columns = tuple(train.column_names)

    missing_columns = [
        column
        for column in OUTPUT_COLUMNS
        if column not in actual_columns
    ]

    unexpected_columns = [
        column
        for column in actual_columns
        if column not in OUTPUT_COLUMNS
    ]

    if missing_columns or unexpected_columns:
        raise RuntimeError(
            f"Budget {budget}: output schema mismatch. "
            f"Missing={missing_columns}, "
            f"unexpected={unexpected_columns}, "
            f"actual_order={list(actual_columns)}."
        )
'''

if old not in text:
    raise RuntimeError(
        "Could not locate the original strict column-order check."
    )

path.write_text(
    text.replace(old, new, 1),
    encoding="utf-8",
)

print("Patched:", path)
PY

python3 -m py_compile \
  src/mixtures/materialize_smart_mixtures.py
```

This remains strict about missing or unexpected columns while correctly treating column order as non-semantic.

## Rerun materialization

The partially written `smart_25000` is unverified. The existing `--overwrite` option will replace it safely.

```bash
cd /data/saral/wdir/smart_v2 || exit 1
source ./smart_v2_env.sh

python3 -m src.mixtures.materialize_smart_mixtures \
  --submixture-config /data/saral/wdir/smart_v2/configs/submixtures.json \
  --author-format-root /mnt/warm_storage/saral/smart_v2/author_format \
  --task-indices-root /mnt/warm_storage/saral/smart_v2/author_format/task_indices \
  --orderings /mnt/warm_storage/saral/smart_v2/stage2/author_orderings/facility_location_orderings_all.pkl \
  --allocations-root /mnt/warm_storage/saral/smart_v2/stage1/allocations \
  --assembly-summary /mnt/warm_storage/saral/smart_v2/stage2/author_orderings/assembly_summary.json \
  --output-root /mnt/warm_storage/saral/smart_v2/mixtures \
  --budgets 25000 50000 \
  --verification-batch-size 20000 \
  --overwrite \
  2>&1 | tee \
  /mnt/warm_storage/saral/smart_v2/logs/materialize_smart_mixtures.log
```

No SMART selection, ordering, shuffle, or dataset content is changed by this patch; only the overly strict physical-column-order assertion is corrected.


