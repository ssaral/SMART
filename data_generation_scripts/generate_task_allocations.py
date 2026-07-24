"""Convert SMART Graph Cut gains into exact task-level budgets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np


EXPECTED_TASK_COUNT = 309


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply the SMART repository's second-order Taylor "
            "softmax and cyclic integer allocation."
        )
    )
    parser.add_argument(
        "--graph-cut-order",
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
        "--budgets",
        type=int,
        nargs="+",
        default=[25_000, 50_000],
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
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
    payload: dict[str, Any],
    path: Path,
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")

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


def load_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        return list(csv.DictReader(handle))


def load_manifest_counts(
    path: Path,
) -> dict[str, int]:
    rows = load_csv(path)

    required = {
        "task_id",
        "valid_train_count",
    }

    if not rows:
        raise RuntimeError("Clean task manifest is empty.")

    missing = required - set(rows[0])

    if missing:
        raise ValueError(
            "Clean manifest is missing columns: "
            + ", ".join(sorted(missing))
        )

    result: dict[str, int] = {}

    for row in rows:
        task_id = row["task_id"]

        if task_id in result:
            raise ValueError(
                f"Duplicate task ID in manifest: {task_id}"
            )

        count = int(row["valid_train_count"])

        if count <= 0:
            raise ValueError(
                f"Task has no valid training rows: {task_id}"
            )

        result[task_id] = count

    return result


def load_graph_cut_rows(
    path: Path,
    manifest_counts: dict[str, int],
) -> list[dict[str, Any]]:
    raw_rows = load_csv(path)

    if not raw_rows:
        raise RuntimeError("Graph Cut order CSV is empty.")

    required = {
        "graph_cut_rank",
        "task_index",
        "task_id",
        "corpus",
        "task_name",
        "valid_train_count",
        "marginal_gain",
    }

    missing = required - set(raw_rows[0])

    if missing:
        raise ValueError(
            "Graph Cut CSV is missing columns: "
            + ", ".join(sorted(missing))
        )

    rows: list[dict[str, Any]] = []

    for raw in raw_rows:
        task_id = raw["task_id"]

        if task_id not in manifest_counts:
            raise KeyError(
                f"Graph Cut task not found in manifest: {task_id}"
            )

        graph_count = int(raw["valid_train_count"])
        manifest_count = manifest_counts[task_id]

        if graph_count != manifest_count:
            raise RuntimeError(
                f"Valid-row mismatch for {task_id}: "
                f"Graph Cut={graph_count}, "
                f"manifest={manifest_count}."
            )

        rows.append(
            {
                "graph_cut_rank": int(
                    raw["graph_cut_rank"]
                ),
                "task_index": int(raw["task_index"]),
                "task_id": task_id,
                "corpus": raw["corpus"],
                "task_name": raw["task_name"],
                "valid_train_count": manifest_count,
                "marginal_gain": float(
                    raw["marginal_gain"]
                ),
            }
        )

    rows.sort(
        key=lambda row: row["graph_cut_rank"]
    )

    if len(rows) != EXPECTED_TASK_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_TASK_COUNT} Graph Cut rows; "
            f"found {len(rows)}."
        )

    expected_ranks = list(
        range(1, EXPECTED_TASK_COUNT + 1)
    )
    actual_ranks = [
        row["graph_cut_rank"]
        for row in rows
    ]

    if actual_ranks != expected_ranks:
        raise RuntimeError(
            "Graph Cut ranks are not contiguous from 1 to 309."
        )

    task_ids = [
        row["task_id"]
        for row in rows
    ]
    task_indices = [
        row["task_index"]
        for row in rows
    ]

    if len(set(task_ids)) != EXPECTED_TASK_COUNT:
        raise RuntimeError(
            "Graph Cut order contains duplicate task IDs."
        )

    if len(set(task_indices)) != EXPECTED_TASK_COUNT:
        raise RuntimeError(
            "Graph Cut order contains duplicate task indices."
        )

    gains = np.asarray(
        [row["marginal_gain"] for row in rows],
        dtype=np.float64,
    )

    if not np.isfinite(gains).all():
        raise RuntimeError(
            "Graph Cut gains contain non-finite values."
        )

    return rows


def cyclic_add(
    allocations: np.ndarray,
    capacities: np.ndarray,
    amount: int,
) -> np.ndarray:
    """Redistribute amount one unit at a time in greedy order."""

    if amount < 0:
        raise ValueError("Cyclic amount cannot be negative.")

    additions = np.zeros_like(
        allocations,
        dtype=np.int64,
    )

    remaining = int(amount)

    while remaining > 0:
        progress = False

        for index in range(allocations.shape[0]):
            if remaining == 0:
                break

            if allocations[index] >= capacities[index]:
                continue

            allocations[index] += 1
            additions[index] += 1
            remaining -= 1
            progress = True

        if not progress:
            raise RuntimeError(
                "Insufficient remaining task capacity to "
                f"redistribute {remaining:,} examples."
            )

    return additions


def allocate_budget(
    weights: np.ndarray,
    capacities: np.ndarray,
    budget: int,
) -> dict[str, Any]:
    if budget <= 0:
        raise ValueError("Budget must be positive.")

    total_capacity = int(capacities.sum())

    if budget > total_capacity:
        raise ValueError(
            f"Budget {budget:,} exceeds total capacity "
            f"{total_capacity:,}."
        )

    raw_allocations = weights * budget
    floor_allocations = np.floor(
        raw_allocations
    ).astype(np.int64)

    pre_cap_allocations = floor_allocations.copy()

    initial_remainder = int(
        budget - pre_cap_allocations.sum()
    )

    # At this stage capacities are intentionally ignored because the
    # repository first integerizes and then handles overflow.
    remainder_additions = np.zeros_like(
        pre_cap_allocations,
        dtype=np.int64,
    )

    for offset in range(initial_remainder):
        index = offset % pre_cap_allocations.shape[0]
        pre_cap_allocations[index] += 1
        remainder_additions[index] += 1

    overflow_by_task = np.maximum(
        pre_cap_allocations - capacities,
        0,
    )

    overflow_total = int(
        overflow_by_task.sum()
    )

    final_allocations = np.minimum(
        pre_cap_allocations,
        capacities,
    ).astype(
        np.int64,
        copy=False,
    )

    redistribution_additions = cyclic_add(
        allocations=final_allocations,
        capacities=capacities,
        amount=overflow_total,
    )

    if int(final_allocations.sum()) != budget:
        raise RuntimeError(
            f"Final allocation sums to "
            f"{int(final_allocations.sum()):,}; "
            f"expected {budget:,}."
        )

    if np.any(final_allocations < 0):
        raise RuntimeError(
            "Final allocation contains negative values."
        )

    if np.any(final_allocations > capacities):
        raise RuntimeError(
            "Final allocation exceeds a task capacity."
        )

    return {
        "budget": budget,
        "raw": raw_allocations,
        "floor": floor_allocations,
        "initial_remainder": initial_remainder,
        "remainder_additions": remainder_additions,
        "pre_cap": pre_cap_allocations,
        "overflow_by_task": overflow_by_task,
        "overflow_total": overflow_total,
        "redistribution_additions": (
            redistribution_additions
        ),
        "final": final_allocations,
    }


def main() -> int:
    args = parse_args()

    if args.temperature <= 0:
        raise ValueError(
            "--temperature must be positive."
        )

    budgets = list(dict.fromkeys(args.budgets))

    if not budgets:
        raise ValueError(
            "At least one budget must be provided."
        )

    if any(budget <= 0 for budget in budgets):
        raise ValueError(
            "Every requested budget must be positive."
        )

    graph_cut_path = args.graph_cut_order.resolve()
    manifest_path = args.manifest.resolve()
    output_root = args.output_root.resolve()

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest_counts = load_manifest_counts(
        manifest_path
    )
    rows = load_graph_cut_rows(
        graph_cut_path,
        manifest_counts,
    )

    gains = np.asarray(
        [row["marginal_gain"] for row in rows],
        dtype=np.float64,
    )

    capacities = np.asarray(
        [row["valid_train_count"] for row in rows],
        dtype=np.int64,
    )

    scaled_gains = gains / args.temperature

    # Exact second-order Taylor approximation used by the
    # inspected SMART repository.
    taylor_scores = (
        1.0
        + scaled_gains
        + 0.5 * scaled_gains * scaled_gains
    )

    if not np.isfinite(taylor_scores).all():
        raise RuntimeError(
            "Taylor scores contain non-finite values."
        )

    if np.any(taylor_scores <= 0):
        bad = np.flatnonzero(
            taylor_scores <= 0
        ).tolist()

        raise RuntimeError(
            f"Nonpositive Taylor scores at rows {bad}."
        )

    weights = taylor_scores / taylor_scores.sum(
        dtype=np.float64
    )

    if not np.isclose(
        weights.sum(dtype=np.float64),
        1.0,
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError(
            "Normalized task weights do not sum to one."
        )

    allocation_results = {
        budget: allocate_budget(
            weights=weights,
            capacities=capacities,
            budget=budget,
        )
        for budget in budgets
    }

    output_rows: list[dict[str, Any]] = []

    for index, row in enumerate(rows):
        output_row: dict[str, Any] = {
            **row,
            "scaled_gain": float(
                scaled_gains[index]
            ),
            "taylor_score": float(
                taylor_scores[index]
            ),
            "normalized_weight": float(
                weights[index]
            ),
        }

        for budget in budgets:
            result = allocation_results[budget]
            suffix = str(budget)

            output_row.update(
                {
                    f"raw_allocation_{suffix}": float(
                        result["raw"][index]
                    ),
                    f"floor_allocation_{suffix}": int(
                        result["floor"][index]
                    ),
                    f"remainder_addition_{suffix}": int(
                        result["remainder_additions"][index]
                    ),
                    f"pre_cap_allocation_{suffix}": int(
                        result["pre_cap"][index]
                    ),
                    f"overflow_{suffix}": int(
                        result["overflow_by_task"][index]
                    ),
                    f"redistribution_addition_{suffix}": int(
                        result[
                            "redistribution_additions"
                        ][index]
                    ),
                    f"final_allocation_{suffix}": int(
                        result["final"][index]
                    ),
                }
            )

        output_rows.append(output_row)

    allocation_csv_path = (
        output_root / "task_allocations.csv"
    )
    summary_path = (
        output_root / "allocation_summary.json"
    )

    with allocation_csv_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(output_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(output_rows)

    budget_summaries: dict[str, Any] = {}

    for budget in budgets:
        result = allocation_results[budget]
        final = result["final"]

        budget_summaries[str(budget)] = {
            "requested_budget": budget,
            "final_sum": int(final.sum()),
            "initial_floor_sum": int(
                result["floor"].sum()
            ),
            "initial_remainder": int(
                result["initial_remainder"]
            ),
            "overflow_total": int(
                result["overflow_total"]
            ),
            "task_count_with_nonzero_allocation": int(
                np.sum(final > 0)
            ),
            "task_count_with_zero_allocation": int(
                np.sum(final == 0)
            ),
            "task_count_at_capacity": int(
                np.sum(final == capacities)
            ),
            "minimum_allocation": int(
                final.min()
            ),
            "maximum_allocation": int(
                final.max()
            ),
            "minimum_graph_cut_rank_with_allocation": int(
                np.flatnonzero(final > 0)[0] + 1
            ),
            "maximum_graph_cut_rank_with_allocation": int(
                np.flatnonzero(final > 0)[-1] + 1
            ),
        }

    monotonic_checks: dict[str, Any] = {}

    sorted_budgets = sorted(budgets)

    for smaller, larger in zip(
        sorted_budgets,
        sorted_budgets[1:],
    ):
        smaller_alloc = allocation_results[
            smaller
        ]["final"]
        larger_alloc = allocation_results[
            larger
        ]["final"]

        violations = np.flatnonzero(
            larger_alloc < smaller_alloc
        )

        monotonic_checks[
            f"{smaller}_to_{larger}"
        ] = {
            "violation_count": int(
                violations.size
            ),
            "violating_task_ids": [
                rows[int(index)]["task_id"]
                for index in violations
            ],
        }

    summary = {
        "format_version": 1,
        "stage": "SMART_stage1_task_allocation",
        "status": "complete",
        "configuration": {
            "task_count": len(rows),
            "budgets": budgets,
            "temperature": args.temperature,
            "score_function": (
                "1 + (gain / temperature) + "
                "0.5 * (gain / temperature)^2"
            ),
            "normalization": (
                "Divide Taylor scores by their total."
            ),
            "integerization": (
                "Floor allocations, then distribute the "
                "remainder cyclically in Graph Cut order."
            ),
            "overflow_handling": (
                "Cap by valid task size, then redistribute "
                "overflow cyclically in Graph Cut order."
            ),
            "approximation": False,
        },
        "inputs": {
            "graph_cut_order": str(
                graph_cut_path
            ),
            "graph_cut_order_sha256": (
                sha256_file(graph_cut_path)
            ),
            "clean_task_manifest": str(
                manifest_path
            ),
            "clean_task_manifest_sha256": (
                sha256_file(manifest_path)
            ),
        },
        "weights": {
            "sum": float(
                weights.sum(dtype=np.float64)
            ),
            "minimum": float(weights.min()),
            "maximum": float(weights.max()),
            "minimum_taylor_score": float(
                taylor_scores.min()
            ),
            "maximum_taylor_score": float(
                taylor_scores.max()
            ),
        },
        "budgets": budget_summaries,
        "cross_budget_monotonicity": (
            monotonic_checks
        ),
        "outputs": {
            "task_allocations_csv": str(
                allocation_csv_path
            ),
        },
    }

    atomic_write_json(
        summary,
        summary_path,
    )

    print("=== SMART task allocations ===")
    print(f"Tasks:       {len(rows)}")
    print(f"Temperature: {args.temperature}")
    print(
        f"Gain range:  "
        f"{gains.min():.9g} to "
        f"{gains.max():.9g}"
    )
    print(
        f"Weight range:"
        f" {weights.min():.12g} to "
        f"{weights.max():.12g}"
    )

    for budget in budgets:
        info = budget_summaries[str(budget)]

        print()
        print(f"Budget {budget:,}:")
        print(
            f"  Final sum:               "
            f"{info['final_sum']:,}"
        )
        print(
            f"  Initial remainder:       "
            f"{info['initial_remainder']}"
        )
        print(
            f"  Overflow redistributed:  "
            f"{info['overflow_total']:,}"
        )
        print(
            f"  Tasks with allocation:   "
            f"{info['task_count_with_nonzero_allocation']}"
        )
        print(
            f"  Tasks at capacity:       "
            f"{info['task_count_at_capacity']}"
        )
        print(
            f"  Allocation range:        "
            f"{info['minimum_allocation']} to "
            f"{info['maximum_allocation']}"
        )

    print()
    print(f"Allocation CSV: {allocation_csv_path}")
    print(f"Summary JSON:   {summary_path}")
    print("SMART task allocation passed.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# python3 -m py_compile data_generation_scripts/generate_task_allocations.py

# To run this script:
# mkdir -p  /mnt/warm_storage/saral/smart/artifacts/stage1_allocations
# python3 data_generation_scripts/generate_task_allocations.py \
#   --graph-cut-order /mnt/warm_storage/saral/smart/artifacts/stage1_graph_cut/graph_cut_order.csv \
#   --manifest /mnt/warm_storage/saral/smart/prepared_data/clean_task_manifest.csv \
#   --output-root /mnt/warm_storage/saral/smart/artifacts/stage1_allocations \
#   --budgets 25000 50000 \
#   --temperature 1.0 \
#   2>&1 | tee \
#   /mnt/warm_storage/saral/smart/artifacts/stage1_allocations/allocation.log
