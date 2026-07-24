"""Validate candidate-restricted Facility Location against exact runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import resource
import time
from pathlib import Path
from typing import Any

import numpy as np
import submodlib.functions as submod_fn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--allocations",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--embedding-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--exact-benchmark-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--task-id",
        required=True,
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--candidate-size",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=23,
    )
    parser.add_argument(
        "--coverage-block-rows",
        type=int,
        default=2048,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--show-progress",
        action="store_true",
    )

    return parser.parse_args()


def safe_name(value: str) -> str:
    return re.sub(
        r"[^A-Za-z0-9_.-]+",
        "__",
        value,
    )


def peak_rss_gib() -> float:
    # Linux ru_maxrss is reported in KiB.
    return (
        resource.getrusage(
            resource.RUSAGE_SELF
        ).ru_maxrss
        * 1024
        / (1024 ** 3)
    )


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


def sha256_array(array: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(
        np.ascontiguousarray(array).tobytes()
    )
    return digest.hexdigest()


def load_task_allocation(
    path: Path,
    task_id: str,
) -> dict[str, Any]:
    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        matches = [
            row
            for row in csv.DictReader(handle)
            if row["task_id"] == task_id
        ]

    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one allocation row for {task_id}; "
            f"found {len(matches)}."
        )

    row = matches[0]

    return {
        "task_index": int(row["task_index"]),
        "task_id": row["task_id"],
        "valid_train_count": int(
            row["valid_train_count"]
        ),
        "allocation_25000": int(
            row["final_allocation_25000"]
        ),
        "allocation_50000": int(
            row["final_allocation_50000"]
        ),
    }


def resolve_task_directory(
    embedding_root: Path,
    task_index: int,
    task_id: str,
) -> Path:
    expected = (
        embedding_root
        / "tasks"
        / f"{task_index:04d}_{safe_name(task_id)}"
    )

    if expected.is_dir():
        return expected

    candidates = sorted(
        (embedding_root / "tasks").glob(
            f"{task_index:04d}_*"
        )
    )

    if len(candidates) != 1:
        raise FileNotFoundError(
            "Could not uniquely resolve task directory: "
            f"{candidates}"
        )

    return candidates[0]


def find_exact_report(
    root: Path,
    task_id: str,
    sample_size: int,
) -> tuple[dict[str, Any], Path]:
    matches: list[tuple[dict[str, Any], Path]] = []

    for path in sorted(root.glob("*_report.json")):
        with path.open(
            "r",
            encoding="utf-8",
        ) as handle:
            report = json.load(handle)

        if (
            report.get("task", {}).get("task_id")
            == task_id
            and report.get("benchmark", {}).get(
                "sample_size"
            )
            == sample_size
        ):
            matches.append((report, path))

    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one exact report for task={task_id}, "
            f"sample_size={sample_size}; found "
            f"{len(matches)}."
        )

    return matches[0]


def load_exact_selection(
    report: dict[str, Any],
) -> list[dict[str, Any]]:
    path = Path(
        report["outputs"]["selection_csv"]
    )

    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        raw_rows = list(csv.DictReader(handle))

    rows = [
        {
            "selection_rank": int(
                row["selection_rank"]
            ),
            "sample_index": int(
                row["sample_index"]
            ),
            "full_local_index": int(
                row["full_local_index"]
            ),
            "source_index": int(
                row["source_index"]
            ),
            "marginal_gain": float(
                row["marginal_gain"]
            ),
            "cumulative_gain": float(
                row["cumulative_gain"]
            ),
        }
        for row in raw_rows
    ]

    rows.sort(
        key=lambda row: row["selection_rank"]
    )

    return rows


def normalize_rows(
    matrix: np.ndarray,
) -> np.ndarray:
    norms = np.linalg.norm(
        matrix,
        axis=1,
        keepdims=True,
    )

    if np.any(norms == 0):
        raise RuntimeError(
            "Zero-norm embedding encountered."
        )

    return (
        matrix / norms
    ).astype(
        np.float32,
        copy=False,
    )


def coverage_statistics(
    represented_embeddings: np.ndarray,
    selected_embeddings: np.ndarray,
    block_rows: int,
) -> dict[str, float]:
    represented = normalize_rows(
        represented_embeddings
    )
    selected = normalize_rows(
        selected_embeddings
    )

    coverage = np.empty(
        represented.shape[0],
        dtype=np.float32,
    )

    selected_transpose = np.ascontiguousarray(
        selected.T
    )

    for start in range(
        0,
        represented.shape[0],
        block_rows,
    ):
        end = min(
            start + block_rows,
            represented.shape[0],
        )

        similarities = (
            represented[start:end]
            @ selected_transpose
        )

        coverage[start:end] = np.max(
            similarities,
            axis=1,
        )

    return {
        "sum": float(
            coverage.sum(dtype=np.float64)
        ),
        "mean": float(
            coverage.mean(dtype=np.float64)
        ),
        "minimum": float(coverage.min()),
        "p01": float(
            np.quantile(coverage, 0.01)
        ),
        "p05": float(
            np.quantile(coverage, 0.05)
        ),
        "median": float(
            np.quantile(coverage, 0.50)
        ),
        "p95": float(
            np.quantile(coverage, 0.95)
        ),
        "maximum": float(coverage.max()),
    }


def prefix_metrics(
    name: str,
    budget: int,
    candidate_selected_sample_indices: np.ndarray,
    exact_rows: list[dict[str, Any]],
    candidate_cumulative_gains: np.ndarray,
    represented_embeddings: np.ndarray,
    block_rows: int,
) -> dict[str, Any]:
    if budget <= 0:
        raise ValueError(
            f"{name} budget must be positive."
        )

    if budget > len(exact_rows):
        raise RuntimeError(
            f"Exact selection has only {len(exact_rows)} "
            f"rows, but {name} requires {budget}."
        )

    if budget > candidate_selected_sample_indices.size:
        raise RuntimeError(
            "Candidate selection is shorter than the "
            f"{name} budget."
        )

    exact_selected = np.asarray(
        [
            row["sample_index"]
            for row in exact_rows[:budget]
        ],
        dtype=np.int64,
    )

    candidate_selected = (
        candidate_selected_sample_indices[:budget]
    )

    exact_set = set(
        int(index)
        for index in exact_selected
    )
    candidate_set = set(
        int(index)
        for index in candidate_selected
    )

    overlap = len(
        exact_set & candidate_set
    )

    exact_objective = float(
        exact_rows[budget - 1][
            "cumulative_gain"
        ]
    )

    candidate_objective = float(
        candidate_cumulative_gains[
            budget - 1
        ]
    )

    exact_coverage = coverage_statistics(
        represented_embeddings,
        represented_embeddings[
            exact_selected
        ],
        block_rows,
    )

    candidate_coverage = coverage_statistics(
        represented_embeddings,
        represented_embeddings[
            candidate_selected
        ],
        block_rows,
    )

    return {
        "name": name,
        "budget": budget,
        "exact_objective": exact_objective,
        "candidate_objective": (
            candidate_objective
        ),
        "objective_ratio": (
            candidate_objective
            / exact_objective
        ),
        "relative_objective_regret": (
            1.0
            - (
                candidate_objective
                / exact_objective
            )
        ),
        "selection_overlap_count": overlap,
        "selection_overlap_fraction": (
            overlap / budget
        ),
        "exact_manual_coverage": (
            exact_coverage
        ),
        "candidate_manual_coverage": (
            candidate_coverage
        ),
        "manual_coverage_sum_ratio": (
            candidate_coverage["sum"]
            / exact_coverage["sum"]
        ),
        "manual_coverage_p05_delta": (
            candidate_coverage["p05"]
            - exact_coverage["p05"]
        ),
    }


def main() -> int:
    args = parse_args()

    if args.sample_size < 2:
        raise ValueError(
            "--sample-size must be at least 2."
        )

    if args.candidate_size < 2:
        raise ValueError(
            "--candidate-size must be at least 2."
        )

    allocation_path = args.allocations.resolve()
    embedding_root = args.embedding_root.resolve()
    exact_root = args.exact_benchmark_root.resolve()
    output_root = args.output_root.resolve()

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    allocation = load_task_allocation(
        allocation_path,
        args.task_id,
    )

    task_index = allocation["task_index"]
    full_count = allocation[
        "valid_train_count"
    ]
    budget_25k = allocation[
        "allocation_25000"
    ]
    budget_50k = allocation[
        "allocation_50000"
    ]

    if args.sample_size > full_count:
        raise ValueError(
            f"Sample size {args.sample_size:,} exceeds "
            f"task size {full_count:,}."
        )

    if args.candidate_size > args.sample_size:
        raise ValueError(
            f"Candidate size {args.candidate_size:,} "
            f"exceeds represented sample size "
            f"{args.sample_size:,}."
        )

    if args.candidate_size <= budget_50k:
        raise ValueError(
            f"Candidate size must exceed the 50K "
            f"allocation because submodlib requires "
            f"budget < ground-set size: "
            f"candidate_size={args.candidate_size}, "
            f"budget={budget_50k}."
        )

    exact_report, exact_report_path = (
        find_exact_report(
            exact_root,
            args.task_id,
            args.sample_size,
        )
    )

    exact_rows = load_exact_selection(
        exact_report
    )

    if len(exact_rows) != budget_50k:
        raise RuntimeError(
            f"Exact benchmark contains "
            f"{len(exact_rows)} selections; expected "
            f"the 50K allocation {budget_50k}."
        )

    task_directory = resolve_task_directory(
        embedding_root,
        task_index,
        args.task_id,
    )

    embeddings_path = (
        task_directory / "prompt_embeddings.npy"
    )
    source_indices_path = (
        task_directory / "source_indices.npy"
    )

    embeddings_memmap = np.load(
        embeddings_path,
        mmap_mode="r",
    )
    source_indices_memmap = np.load(
        source_indices_path,
        mmap_mode="r",
    )

    if embeddings_memmap.shape != (
        full_count,
        1024,
    ):
        raise RuntimeError(
            f"Unexpected embedding shape: "
            f"{embeddings_memmap.shape}"
        )

    represented_full_local_indices = (
        np.arange(
            args.sample_size,
            dtype=np.int64,
        )
        * full_count
        // args.sample_size
    )

    if (
        np.unique(
            represented_full_local_indices
        ).size
        != args.sample_size
    ):
        raise RuntimeError(
            "Represented sample contains duplicate "
            "full-local indices."
        )

    represented_embeddings = np.array(
        embeddings_memmap[
            represented_full_local_indices
        ],
        dtype=np.float32,
        order="C",
        copy=True,
    )

    represented_source_indices = np.array(
        source_indices_memmap[
            represented_full_local_indices
        ],
        dtype=np.int64,
        copy=True,
    )

    if not np.isfinite(
        represented_embeddings
    ).all():
        raise RuntimeError(
            "Represented embeddings contain "
            "non-finite values."
        )

    # One deterministic permutation per task. Candidate
    # sets formed from prefixes are therefore nested.
    seed_sequence = np.random.SeedSequence(
        [args.seed, task_index]
    )
    rng = np.random.default_rng(
        seed_sequence
    )

    candidate_permutation = rng.permutation(
        args.sample_size
    )

    candidate_sample_indices = np.sort(
        candidate_permutation[
            : args.candidate_size
        ]
    ).astype(
        np.int64,
        copy=False,
    )

    candidate_embeddings = np.array(
        represented_embeddings[
            candidate_sample_indices
        ],
        dtype=np.float32,
        order="C",
        copy=True,
    )

    candidate_full_local_indices = (
        represented_full_local_indices[
            candidate_sample_indices
        ]
    )

    candidate_source_indices = (
        represented_source_indices[
            candidate_sample_indices
        ]
    )

    exact_50k_sample_indices = np.asarray(
        [
            row["sample_index"]
            for row in exact_rows
        ],
        dtype=np.int64,
    )

    exact_25k_sample_indices = (
        exact_50k_sample_indices[
            :budget_25k
        ]
    )

    candidate_pool_set = set(
        int(index)
        for index in candidate_sample_indices
    )

    exact_25k_available = sum(
        int(index) in candidate_pool_set
        for index in exact_25k_sample_indices
    )

    exact_50k_available = sum(
        int(index) in candidate_pool_set
        for index in exact_50k_sample_indices
    )

    print(
        "=== Candidate-restricted Facility Location ==="
    )
    print(f"Task:                    {args.task_id}")
    print(f"Full task size:          {full_count:,}")
    print(
        f"Represented-set size:    "
        f"{args.sample_size:,}"
    )
    print(
        f"Candidate-set size:      "
        f"{args.candidate_size:,}"
    )
    print(f"25K allocation:          {budget_25k}")
    print(f"50K allocation:          {budget_50k}")
    print(f"Seed:                    {args.seed}")
    print(
        f"Cross-kernel entries:    "
        f"{args.sample_size * args.candidate_size:,}"
    )
    print(
        f"Raw float32 cross-kernel:"
        f" {args.sample_size * args.candidate_size * 4 / (1024 ** 3):.3f} GiB"
    )
    print(
        f"Exact 50K choices present in pool: "
        f"{exact_50k_available}/{budget_50k}"
    )
    print()

    initial_rss = peak_rss_gib()
    constructor_start = time.perf_counter()

    # Scalable adaptation:
    # represented set U = complete task/sample
    # selectable ground set V = candidate pool
    objective = (
        submod_fn.facilityLocation
        .FacilityLocationFunction(
            n=args.candidate_size,
            mode="dense",
            separate_rep=True,
            n_rep=args.sample_size,
            data=candidate_embeddings,
            data_rep=represented_embeddings,
            metric="cosine",
            create_dense_cpp_kernel_in_python=False,
        )
    )

    constructor_seconds = (
        time.perf_counter()
        - constructor_start
    )

    constructor_rss = peak_rss_gib()

    maximize_start = time.perf_counter()

    greedy_result = objective.maximize(
        budget=budget_50k,
        optimizer="LazyGreedy",
        stopIfZeroGain=False,
        stopIfNegativeGain=False,
        verbose=False,
        show_progress=args.show_progress,
    )

    maximize_seconds = (
        time.perf_counter()
        - maximize_start
    )

    final_rss = peak_rss_gib()

    greedy_result = [
        (int(index), float(gain))
        for index, gain in greedy_result
    ]

    if len(greedy_result) != budget_50k:
        raise RuntimeError(
            f"Expected {budget_50k} selections; "
            f"received {len(greedy_result)}."
        )

    selected_candidate_indices = np.asarray(
        [
            index
            for index, _ in greedy_result
        ],
        dtype=np.int64,
    )

    gains = np.asarray(
        [
            gain
            for _, gain in greedy_result
        ],
        dtype=np.float64,
    )

    if (
        np.unique(
            selected_candidate_indices
        ).size
        != budget_50k
    ):
        raise RuntimeError(
            "Candidate Facility Location returned "
            "duplicate selections."
        )

    if not np.isfinite(gains).all():
        raise RuntimeError(
            "Candidate Facility Location returned "
            "non-finite gains."
        )

    selected_sample_indices = (
        candidate_sample_indices[
            selected_candidate_indices
        ]
    )

    selected_full_local_indices = (
        candidate_full_local_indices[
            selected_candidate_indices
        ]
    )

    selected_source_indices = (
        candidate_source_indices[
            selected_candidate_indices
        ]
    )

    cumulative_gains = np.cumsum(
        gains,
        dtype=np.float64,
    )

    metrics_25k = prefix_metrics(
        name="25000",
        budget=budget_25k,
        candidate_selected_sample_indices=(
            selected_sample_indices
        ),
        exact_rows=exact_rows,
        candidate_cumulative_gains=(
            cumulative_gains
        ),
        represented_embeddings=(
            represented_embeddings
        ),
        block_rows=args.coverage_block_rows,
    )

    metrics_50k = prefix_metrics(
        name="50000",
        budget=budget_50k,
        candidate_selected_sample_indices=(
            selected_sample_indices
        ),
        exact_rows=exact_rows,
        candidate_cumulative_gains=(
            cumulative_gains
        ),
        represented_embeddings=(
            represented_embeddings
        ),
        block_rows=args.coverage_block_rows,
    )

    full_candidate_equivalence = None

    if args.candidate_size == args.sample_size:
        exact_order = np.asarray(
            [
                row["sample_index"]
                for row in exact_rows
            ],
            dtype=np.int64,
        )

        full_candidate_equivalence = {
            "order_identical": bool(
                np.array_equal(
                    selected_sample_indices,
                    exact_order,
                )
            ),
            "maximum_gain_absolute_difference": float(
                np.max(
                    np.abs(
                        gains
                        - np.asarray(
                            [
                                row["marginal_gain"]
                                for row in exact_rows
                            ],
                            dtype=np.float64,
                        )
                    )
                )
            ),
        }

    run_name = (
        f"{task_index:04d}_"
        f"{safe_name(args.task_id)}_"
        f"rep{args.sample_size}_"
        f"cand{args.candidate_size}_"
        f"k{budget_50k}"
    )

    selection_path = (
        output_root
        / f"{run_name}_selection.csv"
    )
    report_path = (
        output_root
        / f"{run_name}_report.json"
    )

    with selection_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        fieldnames = [
            "selection_rank",
            "candidate_index",
            "represented_sample_index",
            "full_local_index",
            "source_index",
            "marginal_gain",
            "cumulative_gain",
        ]

        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )
        writer.writeheader()

        for rank in range(budget_50k):
            writer.writerow(
                {
                    "selection_rank": rank + 1,
                    "candidate_index": int(
                        selected_candidate_indices[
                            rank
                        ]
                    ),
                    "represented_sample_index": int(
                        selected_sample_indices[
                            rank
                        ]
                    ),
                    "full_local_index": int(
                        selected_full_local_indices[
                            rank
                        ]
                    ),
                    "source_index": int(
                        selected_source_indices[
                            rank
                        ]
                    ),
                    "marginal_gain": float(
                        gains[rank]
                    ),
                    "cumulative_gain": float(
                        cumulative_gains[rank]
                    ),
                }
            )

    report = {
        "status": "complete",
        "method": (
            "candidate_restricted_facility_location"
        ),
        "approximation": {
            "represented_set": (
                "All rows in the benchmark task/sample."
            ),
            "selectable_ground_set": (
                "Nested deterministic random candidate "
                "pool."
            ),
            "objective": (
                "Facility Location with separate_rep=True."
            ),
            "seed": args.seed,
            "candidate_pool_is_nested": True,
        },
        "task": {
            "task_id": args.task_id,
            "task_index": task_index,
            "full_valid_train_count": full_count,
        },
        "benchmark": {
            "represented_size": args.sample_size,
            "candidate_size": args.candidate_size,
            "allocation_25000": budget_25k,
            "allocation_50000": budget_50k,
            "exact_report": str(
                exact_report_path
            ),
        },
        "candidate_pool": {
            "exact_25000_selected_items_available": (
                exact_25k_available
            ),
            "exact_25000_available_fraction": (
                exact_25k_available
                / budget_25k
            ),
            "exact_50000_selected_items_available": (
                exact_50k_available
            ),
            "exact_50000_available_fraction": (
                exact_50k_available
                / budget_50k
            ),
            "sample_indices_sha256": (
                sha256_array(
                    candidate_sample_indices
                )
            ),
        },
        "timing_seconds": {
            "construct_objective": (
                constructor_seconds
            ),
            "maximize": maximize_seconds,
            "total": (
                constructor_seconds
                + maximize_seconds
            ),
        },
        "memory": {
            "raw_float32_cross_kernel_gib": (
                args.sample_size
                * args.candidate_size
                * 4
                / (1024 ** 3)
            ),
            "peak_rss_at_start_gib": (
                initial_rss
            ),
            "peak_rss_after_constructor_gib": (
                constructor_rss
            ),
            "peak_rss_after_maximize_gib": (
                final_rss
            ),
        },
        "metrics": {
            "25000": metrics_25k,
            "50000": metrics_50k,
        },
        "full_candidate_equivalence": (
            full_candidate_equivalence
        ),
        "result": {
            "selection_count": (
                budget_50k
            ),
            "minimum_gain": float(
                gains.min()
            ),
            "maximum_gain": float(
                gains.max()
            ),
            "selected_sample_indices_sha256": (
                sha256_array(
                    selected_sample_indices
                )
            ),
            "selected_source_indices_sha256": (
                sha256_array(
                    selected_source_indices
                )
            ),
        },
        "outputs": {
            "selection_csv": str(
                selection_path
            ),
        },
        "environment": {
            "omp_num_threads": (
                os.environ.get(
                    "OMP_NUM_THREADS"
                )
            ),
            "mkl_num_threads": (
                os.environ.get(
                    "MKL_NUM_THREADS"
                )
            ),
            "openblas_num_threads": (
                os.environ.get(
                    "OPENBLAS_NUM_THREADS"
                )
            ),
            "cpu_count": os.cpu_count(),
        },
    }

    atomic_write_json(
        report,
        report_path,
    )

    print()
    print("=== Candidate validation result ===")
    print(
        f"Constructor:                 "
        f"{constructor_seconds:.3f} seconds"
    )
    print(
        f"LazyGreedy:                 "
        f"{maximize_seconds:.3f} seconds"
    )
    print(
        f"Peak RSS:                   "
        f"{final_rss:.3f} GiB"
    )
    print(
        f"25K objective ratio:        "
        f"{metrics_25k['objective_ratio']:.8f}"
    )
    print(
        f"50K objective ratio:        "
        f"{metrics_50k['objective_ratio']:.8f}"
    )
    print(
        f"25K selection overlap:      "
        f"{metrics_25k['selection_overlap_fraction']:.4f}"
    )
    print(
        f"50K selection overlap:      "
        f"{metrics_50k['selection_overlap_fraction']:.4f}"
    )
    print(
        f"25K manual coverage ratio:  "
        f"{metrics_25k['manual_coverage_sum_ratio']:.8f}"
    )
    print(
        f"50K manual coverage ratio:  "
        f"{metrics_50k['manual_coverage_sum_ratio']:.8f}"
    )

    if full_candidate_equivalence is not None:
        print(
            f"Exact order identical:       "
            f"{full_candidate_equivalence['order_identical']}"
        )
        print(
            f"Maximum gain difference:     "
            f"{full_candidate_equivalence['maximum_gain_absolute_difference']:.9g}"
        )

    print(f"Report:                     {report_path}")
    print(f"Selection:                  {selection_path}")
    print(
        "Candidate-restricted Facility Location "
        "validation passed."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
