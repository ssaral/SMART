"""Benchmark the authors' exact dense Stage 2 Facility Location call."""

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
        "--task-id",
        required=True,
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=0,
        help="Use 0 for the complete task.",
    )
    parser.add_argument(
        "--budget-column",
        default="final_allocation_50000",
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
    # Linux reports ru_maxrss in KiB.
    return (
        resource.getrusage(
            resource.RUSAGE_SELF
        ).ru_maxrss
        * 1024
        / (1024 ** 3)
    )


def sha256_array(array: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(
        np.ascontiguousarray(array).tobytes()
    )
    return digest.hexdigest()


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


def load_task_row(
    path: Path,
    task_id: str,
) -> dict[str, str]:
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

    return matches[0]


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
            f"Could not uniquely resolve output directory "
            f"for task index {task_index}: {candidates}"
        )

    return candidates[0]


def main() -> int:
    args = parse_args()

    allocation_path = args.allocations.resolve()
    embedding_root = args.embedding_root.resolve()
    output_root = args.output_root.resolve()

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    task_row = load_task_row(
        allocation_path,
        args.task_id,
    )

    if args.budget_column not in task_row:
        raise KeyError(
            f"Missing budget column: {args.budget_column}"
        )

    task_index = int(task_row["task_index"])
    full_count = int(
        task_row["valid_train_count"]
    )
    requested_budget = int(
        task_row[args.budget_column]
    )

    task_directory = resolve_task_directory(
        embedding_root,
        task_index,
        args.task_id,
    )

    embedding_path = (
        task_directory / "prompt_embeddings.npy"
    )
    source_indices_path = (
        task_directory / "source_indices.npy"
    )

    embeddings_memmap = np.load(
        embedding_path,
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

    if source_indices_memmap.shape != (
        full_count,
    ):
        raise RuntimeError(
            f"Unexpected source-index shape: "
            f"{source_indices_memmap.shape}"
        )

    if args.sample_size == 0:
        sample_size = full_count
    else:
        sample_size = args.sample_size

    if not 2 <= sample_size <= full_count:
        raise ValueError(
            f"Sample size must be between 2 and "
            f"{full_count:,}; received {sample_size:,}."
        )

    # Deterministic, approximately uniform positions across
    # the complete task ordering.
    sampled_local_indices = (
        np.arange(
            sample_size,
            dtype=np.int64,
        )
        * full_count
        // sample_size
    )

    if np.unique(
        sampled_local_indices
    ).size != sample_size:
        raise RuntimeError(
            "Deterministic sample contains duplicate indices."
        )

    budget = min(
        requested_budget,
        sample_size - 1,
    )

    if budget <= 0:
        raise RuntimeError(
            "Effective Facility Location budget is zero."
        )

    start_rss = peak_rss_gib()
    load_start = time.perf_counter()

    sampled_embeddings = np.array(
        embeddings_memmap[
            sampled_local_indices
        ],
        dtype=np.float32,
        order="C",
        copy=True,
    )

    sampled_source_indices = np.array(
        source_indices_memmap[
            sampled_local_indices
        ],
        dtype=np.int64,
        copy=True,
    )

    load_seconds = (
        time.perf_counter() - load_start
    )
    after_load_rss = peak_rss_gib()

    if not np.isfinite(
        sampled_embeddings
    ).all():
        raise RuntimeError(
            "Sampled embeddings contain non-finite values."
        )

    print("=== Exact dense Facility Location benchmark ===")
    print(f"Task:                  {args.task_id}")
    print(f"Full task size:        {full_count:,}")
    print(f"Benchmark size:        {sample_size:,}")
    print(f"Requested allocation:  {requested_budget}")
    print(f"Effective FL budget:   {budget}")
    print(f"Embedding shape:       {sampled_embeddings.shape}")
    print(f"Embedding dtype:       {sampled_embeddings.dtype}")
    print(
        f"Raw float32 kernel:    "
        f"{sample_size * sample_size * 4 / (1024 ** 3):.3f} GiB"
    )
    print(
        f"OMP_NUM_THREADS:       "
        f"{os.environ.get('OMP_NUM_THREADS')}"
    )
    print()

    constructor_start = time.perf_counter()

    # Exact call from the authors' fl_ordering.py.
    objective = (
        submod_fn.facilityLocation
        .FacilityLocationFunction(
            n=sampled_embeddings.shape[0],
            separate_rep=False,
            mode="dense",
            data=sampled_embeddings,
            create_dense_cpp_kernel_in_python=False,
        )
    )

    constructor_seconds = (
        time.perf_counter()
        - constructor_start
    )
    after_constructor_rss = peak_rss_gib()

    maximize_start = time.perf_counter()

    greedy_result = objective.maximize(
        budget=budget,
        optimizer="LazyGreedy",
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

    if len(greedy_result) != budget:
        raise RuntimeError(
            f"Expected {budget} selected examples; "
            f"received {len(greedy_result)}."
        )

    selected_sample_indices = np.asarray(
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

    if np.unique(
        selected_sample_indices
    ).size != budget:
        raise RuntimeError(
            "Facility Location returned duplicate indices."
        )

    if not np.isfinite(gains).all():
        raise RuntimeError(
            "Facility Location returned non-finite gains."
        )

    selected_full_local_indices = (
        sampled_local_indices[
            selected_sample_indices
        ]
    )

    selected_source_indices = (
        sampled_source_indices[
            selected_sample_indices
        ]
    )

    run_name = (
        f"{task_index:04d}_"
        f"{safe_name(args.task_id)}_"
        f"n{sample_size}_k{budget}"
    )

    selection_csv_path = (
        output_root
        / f"{run_name}_selection.csv"
    )
    report_path = (
        output_root
        / f"{run_name}_report.json"
    )

    cumulative_gain = np.cumsum(
        gains,
        dtype=np.float64,
    )

    with selection_csv_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        fieldnames = [
            "selection_rank",
            "sample_index",
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

        for rank in range(budget):
            writer.writerow(
                {
                    "selection_rank": rank + 1,
                    "sample_index": int(
                        selected_sample_indices[rank]
                    ),
                    "full_local_index": int(
                        selected_full_local_indices[rank]
                    ),
                    "source_index": int(
                        selected_source_indices[rank]
                    ),
                    "marginal_gain": float(
                        gains[rank]
                    ),
                    "cumulative_gain": float(
                        cumulative_gain[rank]
                    ),
                }
            )

    report = {
        "status": "complete",
        "method": (
            "authors_exact_dense_facility_location"
        ),
        "task": {
            "task_id": args.task_id,
            "task_index": task_index,
            "full_valid_train_count": full_count,
        },
        "benchmark": {
            "sample_size": sample_size,
            "sampling": (
                "deterministic approximately uniform "
                "positions over full local row order"
            ),
            "requested_budget": requested_budget,
            "effective_budget": budget,
            "budget_column": args.budget_column,
        },
        "authors_call": {
            "function": (
                "FacilityLocationFunction"
            ),
            "mode": "dense",
            "separate_rep": False,
            "data_argument": True,
            "create_dense_cpp_kernel_in_python": False,
            "optimizer": "LazyGreedy",
        },
        "memory": {
            "raw_float32_kernel_gib": (
                sample_size
                * sample_size
                * 4
                / (1024 ** 3)
            ),
            "peak_rss_at_start_gib": (
                start_rss
            ),
            "peak_rss_after_data_load_gib": (
                after_load_rss
            ),
            "peak_rss_after_constructor_gib": (
                after_constructor_rss
            ),
            "peak_rss_after_maximize_gib": (
                final_rss
            ),
        },
        "timing_seconds": {
            "load_sample": load_seconds,
            "construct_objective": (
                constructor_seconds
            ),
            "maximize": maximize_seconds,
            "facility_location_total": (
                constructor_seconds
                + maximize_seconds
            ),
        },
        "result": {
            "selection_count": (
                len(greedy_result)
            ),
            "minimum_gain": float(
                gains.min()
            ),
            "maximum_gain": float(
                gains.max()
            ),
            "final_cumulative_gain": float(
                cumulative_gain[-1]
            ),
            "selected_full_local_indices_sha256": (
                sha256_array(
                    selected_full_local_indices
                )
            ),
            "selected_source_indices_sha256": (
                sha256_array(
                    selected_source_indices
                )
            ),
        },
        "inputs": {
            "embedding_path": str(
                embedding_path
            ),
            "source_indices_path": str(
                source_indices_path
            ),
        },
        "outputs": {
            "selection_csv": str(
                selection_csv_path
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
    print("=== Benchmark result ===")
    print(
        f"Data load:             "
        f"{load_seconds:.3f} seconds"
    )
    print(
        f"FL constructor:        "
        f"{constructor_seconds:.3f} seconds"
    )
    print(
        f"LazyGreedy:            "
        f"{maximize_seconds:.3f} seconds"
    )
    print(
        f"FL total:              "
        f"{constructor_seconds + maximize_seconds:.3f} seconds"
    )
    print(
        f"Peak RSS:              "
        f"{final_rss:.3f} GiB"
    )
    print(
        f"Selections:            "
        f"{len(greedy_result)}"
    )
    print(
        f"Gain range:            "
        f"{gains.min():.9g} to "
        f"{gains.max():.9g}"
    )
    print(f"Report:                {report_path}")
    print(f"Selection:             {selection_csv_path}")
    print("Exact Facility Location benchmark passed.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

chmod +x data_generation_scripts/benchmark_exact_facility_location.py
python3 -m py_compile data_generation_scripts/benchmark_exact_facility_location.py
