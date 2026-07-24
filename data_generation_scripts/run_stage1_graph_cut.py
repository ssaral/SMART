"""Run SMART Stage 1 task selection with exact dense Graph Cut."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import submodlib
import submodlib.functions as submod_fn

from local_dataset import load_task_catalog


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the task-level cosine kernel and reproduce the "
            "repository's dense Graph Cut call with LazyGreedy."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--task-embeddings",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--lambda-value",
        type=float,
        default=0.4,
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=None,
        help="Defaults to the complete task count.",
    )
    parser.add_argument(
        "--determinism-runs",
        type=int,
        default=2,
        help=(
            "Repeat Graph Cut and require identical order and gains."
        ),
    )
    parser.add_argument(
        "--kernel-atol",
        type=float,
        default=2e-6,
    )
    return parser.parse_args()


def safe_name(value: str) -> str:
    return re.sub(
        r"[^A-Za-z0-9_.-]+",
        "__",
        value,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


def atomic_save_numpy(
    array: np.ndarray,
    path: Path,
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")

    with temporary.open("wb") as handle:
        np.save(handle, array)

    temporary.replace(path)


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


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def run_graph_cut(
    kernel: np.ndarray,
    budget: int,
    lambda_value: float,
    show_progress: bool,
) -> list[tuple[int, float]]:
    # This mirrors the inspected SMART repository call.
    objective = submod_fn.graphCut.GraphCutFunction(
        n=kernel.shape[0],
        mode="dense",
        ggsijs=kernel,
        lambdaVal=lambda_value,
        separate_rep=False,
    )

    result = objective.maximize(
        budget=budget,
        optimizer="LazyGreedy",
        show_progress=show_progress,
    )

    return [
        (int(index), float(gain))
        for index, gain in result
    ]


def main() -> int:
    args = parse_args()

    if args.determinism_runs < 1:
        raise ValueError(
            "--determinism-runs must be at least 1."
        )

    if args.lambda_value < 0:
        raise ValueError(
            "--lambda-value must be non-negative."
        )

    manifest_path = args.manifest.resolve()
    task_embeddings_path = (
        args.task_embeddings.resolve()
    )
    output_root = args.output_root.resolve()

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    tasks = load_task_catalog(manifest_path)
    task_count = len(tasks)

    budget = (
        task_count
        if args.budget is None
        else args.budget
    )

    if not 1 <= budget <= task_count:
        raise ValueError(
            f"Budget must be in [1, {task_count}], "
            f"received {budget}."
        )

    if not task_embeddings_path.is_file():
        raise FileNotFoundError(
            "Task embedding matrix not found: "
            f"{task_embeddings_path}"
        )

    loaded_embeddings = np.load(
        task_embeddings_path,
        mmap_mode="r",
    )

    if loaded_embeddings.ndim != 2:
        raise RuntimeError(
            "Expected a two-dimensional matrix; "
            f"received {loaded_embeddings.shape}."
        )

    if loaded_embeddings.shape[0] != task_count:
        raise RuntimeError(
            f"Embedding rows={loaded_embeddings.shape[0]}, "
            f"tasks={task_count}."
        )

    if not np.isfinite(loaded_embeddings).all():
        raise RuntimeError(
            "Task embeddings contain non-finite values."
        )

    task_embeddings = np.asarray(
        loaded_embeddings,
        dtype=np.float32,
    )

    task_norms = np.linalg.norm(
        task_embeddings,
        axis=1,
    )

    if np.any(task_norms == 0):
        zero_indices = np.flatnonzero(
            task_norms == 0
        ).tolist()

        raise RuntimeError(
            "Zero-norm task embeddings at indices "
            f"{zero_indices}."
        )

    normalized_embeddings = (
        task_embeddings / task_norms[:, None]
    ).astype(
        np.float32,
        copy=False,
    )

    print("=== SMART Stage 1: task-level Graph Cut ===")
    print(f"Tasks:              {task_count}")
    print(f"Embedding shape:    {task_embeddings.shape}")
    print(f"Budget:             {budget}")
    print(f"Lambda:             {args.lambda_value}")
    print("Optimizer:          LazyGreedy")
    print(
        "Kernel:             dense cosine via "
        "submodlib/sklearn"
    )
    print(f"Output root:        {output_root}")
    print()

    kernel_start = time.perf_counter()

    # Exact repository kernel-construction path.
    similarity_kernel = submodlib.helper.create_kernel(
        X=task_embeddings,
        metric="cosine",
        method="sklearn",
    )

    kernel_seconds = (
        time.perf_counter() - kernel_start
    )

    similarity_kernel = np.asarray(
        similarity_kernel
    )

    expected_kernel_shape = (
        task_count,
        task_count,
    )

    if similarity_kernel.shape != expected_kernel_shape:
        raise RuntimeError(
            f"Kernel shape is {similarity_kernel.shape}; "
            f"expected {expected_kernel_shape}."
        )

    if not np.isfinite(similarity_kernel).all():
        raise RuntimeError(
            "Cosine kernel contains non-finite values."
        )

    manual_kernel = (
        normalized_embeddings
        @ normalized_embeddings.T
    )

    symmetry_error = float(
        np.max(
            np.abs(
                similarity_kernel
                - similarity_kernel.T
            )
        )
    )

    diagonal_error = float(
        np.max(
            np.abs(
                np.diag(similarity_kernel) - 1.0
            )
        )
    )

    manual_cosine_error = float(
        np.max(
            np.abs(
                similarity_kernel
                - manual_kernel
            )
        )
    )

    if symmetry_error > args.kernel_atol:
        raise RuntimeError(
            f"Kernel symmetry error "
            f"{symmetry_error:.9g} exceeds "
            f"{args.kernel_atol:.9g}."
        )

    if diagonal_error > args.kernel_atol:
        raise RuntimeError(
            f"Kernel diagonal error "
            f"{diagonal_error:.9g} exceeds "
            f"{args.kernel_atol:.9g}."
        )

    if manual_cosine_error > args.kernel_atol:
        raise RuntimeError(
            f"Submodlib/manual cosine difference "
            f"{manual_cosine_error:.9g} exceeds "
            f"{args.kernel_atol:.9g}."
        )

    print(
        f"Kernel construction: {kernel_seconds:.3f} seconds"
    )
    print(
        f"Kernel dtype:        {similarity_kernel.dtype}"
    )
    print(
        f"Kernel range:        "
        f"{float(similarity_kernel.min()):.8f} to "
        f"{float(similarity_kernel.max()):.8f}"
    )
    print(
        f"Symmetry error:      {symmetry_error:.3e}"
    )
    print(
        f"Diagonal error:      {diagonal_error:.3e}"
    )
    print(
        f"Manual cosine error: "
        f"{manual_cosine_error:.3e}"
    )
    print()

    graph_cut_runs: list[
        list[tuple[int, float]]
    ] = []

    run_seconds: list[float] = []

    for run_index in range(
        args.determinism_runs
    ):
        start = time.perf_counter()

        result = run_graph_cut(
            kernel=similarity_kernel,
            budget=budget,
            lambda_value=args.lambda_value,
            show_progress=(run_index == 0),
        )

        elapsed = time.perf_counter() - start

        graph_cut_runs.append(result)
        run_seconds.append(elapsed)

        print(
            f"Graph Cut run "
            f"{run_index + 1}/"
            f"{args.determinism_runs}: "
            f"{len(result)} selections in "
            f"{elapsed:.3f} seconds"
        )

    greedy_result = graph_cut_runs[0]

    if len(greedy_result) != budget:
        raise RuntimeError(
            f"LazyGreedy returned "
            f"{len(greedy_result)} items; "
            f"expected {budget}."
        )

    selected_indices = [
        index
        for index, _ in greedy_result
    ]

    marginal_gains = np.asarray(
        [
            gain
            for _, gain in greedy_result
        ],
        dtype=np.float64,
    )

    if len(set(selected_indices)) != budget:
        raise RuntimeError(
            "Graph Cut returned duplicate task indices."
        )

    if any(
        index < 0 or index >= task_count
        for index in selected_indices
    ):
        raise RuntimeError(
            "Graph Cut returned an out-of-range "
            "task index."
        )

    if (
        budget == task_count
        and set(selected_indices)
        != set(range(task_count))
    ):
        raise RuntimeError(
            "Full-budget Graph Cut did not select "
            "all tasks."
        )

    if not np.isfinite(marginal_gains).all():
        raise RuntimeError(
            "Graph Cut returned non-finite gains."
        )

    for run_number, candidate in enumerate(
        graph_cut_runs[1:],
        start=2,
    ):
        candidate_indices = [
            index
            for index, _ in candidate
        ]

        candidate_gains = np.asarray(
            [
                gain
                for _, gain in candidate
            ],
            dtype=np.float64,
        )

        if candidate_indices != selected_indices:
            raise RuntimeError(
                "Graph Cut order changed on "
                f"determinism run {run_number}."
            )

        if not np.array_equal(
            candidate_gains,
            marginal_gains,
        ):
            maximum_difference = float(
                np.max(
                    np.abs(
                        candidate_gains
                        - marginal_gains
                    )
                )
            )

            raise RuntimeError(
                "Graph Cut gains changed on "
                f"determinism run {run_number}; "
                f"maximum difference="
                f"{maximum_difference:.9g}."
            )

    cumulative_gains = np.cumsum(
        marginal_gains,
        dtype=np.float64,
    )

    normalized_path = (
        output_root
        / "task_embeddings_l2_normalized.npy"
    )

    kernel_path = (
        output_root
        / "task_similarity_cosine.npy"
    )

    order_csv_path = (
        output_root
        / "graph_cut_order.csv"
    )

    result_json_path = (
        output_root
        / "graph_cut_result.json"
    )

    atomic_save_numpy(
        normalized_embeddings,
        normalized_path,
    )

    atomic_save_numpy(
        similarity_kernel,
        kernel_path,
    )

    selection_rows: list[dict[str, Any]] = []

    for rank, (task_index, gain) in enumerate(
        greedy_result,
        start=1,
    ):
        task = tasks[task_index]

        selection_rows.append(
            {
                "graph_cut_rank": rank,
                "task_index": task_index,
                "task_id": task.task_id,
                "corpus": task.corpus,
                "task_name": task.task_name,
                "valid_train_count": (
                    task.valid_train_count
                ),
                "marginal_gain": gain,
                "cumulative_gain": float(
                    cumulative_gains[rank - 1]
                ),
            }
        )

    with order_csv_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                selection_rows[0].keys()
            ),
        )
        writer.writeheader()
        writer.writerows(selection_rows)

    nonpositive_gain_count = int(
        np.sum(marginal_gains <= 0)
    )

    negative_gain_count = int(
        np.sum(marginal_gains < 0)
    )

    payload = {
        "format_version": 1,
        "stage": "SMART_stage1_task_graph_cut",
        "status": "complete",
        "configuration": {
            "task_count": task_count,
            "budget": budget,
            "lambda_value": (
                args.lambda_value
            ),
            "optimizer": "LazyGreedy",
            "kernel_metric": "cosine",
            "kernel_method": "sklearn",
            "kernel_mode": "dense",
            "separate_rep": False,
            "determinism_runs": (
                args.determinism_runs
            ),
        },
        "inputs": {
            "manifest": str(manifest_path),
            "manifest_sha256": (
                sha256_file(manifest_path)
            ),
            "task_embeddings": str(
                task_embeddings_path
            ),
            "task_embeddings_sha256": (
                sha256_file(
                    task_embeddings_path
                )
            ),
            "task_embedding_shape": list(
                task_embeddings.shape
            ),
            "task_embedding_dtype": str(
                task_embeddings.dtype
            ),
            "task_mean_norm_minimum": float(
                task_norms.min()
            ),
            "task_mean_norm_maximum": float(
                task_norms.max()
            ),
        },
        "kernel": {
            "path": str(kernel_path),
            "shape": list(
                similarity_kernel.shape
            ),
            "dtype": str(
                similarity_kernel.dtype
            ),
            "minimum": float(
                similarity_kernel.min()
            ),
            "maximum": float(
                similarity_kernel.max()
            ),
            "symmetry_max_absolute_error": (
                symmetry_error
            ),
            "diagonal_max_absolute_error": (
                diagonal_error
            ),
            "manual_cosine_max_absolute_error": (
                manual_cosine_error
            ),
            "construction_seconds": (
                kernel_seconds
            ),
        },
        "graph_cut": {
            "selection_count": len(
                greedy_result
            ),
            "minimum_marginal_gain": float(
                marginal_gains.min()
            ),
            "maximum_marginal_gain": float(
                marginal_gains.max()
            ),
            "final_cumulative_gain": float(
                cumulative_gains[-1]
            ),
            "nonpositive_gain_count": (
                nonpositive_gain_count
            ),
            "negative_gain_count": (
                negative_gain_count
            ),
            "run_seconds": run_seconds,
            "deterministic_order": True,
            "deterministic_gains": True,
            "order_csv": str(
                order_csv_path
            ),
            "selections": selection_rows,
        },
        "outputs": {
            "normalized_task_embeddings": str(
                normalized_path
            ),
            "cosine_kernel": str(
                kernel_path
            ),
            "graph_cut_order_csv": str(
                order_csv_path
            ),
        },
        "environment": {
            "numpy": np.__version__,
            "submodlib": package_version(
                "submodlib"
            ),
            "scikit_learn": package_version(
                "scikit-learn"
            ),
        },
    }

    atomic_write_json(
        payload,
        result_json_path,
    )

    print()
    print("=== Graph Cut summary ===")
    print(
        f"Selected tasks:       "
        f"{len(greedy_result)}/{task_count}"
    )
    print(
        f"Gain range:           "
        f"{float(marginal_gains.min()):.9g} to "
        f"{float(marginal_gains.max()):.9g}"
    )
    print(
        f"Nonpositive gains:    "
        f"{nonpositive_gain_count}"
    )
    print(
        f"Negative gains:       "
        f"{negative_gain_count}"
    )
    print(
        f"Final cumulative gain:"
        f"{float(cumulative_gains[-1]):.9g}"
    )
    print()
    print("First 10 selected tasks:")

    for row in selection_rows[:10]:
        print(
            f"  {row['graph_cut_rank']:3d}. "
            f"{row['task_id']} "
            f"gain={row['marginal_gain']:.9g}"
        )

    print()
    print(f"Order CSV:   {order_csv_path}")
    print(f"Result JSON: {result_json_path}")
    print("SMART Stage 1 Graph Cut passed.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# Compilation command:
# python3 -m py_compile  data_generation_scripts/run_stage1_graph_cut.py 

# To run, we set below thread counts:
# export OMP_NUM_THREADS=1
# export MKL_NUM_THREADS=1
# export OPENBLAS_NUM_THREADS=1

# python3 data_generation_scripts/run_stage1_graph_cut.py \
#   --manifest /mnt/warm_storage/saral/smart/prepared_data/clean_task_manifest.csv \
#   --task-embeddings /mnt/warm_storage/saral/smart/artifacts/prompt_embeddings/gte-large/task_embeddings.npy \
#   --output-root /mnt/warm_storage/saral/smart/artifacts/stage1_graph_cut \
#   --lambda-value 0.4 \
#   --budget 309 \
#   --determinism-runs 2 \
#   2>&1 | tee \
#   /mnt/warm_storage/saral/smart/artifacts/stage1_graph_cut/stage1_graph_cut.log
