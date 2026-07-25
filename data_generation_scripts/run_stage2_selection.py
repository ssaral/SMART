"""Run production SMART Stage 2 instance selection."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import submodlib.functions as submod_fn


EXPECTED_TASK_COUNT = 309
EXPECTED_BUDGET_25K = 25_000
EXPECTED_BUDGET_50K = 50_000
EMBEDDING_DIMENSION = 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run exact or candidate-restricted Facility Location "
            "for every SMART task."
        )
    )

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
        "--output-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--exact-threshold",
        type=int,
        default=5000,
    )
    parser.add_argument(
        "--candidate-size",
        type=int,
        default=2048,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=23,
    )
    parser.add_argument(
        "--coverage-block-rows",
        type=int,
        default=4096,
    )
    parser.add_argument(
        "--task-id",
        action="append",
        default=[],
        help=(
            "Optional task filter. May be supplied multiple times. "
            "Omit to process all tasks."
        ),
    )
    parser.add_argument(
        "--show-progress",
        action="store_true",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute tasks even if valid outputs exist.",
    )

    return parser.parse_args()


def safe_name(value: str) -> str:
    return re.sub(
        r"[^A-Za-z0-9_.-]+",
        "__",
        value,
    )


def atomic_save_numpy(
    array: np.ndarray,
    path: Path,
) -> None:
    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    with temporary.open("wb") as handle:
        np.save(handle, array)

    temporary.replace(path)


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


def current_rss_gib() -> float | None:
    path = Path("/proc/self/status")

    if not path.is_file():
        return None

    for line in path.read_text(
        encoding="utf-8",
    ).splitlines():
        if line.startswith("VmRSS:"):
            fields = line.split()

            if len(fields) >= 2:
                return (
                    int(fields[1])
                    * 1024
                    / (1024 ** 3)
                )

    return None


def load_allocations(
    path: Path,
) -> list[dict[str, Any]]:
    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        raw_rows = list(csv.DictReader(handle))

    if len(raw_rows) != EXPECTED_TASK_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_TASK_COUNT} allocation rows; "
            f"found {len(raw_rows)}."
        )

    rows: list[dict[str, Any]] = []

    for raw in raw_rows:
        row = {
            "graph_cut_rank": int(
                raw["graph_cut_rank"]
            ),
            "task_index": int(
                raw["task_index"]
            ),
            "task_id": raw["task_id"],
            "corpus": raw["corpus"],
            "task_name": raw["task_name"],
            "valid_train_count": int(
                raw["valid_train_count"]
            ),
            "allocation_25000": int(
                raw["final_allocation_25000"]
            ),
            "allocation_50000": int(
                raw["final_allocation_50000"]
            ),
        }

        if not (
            0
            < row["allocation_25000"]
            <= row["allocation_50000"]
            < row["valid_train_count"]
        ):
            raise RuntimeError(
                f"Invalid allocations for {row['task_id']}: "
                f"n={row['valid_train_count']}, "
                f"k25={row['allocation_25000']}, "
                f"k50={row['allocation_50000']}."
            )

        rows.append(row)

    if len(
        {
            row["task_id"]
            for row in rows
        }
    ) != EXPECTED_TASK_COUNT:
        raise RuntimeError(
            "Allocation table contains duplicate task IDs."
        )

    if sum(
        row["allocation_25000"]
        for row in rows
    ) != EXPECTED_BUDGET_25K:
        raise RuntimeError(
            "25K allocations do not sum to 25,000."
        )

    if sum(
        row["allocation_50000"]
        for row in rows
    ) != EXPECTED_BUDGET_50K:
        raise RuntimeError(
            "50K allocations do not sum to 50,000."
        )

    return rows


def resolve_embedding_directory(
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
            f"Unable to resolve embedding directory for "
            f"{task_id}: {candidates}"
        )

    return candidates[0]


def task_output_directory(
    output_root: Path,
    task_index: int,
    task_id: str,
) -> Path:
    return (
        output_root
        / "tasks"
        / f"{task_index:04d}_{safe_name(task_id)}"
    )


def deterministic_candidate_indices(
    task_size: int,
    candidate_size: int,
    task_index: int,
    seed: int,
) -> np.ndarray:
    seed_sequence = np.random.SeedSequence(
        [seed, task_index]
    )

    rng = np.random.default_rng(
        seed_sequence
    )

    permutation = rng.permutation(
        task_size
    )

    # Sorting does not change membership. It makes saved
    # candidate arrays easier to inspect and reproduce.
    return np.sort(
        permutation[:candidate_size]
    ).astype(
        np.int64,
        copy=False,
    )


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
            "Zero-norm prompt embedding encountered."
        )

    return (
        matrix / norms
    ).astype(
        np.float32,
        copy=False,
    )


def facility_coverage_sum(
    represented_embeddings: np.ndarray,
    selected_indices: np.ndarray,
    block_rows: int,
) -> float:
    represented = normalize_rows(
        represented_embeddings
    )

    selected = normalize_rows(
        np.asarray(
            represented_embeddings[
                selected_indices
            ],
            dtype=np.float32,
        )
    )

    selected_transpose = np.ascontiguousarray(
        selected.T
    )

    total = 0.0

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

        total += float(
            np.max(
                similarities,
                axis=1,
            ).sum(
                dtype=np.float64
            )
        )

    return total


def output_is_complete(
    output_dir: Path,
    row: dict[str, Any],
    exact_threshold: int,
    candidate_size: int,
    seed: int,
) -> bool:
    metadata_path = output_dir / "metadata.json"

    if not metadata_path.is_file():
        return False

    required = [
        output_dir
        / "selected_full_local_indices.npy",
        output_dir
        / "selected_source_indices.npy",
        output_dir
        / "marginal_gains.npy",
        output_dir
        / "selection.csv",
    ]

    if not all(path.is_file() for path in required):
        return False

    try:
        metadata = json.loads(
            metadata_path.read_text(
                encoding="utf-8"
            )
        )

        if metadata.get("status") != "complete":
            return False

        if metadata["task"]["task_id"] != row["task_id"]:
            return False

        if (
            metadata["task"]["valid_train_count"]
            != row["valid_train_count"]
        ):
            return False

        if (
            metadata["allocations"]["25000"]
            != row["allocation_25000"]
        ):
            return False

        if (
            metadata["allocations"]["50000"]
            != row["allocation_50000"]
        ):
            return False

        config = metadata["configuration"]

        if (
            config["exact_threshold"]
            != exact_threshold
            or config["candidate_size"]
            != candidate_size
            or config["seed"] != seed
        ):
            return False

        selected = np.load(
            required[0],
            mmap_mode="r",
        )
        sources = np.load(
            required[1],
            mmap_mode="r",
        )
        gains = np.load(
            required[2],
            mmap_mode="r",
        )

        expected_shape = (
            row["allocation_50000"],
        )

        return (
            selected.shape == expected_shape
            and sources.shape == expected_shape
            and gains.shape == expected_shape
        )

    except Exception:
        return False


def run_facility_location(
    represented_embeddings: np.ndarray,
    task_index: int,
    k50: int,
    exact_threshold: int,
    candidate_size: int,
    seed: int,
    show_progress: bool,
) -> dict[str, Any]:
    task_size = represented_embeddings.shape[0]

    constructor_start = time.perf_counter()

    if task_size <= exact_threshold:
        method = "authors_exact_dense"

        objective = (
            submod_fn.facilityLocation
            .FacilityLocationFunction(
                n=task_size,
                separate_rep=False,
                mode="dense",
                data=represented_embeddings,
                create_dense_cpp_kernel_in_python=False,
            )
        )

        selectable_full_indices = np.arange(
            task_size,
            dtype=np.int64,
        )

        effective_candidate_size = task_size
        candidate_indices = None

    else:
        method = "candidate_restricted"

        effective_candidate_size = min(
            candidate_size,
            task_size,
        )

        if effective_candidate_size <= k50:
            raise RuntimeError(
                f"Candidate size "
                f"{effective_candidate_size} must exceed "
                f"selection budget {k50}."
            )

        candidate_indices = (
            deterministic_candidate_indices(
                task_size=task_size,
                candidate_size=(
                    effective_candidate_size
                ),
                task_index=task_index,
                seed=seed,
            )
        )

        candidate_embeddings = np.array(
            represented_embeddings[
                candidate_indices
            ],
            dtype=np.float32,
            order="C",
            copy=True,
        )

        objective = (
            submod_fn.facilityLocation
            .FacilityLocationFunction(
                n=effective_candidate_size,
                mode="dense",
                separate_rep=True,
                n_rep=task_size,
                data=candidate_embeddings,
                data_rep=represented_embeddings,
                metric="cosine",
                create_dense_cpp_kernel_in_python=False,
            )
        )

        selectable_full_indices = (
            candidate_indices
        )

    constructor_seconds = (
        time.perf_counter()
        - constructor_start
    )

    rss_after_constructor = current_rss_gib()

    maximize_start = time.perf_counter()

    greedy_result = objective.maximize(
        budget=k50,
        optimizer="LazyGreedy",
        stopIfZeroGain=False,
        stopIfNegativeGain=False,
        verbose=False,
        show_progress=show_progress,
    )

    maximize_seconds = (
        time.perf_counter()
        - maximize_start
    )

    rss_after_maximize = current_rss_gib()

    greedy_result = [
        (int(index), float(gain))
        for index, gain in greedy_result
    ]

    if len(greedy_result) != k50:
        raise RuntimeError(
            f"Facility Location returned "
            f"{len(greedy_result)} items; "
            f"expected {k50}."
        )

    selected_objective_indices = np.asarray(
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
        selected_objective_indices
    ).size != k50:
        raise RuntimeError(
            "Facility Location returned duplicate indices."
        )

    if not np.isfinite(gains).all():
        raise RuntimeError(
            "Facility Location returned non-finite gains."
        )

    selected_full_indices = (
        selectable_full_indices[
            selected_objective_indices
        ]
    )

    del objective
    gc.collect()

    return {
        "method": method,
        "candidate_size": (
            effective_candidate_size
        ),
        "candidate_indices": candidate_indices,
        "selected_full_indices": (
            selected_full_indices
        ),
        "gains": gains,
        "constructor_seconds": (
            constructor_seconds
        ),
        "maximize_seconds": maximize_seconds,
        "rss_after_constructor_gib": (
            rss_after_constructor
        ),
        "rss_after_maximize_gib": (
            rss_after_maximize
        ),
    }


def process_task(
    row: dict[str, Any],
    embedding_root: Path,
    output_root: Path,
    exact_threshold: int,
    candidate_size: int,
    seed: int,
    block_rows: int,
    show_progress: bool,
) -> dict[str, Any]:
    task_id = row["task_id"]
    task_index = row["task_index"]
    task_size = row["valid_train_count"]
    k25 = row["allocation_25000"]
    k50 = row["allocation_50000"]

    embedding_dir = resolve_embedding_directory(
        embedding_root,
        task_index,
        task_id,
    )

    embeddings_path = (
        embedding_dir
        / "prompt_embeddings.npy"
    )
    source_indices_path = (
        embedding_dir
        / "source_indices.npy"
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
        task_size,
        EMBEDDING_DIMENSION,
    ):
        raise RuntimeError(
            f"{task_id}: embedding shape "
            f"{embeddings_memmap.shape} does not match "
            f"({task_size}, {EMBEDDING_DIMENSION})."
        )

    if source_indices_memmap.shape != (
        task_size,
    ):
        raise RuntimeError(
            f"{task_id}: source-index shape mismatch."
        )

    if not np.isfinite(
        embeddings_memmap
    ).all():
        raise RuntimeError(
            f"{task_id}: embeddings contain "
            "non-finite values."
        )

    # The NPY arrays are already contiguous float32.
    # np.asarray preserves the memory mapping rather than
    # making an unnecessary task-sized copy.
    represented_embeddings = np.asarray(
        embeddings_memmap,
        dtype=np.float32,
        order="C",
    )

    rss_before = current_rss_gib()
    task_start = time.perf_counter()

    result = run_facility_location(
        represented_embeddings=(
            represented_embeddings
        ),
        task_index=task_index,
        k50=k50,
        exact_threshold=exact_threshold,
        candidate_size=candidate_size,
        seed=seed,
        show_progress=show_progress,
    )

    selected_full_indices = result[
        "selected_full_indices"
    ]
    gains = result["gains"]

    if np.any(
        selected_full_indices < 0
    ) or np.any(
        selected_full_indices >= task_size
    ):
        raise RuntimeError(
            f"{task_id}: selected index outside task."
        )

    if np.unique(
        selected_full_indices
    ).size != k50:
        raise RuntimeError(
            f"{task_id}: selected full indices are "
            "not unique."
        )

    selected_source_indices = np.asarray(
        source_indices_memmap[
            selected_full_indices
        ],
        dtype=np.int64,
    )

    if np.unique(
        selected_source_indices
    ).size != k50:
        raise RuntimeError(
            f"{task_id}: selected source indices are "
            "not unique."
        )

    cumulative_gains = np.cumsum(
        gains,
        dtype=np.float64,
    )

    coverage_start = time.perf_counter()

    coverage_25k = facility_coverage_sum(
        represented_embeddings,
        selected_full_indices[:k25],
        block_rows,
    )

    coverage_50k = facility_coverage_sum(
        represented_embeddings,
        selected_full_indices,
        block_rows,
    )

    coverage_seconds = (
        time.perf_counter()
        - coverage_start
    )

    objective_25k = float(
        cumulative_gains[k25 - 1]
    )
    objective_50k = float(
        cumulative_gains[k50 - 1]
    )

    for name, objective, manual in [
        ("25K", objective_25k, coverage_25k),
        ("50K", objective_50k, coverage_50k),
    ]:
        if not np.isclose(
            objective,
            manual,
            rtol=2e-5,
            atol=2e-3,
        ):
            raise RuntimeError(
                f"{task_id}: {name} cumulative gain "
                f"{objective:.9g} does not match manual "
                f"coverage {manual:.9g}."
            )

    output_dir = task_output_directory(
        output_root,
        task_index,
        task_id,
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    selected_path = (
        output_dir
        / "selected_full_local_indices.npy"
    )
    selected_source_path = (
        output_dir
        / "selected_source_indices.npy"
    )
    gains_path = (
        output_dir
        / "marginal_gains.npy"
    )
    candidate_path = (
        output_dir
        / "candidate_full_local_indices.npy"
    )
    selection_csv_path = (
        output_dir / "selection.csv"
    )
    metadata_path = (
        output_dir / "metadata.json"
    )

    atomic_save_numpy(
        selected_full_indices.astype(
            np.int64,
            copy=False,
        ),
        selected_path,
    )

    atomic_save_numpy(
        selected_source_indices.astype(
            np.int64,
            copy=False,
        ),
        selected_source_path,
    )

    atomic_save_numpy(
        gains.astype(
            np.float64,
            copy=False,
        ),
        gains_path,
    )

    if result["candidate_indices"] is not None:
        atomic_save_numpy(
            result["candidate_indices"],
            candidate_path,
        )
    elif candidate_path.exists():
        candidate_path.unlink()

    with selection_csv_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        fieldnames = [
            "selection_rank",
            "full_local_index",
            "source_index",
            "marginal_gain",
            "cumulative_gain",
            "included_in_25000",
            "included_in_50000",
        ]

        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )
        writer.writeheader()

        for index in range(k50):
            writer.writerow(
                {
                    "selection_rank": index + 1,
                    "full_local_index": int(
                        selected_full_indices[index]
                    ),
                    "source_index": int(
                        selected_source_indices[index]
                    ),
                    "marginal_gain": float(
                        gains[index]
                    ),
                    "cumulative_gain": float(
                        cumulative_gains[index]
                    ),
                    "included_in_25000": (
                        index < k25
                    ),
                    "included_in_50000": True,
                }
            )

    total_seconds = (
        time.perf_counter() - task_start
    )

    raw_kernel_entries = (
        task_size
        * result["candidate_size"]
    )

    metadata = {
        "status": "complete",
        "stage": "SMART_stage2_selection",
        "task": {
            "task_index": task_index,
            "task_id": task_id,
            "corpus": row["corpus"],
            "task_name": row["task_name"],
            "graph_cut_rank": (
                row["graph_cut_rank"]
            ),
            "valid_train_count": task_size,
        },
        "allocations": {
            "25000": k25,
            "50000": k50,
            "prefix_nested": True,
        },
        "configuration": {
            "exact_threshold": exact_threshold,
            "candidate_size": candidate_size,
            "seed": seed,
            "optimizer": "LazyGreedy",
            "similarity": "cosine",
            "represented_set": (
                "all valid task examples"
            ),
            "stop_if_zero_gain": False,
            "stop_if_negative_gain": False,
        },
        "method": {
            "name": result["method"],
            "effective_candidate_size": (
                result["candidate_size"]
            ),
            "approximation": (
                result["method"]
                == "candidate_restricted"
            ),
            "raw_similarity_entries": (
                raw_kernel_entries
            ),
            "raw_float32_similarity_gib": (
                raw_kernel_entries
                * 4
                / (1024 ** 3)
            ),
        },
        "objective_checks": {
            "cumulative_gain_25000": (
                objective_25k
            ),
            "manual_coverage_25000": (
                coverage_25k
            ),
            "cumulative_gain_50000": (
                objective_50k
            ),
            "manual_coverage_50000": (
                coverage_50k
            ),
        },
        "timing_seconds": {
            "construct_objective": (
                result[
                    "constructor_seconds"
                ]
            ),
            "maximize": (
                result["maximize_seconds"]
            ),
            "manual_coverage": (
                coverage_seconds
            ),
            "total": total_seconds,
        },
        "memory": {
            "rss_before_gib": rss_before,
            "rss_after_constructor_gib": (
                result[
                    "rss_after_constructor_gib"
                ]
            ),
            "rss_after_maximize_gib": (
                result[
                    "rss_after_maximize_gib"
                ]
            ),
        },
        "inputs": {
            "prompt_embeddings": str(
                embeddings_path
            ),
            "source_indices": str(
                source_indices_path
            ),
        },
        "outputs": {
            "selected_full_local_indices": str(
                selected_path
            ),
            "selected_source_indices": str(
                selected_source_path
            ),
            "marginal_gains": str(
                gains_path
            ),
            "selection_csv": str(
                selection_csv_path
            ),
            "candidate_full_local_indices": (
                str(candidate_path)
                if candidate_path.is_file()
                else None
            ),
        },
    }

    atomic_write_json(
        metadata,
        metadata_path,
    )

    metadata["outputs"]["metadata"] = str(
        metadata_path
    )

    return metadata


def main() -> int:
    args = parse_args()

    if args.exact_threshold < 2:
        raise ValueError(
            "--exact-threshold must be at least 2."
        )

    if args.candidate_size < 2:
        raise ValueError(
            "--candidate-size must be at least 2."
        )

    allocation_path = args.allocations.resolve()
    embedding_root = args.embedding_root.resolve()
    output_root = args.output_root.resolve()

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = load_allocations(
        allocation_path
    )

    requested_task_ids = set(
        args.task_id
    )

    if requested_task_ids:
        known = {
            row["task_id"]
            for row in rows
        }

        unknown = (
            requested_task_ids - known
        )

        if unknown:
            raise KeyError(
                "Unknown task IDs: "
                + ", ".join(sorted(unknown))
            )

        rows_to_process = [
            row
            for row in rows
            if row["task_id"]
            in requested_task_ids
        ]
    else:
        rows_to_process = list(rows)

    # Small tasks first gives quick validation and ensures
    # completed work accumulates before the largest tasks.
    rows_to_process.sort(
        key=lambda row: (
            row["valid_train_count"],
            row["task_index"],
        )
    )

    print("=== SMART Stage 2 production ===")
    print(
        f"Tasks requested:       "
        f"{len(rows_to_process)}"
    )
    print(
        f"Exact threshold:       "
        f"{args.exact_threshold:,}"
    )
    print(
        f"Candidate size:        "
        f"{args.candidate_size:,}"
    )
    print(f"Seed:                  {args.seed}")
    print(
        f"OMP_NUM_THREADS:       "
        f"{os.environ.get('OMP_NUM_THREADS')}"
    )
    print(f"Output root:           {output_root}")
    print()

    run_start = time.perf_counter()
    completed = 0
    skipped = 0
    results: list[dict[str, Any]] = []

    for position, row in enumerate(
        rows_to_process,
        start=1,
    ):
        output_dir = task_output_directory(
            output_root,
            row["task_index"],
            row["task_id"],
        )

        if (
            not args.force
            and output_is_complete(
                output_dir=output_dir,
                row=row,
                exact_threshold=(
                    args.exact_threshold
                ),
                candidate_size=(
                    args.candidate_size
                ),
                seed=args.seed,
            )
        ):
            metadata = json.loads(
                (
                    output_dir
                    / "metadata.json"
                ).read_text(
                    encoding="utf-8"
                )
            )

            results.append(metadata)
            skipped += 1

            print(
                f"[{position:3d}/"
                f"{len(rows_to_process)}] "
                f"SKIP {row['task_id']}"
            )

            continue

        expected_method = (
            "exact"
            if row["valid_train_count"]
            <= args.exact_threshold
            else "candidate"
        )

        print(
            f"[{position:3d}/"
            f"{len(rows_to_process)}] "
            f"START {row['task_id']} "
            f"n={row['valid_train_count']:,} "
            f"k25={row['allocation_25000']} "
            f"k50={row['allocation_50000']} "
            f"method={expected_method}",
            flush=True,
        )

        metadata = process_task(
            row=row,
            embedding_root=embedding_root,
            output_root=output_root,
            exact_threshold=(
                args.exact_threshold
            ),
            candidate_size=(
                args.candidate_size
            ),
            seed=args.seed,
            block_rows=(
                args.coverage_block_rows
            ),
            show_progress=args.show_progress,
        )

        results.append(metadata)
        completed += 1

        elapsed = (
            time.perf_counter() - run_start
        )

        processed = completed + skipped
        seconds_per_task = (
            elapsed / processed
        )
        remaining = (
            len(rows_to_process) - processed
        )
        eta_seconds = (
            seconds_per_task * remaining
        )

        print(
            f"[{position:3d}/"
            f"{len(rows_to_process)}] "
            f"DONE  {row['task_id']} "
            f"method={metadata['method']['name']} "
            f"seconds="
            f"{metadata['timing_seconds']['total']:.2f} "
            f"ETA={eta_seconds / 3600:.2f}h",
            flush=True,
        )

        del metadata
        gc.collect()

    # A filtered smoke run does not write the final global
    # production catalog.
    if requested_task_ids:
        print()
        print("Filtered Stage 2 run passed.")
        print(f"Completed: {completed}")
        print(f"Skipped:   {skipped}")
        return 0

    results_by_task = {
        result["task"]["task_id"]: result
        for result in results
    }

    if len(results_by_task) != EXPECTED_TASK_COUNT:
        raise RuntimeError(
            f"Only {len(results_by_task)} complete task "
            "results were collected."
        )

    ordered_results = [
        results_by_task[row["task_id"]]
        for row in sorted(
            rows,
            key=lambda item: item[
                "task_index"
            ],
        )
    ]

    total_25k = sum(
        result["allocations"]["25000"]
        for result in ordered_results
    )
    total_50k = sum(
        result["allocations"]["50000"]
        for result in ordered_results
    )

    if total_25k != EXPECTED_BUDGET_25K:
        raise RuntimeError(
            f"Stage 2 25K total is {total_25k}."
        )

    if total_50k != EXPECTED_BUDGET_50K:
        raise RuntimeError(
            f"Stage 2 50K total is {total_50k}."
        )

    method_counts = Counter(
        result["method"]["name"]
        for result in ordered_results
    )

    total_runtime = sum(
        result["timing_seconds"]["total"]
        for result in ordered_results
    )

    catalog_path = (
        output_root
        / "stage2_selection_catalog.json"
    )
    summary_path = (
        output_root
        / "stage2_selection_summary.json"
    )

    catalog = {
        "format_version": 1,
        "stage": "SMART_stage2_selection",
        "status": "complete",
        "configuration": {
            "task_count": EXPECTED_TASK_COUNT,
            "exact_threshold": (
                args.exact_threshold
            ),
            "candidate_size": (
                args.candidate_size
            ),
            "seed": args.seed,
            "allocation_25000": (
                EXPECTED_BUDGET_25K
            ),
            "allocation_50000": (
                EXPECTED_BUDGET_50K
            ),
            "prefix_nested": True,
        },
        "inputs": {
            "task_allocations": str(
                allocation_path
            ),
            "task_allocations_sha256": (
                sha256_file(
                    allocation_path
                )
            ),
            "embedding_root": str(
                embedding_root
            ),
        },
        "tasks": ordered_results,
    }

    atomic_write_json(
        catalog,
        catalog_path,
    )

    summary = {
        "format_version": 1,
        "stage": "SMART_stage2_selection",
        "status": "complete",
        "task_count": EXPECTED_TASK_COUNT,
        "allocation_totals": {
            "25000": total_25k,
            "50000": total_50k,
        },
        "method_counts": dict(
            sorted(method_counts.items())
        ),
        "total_task_runtime_seconds": (
            total_runtime
        ),
        "catalog": str(catalog_path),
        "catalog_sha256": (
            sha256_file(catalog_path)
        ),
    }

    atomic_write_json(
        summary,
        summary_path,
    )

    print()
    print("=== Stage 2 summary ===")
    print("Status:          complete")
    print(
        f"Tasks:           "
        f"{EXPECTED_TASK_COUNT}"
    )
    print(f"25K total:       {total_25k:,}")
    print(f"50K total:       {total_50k:,}")
    print(
        f"Methods:         "
        f"{dict(method_counts)}"
    )
    print(
        f"Task runtime:    "
        f"{total_runtime / 3600:.2f} hours"
    )
    print(f"Catalog:         {catalog_path}")
    print(f"Summary:         {summary_path}")
    print("SMART Stage 2 production passed.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
