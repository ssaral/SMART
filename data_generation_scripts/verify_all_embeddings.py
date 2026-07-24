"""Comprehensively verify all local SMART embedding artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from local_dataset import (
    iter_valid_rows,
    load_task_catalog,
)


EXPECTED_TASK_COUNT = 309
EXPECTED_TOTAL_ROWS = 6_266_471
EXPECTED_DIMENSION = 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify shapes, row alignment, numerical validity, "
            "and task means for all SMART prompt embeddings."
        )
    )
    parser.add_argument(
        "--manifest",
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
        default=None,
    )
    parser.add_argument(
        "--chunk-rows",
        type=int,
        default=8192,
        help=(
            "Number of embedding rows processed in memory at once."
        ),
    )
    parser.add_argument(
        "--mean-atol",
        type=float,
        default=2e-6,
    )
    parser.add_argument(
        "--mean-rtol",
        type=float,
        default=2e-5,
    )
    return parser.parse_args()


def safe_name(value: str) -> str:
    return re.sub(
        r"[^A-Za-z0-9_.-]+",
        "__",
        value,
    )


def task_directory(
    root: Path,
    task_index: int,
    task_id: str,
) -> Path:
    return (
        root
        / "tasks"
        / f"{task_index:04d}_{safe_name(task_id)}"
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


def add_failure(
    failures: list[dict[str, Any]],
    category: str,
    message: str,
    task_id: str | None = None,
) -> None:
    record: dict[str, Any] = {
        "category": category,
        "message": message,
    }

    if task_id is not None:
        record["task_id"] = task_id

    failures.append(record)


def load_json(path: Path) -> Any:
    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        return json.load(handle)


def main() -> int:
    args = parse_args()

    if args.chunk_rows <= 0:
        raise ValueError("--chunk-rows must be positive.")

    embedding_root = args.embedding_root.resolve()
    manifest_path = args.manifest.resolve()

    output_root = (
        args.output_root.resolve()
        if args.output_root is not None
        else embedding_root / "verification"
    )
    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_path = output_root / "verification_report.json"
    task_csv_path = output_root / "task_verification.csv"
    failure_path = output_root / "verification_failures.jsonl"

    run_summary_path = (
        embedding_root / "run_summary.json"
    )
    embedding_catalog_path = (
        embedding_root / "embedding_catalog.json"
    )
    task_matrix_path = (
        embedding_root / "task_embeddings.npy"
    )

    failures: list[dict[str, Any]] = []
    warnings: list[str] = []
    task_results: list[dict[str, Any]] = []

    start_time = time.perf_counter()

    print("=== SMART embedding verification ===")
    print(f"Manifest:       {manifest_path}")
    print(f"Embedding root: {embedding_root}")
    print(f"Chunk rows:     {args.chunk_rows:,}")
    print()

    required_top_level = [
        run_summary_path,
        embedding_catalog_path,
        task_matrix_path,
    ]

    for path in required_top_level:
        if not path.is_file():
            add_failure(
                failures,
                "missing_top_level_file",
                f"Missing required file: {path}",
            )

    if failures:
        for failure in failures:
            print(
                "ERROR:",
                failure["message"],
                file=sys.stderr,
            )

        with failure_path.open(
            "w",
            encoding="utf-8",
        ) as handle:
            for failure in failures:
                handle.write(
                    json.dumps(failure) + "\n"
                )

        return 2

    tasks = load_task_catalog(manifest_path)
    run_summary = load_json(run_summary_path)
    embedding_catalog = load_json(
        embedding_catalog_path
    )

    if len(tasks) != EXPECTED_TASK_COUNT:
        add_failure(
            failures,
            "task_count",
            f"Loaded {len(tasks)} tasks; expected "
            f"{EXPECTED_TASK_COUNT}.",
        )

    total_manifest_rows = sum(
        task.valid_train_count
        for task in tasks
    )

    if total_manifest_rows != EXPECTED_TOTAL_ROWS:
        add_failure(
            failures,
            "manifest_row_count",
            f"Manifest contains {total_manifest_rows:,} "
            f"valid rows; expected "
            f"{EXPECTED_TOTAL_ROWS:,}.",
        )

    if run_summary.get("status") != "complete":
        add_failure(
            failures,
            "run_summary_status",
            "run_summary.json does not report "
            "status='complete'.",
        )

    if (
        run_summary.get("task_count")
        != EXPECTED_TASK_COUNT
    ):
        add_failure(
            failures,
            "run_summary_task_count",
            "run_summary.json task count does not "
            "equal 309.",
        )

    if (
        run_summary.get("total_valid_train_rows")
        != EXPECTED_TOTAL_ROWS
    ):
        add_failure(
            failures,
            "run_summary_row_count",
            "run_summary.json valid-row count does not "
            f"equal {EXPECTED_TOTAL_ROWS:,}.",
        )

    worker_results = run_summary.get(
        "worker_results",
        [],
    )

    assigned_task_total = sum(
        int(worker.get("assigned_tasks", 0))
        for worker in worker_results
    )
    accounted_task_total = sum(
        int(worker.get("completed_tasks", 0))
        + int(worker.get("skipped_tasks", 0))
        for worker in worker_results
    )
    worker_row_total = sum(
        int(worker.get("processed_rows", 0))
        for worker in worker_results
    )

    if assigned_task_total != EXPECTED_TASK_COUNT:
        add_failure(
            failures,
            "worker_assigned_tasks",
            f"Workers were assigned {assigned_task_total} "
            "tasks rather than 309.",
        )

    if accounted_task_total != EXPECTED_TASK_COUNT:
        add_failure(
            failures,
            "worker_accounted_tasks",
            f"Workers accounted for "
            f"{accounted_task_total} tasks rather "
            "than 309.",
        )

    if worker_row_total != EXPECTED_TOTAL_ROWS:
        add_failure(
            failures,
            "worker_row_total",
            f"Workers report {worker_row_total:,} rows; "
            f"expected {EXPECTED_TOTAL_ROWS:,}.",
        )

    if (
        embedding_catalog.get("task_count")
        != EXPECTED_TASK_COUNT
    ):
        add_failure(
            failures,
            "embedding_catalog_task_count",
            "embedding_catalog.json does not contain "
            "309 tasks.",
        )

    task_matrix = np.load(
        task_matrix_path,
        mmap_mode="r",
    )

    if task_matrix.shape != (
        EXPECTED_TASK_COUNT,
        EXPECTED_DIMENSION,
    ):
        add_failure(
            failures,
            "task_matrix_shape",
            f"task_embeddings.npy has shape "
            f"{task_matrix.shape}; expected "
            f"({EXPECTED_TASK_COUNT}, "
            f"{EXPECTED_DIMENSION}).",
        )

    if task_matrix.dtype != np.float32:
        add_failure(
            failures,
            "task_matrix_dtype",
            f"task_embeddings.npy has dtype "
            f"{task_matrix.dtype}; expected float32.",
        )

    if not np.isfinite(task_matrix).all():
        add_failure(
            failures,
            "task_matrix_nonfinite",
            "task_embeddings.npy contains non-finite "
            "values.",
        )

    metadata_files = sorted(
        (embedding_root / "tasks").glob(
            "*/metadata.json"
        )
    )

    if len(metadata_files) != EXPECTED_TASK_COUNT:
        add_failure(
            failures,
            "metadata_file_count",
            f"Found {len(metadata_files)} task metadata "
            "files; expected 309.",
        )

    stale_error_logs = sorted(
        embedding_root.glob(
            "worker_cuda_*.error.log"
        )
    )

    if stale_error_logs:
        warnings.append(
            "Worker error logs are present. They may be "
            "stale files from the earlier failed launch: "
            + ", ".join(
                path.name for path in stale_error_logs
            )
        )

    global_min_norm = math.inf
    global_max_norm = -math.inf
    global_zero_norms = 0
    total_verified_rows = 0
    exact_matrix_mean_matches = 0
    source_index_matches = 0
    maximum_mean_error = 0.0

    print(
        f"Verifying {len(tasks)} tasks and "
        f"{total_manifest_rows:,} prompt embeddings..."
    )
    print()

    for position, task in enumerate(
        tasks,
        start=1,
    ):
        current_failures_before = len(failures)

        output_dir = task_directory(
            embedding_root,
            task.task_index,
            task.task_id,
        )

        embeddings_path = (
            output_dir / "prompt_embeddings.npy"
        )
        source_indices_path = (
            output_dir / "source_indices.npy"
        )
        task_mean_path = (
            output_dir / "task_mean.npy"
        )
        metadata_path = (
            output_dir / "metadata.json"
        )

        required_paths = [
            embeddings_path,
            source_indices_path,
            task_mean_path,
            metadata_path,
        ]

        missing_paths = [
            path
            for path in required_paths
            if not path.is_file()
        ]

        if missing_paths:
            add_failure(
                failures,
                "missing_task_file",
                "Missing task output files: "
                + ", ".join(
                    str(path)
                    for path in missing_paths
                ),
                task.task_id,
            )
            continue

        try:
            metadata = load_json(metadata_path)
            embeddings = np.load(
                embeddings_path,
                mmap_mode="r",
            )
            source_indices = np.load(
                source_indices_path,
                mmap_mode="r",
            )
            saved_mean = np.load(
                task_mean_path,
                mmap_mode="r",
            )
        except Exception as exc:
            add_failure(
                failures,
                "task_load_error",
                repr(exc),
                task.task_id,
            )
            continue

        expected_embedding_shape = (
            task.valid_train_count,
            EXPECTED_DIMENSION,
        )

        if embeddings.shape != expected_embedding_shape:
            add_failure(
                failures,
                "embedding_shape",
                f"Embedding shape is {embeddings.shape}; "
                f"expected {expected_embedding_shape}.",
                task.task_id,
            )

        if embeddings.dtype != np.float32:
            add_failure(
                failures,
                "embedding_dtype",
                f"Embedding dtype is "
                f"{embeddings.dtype}; expected float32.",
                task.task_id,
            )

        if source_indices.shape != (
            task.valid_train_count,
        ):
            add_failure(
                failures,
                "source_index_shape",
                f"source_indices shape is "
                f"{source_indices.shape}; expected "
                f"({task.valid_train_count},).",
                task.task_id,
            )

        if source_indices.dtype != np.int64:
            add_failure(
                failures,
                "source_index_dtype",
                f"source_indices dtype is "
                f"{source_indices.dtype}; expected int64.",
                task.task_id,
            )

        if saved_mean.shape != (
            EXPECTED_DIMENSION,
        ):
            add_failure(
                failures,
                "task_mean_shape",
                f"task mean shape is "
                f"{saved_mean.shape}; expected "
                f"({EXPECTED_DIMENSION},).",
                task.task_id,
            )

        if saved_mean.dtype != np.float32:
            add_failure(
                failures,
                "task_mean_dtype",
                f"task mean dtype is "
                f"{saved_mean.dtype}; expected float32.",
                task.task_id,
            )

        metadata_task = metadata.get("task", {})

        if metadata_task.get("task_id") != task.task_id:
            add_failure(
                failures,
                "metadata_task_id",
                "metadata task ID does not match "
                "the clean manifest.",
                task.task_id,
            )

        if (
            metadata_task.get("task_index")
            != task.task_index
        ):
            add_failure(
                failures,
                "metadata_task_index",
                "metadata task index does not match "
                "the clean manifest.",
                task.task_id,
            )

        if (
            metadata_task.get("valid_train_count")
            != task.valid_train_count
        ):
            add_failure(
                failures,
                "metadata_row_count",
                "metadata valid-row count does not "
                "match the clean manifest.",
                task.task_id,
            )

        if (
            embeddings.shape
            != expected_embedding_shape
            or saved_mean.shape
            != (EXPECTED_DIMENSION,)
        ):
            continue

        expected_source_indices = np.fromiter(
            (
                example["source_index"]
                for example in iter_valid_rows(
                    task,
                    "train",
                )
            ),
            dtype=np.int64,
        )

        indices_match = (
            expected_source_indices.shape
            == source_indices.shape
            and np.array_equal(
                expected_source_indices,
                source_indices,
            )
        )

        if indices_match:
            source_index_matches += 1
        else:
            message = (
                "Saved source indices do not match a "
                "fresh scan of valid source rows."
            )

            if (
                expected_source_indices.shape
                == source_indices.shape
            ):
                mismatches = np.flatnonzero(
                    expected_source_indices
                    != source_indices
                )

                if mismatches.size:
                    first = int(mismatches[0])
                    message += (
                        f" First mismatch at embedding "
                        f"row {first}: expected "
                        f"{int(expected_source_indices[first])}, "
                        f"saved "
                        f"{int(source_indices[first])}."
                    )
            else:
                message += (
                    f" Expected shape "
                    f"{expected_source_indices.shape}, "
                    f"saved shape {source_indices.shape}."
                )

            add_failure(
                failures,
                "source_index_alignment",
                message,
                task.task_id,
            )

        if (
            source_indices.size > 1
            and not np.all(
                source_indices[1:]
                > source_indices[:-1]
            )
        ):
            add_failure(
                failures,
                "source_index_order",
                "Saved source indices are not strictly "
                "increasing.",
                task.task_id,
            )

        if np.array_equal(
            np.asarray(task_matrix[task.task_index]),
            np.asarray(saved_mean),
        ):
            exact_matrix_mean_matches += 1
        else:
            add_failure(
                failures,
                "task_matrix_mean_mismatch",
                "The task mean does not exactly match "
                "its row in task_embeddings.npy.",
                task.task_id,
            )

        # Reproduce the exact mean operation used by embed_all_tasks.py.
        # This is the artifact-integrity check.
        replication_mean = embeddings.mean(
            axis=0,
            dtype=np.float32,
        )
        
        replication_mean_error = float(
            np.max(
                np.abs(
                    replication_mean
                    - np.asarray(
                        saved_mean,
                        dtype=np.float32,
                    )
                )
            )
        )
        
        if not np.array_equal(
            replication_mean,
            np.asarray(
                saved_mean,
                dtype=np.float32,
            ),
        ):
            add_failure(
                failures,
                "task_mean_replication",
                (
                    "Recomputing the task mean with the same "
                    "float32 NumPy reduction did not reproduce "
                    "the saved mean. Maximum absolute difference: "
                    f"{replication_mean_error:.9g}."
                ),
                task.task_id,
            )
        component_sum = np.zeros(
            EXPECTED_DIMENSION,
            dtype=np.float64,
        )

        task_min_norm = math.inf
        task_max_norm = -math.inf
        task_zero_norms = 0
        task_nonfinite_values = 0

        for start in range(
            0,
            task.valid_train_count,
            args.chunk_rows,
        ):
            end = min(
                start + args.chunk_rows,
                task.valid_train_count,
            )

            chunk = np.asarray(
                embeddings[start:end],
                dtype=np.float32,
            )

            finite_mask = np.isfinite(chunk)
            nonfinite_count = int(
                chunk.size - finite_mask.sum()
            )
            task_nonfinite_values += nonfinite_count

            if nonfinite_count:
                continue

            norms = np.linalg.norm(
                chunk,
                axis=1,
            )

            zero_count = int(
                np.sum(norms == 0)
            )
            task_zero_norms += zero_count

            if norms.size:
                task_min_norm = min(
                    task_min_norm,
                    float(norms.min()),
                )
                task_max_norm = max(
                    task_max_norm,
                    float(norms.max()),
                )

            component_sum += chunk.sum(
                axis=0,
                dtype=np.float64,
            )

        if task_nonfinite_values:
            add_failure(
                failures,
                "embedding_nonfinite",
                f"Found {task_nonfinite_values:,} "
                "non-finite embedding values.",
                task.task_id,
            )

        if task_zero_norms:
            add_failure(
                failures,
                "embedding_zero_norm",
                f"Found {task_zero_norms:,} "
                "zero-norm embeddings.",
                task.task_id,
            )

        if not np.isfinite(saved_mean).all():
            add_failure(
                failures,
                "task_mean_nonfinite",
                "Saved task mean contains non-finite "
                "values.",
                task.task_id,
            )

        if task.valid_train_count > 0:
            independent_mean = (
                component_sum
                / task.valid_train_count
            )

            mean_error = float(
                np.max(
                    np.abs(
                        independent_mean
                        - np.asarray(
                            saved_mean,
                            dtype=np.float64,
                        )
                    )
                )
            )

            maximum_mean_error = max(
                maximum_mean_error,
                mean_error,
            )

            # This float64 calculation is a higher-precision reference,
            # not an exact reproduction of the float32 reduction used
            # during embedding generation.
            #
            # Small differences are expected because reduction precision,
            # grouping and addition order differ.
            high_precision_drift_limit = 2e-5
            
            if mean_error > high_precision_drift_limit:
                add_failure(
                    failures,
                    "task_mean_high_precision_drift",
                    (
                        "The saved float32 task mean differs unusually "
                        "from the chunked float64 reference. Maximum "
                        f"absolute difference: {mean_error:.9g}; "
                        f"limit: {high_precision_drift_limit:.9g}."
                    ),
                    task.task_id,
                )

            # mean_matches = np.allclose(
            #     independent_mean,
            #     np.asarray(
            #         saved_mean,
            #         dtype=np.float64,
            #     ),
            #     rtol=args.mean_rtol,
            #     atol=args.mean_atol,
            # )

            # if not mean_matches:
            #     add_failure(
            #         failures,
            #         "task_mean_recomputation",
            #         f"Independently recomputed mean does "
            #         f"not match saved mean. Maximum "
            #         f"absolute difference: "
            #         f"{mean_error:.9g}.",
            #         task.task_id,
            #     )

        if task_min_norm != math.inf:
            global_min_norm = min(
                global_min_norm,
                task_min_norm,
            )

        if task_max_norm != -math.inf:
            global_max_norm = max(
                global_max_norm,
                task_max_norm,
            )

        global_zero_norms += task_zero_norms
        total_verified_rows += task.valid_train_count

        task_passed = (
            len(failures)
            == current_failures_before
        )

        task_results.append(
            {
                "task_index": task.task_index,
                "task_id": task.task_id,
                "valid_train_count": (
                    task.valid_train_count
                ),
                "embedding_shape": (
                    f"{embeddings.shape[0]}x"
                    f"{embeddings.shape[1]}"
                ),
                "source_indices_match": (
                    indices_match
                ),
                "minimum_embedding_norm": (
                    task_min_norm
                ),
                "maximum_embedding_norm": (
                    task_max_norm
                ),
                "zero_norm_count": (
                    task_zero_norms
                ),
                "nonfinite_value_count": (
                    task_nonfinite_values
                ),
                "maximum_mean_absolute_error": (
                    mean_error
                ),
                "replication_mean_max_absolute_error": (
                replication_mean_error
                ),
                "passed": task_passed,
            }
        )

        status = "PASS" if task_passed else "FAIL"

        print(
            f"[{position:3d}/{len(tasks)}] "
            f"{status} {task.task_id} "
            f"rows={task.valid_train_count:,} "
            f"mean_error={mean_error:.3e}"
        )

    if total_verified_rows != EXPECTED_TOTAL_ROWS:
        add_failure(
            failures,
            "verified_row_total",
            f"Verified task outputs account for "
            f"{total_verified_rows:,} rows rather than "
            f"{EXPECTED_TOTAL_ROWS:,}.",
        )

    task_fieldnames = [
        "task_index",
        "task_id",
        "valid_train_count",
        "embedding_shape",
        "source_indices_match",
        "minimum_embedding_norm",
        "maximum_embedding_norm",
        "zero_norm_count",
        "nonfinite_value_count",
        "maximum_mean_absolute_error",
        "replication_mean_max_absolute_error",
        "passed",
    ]

    with task_csv_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=task_fieldnames,
        )
        writer.writeheader()
        writer.writerows(task_results)

    with failure_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        for failure in failures:
            handle.write(
                json.dumps(
                    failure,
                    ensure_ascii=False,
                )
                + "\n"
            )

    elapsed_seconds = (
        time.perf_counter() - start_time
    )

    report = {
        "status": (
            "verified"
            if not failures
            else "failed"
        ),
        "task_count": len(tasks),
        "expected_task_count": (
            EXPECTED_TASK_COUNT
        ),
        "total_valid_train_rows": (
            total_manifest_rows
        ),
        "total_verified_embedding_rows": (
            total_verified_rows
        ),
        "embedding_dimension": (
            EXPECTED_DIMENSION
        ),
        "task_embedding_matrix": {
            "path": str(task_matrix_path),
            "shape": list(task_matrix.shape),
            "dtype": str(task_matrix.dtype),
            "finite": bool(
                np.isfinite(task_matrix).all()
            ),
            "sha256": sha256_file(
                task_matrix_path
            ),
        },
        "alignment": {
            "tasks_with_matching_source_indices": (
                source_index_matches
            ),
            "tasks_with_exact_matrix_mean_match": (
                exact_matrix_mean_matches
            ),
        },
        "numerical_checks": {
            "global_minimum_embedding_norm": (
                global_min_norm
                if global_min_norm != math.inf
                else None
            ),
            "global_maximum_embedding_norm": (
                global_max_norm
                if global_max_norm != -math.inf
                else None
            ),
            "global_zero_norm_count": (
                global_zero_norms
            ),
            "maximum_independent_mean_absolute_error": (
                maximum_mean_error
            ),
            "mean_absolute_tolerance": (
                args.mean_atol
            ),
            "mean_relative_tolerance": (
                args.mean_rtol
            ),
            "maximum_float64_reference_drift": (
            maximum_mean_error
            ),
        "float64_reference_drift_limit": 2e-5,
        },
        "run_summary_checks": {
            "status": run_summary.get("status"),
            "assigned_task_total": (
                assigned_task_total
            ),
            "accounted_task_total": (
                accounted_task_total
            ),
            "worker_processed_row_total": (
                worker_row_total
            ),
        },
        "checksums": {
            "clean_task_manifest_sha256": (
                sha256_file(manifest_path)
            ),
            "run_summary_sha256": (
                sha256_file(run_summary_path)
            ),
            "embedding_catalog_sha256": (
                sha256_file(
                    embedding_catalog_path
                )
            ),
        },
        "failure_count": len(failures),
        "warning_count": len(warnings),
        "warnings": warnings,
        "elapsed_seconds": elapsed_seconds,
        "outputs": {
            "task_verification_csv": str(
                task_csv_path
            ),
            "failure_jsonl": str(
                failure_path
            ),
        },
    }

    atomic_write_json(
        report,
        report_path,
    )

    print()
    print("=== Verification summary ===")
    print(f"Status:                    {report['status']}")
    print(f"Tasks checked:             {len(tasks)}")
    print(
        f"Embedding rows checked:    "
        f"{total_verified_rows:,}"
    )
    print(
        "Matching source indices:   "
        f"{source_index_matches}/{len(tasks)}"
    )
    print(
        "Exact matrix/mean matches: "
        f"{exact_matrix_mean_matches}/{len(tasks)}"
    )
    print(
        "Maximum mean error:        "
        f"{maximum_mean_error:.9g}"
    )
    print(
        "Global embedding norm:     "
        f"{global_min_norm:.6f} to "
        f"{global_max_norm:.6f}"
    )
    print(f"Failures:                  {len(failures)}")
    print(f"Warnings:                  {len(warnings)}")
    print(
        f"Elapsed:                   "
        f"{elapsed_seconds / 60:.2f} minutes"
    )
    print(f"Report:                    {report_path}")

    if warnings:
        print()
        print("Warnings:")

        for warning in warnings:
            print(f"  - {warning}")

    if failures:
        print()
        print(
            f"Failure details: {failure_path}",
            file=sys.stderr,
        )
        return 2

    print()
    print("All embedding verification checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# To run this script: use the command below:
# python3 data_generation_scripts/verify_all_embeddings.py \
#   --manifest /mnt/warm_storage/saral/smart/prepared_data/clean_task_manifest.csv \
#   --embedding-root /mnt/warm_storage/saral/smart/artifacts/prompt_embeddings/gte-large \
#   --output-root /mnt/warm_storage/saral/smart/artifacts/prompt_embeddings/gte-large/verification \
#   --chunk-rows 8192 \
#   2>&1 | tee \
#   /mnt/warm_storage/saral/smart/artifacts/prompt_embeddings/gte-large/verification/verification.log
