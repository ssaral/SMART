"""Generate resumable GTE-large prompt embeddings for all SMART tasks."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import re
import sys
import time
import traceback
from concurrent.futures import (
    ProcessPoolExecutor,
    as_completed,
)
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from local_dataset import (
    TaskSpec,
    iter_valid_rows,
    load_task_catalog,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--inventory-manifest",
        type=Path,
        default=None,
        help=(
            "Original task manifest containing train_avg_input_chars. "
            "Used only to balance work across GPUs."
        ),
    )
    parser.add_argument(
        "--encoder",
        required=True,
        help="Local GTE-large snapshot path.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--devices",
        default="auto",
        help=(
            "'auto' for all visible GPUs, or a comma-separated list "
            "such as '0,1,2,3'."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=23,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute tasks even when valid outputs already exist.",
    )

    return parser.parse_args()


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", value)


def task_directory(
    root: Path,
    task: TaskSpec,
) -> Path:
    return (
        root
        / "tasks"
        / f"{task.task_index:04d}_{safe_name(task.task_id)}"
    )


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


def parse_devices(value: str) -> list[int]:
    if value == "auto":
        count = torch.cuda.device_count()

        if count == 0:
            raise RuntimeError("No visible CUDA devices.")

        return list(range(count))

    devices = [
        int(item.strip())
        for item in value.split(",")
        if item.strip()
    ]

    if not devices:
        raise ValueError("No CUDA devices were specified.")

    visible_count = torch.cuda.device_count()

    for device in devices:
        if device < 0 or device >= visible_count:
            raise ValueError(
                f"cuda:{device} is invalid; "
                f"{visible_count} devices are visible."
            )

    if len(set(devices)) != len(devices):
        raise ValueError("Duplicate CUDA devices were specified.")

    return devices


def load_average_input_chars(
    path: Path | None,
) -> dict[str, float]:
    if path is None:
        return {}

    path = path.resolve()

    if not path.is_file():
        raise FileNotFoundError(
            f"Inventory manifest not found: {path}"
        )

    values: dict[str, float] = {}

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle)

        required = {
            "task_id",
            "train_avg_input_chars",
        }

        missing = required - set(reader.fieldnames or [])

        if missing:
            raise ValueError(
                "Inventory manifest is missing columns: "
                + ", ".join(sorted(missing))
            )

        for row in reader:
            values[row["task_id"]] = float(
                row["train_avg_input_chars"]
            )

    return values


def estimated_cost(
    task: TaskSpec,
    average_chars: dict[str, float],
) -> float:
    average = max(
        1.0,
        average_chars.get(task.task_id, 1.0),
    )

    return task.valid_train_count * average


def create_shards(
    tasks: Sequence[TaskSpec],
    devices: Sequence[int],
    average_chars: dict[str, float],
) -> tuple[list[list[TaskSpec]], list[float]]:
    shards: list[list[TaskSpec]] = [
        [] for _ in devices
    ]
    loads = [0.0 for _ in devices]

    ordered_tasks = sorted(
        tasks,
        key=lambda task: (
            estimated_cost(task, average_chars),
            task.valid_train_count,
        ),
        reverse=True,
    )

    for task in ordered_tasks:
        shard_index = min(
            range(len(shards)),
            key=lambda index: loads[index],
        )

        shards[shard_index].append(task)
        loads[shard_index] += estimated_cost(
            task,
            average_chars,
        )

    return shards, loads


def existing_output_is_valid(
    output_dir: Path,
    task: TaskSpec,
    embedding_dimension: int,
) -> bool:
    metadata_path = output_dir / "metadata.json"
    embeddings_path = output_dir / "prompt_embeddings.npy"
    indices_path = output_dir / "source_indices.npy"
    mean_path = output_dir / "task_mean.npy"

    required_paths = (
        metadata_path,
        embeddings_path,
        indices_path,
        mean_path,
    )

    if not all(path.is_file() for path in required_paths):
        return False

    try:
        with metadata_path.open(
            "r",
            encoding="utf-8",
        ) as handle:
            metadata = json.load(handle)

        if metadata["task"]["task_id"] != task.task_id:
            return False

        if (
            metadata["task"]["valid_train_count"]
            != task.valid_train_count
        ):
            return False

        embeddings = np.load(
            embeddings_path,
            mmap_mode="r",
        )
        source_indices = np.load(
            indices_path,
            mmap_mode="r",
        )
        task_mean = np.load(
            mean_path,
            mmap_mode="r",
        )

        if embeddings.shape != (
            task.valid_train_count,
            embedding_dimension,
        ):
            return False

        if embeddings.dtype != np.float32:
            return False

        if source_indices.shape != (
            task.valid_train_count,
        ):
            return False

        if task_mean.shape != (
            embedding_dimension,
        ):
            return False

        if task_mean.dtype != np.float32:
            return False

    except Exception:
        return False

    return True


def collect_task_rows(
    task: TaskSpec,
) -> tuple[list[str], np.ndarray]:
    prompts: list[str] = []
    source_indices: list[int] = []

    for example in iter_valid_rows(task, "train"):
        prompts.append(example["inputs"])
        source_indices.append(example["source_index"])

    if len(prompts) != task.valid_train_count:
        raise RuntimeError(
            f"{task.task_id}: expected "
            f"{task.valid_train_count:,} valid prompts, "
            f"but collected {len(prompts):,}."
        )

    return (
        prompts,
        np.asarray(source_indices, dtype=np.int64),
    )


def worker_main(
    device_index: int,
    tasks: list[TaskSpec],
    encoder: str,
    output_root: str,
    batch_size: int,
    seed: int,
    overwrite: bool,
) -> dict[str, Any]:
    output_root_path = Path(output_root).resolve()
    worker_log_path = (
        output_root_path
        / f"worker_cuda_{device_index}.log"
    )

    worker_log_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    def log(message: str) -> None:
        line = (
            f"[cuda:{device_index}] "
            f"{time.strftime('%Y-%m-%d %H:%M:%S')} "
            f"{message}"
        )

        print(line, flush=True)

        with worker_log_path.open(
            "a",
            encoding="utf-8",
        ) as handle:
            handle.write(line + "\n")

    try:
        torch.cuda.set_device(device_index)
        torch.manual_seed(seed + device_index)
        torch.cuda.manual_seed_all(seed + device_index)

        log(
            f"Loading encoder from {encoder}; "
            f"assigned tasks={len(tasks)}"
        )

        model = SentenceTransformer(
            encoder,
            device=f"cuda:{device_index}",
            local_files_only=True,
        )
        model.eval()

        embedding_dimension = int(
            model.get_sentence_embedding_dimension()
        )
        max_sequence_length = int(
            model.max_seq_length
        )

        log(
            f"Encoder ready: dimension={embedding_dimension}, "
            f"max_sequence_length={max_sequence_length}"
        )

        completed = 0
        skipped = 0
        total_rows = 0
        start_time = time.perf_counter()

        for position, task in enumerate(tasks, start=1):
            output_dir = task_directory(
                output_root_path,
                task,
            )
            output_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            if (
                not overwrite
                and existing_output_is_valid(
                    output_dir,
                    task,
                    embedding_dimension,
                )
            ):
                skipped += 1
                total_rows += task.valid_train_count

                log(
                    f"[{position}/{len(tasks)}] SKIP "
                    f"{task.task_id} "
                    f"rows={task.valid_train_count:,}"
                )
                continue

            for temporary in output_dir.glob("*.tmp"):
                temporary.unlink(missing_ok=True)

            log(
                f"[{position}/{len(tasks)}] START "
                f"{task.task_id} "
                f"rows={task.valid_train_count:,}"
            )

            prompts, source_indices = collect_task_rows(task)

            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(
                device_index
            )

            encode_start = time.perf_counter()

            with torch.inference_mode():
                embeddings = model.encode(
                    prompts,
                    batch_size=batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=False,
                )

            encode_seconds = (
                time.perf_counter() - encode_start
            )

            embeddings = np.asarray(
                embeddings,
                dtype=np.float32,
            )

            expected_shape = (
                task.valid_train_count,
                embedding_dimension,
            )

            if embeddings.shape != expected_shape:
                raise RuntimeError(
                    f"{task.task_id}: expected shape "
                    f"{expected_shape}, got "
                    f"{embeddings.shape}."
                )

            if not np.isfinite(embeddings).all():
                raise RuntimeError(
                    f"{task.task_id}: embeddings contain "
                    "non-finite values."
                )

            norms = np.linalg.norm(
                embeddings,
                axis=1,
            )

            zero_norm_count = int(
                np.sum(norms == 0)
            )

            if zero_norm_count:
                raise RuntimeError(
                    f"{task.task_id}: found "
                    f"{zero_norm_count:,} zero-norm vectors."
                )

            # Preserve NumPy's float32 mean behavior used on
            # float32 Sentence Transformer outputs.
            task_mean = embeddings.mean(
                axis=0
            ).astype(
                np.float32,
                copy=False,
            )

            peak_memory = int(
                torch.cuda.max_memory_allocated(
                    device_index
                )
            )

            embeddings_path = (
                output_dir / "prompt_embeddings.npy"
            )
            indices_path = (
                output_dir / "source_indices.npy"
            )
            mean_path = output_dir / "task_mean.npy"
            metadata_path = output_dir / "metadata.json"

            atomic_save_numpy(
                embeddings,
                embeddings_path,
            )
            atomic_save_numpy(
                source_indices,
                indices_path,
            )
            atomic_save_numpy(
                task_mean,
                mean_path,
            )

            metadata = {
                "format_version": 1,
                "task": {
                    "task_index": task.task_index,
                    "task_id": task.task_id,
                    "task_name": task.task_name,
                    "corpus": task.corpus,
                    "source_file": task.source_file,
                    "valid_train_count": (
                        task.valid_train_count
                    ),
                    "template_type": (
                        task.template_type
                    ),
                },
                "encoder": {
                    "name_or_path": encoder,
                    "embedding_dimension": (
                        embedding_dimension
                    ),
                    "maximum_sequence_length": (
                        max_sequence_length
                    ),
                    "input_field": "inputs",
                    "normalize_embeddings_argument": (
                        False
                    ),
                },
                "execution": {
                    "device": f"cuda:{device_index}",
                    "batch_size": batch_size,
                    "seed": seed,
                    "encoding_seconds": encode_seconds,
                    "prompts_per_second": (
                        task.valid_train_count
                        / encode_seconds
                    ),
                    "peak_allocated_gpu_memory_bytes": (
                        peak_memory
                    ),
                },
                "embedding_array": {
                    "path": str(embeddings_path),
                    "shape": list(embeddings.shape),
                    "dtype": str(embeddings.dtype),
                    "minimum_norm": float(
                        norms.min()
                    ),
                    "mean_norm": float(
                        norms.mean()
                    ),
                    "maximum_norm": float(
                        norms.max()
                    ),
                },
                "source_indices": {
                    "path": str(indices_path),
                    "shape": list(
                        source_indices.shape
                    ),
                    "dtype": str(
                        source_indices.dtype
                    ),
                },
                "task_mean": {
                    "path": str(mean_path),
                    "shape": list(
                        task_mean.shape
                    ),
                    "dtype": str(
                        task_mean.dtype
                    ),
                    "norm": float(
                        np.linalg.norm(task_mean)
                    ),
                },
            }

            # Written last: metadata.json is the completion marker.
            atomic_write_json(
                metadata,
                metadata_path,
            )

            completed += 1
            total_rows += task.valid_train_count

            log(
                f"[{position}/{len(tasks)}] DONE "
                f"{task.task_id} "
                f"seconds={encode_seconds:.2f} "
                f"rate={task.valid_train_count / encode_seconds:.2f}/s "
                f"peak_vram={peak_memory / 1024**3:.2f}GiB"
            )

            del prompts
            del source_indices
            del embeddings
            del task_mean
            del norms

            gc.collect()
            torch.cuda.empty_cache()

        elapsed = time.perf_counter() - start_time

        result = {
            "device": device_index,
            "assigned_tasks": len(tasks),
            "completed_tasks": completed,
            "skipped_tasks": skipped,
            "processed_rows": total_rows,
            "elapsed_seconds": elapsed,
        }

        log(
            f"WORKER COMPLETE: completed={completed}, "
            f"skipped={skipped}, rows={total_rows:,}, "
            f"elapsed={elapsed:.2f}s"
        )

        return result

    except Exception:
        error_path = (
            output_root_path
            / f"worker_cuda_{device_index}.error.log"
        )

        error_text = traceback.format_exc()

        with error_path.open(
            "w",
            encoding="utf-8",
        ) as handle:
            handle.write(error_text)

        log(
            f"WORKER FAILED. Traceback written to "
            f"{error_path}"
        )

        raise


def build_final_outputs(
    tasks: Sequence[TaskSpec],
    output_root: Path,
    encoder: str,
    devices: Sequence[int],
    worker_results: Sequence[dict[str, Any]],
) -> None:
    task_means: list[np.ndarray] = []
    task_records: list[dict[str, Any]] = []
    embedding_dimension: int | None = None

    for task in sorted(
        tasks,
        key=lambda item: item.task_index,
    ):
        output_dir = task_directory(
            output_root,
            task,
        )

        metadata_path = output_dir / "metadata.json"
        mean_path = output_dir / "task_mean.npy"
        embeddings_path = (
            output_dir / "prompt_embeddings.npy"
        )
        source_indices_path = (
            output_dir / "source_indices.npy"
        )

        if not metadata_path.is_file():
            raise RuntimeError(
                f"Missing metadata for {task.task_id}"
            )

        with metadata_path.open(
            "r",
            encoding="utf-8",
        ) as handle:
            metadata = json.load(handle)

        mean = np.load(mean_path)
        embeddings = np.load(
            embeddings_path,
            mmap_mode="r",
        )
        source_indices = np.load(
            source_indices_path,
            mmap_mode="r",
        )

        if embedding_dimension is None:
            embedding_dimension = int(
                mean.shape[0]
            )

        if mean.shape != (embedding_dimension,):
            raise RuntimeError(
                f"Invalid task mean shape for "
                f"{task.task_id}: {mean.shape}"
            )

        if embeddings.shape != (
            task.valid_train_count,
            embedding_dimension,
        ):
            raise RuntimeError(
                f"Invalid prompt embedding shape for "
                f"{task.task_id}: {embeddings.shape}"
            )

        if source_indices.shape != (
            task.valid_train_count,
        ):
            raise RuntimeError(
                f"Invalid source-index shape for "
                f"{task.task_id}: "
                f"{source_indices.shape}"
            )

        task_means.append(
            np.asarray(mean, dtype=np.float32)
        )

        task_records.append(
            {
                "task_index": task.task_index,
                "task_id": task.task_id,
                "corpus": task.corpus,
                "valid_train_count": (
                    task.valid_train_count
                ),
                "output_directory": str(
                    output_dir
                ),
                "metadata": metadata,
            }
        )

    task_embedding_matrix = np.stack(
        task_means,
        axis=0,
    ).astype(
        np.float32,
        copy=False,
    )

    task_embeddings_path = (
        output_root / "task_embeddings.npy"
    )
    task_catalog_path = (
        output_root / "embedding_catalog.json"
    )
    run_summary_path = (
        output_root / "run_summary.json"
    )

    atomic_save_numpy(
        task_embedding_matrix,
        task_embeddings_path,
    )

    atomic_write_json(
        {
            "format_version": 1,
            "task_count": len(task_records),
            "encoder": encoder,
            "task_embedding_matrix": {
                "path": str(task_embeddings_path),
                "shape": list(
                    task_embedding_matrix.shape
                ),
                "dtype": str(
                    task_embedding_matrix.dtype
                ),
            },
            "tasks": task_records,
        },
        task_catalog_path,
    )

    atomic_write_json(
        {
            "status": "complete",
            "task_count": len(tasks),
            "total_valid_train_rows": sum(
                task.valid_train_count
                for task in tasks
            ),
            "encoder": encoder,
            "devices": list(devices),
            "worker_results": list(
                worker_results
            ),
            "task_embeddings": {
                "path": str(task_embeddings_path),
                "shape": list(
                    task_embedding_matrix.shape
                ),
            },
            "embedding_catalog": str(
                task_catalog_path
            ),
        },
        run_summary_path,
    )


def main() -> int:
    args = parse_args()

    if args.batch_size <= 0:
        raise ValueError(
            "--batch-size must be positive."
        )

    output_root = args.output_root.resolve()
    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )
    (output_root / "tasks").mkdir(
        parents=True,
        exist_ok=True,
    )

    encoder_path = str(
        Path(args.encoder).resolve()
    )

    if not Path(encoder_path).is_dir():
        raise FileNotFoundError(
            f"Local encoder snapshot not found: "
            f"{encoder_path}"
        )

    tasks = load_task_catalog(
        args.manifest
    )
    devices = parse_devices(
        args.devices
    )
    average_chars = load_average_input_chars(
        args.inventory_manifest
    )

    shards, estimated_loads = create_shards(
        tasks,
        devices,
        average_chars,
    )

    plan = {
        "task_count": len(tasks),
        "total_valid_train_rows": sum(
            task.valid_train_count
            for task in tasks
        ),
        "encoder": encoder_path,
        "batch_size": args.batch_size,
        "devices": list(devices),
        "shards": [
            {
                "device": device,
                "estimated_load": (
                    estimated_loads[index]
                ),
                "task_count": len(
                    shards[index]
                ),
                "valid_train_rows": sum(
                    task.valid_train_count
                    for task in shards[index]
                ),
                "tasks": [
                    {
                        "task_index": task.task_index,
                        "task_id": task.task_id,
                        "valid_train_count": (
                            task.valid_train_count
                        ),
                    }
                    for task in shards[index]
                ],
            }
            for index, device in enumerate(devices)
        ],
    }

    atomic_write_json(
        plan,
        output_root / "run_plan.json",
    )

    print("=== Full GTE embedding run ===")
    print(f"Tasks:              {len(tasks)}")
    print(
        f"Valid train rows:   "
        f"{plan['total_valid_train_rows']:,}"
    )
    print(f"Visible GPUs:       {devices}")
    print(f"Batch size/GPU:     {args.batch_size}")
    print(f"Encoder:            {encoder_path}")
    print(f"Output root:        {output_root}")
    print()

    for shard in plan["shards"]:
        print(
            f"cuda:{shard['device']} -> "
            f"{shard['task_count']} tasks, "
            f"{shard['valid_train_rows']:,} rows"
        )

    context = get_context("spawn")
    worker_results: list[dict[str, Any]] = []

    with ProcessPoolExecutor(
        max_workers=len(devices),
        mp_context=context,
    ) as executor:
        futures = {
            executor.submit(
                worker_main,
                device,
                shards[index],
                encoder_path,
                str(output_root),
                args.batch_size,
                args.seed,
                args.overwrite,
            ): device
            for index, device in enumerate(devices)
        }

        for future in as_completed(futures):
            device = futures[future]

            try:
                worker_results.append(
                    future.result()
                )
            except Exception as exc:
                print(
                    f"ERROR: cuda:{device} worker failed: "
                    f"{exc}",
                    file=sys.stderr,
                )
                return 2

    worker_results.sort(
        key=lambda result: result["device"]
    )

    build_final_outputs(
        tasks=tasks,
        output_root=output_root,
        encoder=encoder_path,
        devices=devices,
        worker_results=worker_results,
    )

    print()
    print("All 309 task embedding outputs verified.")
    print(
        "Task embedding matrix: "
        f"{output_root / 'task_embeddings.npy'}"
    )
    print(
        "Run summary: "
        f"{output_root / 'run_summary.json'}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
