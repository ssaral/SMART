## Step 5 — Generate row-aligned GTE-large prompt embeddings

The authors embed only the `inputs` field with `thenlper/gte-large`, use batch size 128, and later average those embeddings to obtain task representations.  

This implementation writes one float32 matrix per corpus:

```text
embedding row i == author-format train dataset row i
```

That alignment lets the task-index pickles index the embeddings directly.

### 5.1 Set the encoder path

Reuse the immutable GTE snapshot:

```bash
cd /data/saral/wdir/smart_v2 || exit 1
source ./smart_v2_env.sh

export GTE_MODEL=/mnt/warm_storage/saral/smart/cache/sentence_transformers/models--thenlper--gte-large/snapshots/4bef63f39fcc5e2d6b0aae83089f307af4970164

test -f "$GTE_MODEL/modules.json" || {
  echo "Invalid GTE model path: $GTE_MODEL"
  exit 1
}

python3 - <<'PY'
import torch
from sentence_transformers import SentenceTransformer

model_path = (
    "/mnt/warm_storage/saral/smart/cache/"
    "sentence_transformers/"
    "models--thenlper--gte-large/"
    "snapshots/"
    "4bef63f39fcc5e2d6b0aae83089f307af4970164"
)

model = SentenceTransformer(model_path, device="cpu")

print("Embedding dimension:", model.get_sentence_embedding_dimension())
print("Maximum sequence length:", model.max_seq_length)
print("Visible CUDA devices:", torch.cuda.device_count())
PY
```

Expected:

```text
Embedding dimension: 1024
Maximum sequence length: 512
Visible CUDA devices: 4
```

---

## 5.2 Create the multi-GPU embedding script

```bash
cat > src/embeddings/embed_author_format.py <<'PY'
"""Generate row-aligned GTE-large prompt embeddings.

For every corpus, output row i corresponds exactly to row i of the
author-format training split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import platform
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import sentence_transformers
import torch
from datasets import load_from_disk
from sentence_transformers import SentenceTransformer


EXPECTED_CORPORA = (
    "cot",
    "flan2021",
    "sglue",
    "t0",
    "tulu",
)

EXPECTED_TOTAL_ROWS = 6_266_471
EXPECTED_DIMENSION = 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--author-format-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--task-indices-summary",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--devices",
        type=int,
        nargs="+",
        required=True,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--chunk-rows",
        type=int,
        default=8192,
    )
    parser.add_argument(
        "--shard-rows",
        type=int,
        default=250_000,
    )
    parser.add_argument(
        "--embedding-dimension",
        type=int,
        default=EXPECTED_DIMENSION,
    )

    return parser.parse_args()


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


def sha256_text(value: str) -> str:
    return hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()


def load_expected_rows(
    summary_path: Path,
) -> dict[str, int]:
    with summary_path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        summary = json.load(handle)

    if summary.get("status") != "complete":
        raise RuntimeError(
            "Task-index summary is not complete."
        )

    result: dict[str, int] = {}

    for item in summary["corpora"]:
        corpus = item["corpus"]

        if corpus not in EXPECTED_CORPORA:
            raise ValueError(
                f"Unexpected corpus: {corpus}"
            )

        result[corpus] = int(
            item["train_rows"]
        )

    if set(result) != set(EXPECTED_CORPORA):
        raise RuntimeError(
            "Task-index summary does not contain "
            "all five corpora."
        )

    total = sum(result.values())

    if total != EXPECTED_TOTAL_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_TOTAL_ROWS:,} rows; "
            f"summary contains {total:,}."
        )

    return result


def validate_existing_matrix(
    path: Path,
    expected_shape: tuple[int, int],
) -> None:
    matrix = np.load(
        path,
        mmap_mode="r",
    )

    if matrix.shape != expected_shape:
        raise RuntimeError(
            f"{path}: shape {matrix.shape} does not "
            f"match {expected_shape}."
        )

    if matrix.dtype != np.float32:
        raise RuntimeError(
            f"{path}: dtype {matrix.dtype} is not float32."
        )


def prepare_output_matrices(
    output_root: Path,
    expected_rows: dict[str, int],
    dimension: int,
) -> dict[str, Path]:
    result: dict[str, Path] = {}

    for corpus in EXPECTED_CORPORA:
        corpus_root = output_root / corpus
        corpus_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        matrix_path = (
            corpus_root
            / "train_prompt_embeddings.npy"
        )

        expected_shape = (
            expected_rows[corpus],
            dimension,
        )

        if matrix_path.exists():
            validate_existing_matrix(
                matrix_path,
                expected_shape,
            )
        else:
            matrix = np.lib.format.open_memmap(
                matrix_path,
                mode="w+",
                dtype=np.float32,
                shape=expected_shape,
            )
            matrix.flush()
            del matrix

        result[corpus] = matrix_path

    return result


def create_jobs(
    expected_rows: dict[str, int],
    matrix_paths: dict[str, Path],
    author_root: Path,
    output_root: Path,
    shard_rows: int,
) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []

    for corpus in EXPECTED_CORPORA:
        row_count = expected_rows[corpus]

        for start in range(
            0,
            row_count,
            shard_rows,
        ):
            end = min(
                start + shard_rows,
                row_count,
            )

            job_id = (
                f"{corpus}_{start:09d}_{end:09d}"
            )

            jobs.append(
                {
                    "job_id": job_id,
                    "corpus": corpus,
                    "start": start,
                    "end": end,
                    "row_count": end - start,
                    "dataset_path": str(
                        author_root / corpus
                    ),
                    "matrix_path": str(
                        matrix_paths[corpus]
                    ),
                    "status_path": str(
                        output_root
                        / "status"
                        / f"{job_id}.json"
                    ),
                }
            )

    return jobs


def assign_jobs(
    jobs: list[dict[str, Any]],
    devices: list[int],
) -> dict[int, list[dict[str, Any]]]:
    assignments = {
        device: []
        for device in devices
    }
    loads = {
        device: 0
        for device in devices
    }

    for job in sorted(
        jobs,
        key=lambda item: item["row_count"],
        reverse=True,
    ):
        device = min(
            devices,
            key=lambda value: loads[value],
        )

        assignments[device].append(job)
        loads[device] += job["row_count"]

    return assignments


def status_matches_complete(
    status_path: Path,
    job: dict[str, Any],
    model_identity: str,
) -> bool:
    if not status_path.is_file():
        return False

    try:
        with status_path.open(
            "r",
            encoding="utf-8",
        ) as handle:
            status = json.load(handle)
    except Exception:
        return False

    return (
        status.get("status") == "complete"
        and status.get("job_id")
        == job["job_id"]
        and int(status.get("start", -1))
        == job["start"]
        and int(status.get("end", -1))
        == job["end"]
        and status.get("model_identity")
        == model_identity
    )


def worker_main(
    device: int,
    jobs: list[dict[str, Any]],
    model_path: str,
    model_identity: str,
    output_root: str,
    batch_size: int,
    chunk_rows: int,
    embedding_dimension: int,
) -> None:
    worker_start = time.perf_counter()

    output_root_path = Path(output_root)
    worker_status_path = (
        output_root_path
        / "workers"
        / f"worker_cuda_{device}.json"
    )
    worker_error_path = (
        output_root_path
        / "workers"
        / f"worker_cuda_{device}.error.log"
    )

    worker_status_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available."
            )

        if device >= torch.cuda.device_count():
            raise RuntimeError(
                f"Requested cuda:{device}, but only "
                f"{torch.cuda.device_count()} devices "
                "are visible."
            )

        torch.cuda.set_device(device)
        torch.set_grad_enabled(False)

        model = SentenceTransformer(
            model_path,
            device=f"cuda:{device}",
        )
        model.eval()

        dimension = (
            model.get_sentence_embedding_dimension()
        )

        if dimension != embedding_dimension:
            raise RuntimeError(
                f"Model dimension {dimension} does not "
                f"match expected {embedding_dimension}."
            )

        dataset_cache: dict[str, Any] = {}
        completed_jobs = 0
        skipped_jobs = 0
        completed_rows = 0

        print(
            f"[cuda:{device}] Loaded encoder; "
            f"{len(jobs)} assigned shards",
            flush=True,
        )

        for job_number, job in enumerate(
            jobs,
            start=1,
        ):
            status_path = Path(
                job["status_path"]
            )

            if status_matches_complete(
                status_path,
                job,
                model_identity,
            ):
                skipped_jobs += 1

                print(
                    f"[cuda:{device}] Skip "
                    f"{job['job_id']} "
                    f"({job_number}/{len(jobs)})",
                    flush=True,
                )
                continue

            corpus = job["corpus"]

            if corpus not in dataset_cache:
                dataset = load_from_disk(
                    job["dataset_path"]
                )
                dataset_cache[corpus] = (
                    dataset["train"]
                )

            train = dataset_cache[corpus]

            if len(train) < job["end"]:
                raise RuntimeError(
                    f"{job['job_id']}: dataset contains "
                    f"{len(train):,} rows, but shard ends "
                    f"at {job['end']:,}."
                )

            matrix = np.load(
                job["matrix_path"],
                mmap_mode="r+",
            )

            job_start_time = time.perf_counter()

            print(
                f"[cuda:{device}] Start "
                f"{job['job_id']} "
                f"({job_number}/{len(jobs)})",
                flush=True,
            )

            for chunk_start in range(
                job["start"],
                job["end"],
                chunk_rows,
            ):
                chunk_end = min(
                    chunk_start + chunk_rows,
                    job["end"],
                )

                prompts = train[
                    chunk_start:chunk_end
                ]["inputs"]

                embeddings = model.encode(
                    prompts,
                    batch_size=batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    convert_to_tensor=False,
                    normalize_embeddings=False,
                )

                embeddings = np.asarray(
                    embeddings,
                    dtype=np.float32,
                )

                expected_shape = (
                    chunk_end - chunk_start,
                    embedding_dimension,
                )

                if embeddings.shape != expected_shape:
                    raise RuntimeError(
                        f"{job['job_id']}: encoded shape "
                        f"{embeddings.shape}, expected "
                        f"{expected_shape}."
                    )

                if not np.isfinite(
                    embeddings
                ).all():
                    raise RuntimeError(
                        f"{job['job_id']}: non-finite "
                        "embedding values."
                    )

                matrix[
                    chunk_start:chunk_end
                ] = embeddings

                print(
                    f"[cuda:{device}] "
                    f"{job['job_id']} "
                    f"{chunk_end - job['start']:,}/"
                    f"{job['row_count']:,}",
                    flush=True,
                )

            matrix.flush()
            del matrix

            elapsed = (
                time.perf_counter()
                - job_start_time
            )

            atomic_write_json(
                {
                    "status": "complete",
                    "job_id": job["job_id"],
                    "corpus": corpus,
                    "start": job["start"],
                    "end": job["end"],
                    "row_count": job["row_count"],
                    "model_identity": (
                        model_identity
                    ),
                    "device": device,
                    "elapsed_seconds": elapsed,
                    "rows_per_second": (
                        job["row_count"] / elapsed
                    ),
                },
                status_path,
            )

            completed_jobs += 1
            completed_rows += job["row_count"]

            print(
                f"[cuda:{device}] Complete "
                f"{job['job_id']} in "
                f"{elapsed:.2f}s",
                flush=True,
            )

        worker_elapsed = (
            time.perf_counter()
            - worker_start
        )

        atomic_write_json(
            {
                "status": "complete",
                "device": device,
                "assigned_jobs": len(jobs),
                "completed_jobs": completed_jobs,
                "skipped_jobs": skipped_jobs,
                "completed_rows_this_run": (
                    completed_rows
                ),
                "model_max_seq_length": (
                    model.max_seq_length
                ),
                "embedding_dimension": dimension,
                "elapsed_seconds": worker_elapsed,
            },
            worker_status_path,
        )

    except Exception:
        error_text = traceback.format_exc()

        worker_error_path.write_text(
            error_text,
            encoding="utf-8",
        )

        atomic_write_json(
            {
                "status": "failed",
                "device": device,
                "error_log": str(
                    worker_error_path
                ),
            },
            worker_status_path,
        )

        print(
            error_text,
            file=sys.stderr,
            flush=True,
        )
        raise


def main() -> int:
    args = parse_args()

    if args.batch_size <= 0:
        raise ValueError(
            "--batch-size must be positive."
        )

    if args.chunk_rows <= 0:
        raise ValueError(
            "--chunk-rows must be positive."
        )

    if args.shard_rows <= 0:
        raise ValueError(
            "--shard-rows must be positive."
        )

    if not args.devices:
        raise ValueError(
            "At least one CUDA device is required."
        )

    if len(set(args.devices)) != len(
        args.devices
    ):
        raise ValueError(
            "CUDA devices must be unique."
        )

    author_root = (
        args.author_format_root.resolve()
    )
    summary_path = (
        args.task_indices_summary.resolve()
    )
    model_path = args.model_path.resolve()
    output_root = args.output_root.resolve()

    if not (model_path / "modules.json").is_file():
        raise FileNotFoundError(
            f"Not a SentenceTransformer snapshot: "
            f"{model_path}"
        )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    expected_rows = load_expected_rows(
        summary_path
    )

    model_identity = sha256_text(
        str(model_path)
    )

    matrix_paths = prepare_output_matrices(
        output_root=output_root,
        expected_rows=expected_rows,
        dimension=args.embedding_dimension,
    )

    jobs = create_jobs(
        expected_rows=expected_rows,
        matrix_paths=matrix_paths,
        author_root=author_root,
        output_root=output_root,
        shard_rows=args.shard_rows,
    )

    assignments = assign_jobs(
        jobs,
        args.devices,
    )

    print("=== SMART-v2 prompt embeddings ===")
    print(f"Model:       {model_path}")
    print(
        f"Rows:        "
        f"{sum(expected_rows.values()):,}"
    )
    print(
        f"Dimension:   "
        f"{args.embedding_dimension}"
    )
    print(f"Batch size:  {args.batch_size}")
    print(f"Chunk rows:  {args.chunk_rows:,}")
    print(f"Shard rows:  {args.shard_rows:,}")
    print(f"Devices:     {args.devices}")
    print(f"Shards:      {len(jobs)}")

    for device in args.devices:
        assigned_rows = sum(
            item["row_count"]
            for item in assignments[device]
        )

        print(
            f"  cuda:{device}: "
            f"{len(assignments[device])} shards, "
            f"{assigned_rows:,} rows"
        )

    start_time = time.perf_counter()

    context = mp.get_context("spawn")
    processes: list[mp.Process] = []

    for device in args.devices:
        process = context.Process(
            target=worker_main,
            kwargs={
                "device": device,
                "jobs": assignments[device],
                "model_path": str(model_path),
                "model_identity": model_identity,
                "output_root": str(
                    output_root
                ),
                "batch_size": args.batch_size,
                "chunk_rows": args.chunk_rows,
                "embedding_dimension": (
                    args.embedding_dimension
                ),
            },
            name=f"gte-cuda-{device}",
        )

        process.start()
        processes.append(process)

    for process in processes:
        process.join()

    failed_processes = [
        {
            "name": process.name,
            "exitcode": process.exitcode,
        }
        for process in processes
        if process.exitcode != 0
    ]

    if failed_processes:
        raise RuntimeError(
            "Embedding workers failed: "
            f"{failed_processes}"
        )

    incomplete_jobs: list[str] = []

    for job in jobs:
        if not status_matches_complete(
            Path(job["status_path"]),
            job,
            model_identity,
        ):
            incomplete_jobs.append(
                job["job_id"]
            )

    if incomplete_jobs:
        raise RuntimeError(
            "Incomplete embedding shards: "
            f"{incomplete_jobs}"
        )

    for corpus in EXPECTED_CORPORA:
        validate_existing_matrix(
            matrix_paths[corpus],
            (
                expected_rows[corpus],
                args.embedding_dimension,
            ),
        )

    elapsed = (
        time.perf_counter()
        - start_time
    )

    output_files = []

    for corpus in EXPECTED_CORPORA:
        path = matrix_paths[corpus]

        output_files.append(
            {
                "corpus": corpus,
                "row_count": (
                    expected_rows[corpus]
                ),
                "shape": [
                    expected_rows[corpus],
                    args.embedding_dimension,
                ],
                "dtype": "float32",
                "path": str(path),
                "size_bytes": (
                    path.stat().st_size
                ),
                "row_alignment": (
                    "Embedding row i equals "
                    "author-format train row i."
                ),
            }
        )

    run_summary_path = (
        output_root
        / "embedding_run_summary.json"
    )

    atomic_write_json(
        {
            "format_version": 1,
            "stage": (
                "smart_v2_prompt_embeddings"
            ),
            "status": "complete",
            "model_path": str(model_path),
            "model_identity": model_identity,
            "sentence_transformers_version": (
                sentence_transformers.__version__
            ),
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "python_version": (
                platform.python_version()
            ),
            "host": platform.node(),
            "configuration": {
                "devices": args.devices,
                "batch_size": args.batch_size,
                "chunk_rows": args.chunk_rows,
                "shard_rows": args.shard_rows,
                "embedding_dimension": (
                    args.embedding_dimension
                ),
                "normalize_embeddings_argument": (
                    False
                ),
                "input_field": "inputs",
                "output_dtype": "float32",
            },
            "total_rows": sum(
                expected_rows.values()
            ),
            "total_shards": len(jobs),
            "elapsed_seconds": elapsed,
            "outputs": output_files,
        },
        run_summary_path,
    )

    print()
    print("=== Embedding generation complete ===")
    print(
        f"Rows:       "
        f"{sum(expected_rows.values()):,}"
    )
    print(f"Shards:     {len(jobs)}")
    print(f"Elapsed:    {elapsed / 60:.2f} min")
    print(f"Summary:    {run_summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
PY

python3 -m py_compile \
  src/embeddings/embed_author_format.py
```

---

## 5.3 Run the embedding job

```bash
cd /data/saral/wdir/smart_v2 || exit 1
source ./smart_v2_env.sh

export GTE_MODEL=/mnt/warm_storage/saral/smart/cache/sentence_transformers/models--thenlper--gte-large/snapshots/4bef63f39fcc5e2d6b0aae83089f307af4970164

mkdir -p \
  /mnt/warm_storage/saral/smart_v2/embeddings/gte-large

python3 -m src.embeddings.embed_author_format \
  --author-format-root /mnt/warm_storage/saral/smart_v2/author_format \
  --task-indices-summary /mnt/warm_storage/saral/smart_v2/author_format/task_indices/task_indices_summary.json \
  --model-path "$GTE_MODEL" \
  --output-root /mnt/warm_storage/saral/smart_v2/embeddings/gte-large \
  --devices 0 1 2 3 \
  --batch-size 128 \
  --chunk-rows 8192 \
  --shard-rows 250000 \
  --embedding-dimension 1024 \
  2>&1 | tee \
  /mnt/warm_storage/saral/smart_v2/logs/embed_gte_large.log
```

The job is resumable. Completed shard status files are skipped on rerun.

Expected outputs:

```text
/mnt/warm_storage/saral/smart_v2/embeddings/gte-large/
├── cot/train_prompt_embeddings.npy
├── flan2021/train_prompt_embeddings.npy
├── sglue/train_prompt_embeddings.npy
├── t0/train_prompt_embeddings.npy
├── tulu/train_prompt_embeddings.npy
├── status/
├── workers/
└── embedding_run_summary.json
```

---

## 5.4 Create the verifier

```bash
cat > src/embeddings/verify_author_embeddings.py <<'PY'
"""Verify SMART-v2 row-aligned prompt embeddings."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from datasets import load_from_disk
from sentence_transformers import SentenceTransformer


EXPECTED_CORPORA = (
    "cot",
    "flan2021",
    "sglue",
    "t0",
    "tulu",
)

EXPECTED_TOTAL_ROWS = 6_266_471
EXPECTED_DIMENSION = 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--author-format-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--embedding-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
    )
    parser.add_argument(
        "--scan-rows",
        type=int,
        default=100_000,
    )
    parser.add_argument(
        "--samples-per-corpus",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )

    return parser.parse_args()


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


def deterministic_indices(
    corpus: str,
    row_count: int,
    sample_count: int,
) -> list[int]:
    scored: list[tuple[int, int]] = []

    candidates = max(
        sample_count * 20,
        sample_count,
    )

    for value in range(candidates):
        digest = hashlib.sha256(
            f"23|{corpus}|{value}".encode(
                "utf-8"
            )
        ).digest()

        score = int.from_bytes(
            digest[:8],
            byteorder="big",
            signed=False,
        )

        index = score % row_count
        scored.append((score, index))

    result: list[int] = []
    seen: set[int] = set()

    for _, index in sorted(scored):
        if index in seen:
            continue

        seen.add(index)
        result.append(index)

        if len(result) == sample_count:
            break

    if row_count > 0:
        for fixed in (
            0,
            row_count - 1,
        ):
            if fixed not in seen:
                result.append(fixed)
                seen.add(fixed)

    return sorted(result)


def main() -> int:
    args = parse_args()

    author_root = (
        args.author_format_root.resolve()
    )
    embedding_root = (
        args.embedding_root.resolve()
    )
    model_path = args.model_path.resolve()

    model = SentenceTransformer(
        str(model_path),
        device=args.device,
    )
    model.eval()

    dimension = (
        model.get_sentence_embedding_dimension()
    )

    if dimension != EXPECTED_DIMENSION:
        raise RuntimeError(
            f"Unexpected model dimension: {dimension}"
        )

    corpus_reports: list[dict[str, Any]] = []
    total_rows = 0
    global_min_norm = float("inf")
    global_max_norm = 0.0
    maximum_reencode_error = 0.0
    minimum_reencode_cosine = 1.0

    start_time = time.perf_counter()

    for corpus in EXPECTED_CORPORA:
        dataset = load_from_disk(
            str(author_root / corpus)
        )
        train = dataset["train"]

        matrix_path = (
            embedding_root
            / corpus
            / "train_prompt_embeddings.npy"
        )

        matrix = np.load(
            matrix_path,
            mmap_mode="r",
        )

        expected_shape = (
            len(train),
            EXPECTED_DIMENSION,
        )

        if matrix.shape != expected_shape:
            raise RuntimeError(
                f"{corpus}: matrix shape "
                f"{matrix.shape} != {expected_shape}"
            )

        if matrix.dtype != np.float32:
            raise RuntimeError(
                f"{corpus}: matrix dtype "
                f"{matrix.dtype} is not float32."
            )

        finite = True
        zero_norm_count = 0
        minimum_norm = float("inf")
        maximum_norm = 0.0

        for start in range(
            0,
            len(train),
            args.scan_rows,
        ):
            end = min(
                start + args.scan_rows,
                len(train),
            )

            block = np.asarray(
                matrix[start:end],
                dtype=np.float32,
            )

            if not np.isfinite(block).all():
                finite = False
                raise RuntimeError(
                    f"{corpus}: non-finite values in "
                    f"rows {start}:{end}."
                )

            norms = np.linalg.norm(
                block,
                axis=1,
            )

            zero_norm_count += int(
                np.sum(norms == 0.0)
            )
            minimum_norm = min(
                minimum_norm,
                float(norms.min()),
            )
            maximum_norm = max(
                maximum_norm,
                float(norms.max()),
            )

        if zero_norm_count:
            raise RuntimeError(
                f"{corpus}: found "
                f"{zero_norm_count} zero vectors."
            )

        sample_indices = deterministic_indices(
            corpus=corpus,
            row_count=len(train),
            sample_count=args.samples_per_corpus,
        )

        prompts = train.select(
            sample_indices
        )["inputs"]

        reencoded = model.encode(
            prompts,
            batch_size=args.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            convert_to_tensor=False,
            normalize_embeddings=False,
        ).astype(
            np.float32,
            copy=False,
        )

        stored = np.asarray(
            matrix[sample_indices],
            dtype=np.float32,
        )

        maximum_absolute_error = float(
            np.max(
                np.abs(
                    stored - reencoded
                )
            )
        )

        stored_norms = np.linalg.norm(
            stored,
            axis=1,
        )
        reencoded_norms = np.linalg.norm(
            reencoded,
            axis=1,
        )

        cosines = np.sum(
            stored * reencoded,
            axis=1,
        ) / (
            stored_norms
            * reencoded_norms
        )

        minimum_cosine = float(
            cosines.min()
        )

        if minimum_cosine < 0.99999:
            raise RuntimeError(
                f"{corpus}: re-encoding cosine "
                f"{minimum_cosine:.9f} is too low."
            )

        if maximum_absolute_error > 1e-4:
            raise RuntimeError(
                f"{corpus}: maximum re-encoding "
                f"error {maximum_absolute_error:.9g} "
                "is too large."
            )

        corpus_reports.append(
            {
                "corpus": corpus,
                "row_count": len(train),
                "shape": list(matrix.shape),
                "dtype": str(matrix.dtype),
                "finite": finite,
                "zero_norm_count": (
                    zero_norm_count
                ),
                "minimum_norm": minimum_norm,
                "maximum_norm": maximum_norm,
                "sample_indices": sample_indices,
                "maximum_reencoding_absolute_error": (
                    maximum_absolute_error
                ),
                "minimum_reencoding_cosine": (
                    minimum_cosine
                ),
            }
        )

        total_rows += len(train)
        global_min_norm = min(
            global_min_norm,
            minimum_norm,
        )
        global_max_norm = max(
            global_max_norm,
            maximum_norm,
        )
        maximum_reencode_error = max(
            maximum_reencode_error,
            maximum_absolute_error,
        )
        minimum_reencode_cosine = min(
            minimum_reencode_cosine,
            minimum_cosine,
        )

        print(
            f"{corpus:10s} "
            f"rows={len(train):,} "
            f"norm={minimum_norm:.8f}.."
            f"{maximum_norm:.8f} "
            f"reencode_error="
            f"{maximum_absolute_error:.3e} "
            f"cosine={minimum_cosine:.9f}"
        )

    if total_rows != EXPECTED_TOTAL_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_TOTAL_ROWS:,} rows; "
            f"verified {total_rows:,}."
        )

    elapsed = (
        time.perf_counter()
        - start_time
    )

    report_path = (
        embedding_root
        / "verification_report.json"
    )

    atomic_write_json(
        {
            "format_version": 1,
            "stage": (
                "smart_v2_prompt_embedding_verification"
            ),
            "status": "verified",
            "total_rows": total_rows,
            "embedding_dimension": (
                EXPECTED_DIMENSION
            ),
            "global_minimum_norm": (
                global_min_norm
            ),
            "global_maximum_norm": (
                global_max_norm
            ),
            "maximum_reencoding_absolute_error": (
                maximum_reencode_error
            ),
            "minimum_reencoding_cosine": (
                minimum_reencode_cosine
            ),
            "elapsed_seconds": elapsed,
            "corpora": corpus_reports,
        },
        report_path,
    )

    print()
    print("=== Verification summary ===")
    print("Status:                 verified")
    print(
        f"Embedding rows:         "
        f"{total_rows:,}"
    )
    print(
        f"Embedding dimension:    "
        f"{EXPECTED_DIMENSION}"
    )
    print(
        f"Global norm range:      "
        f"{global_min_norm:.8f} to "
        f"{global_max_norm:.8f}"
    )
    print(
        f"Maximum reencode error: "
        f"{maximum_reencode_error:.3e}"
    )
    print(
        f"Minimum reencode cosine:"
        f" {minimum_reencode_cosine:.9f}"
    )
    print(
        f"Elapsed:                "
        f"{elapsed / 60:.2f} minutes"
    )
    print(f"Report:                 {report_path}")
    print()
    print(
        "All prompt embedding checks passed."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
PY

python3 -m py_compile \
  src/embeddings/verify_author_embeddings.py
```

---

## 5.5 Run verification

```bash
cd /data/saral/wdir/smart_v2 || exit 1
source ./smart_v2_env.sh

export GTE_MODEL=/mnt/warm_storage/saral/smart/cache/sentence_transformers/models--thenlper--gte-large/snapshots/4bef63f39fcc5e2d6b0aae83089f307af4970164

python3 -m src.embeddings.verify_author_embeddings \
  --author-format-root /mnt/warm_storage/saral/smart_v2/author_format \
  --embedding-root /mnt/warm_storage/saral/smart_v2/embeddings/gte-large \
  --model-path "$GTE_MODEL" \
  --device cuda:0 \
  --scan-rows 100000 \
  --samples-per-corpus 8 \
  --batch-size 128 \
  2>&1 | tee \
  /mnt/warm_storage/saral/smart_v2/logs/verify_gte_large.log
```

Acceptance conditions:

```text
status                       = verified
total embedding rows         = 6,266,471
embedding dimension          = 1,024
dtype                        = float32
non-finite values            = 0
zero-norm vectors            = 0
minimum re-encoding cosine   >= 0.99999
maximum re-encoding error    <= 1e-4
```

After this passes, the next step is computing the 309 task-mean embeddings directly from these matrices and the authors-compatible task-index pickles.
