"""Smoke-test GTE-large prompt embeddings on one local SMART task."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import sentence_transformers
from sentence_transformers import SentenceTransformer

from local_dataset import (
    iter_valid_rows,
    load_task_catalog,
    task_lookup,
)


DEFAULT_MANIFEST = (
    "/mnt/warm_storage/saral/smart/"
    "prepared_data/clean_task_manifest.csv"
)

DEFAULT_OUTPUT_ROOT = (
    "/mnt/warm_storage/saral/smart/"
    "artifacts/embedding_smoke"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Embed the valid training prompts of one local SMART task "
            "with GTE-large."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(DEFAULT_MANIFEST),
    )
    parser.add_argument(
        "--task-id",
        default="sglue::axg",
    )
    parser.add_argument(
        "--encoder",
        default="thenlper/gte-large",
        help=(
            "Hugging Face model ID or local Sentence Transformer path."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(DEFAULT_OUTPUT_ROOT),
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
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
        "--token-length-batch-size",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Do not attempt to access the Hugging Face Hub.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", value)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


def batched(
    values: list[str],
    batch_size: int,
) -> Iterable[list[str]]:
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


def get_token_lengths(
    model: SentenceTransformer,
    prompts: list[str],
    batch_size: int,
) -> list[int]:
    tokenizer = model.tokenizer
    lengths: list[int] = []

    for prompt_batch in batched(prompts, batch_size):
        encoded = tokenizer(
            prompt_batch,
            add_special_tokens=True,
            padding=False,
            truncation=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )

        lengths.extend(
            len(input_ids)
            for input_ids in encoded["input_ids"]
        )

    return lengths


def write_json_atomic(
    payload: dict[str, Any],
    path: Path,
) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")

    with temporary_path.open(
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

    temporary_path.replace(path)


def save_numpy_atomic(
    array: np.ndarray,
    path: Path,
) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")

    with temporary_path.open("wb") as handle:
        np.save(handle, array)

    temporary_path.replace(path)


def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")

    if args.token_length_batch_size <= 0:
        raise ValueError(
            "--token-length-batch-size must be positive."
        )

    if args.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"Requested {args.device}, but CUDA is unavailable."
            )

        device_index = int(args.device.split(":")[1])

        if device_index >= torch.cuda.device_count():
            raise RuntimeError(
                f"Requested {args.device}, but only "
                f"{torch.cuda.device_count()} CUDA devices are visible."
            )

    tasks = load_task_catalog(args.manifest)
    lookup = task_lookup(tasks)

    if args.task_id not in lookup:
        raise KeyError(f"Task not found: {args.task_id}")

    task = lookup[args.task_id]

    prompts: list[str] = []
    row_map: list[dict[str, Any]] = []

    for local_index, example in enumerate(
        iter_valid_rows(task, "train")
    ):
        prompts.append(example["inputs"])
        row_map.append(
            {
                "embedding_index": local_index,
                "task_index": example["task_index"],
                "task_id": example["task_id"],
                "corpus": example["corpus"],
                "split": example["split"],
                "source_file": example["source_file"],
                "source_index": example["source_index"],
            }
        )

    if len(prompts) != task.valid_train_count:
        raise RuntimeError(
            f"Task manifest reports {task.valid_train_count:,} "
            f"valid rows, but the adapter yielded {len(prompts):,}."
        )

    output_dir = (
        args.output_root.resolve()
        / safe_name(args.task_id)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== GTE embedding smoke test ===")
    print(f"Task ID:          {task.task_id}")
    print(f"Source file:      {task.source_file}")
    print(f"Valid prompts:    {len(prompts):,}")
    print(f"Encoder:          {args.encoder}")
    print(f"Device:           {args.device}")
    print(f"Batch size:       {args.batch_size}")
    print(f"Output directory: {output_dir}")
    print()

    load_start = time.perf_counter()

    model = SentenceTransformer(
        args.encoder,
        device=args.device,
        cache_folder=os.environ.get(
            "SENTENCE_TRANSFORMERS_HOME"
        ),
        local_files_only=args.local_files_only,
    )
    model.eval()

    load_seconds = time.perf_counter() - load_start

    max_seq_length = int(model.max_seq_length)
    embedding_dimension = int(
        model.get_sentence_embedding_dimension()
    )

    print(f"Model load time:        {load_seconds:.2f} seconds")
    print(f"Maximum sequence length: {max_seq_length}")
    print(f"Embedding dimension:     {embedding_dimension}")
    print()

    print("Calculating untruncated token lengths...")

    token_length_start = time.perf_counter()

    token_lengths = get_token_lengths(
        model=model,
        prompts=prompts,
        batch_size=args.token_length_batch_size,
    )

    token_length_seconds = (
        time.perf_counter() - token_length_start
    )

    truncated_count = sum(
        length > max_seq_length
        for length in token_lengths
    )

    print(
        f"Token-length scan time:  "
        f"{token_length_seconds:.2f} seconds"
    )
    print(f"Minimum token length:    {min(token_lengths)}")
    print(
        f"Mean token length:       "
        f"{float(np.mean(token_lengths)):.2f}"
    )
    print(f"Maximum token length:    {max(token_lengths)}")
    print(
        f"Prompts over encoder limit: "
        f"{truncated_count:,}/{len(prompts):,}"
    )
    print()

    print("Encoding prompts...")

    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(args.device)

    encode_start = time.perf_counter()

    with torch.inference_mode():
        embeddings = model.encode(
            prompts,
            batch_size=args.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )

    encode_seconds = time.perf_counter() - encode_start

    embeddings = np.asarray(embeddings)

    if embeddings.ndim != 2:
        raise RuntimeError(
            f"Expected a two-dimensional embedding array; "
            f"received shape {embeddings.shape}."
        )

    expected_shape = (
        len(prompts),
        embedding_dimension,
    )

    if embeddings.shape != expected_shape:
        raise RuntimeError(
            f"Expected embedding shape {expected_shape}; "
            f"received {embeddings.shape}."
        )

    if not np.isfinite(embeddings).all():
        bad_count = int(
            embeddings.size
            - np.isfinite(embeddings).sum()
        )
        raise RuntimeError(
            f"Embedding array contains {bad_count:,} "
            "non-finite values."
        )

    norms = np.linalg.norm(embeddings, axis=1)

    if np.any(norms == 0):
        zero_norm_count = int(np.sum(norms == 0))
        raise RuntimeError(
            f"Found {zero_norm_count:,} zero-norm embeddings."
        )

    peak_gpu_memory_bytes = None

    if args.device.startswith("cuda"):
        peak_gpu_memory_bytes = int(
            torch.cuda.max_memory_allocated(args.device)
        )

    print()
    print("=== Embedding result ===")
    print(f"Shape:                {embeddings.shape}")
    print(f"Dtype:                {embeddings.dtype}")
    print(f"Encoding time:        {encode_seconds:.2f} seconds")
    print(
        f"Prompts per second:   "
        f"{len(prompts) / encode_seconds:.2f}"
    )
    print(f"Minimum vector norm:  {float(norms.min()):.6f}")
    print(f"Mean vector norm:     {float(norms.mean()):.6f}")
    print(f"Maximum vector norm:  {float(norms.max()):.6f}")

    if peak_gpu_memory_bytes is not None:
        print(
            f"Peak allocated VRAM:  "
            f"{peak_gpu_memory_bytes / 1024**3:.3f} GiB"
        )

    embeddings_path = output_dir / "prompt_embeddings.npy"
    row_map_path = output_dir / "row_map.jsonl"
    metadata_path = output_dir / "metadata.json"

    save_numpy_atomic(embeddings, embeddings_path)

    with row_map_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        for record in row_map:
            handle.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

    metadata: dict[str, Any] = {
        "format_version": 1,
        "stage": "gte_embedding_smoke_test",
        "task": {
            "task_index": task.task_index,
            "task_id": task.task_id,
            "corpus": task.corpus,
            "source_file": task.source_file,
            "valid_train_count": task.valid_train_count,
        },
        "encoder": {
            "requested_name_or_path": args.encoder,
            "maximum_sequence_length": max_seq_length,
            "embedding_dimension": embedding_dimension,
            "normalize_embeddings": False,
            "input_field": "inputs",
        },
        "execution": {
            "device": args.device,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "model_load_seconds": load_seconds,
            "token_length_scan_seconds": token_length_seconds,
            "encoding_seconds": encode_seconds,
            "prompts_per_second": (
                len(prompts) / encode_seconds
            ),
            "peak_allocated_gpu_memory_bytes": (
                peak_gpu_memory_bytes
            ),
        },
        "token_lengths": {
            "minimum": int(min(token_lengths)),
            "mean": float(np.mean(token_lengths)),
            "maximum": int(max(token_lengths)),
            "over_encoder_limit_count": int(truncated_count),
            "over_encoder_limit_fraction": (
                truncated_count / len(prompts)
            ),
        },
        "embedding_array": {
            "path": str(embeddings_path),
            "shape": list(embeddings.shape),
            "dtype": str(embeddings.dtype),
            "minimum_norm": float(norms.min()),
            "mean_norm": float(norms.mean()),
            "maximum_norm": float(norms.max()),
        },
        "row_map": {
            "path": str(row_map_path),
            "row_count": len(row_map),
        },
        "environment": {
            "python_version": sys.version,
            "platform": platform.platform(),
            "torch_version": torch.__version__,
            "sentence_transformers_version": (
                sentence_transformers.__version__
            ),
            "numpy_version": np.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
            "cuda_device_name": (
                torch.cuda.get_device_name(args.device)
                if args.device.startswith("cuda")
                else None
            ),
        },
    }

    metadata["embedding_array"]["sha256"] = sha256_file(
        embeddings_path
    )
    metadata["row_map"]["sha256"] = sha256_file(
        row_map_path
    )

    write_json_atomic(metadata, metadata_path)

    reloaded_embeddings = np.load(
        embeddings_path,
        mmap_mode="r",
    )

    if reloaded_embeddings.shape != embeddings.shape:
        raise RuntimeError(
            "Saved embedding shape does not match the "
            "in-memory embedding shape."
        )

    with row_map_path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        reloaded_row_count = sum(1 for _ in handle)

    if reloaded_row_count != embeddings.shape[0]:
        raise RuntimeError(
            f"Row-map count {reloaded_row_count:,} does not "
            f"match embedding count {embeddings.shape[0]:,}."
        )

    print()
    print(f"Embeddings: {embeddings_path}")
    print(f"Row map:    {row_map_path}")
    print(f"Metadata:   {metadata_path}")
    print()
    print("GTE embedding smoke test passed.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# export CUDA_VISIBLE_DEVICES=0

# python3 data_generation_scripts/smoke_test_gte_embeddings.py \
#   --manifest /mnt/warm_storage/saral/smart/prepared_data/clean_task_manifest.csv \
#   --task-id 'sglue::axg' \
#   --encoder thenlper/gte-large \
#   --device cuda:0 \
#   --batch-size 128 \
#   --output-root /mnt/warm_storage/saral/smart/artifacts/embedding_smoke \
#   2>&1 | tee \
#   /mnt/warm_storage/saral/smart/artifacts/embedding_smoke/gte_axg.log
