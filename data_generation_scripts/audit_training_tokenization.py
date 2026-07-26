"""Audit the exact preprocessing used by SMART instruction_tuner.py."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from datasets import Dataset, load_from_disk
from transformers import AutoConfig, AutoTokenizer


IGNORE_INDEX = -100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit sequence lengths and response supervision using "
            "the exact preprocessing semantics of instruction_tuner.py."
        )
    )

    parser.add_argument(
        "--dataset-25000",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--dataset-50000",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help=(
            "Model specification in LABEL=PATH form. "
            "Supply once for each model."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=4096,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--top-outliers",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
    )

    return parser.parse_args()


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


def parse_models(
    values: list[str],
) -> list[tuple[str, Path]]:
    result: list[tuple[str, Path]] = []
    labels: set[str] = set()

    for value in values:
        if "=" not in value:
            raise ValueError(
                f"Model specification must be LABEL=PATH: {value}"
            )

        label, raw_path = value.split("=", 1)
        label = label.strip()
        path = Path(raw_path.strip()).resolve()

        if not label:
            raise ValueError(
                f"Model label is empty: {value}"
            )

        if label in labels:
            raise ValueError(
                f"Duplicate model label: {label}"
            )

        if not path.exists():
            raise FileNotFoundError(path)

        labels.add(label)
        result.append((label, path))

    return result


def require_dataset_schema(
    dataset: Any,
    expected_train_count: int,
    expected_validation_count: int,
) -> None:
    if set(dataset.keys()) != {
        "train",
        "validation",
    }:
        raise RuntimeError(
            f"Unexpected DatasetDict splits: {dataset.keys()}"
        )

    if len(dataset["train"]) != expected_train_count:
        raise RuntimeError(
            f"Expected {expected_train_count:,} training rows; "
            f"found {len(dataset['train']):,}."
        )

    if (
        len(dataset["validation"])
        != expected_validation_count
    ):
        raise RuntimeError(
            f"Expected {expected_validation_count:,} validation "
            f"rows; found {len(dataset['validation']):,}."
        )

    expected_columns = [
        "prompt",
        "response",
    ]

    for split in ("train", "validation"):
        if dataset[split].column_names != expected_columns:
            raise RuntimeError(
                f"{split} columns are "
                f"{dataset[split].column_names}; expected "
                f"{expected_columns}."
            )


def hash_prompt_response_dataset(
    dataset: Dataset,
    batch_size: int = 1024,
) -> str:
    digest = hashlib.sha256()

    for start in range(
        0,
        len(dataset),
        batch_size,
    ):
        end = min(
            start + batch_size,
            len(dataset),
        )

        batch = dataset[start:end]

        for prompt, response in zip(
            batch["prompt"],
            batch["response"],
        ):
            digest.update(
                prompt.encode(
                    "utf-8",
                    errors="surrogatepass",
                )
            )
            digest.update(b"\0")
            digest.update(
                response.encode(
                    "utf-8",
                    errors="surrogatepass",
                )
            )
            digest.update(b"\n")

    return digest.hexdigest()


def percentile_summary(
    values: np.ndarray,
) -> dict[str, float | int]:
    if values.size == 0:
        raise ValueError(
            "Cannot summarize an empty array."
        )

    return {
        "minimum": int(values.min()),
        "mean": float(
            values.mean(dtype=np.float64)
        ),
        "p50": float(
            np.quantile(values, 0.50)
        ),
        "p90": float(
            np.quantile(values, 0.90)
        ),
        "p95": float(
            np.quantile(values, 0.95)
        ),
        "p99": float(
            np.quantile(values, 0.99)
        ),
        "p99_9": float(
            np.quantile(values, 0.999)
        ),
        "maximum": int(values.max()),
        "total": int(
            values.sum(dtype=np.int64)
        ),
    }


def iter_batches(
    dataset: Dataset,
    batch_size: int,
) -> Iterable[tuple[int, dict[str, list[Any]]]]:
    for start in range(
        0,
        len(dataset),
        batch_size,
    ):
        end = min(
            start + batch_size,
            len(dataset),
        )

        yield start, dataset[start:end]


def tokenize_lengths(
    tokenizer: Any,
    texts: list[str],
) -> list[int]:
    tokenized = tokenizer(
        texts,
        add_special_tokens=True,
        truncation=False,
        padding=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )

    return [
        len(input_ids)
        for input_ids in tokenized["input_ids"]
    ]


def audit_split(
    tokenizer: Any,
    dataset: Dataset,
    split_name: str,
    max_seq_length: int,
    batch_size: int,
    top_outliers: int,
) -> dict[str, Any]:
    row_count = len(dataset)

    full_lengths = np.empty(
        row_count,
        dtype=np.int64,
    )
    prompt_lengths = np.empty(
        row_count,
        dtype=np.int64,
    )
    sequence_lengths = np.empty(
        row_count,
        dtype=np.int64,
    )
    masked_prompt_lengths = np.empty(
        row_count,
        dtype=np.int64,
    )
    supervised_lengths = np.empty(
        row_count,
        dtype=np.int64,
    )
    untruncated_supervised_lengths = np.empty(
        row_count,
        dtype=np.int64,
    )
    lost_supervised_lengths = np.empty(
        row_count,
        dtype=np.int64,
    )

    start_time = time.perf_counter()

    for start, batch in iter_batches(
        dataset,
        batch_size,
    ):
        prompts = batch["prompt"]
        responses = batch["response"]

        for offset, (prompt, response) in enumerate(
            zip(prompts, responses)
        ):
            if not isinstance(prompt, str):
                raise TypeError(
                    f"{split_name} row {start + offset}: "
                    "prompt is not a string."
                )

            if not prompt.strip():
                raise RuntimeError(
                    f"{split_name} row {start + offset}: "
                    "prompt is empty."
                )

            if not isinstance(response, str):
                raise TypeError(
                    f"{split_name} row {start + offset}: "
                    "response is not a string."
                )

            if not response.strip():
                raise RuntimeError(
                    f"{split_name} row {start + offset}: "
                    "response is empty."
                )

        # Exact released-trainer string construction.
        combined = [
            prompt + " " + response
            for prompt, response in zip(
                prompts,
                responses,
            )
        ]

        batch_full_lengths = tokenize_lengths(
            tokenizer,
            combined,
        )
        batch_prompt_lengths = tokenize_lengths(
            tokenizer,
            prompts,
        )

        end = start + len(prompts)

        full_array = np.asarray(
            batch_full_lengths,
            dtype=np.int64,
        )
        prompt_array = np.asarray(
            batch_prompt_lengths,
            dtype=np.int64,
        )

        # The released trainer right-truncates both calls
        # independently to max_seq_length.
        sequence_array = np.minimum(
            full_array,
            max_seq_length,
        )

        prompt_truncated_array = np.minimum(
            prompt_array,
            max_seq_length,
        )

        # labels[:prompt_len] = IGNORE_INDEX. If prompt_len
        # exceeds the concatenated sequence length, all labels
        # are masked.
        masked_array = np.minimum(
            prompt_truncated_array,
            sequence_array,
        )

        supervised_array = (
            sequence_array - masked_array
        )

        untruncated_supervised_array = np.maximum(
            full_array - prompt_array,
            0,
        )

        lost_supervised_array = np.maximum(
            untruncated_supervised_array
            - supervised_array,
            0,
        )

        full_lengths[start:end] = full_array
        prompt_lengths[start:end] = prompt_array
        sequence_lengths[start:end] = sequence_array
        masked_prompt_lengths[start:end] = masked_array
        supervised_lengths[start:end] = supervised_array
        untruncated_supervised_lengths[start:end] = (
            untruncated_supervised_array
        )
        lost_supervised_lengths[start:end] = (
            lost_supervised_array
        )

        processed = end

        if (
            processed == row_count
            or processed % 10_000 == 0
        ):
            elapsed = (
                time.perf_counter()
                - start_time
            )

            print(
                f"    {split_name}: "
                f"{processed:,}/{row_count:,} "
                f"rows in {elapsed / 60:.2f} min",
                flush=True,
            )

    zero_supervised_indices = np.flatnonzero(
        supervised_lengths == 0
    )

    prompt_truncated_indices = np.flatnonzero(
        prompt_lengths >= max_seq_length
    )

    sequence_truncated_indices = np.flatnonzero(
        full_lengths > max_seq_length
    )

    response_loss_indices = np.flatnonzero(
        lost_supervised_lengths > 0
    )

    if top_outliers > 0:
        longest_order = np.argsort(
            full_lengths,
            kind="stable",
        )[::-1][
            :top_outliers
        ]

        most_lost_order = np.argsort(
            lost_supervised_lengths,
            kind="stable",
        )[::-1][
            :top_outliers
        ]
    else:
        longest_order = np.asarray(
            [],
            dtype=np.int64,
        )
        most_lost_order = np.asarray(
            [],
            dtype=np.int64,
        )

    outliers = {
        "longest_sequences": [
            {
                "row_index": int(index),
                "full_tokens": int(
                    full_lengths[index]
                ),
                "prompt_tokens": int(
                    prompt_lengths[index]
                ),
                "retained_sequence_tokens": int(
                    sequence_lengths[index]
                ),
                "retained_supervised_tokens": int(
                    supervised_lengths[index]
                ),
                "lost_supervised_tokens": int(
                    lost_supervised_lengths[index]
                ),
            }
            for index in longest_order
        ],
        "most_supervised_tokens_lost": [
            {
                "row_index": int(index),
                "full_tokens": int(
                    full_lengths[index]
                ),
                "prompt_tokens": int(
                    prompt_lengths[index]
                ),
                "untruncated_supervised_tokens": int(
                    untruncated_supervised_lengths[
                        index
                    ]
                ),
                "retained_supervised_tokens": int(
                    supervised_lengths[index]
                ),
                "lost_supervised_tokens": int(
                    lost_supervised_lengths[index]
                ),
            }
            for index in most_lost_order
            if lost_supervised_lengths[index] > 0
        ],
        "zero_supervised_rows": [
            {
                "row_index": int(index),
                "full_tokens": int(
                    full_lengths[index]
                ),
                "prompt_tokens": int(
                    prompt_lengths[index]
                ),
                "retained_sequence_tokens": int(
                    sequence_lengths[index]
                ),
            }
            for index in zero_supervised_indices[
                :top_outliers
            ]
        ],
    }

    positive_untruncated = (
        untruncated_supervised_lengths > 0
    )

    if np.any(positive_untruncated):
        retention = (
            supervised_lengths[
                positive_untruncated
            ]
            / untruncated_supervised_lengths[
                positive_untruncated
            ]
        )

        retention_summary = {
            "mean": float(
                retention.mean(
                    dtype=np.float64
                )
            ),
            "p01": float(
                np.quantile(retention, 0.01)
            ),
            "p05": float(
                np.quantile(retention, 0.05)
            ),
            "p50": float(
                np.quantile(retention, 0.50)
            ),
            "minimum": float(
                retention.min()
            ),
        }
    else:
        retention_summary = None

    elapsed_seconds = (
        time.perf_counter() - start_time
    )

    return {
        "split": split_name,
        "row_count": row_count,
        "max_seq_length": max_seq_length,
        "timing_seconds": elapsed_seconds,
        "lengths": {
            "full_prompt_response": (
                percentile_summary(
                    full_lengths
                )
            ),
            "prompt": percentile_summary(
                prompt_lengths
            ),
            "retained_sequence": (
                percentile_summary(
                    sequence_lengths
                )
            ),
            "retained_supervised": (
                percentile_summary(
                    supervised_lengths
                )
            ),
            "untruncated_supervised": (
                percentile_summary(
                    untruncated_supervised_lengths
                )
            ),
            "lost_supervised": (
                percentile_summary(
                    lost_supervised_lengths
                )
            ),
        },
        "counts": {
            "sequence_truncated": int(
                sequence_truncated_indices.size
            ),
            "sequence_truncated_fraction": float(
                sequence_truncated_indices.size
                / row_count
            ),
            "prompt_reaches_or_exceeds_limit": int(
                prompt_truncated_indices.size
            ),
            "prompt_reaches_or_exceeds_limit_fraction": float(
                prompt_truncated_indices.size
                / row_count
            ),
            "rows_losing_supervised_tokens": int(
                response_loss_indices.size
            ),
            "rows_losing_supervised_tokens_fraction": float(
                response_loss_indices.size
                / row_count
            ),
            "zero_supervised_tokens": int(
                zero_supervised_indices.size
            ),
            "zero_supervised_tokens_fraction": float(
                zero_supervised_indices.size
                / row_count
            ),
            "supervised_tokens_lt_8": int(
                np.sum(
                    supervised_lengths < 8
                )
            ),
            "supervised_tokens_lt_32": int(
                np.sum(
                    supervised_lengths < 32
                )
            ),
        },
        "supervised_token_retention": (
            retention_summary
        ),
        "outliers": outliers,
    }


def summary_csv_row(
    model_label: str,
    model_path: Path,
    report: dict[str, Any],
) -> dict[str, Any]:
    counts = report["counts"]
    full = report["lengths"][
        "full_prompt_response"
    ]
    prompt = report["lengths"]["prompt"]
    retained = report["lengths"][
        "retained_sequence"
    ]
    supervised = report["lengths"][
        "retained_supervised"
    ]

    return {
        "model": model_label,
        "model_path": str(model_path),
        "split": report["split"],
        "rows": report["row_count"],
        "full_mean": full["mean"],
        "full_p50": full["p50"],
        "full_p95": full["p95"],
        "full_p99": full["p99"],
        "full_max": full["maximum"],
        "prompt_p99": prompt["p99"],
        "prompt_max": prompt["maximum"],
        "retained_mean": retained["mean"],
        "retained_token_total": retained["total"],
        "supervised_mean": supervised["mean"],
        "supervised_p50": supervised["p50"],
        "supervised_p95": supervised["p95"],
        "supervised_min": supervised["minimum"],
        "supervised_token_total": supervised["total"],
        "sequence_truncated": counts[
            "sequence_truncated"
        ],
        "sequence_truncated_fraction": counts[
            "sequence_truncated_fraction"
        ],
        "prompt_at_limit": counts[
            "prompt_reaches_or_exceeds_limit"
        ],
        "rows_losing_supervised_tokens": counts[
            "rows_losing_supervised_tokens"
        ],
        "zero_supervised_tokens": counts[
            "zero_supervised_tokens"
        ],
        "seconds": report["timing_seconds"],
    }


def main() -> int:
    args = parse_args()

    if args.max_seq_length <= 0:
        raise ValueError(
            "--max-seq-length must be positive."
        )

    if args.batch_size <= 0:
        raise ValueError(
            "--batch-size must be positive."
        )

    models = parse_models(args.model)

    dataset_25_path = (
        args.dataset_25000.resolve()
    )
    dataset_50_path = (
        args.dataset_50000.resolve()
    )
    output_root = args.output_root.resolve()

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Loading final SMART datasets...")

    dataset_25 = load_from_disk(
        str(dataset_25_path)
    )
    dataset_50 = load_from_disk(
        str(dataset_50_path)
    )

    expected_validation_count = 183_870

    require_dataset_schema(
        dataset_25,
        expected_train_count=25_000,
        expected_validation_count=(
            expected_validation_count
        ),
    )

    require_dataset_schema(
        dataset_50,
        expected_train_count=50_000,
        expected_validation_count=(
            expected_validation_count
        ),
    )

    print(
        "Verifying that saved validation splits "
        "are identical..."
    )

    validation_hash_25 = (
        hash_prompt_response_dataset(
            dataset_25["validation"]
        )
    )

    validation_hash_50 = (
        hash_prompt_response_dataset(
            dataset_50["validation"]
        )
    )

    if validation_hash_25 != validation_hash_50:
        raise RuntimeError(
            "SMART-25K and SMART-50K validation "
            "splits are not identical."
        )

    split_datasets = [
        (
            "smart_25000_train",
            dataset_25["train"],
        ),
        (
            "smart_50000_train",
            dataset_50["train"],
        ),
        (
            "shared_validation",
            dataset_50["validation"],
        ),
    ]

    all_model_reports: list[
        dict[str, Any]
    ] = []

    csv_rows: list[dict[str, Any]] = []
    fatal_problems: list[str] = []

    for model_label, model_path in models:
        print()
        print(
            "========================================"
        )
        print(f"Model: {model_label}")
        print(f"Path:  {model_path}")
        print(
            "========================================"
        )

        tokenizer = AutoTokenizer.from_pretrained(
            str(model_path),
            trust_remote_code=(
                args.trust_remote_code
            ),
            local_files_only=(
                args.local_files_only
            ),
            use_fast=True,
        )

        config = AutoConfig.from_pretrained(
            str(model_path),
            trust_remote_code=(
                args.trust_remote_code
            ),
            local_files_only=(
                args.local_files_only
            ),
        )

        if tokenizer.eos_token_id is None:
            raise RuntimeError(
                f"{model_label} tokenizer has no EOS token."
            )

        # Exact released-trainer configuration.
        tokenizer.pad_token_id = (
            tokenizer.eos_token_id
        )
        tokenizer.padding_side = "right"

        tokenizer_limit = int(
            tokenizer.model_max_length
        )

        effective_max_seq_length = min(
            args.max_seq_length,
            tokenizer_limit,
        )

        max_position_embeddings = getattr(
            config,
            "max_position_embeddings",
            None,
        )

        model_metadata = {
            "label": model_label,
            "path": str(model_path),
            "model_type": getattr(
                config,
                "model_type",
                None,
            ),
            "architecture": getattr(
                config,
                "architectures",
                None,
            ),
            "tokenizer_class": (
                tokenizer.__class__.__name__
            ),
            "tokenizer_is_fast": bool(
                tokenizer.is_fast
            ),
            "tokenizer_model_max_length": (
                tokenizer_limit
            ),
            "config_max_position_embeddings": (
                max_position_embeddings
            ),
            "requested_max_seq_length": (
                args.max_seq_length
            ),
            "effective_max_seq_length": (
                effective_max_seq_length
            ),
            "padding_side": (
                tokenizer.padding_side
            ),
            "truncation_side": (
                tokenizer.truncation_side
            ),
            "bos_token_id": (
                tokenizer.bos_token_id
            ),
            "eos_token_id": (
                tokenizer.eos_token_id
            ),
            "pad_token_id_after_author_patch": (
                tokenizer.pad_token_id
            ),
            "vocab_size": len(tokenizer),
        }

        if (
            effective_max_seq_length
            != args.max_seq_length
        ):
            fatal_problems.append(
                f"{model_label}: requested max length "
                f"{args.max_seq_length} was reduced to "
                f"{effective_max_seq_length}."
            )

        if tokenizer.truncation_side != "right":
            fatal_problems.append(
                f"{model_label}: tokenizer truncation side "
                f"is {tokenizer.truncation_side!r}, not "
                "'right' as assumed by the released trainer."
            )

        split_reports: list[
            dict[str, Any]
        ] = []

        for split_name, dataset in split_datasets:
            print()
            print(f"  Auditing {split_name}...")

            split_report = audit_split(
                tokenizer=tokenizer,
                dataset=dataset,
                split_name=split_name,
                max_seq_length=(
                    effective_max_seq_length
                ),
                batch_size=args.batch_size,
                top_outliers=(
                    args.top_outliers
                ),
            )

            split_reports.append(
                split_report
            )

            csv_rows.append(
                summary_csv_row(
                    model_label,
                    model_path,
                    split_report,
                )
            )

            counts = split_report["counts"]
            retained = split_report["lengths"][
                "retained_sequence"
            ]
            supervised = split_report["lengths"][
                "retained_supervised"
            ]

            print(
                f"    rows:                 "
                f"{split_report['row_count']:,}"
            )
            print(
                f"    mean retained tokens: "
                f"{retained['mean']:.2f}"
            )
            print(
                f"    p99 retained tokens:  "
                f"{retained['p99']:.2f}"
            )
            print(
                f"    mean response tokens: "
                f"{supervised['mean']:.2f}"
            )
            print(
                f"    truncated rows:       "
                f"{counts['sequence_truncated']:,} "
                f"({counts['sequence_truncated_fraction']:.4%})"
            )
            print(
                f"    prompt at limit:      "
                f"{counts['prompt_reaches_or_exceeds_limit']:,}"
            )
            print(
                f"    zero supervised rows: "
                f"{counts['zero_supervised_tokens']:,}"
            )

            if (
                counts["zero_supervised_tokens"]
                > 0
            ):
                fatal_problems.append(
                    f"{model_label}/{split_name}: "
                    f"{counts['zero_supervised_tokens']} "
                    "rows have zero supervised tokens."
                )

        model_report = {
            "model": model_metadata,
            "splits": split_reports,
        }

        model_report_path = (
            output_root
            / f"{model_label}_tokenization_report.json"
        )

        atomic_write_json(
            model_report,
            model_report_path,
        )

        model_report[
            "report_path"
        ] = str(model_report_path)

        all_model_reports.append(
            model_report
        )

        del tokenizer
        del config

    csv_path = (
        output_root
        / "tokenization_summary.csv"
    )

    if not csv_rows:
        raise RuntimeError(
            "No tokenization reports were generated."
        )

    with csv_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                csv_rows[0].keys()
            ),
        )
        writer.writeheader()
        writer.writerows(csv_rows)

    final_status = (
        "failed"
        if fatal_problems
        else "complete"
    )

    summary = {
        "format_version": 1,
        "stage": "SMART_training_tokenization_audit",
        "status": final_status,
        "author_preprocessing": {
            "concatenation": (
                'prompt + " " + response'
            ),
            "add_special_tokens": True,
            "truncation": True,
            "requested_max_seq_length": (
                args.max_seq_length
            ),
            "prompt_labels_masked_to": (
                IGNORE_INDEX
            ),
            "padding_side": "right",
            "pad_token_equals_eos_token": True,
        },
        "datasets": {
            "smart_25000": str(
                dataset_25_path
            ),
            "smart_50000": str(
                dataset_50_path
            ),
            "validation_count": (
                expected_validation_count
            ),
            "validation_sha256": (
                validation_hash_25
            ),
            "validation_splits_identical": True,
        },
        "models": all_model_reports,
        "fatal_problems": fatal_problems,
        "outputs": {
            "summary_csv": str(csv_path),
        },
    }

    summary_path = (
        output_root
        / "tokenization_audit_summary.json"
    )

    atomic_write_json(
        summary,
        summary_path,
    )

    print()
    print("=== Tokenization audit summary ===")
    print(f"Status: {final_status}")
    print(
        "Validation splits identical: True"
    )
    print(
        f"Fatal problems: {len(fatal_problems)}"
    )

    for problem in fatal_problems:
        print(f"  - {problem}")

    print(f"CSV:     {csv_path}")
    print(f"Summary: {summary_path}")

    if fatal_problems:
        print(
            "Tokenization audit failed. Do not "
            "start training yet."
        )
        return 2

    print(
        "SMART training tokenization audit passed."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
