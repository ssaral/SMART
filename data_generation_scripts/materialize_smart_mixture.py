"""Materialize final SMART 25K and 50K Hugging Face datasets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, TextIO

import ijson
import numpy as np
from datasets import Dataset, DatasetDict, load_dataset


EXPECTED_TASK_COUNT = 309
EXPECTED_TRAIN_COUNTS = {
    25_000: 25_000,
    50_000: 50_000,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--allocations",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--stage2-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=23,
    )
    parser.add_argument(
        "--expected-validation-count",
        type=int,
        default=183_870,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


def safe_name(value: str) -> str:
    return re.sub(
        r"[^A-Za-z0-9_.-]+",
        "__",
        value,
    )


def is_valid_row(row: Any) -> bool:
    if not isinstance(row, dict):
        return False

    inputs = row.get("inputs")
    targets = row.get("targets")

    return (
        isinstance(inputs, str)
        and bool(inputs.strip())
        and isinstance(targets, str)
        and bool(targets.strip())
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


def write_jsonl(
    handle: TextIO,
    record: dict[str, Any],
) -> None:
    handle.write(
        json.dumps(
            record,
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )
    handle.write("\n")


def load_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        return list(csv.DictReader(handle))


def load_manifest(
    path: Path,
) -> dict[str, dict[str, str]]:
    rows = load_csv(path)

    if not rows:
        raise RuntimeError(
            "Clean task manifest is empty."
        )

    required = {
        "task_id",
        "corpus",
        "task_name",
        "source_file",
    }

    missing = required - set(rows[0])

    if missing:
        raise RuntimeError(
            "Manifest is missing columns: "
            + ", ".join(sorted(missing))
        )

    result: dict[str, dict[str, str]] = {}

    for row in rows:
        task_id = row["task_id"]

        if task_id in result:
            raise RuntimeError(
                f"Duplicate manifest task: {task_id}"
            )

        source_file = Path(
            row["source_file"]
        ).resolve()

        if not source_file.is_file():
            raise FileNotFoundError(
                source_file
            )

        row = dict(row)
        row["source_file"] = str(source_file)
        result[task_id] = row

    if len(result) != EXPECTED_TASK_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_TASK_COUNT} manifest tasks; "
            f"found {len(result)}."
        )

    return result


def load_allocations(
    path: Path,
) -> list[dict[str, Any]]:
    raw_rows = load_csv(path)

    if len(raw_rows) != EXPECTED_TASK_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_TASK_COUNT} allocation rows; "
            f"found {len(raw_rows)}."
        )

    rows: list[dict[str, Any]] = []

    for raw in raw_rows:
        row = {
            "task_index": int(
                raw["task_index"]
            ),
            "graph_cut_rank": int(
                raw["graph_cut_rank"]
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
            <= row["valid_train_count"]
        ):
            raise RuntimeError(
                f"Invalid allocation for "
                f"{row['task_id']}."
            )

        rows.append(row)

    rows.sort(
        key=lambda row: row["task_index"]
    )

    if len(
        {
            row["task_id"]
            for row in rows
        }
    ) != EXPECTED_TASK_COUNT:
        raise RuntimeError(
            "Duplicate task IDs in allocation table."
        )

    if sum(
        row["allocation_25000"]
        for row in rows
    ) != 25_000:
        raise RuntimeError(
            "25K allocation sum is not 25,000."
        )

    if sum(
        row["allocation_50000"]
        for row in rows
    ) != 50_000:
        raise RuntimeError(
            "50K allocation sum is not 50,000."
        )

    return rows


def resolve_stage2_directory(
    stage2_root: Path,
    task_index: int,
    task_id: str,
) -> Path:
    expected = (
        stage2_root
        / "tasks"
        / f"{task_index:04d}_{safe_name(task_id)}"
    )

    if expected.is_dir():
        return expected

    candidates = sorted(
        (stage2_root / "tasks").glob(
            f"{task_index:04d}_*"
        )
    )

    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Could not resolve Stage 2 directory for "
            f"{task_id}: {candidates}"
        )

    return candidates[0]


def stream_split(
    source_file: Path,
    split: str,
):
    with source_file.open("rb") as handle:
        yield from enumerate(
            ijson.items(
                handle,
                f"{split}.item",
            )
        )


def save_dataset_atomic(
    dataset: DatasetDict,
    destination: Path,
    overwrite: bool,
) -> None:
    temporary = destination.with_name(
        destination.name + ".tmp"
    )

    if temporary.exists():
        shutil.rmtree(temporary)

    if destination.exists():
        if not overwrite:
            raise FileExistsError(
                f"{destination} already exists. "
                "Use --overwrite to replace it."
            )

        shutil.rmtree(destination)

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset.save_to_disk(
        str(temporary)
    )

    temporary.rename(destination)


def membership_hash(
    keys: set[tuple[str, int]],
) -> str:
    digest = hashlib.sha256()

    for task_id, source_index in sorted(keys):
        digest.update(
            task_id.encode("utf-8")
        )
        digest.update(b"\0")
        digest.update(
            str(source_index).encode("ascii")
        )
        digest.update(b"\n")

    return digest.hexdigest()


def ordered_dataset_hash(
    dataset: Dataset,
) -> str:
    digest = hashlib.sha256()

    for row in dataset:
        digest.update(
            row["task_id"].encode("utf-8")
        )
        digest.update(b"\0")
        digest.update(
            str(row["source_index"]).encode(
                "ascii"
            )
        )
        digest.update(b"\n")

    return digest.hexdigest()


def load_jsonl_dataset(
    path: Path,
    cache_dir: Path,
) -> Dataset:
    return load_dataset(
        "json",
        data_files=str(path),
        split="train",
        cache_dir=str(cache_dir),
    )


def main() -> int:
    args = parse_args()

    manifest_path = args.manifest.resolve()
    allocations_path = args.allocations.resolve()
    stage2_root = args.stage2_root.resolve()
    output_root = args.output_root.resolve()

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw_root = output_root / "raw_audit_jsonl"
    cache_root = output_root / "hf_cache"

    raw_root.mkdir(
        parents=True,
        exist_ok=True,
    )
    cache_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest = load_manifest(
        manifest_path
    )
    allocations = load_allocations(
        allocations_path
    )

    catalog_path = (
        stage2_root
        / "stage2_selection_catalog.json"
    )

    summary_path = (
        stage2_root
        / "stage2_selection_summary.json"
    )

    if not catalog_path.is_file():
        raise FileNotFoundError(
            catalog_path
        )

    if not summary_path.is_file():
        raise FileNotFoundError(
            summary_path
        )

    stage2_catalog = json.loads(
        catalog_path.read_text(
            encoding="utf-8"
        )
    )

    stage2_summary = json.loads(
        summary_path.read_text(
            encoding="utf-8"
        )
    )

    if stage2_catalog.get("status") != "complete":
        raise RuntimeError(
            "Stage 2 catalog is not complete."
        )

    if stage2_summary.get("status") != "complete":
        raise RuntimeError(
            "Stage 2 summary is not complete."
        )

    print("=== Final SMART materialization ===")
    print(f"Tasks:               {len(allocations)}")
    print(
        "Stage 2 configuration:",
        stage2_catalog["configuration"],
    )
    print(f"Output root:         {output_root}")
    print()

    train25_path = (
        raw_root / "smart_25000_train.audit.jsonl"
    )
    train50_path = (
        raw_root / "smart_50000_train.audit.jsonl"
    )
    validation_path = (
        raw_root / "validation.audit.jsonl"
    )

    temporary_paths = {
        train25_path: train25_path.with_suffix(
            ".jsonl.tmp"
        ),
        train50_path: train50_path.with_suffix(
            ".jsonl.tmp"
        ),
        validation_path: validation_path.with_suffix(
            ".jsonl.tmp"
        ),
    }

    for path in temporary_paths.values():
        if path.exists():
            path.unlink()

    train25_keys: set[tuple[str, int]] = set()
    train50_keys: set[tuple[str, int]] = set()
    validation_keys: set[tuple[str, int]] = set()

    method_counts: Counter[str] = Counter()
    train25_count = 0
    train50_count = 0
    validation_count = 0

    with (
        temporary_paths[train25_path].open(
            "w",
            encoding="utf-8",
        ) as train25_handle,
        temporary_paths[train50_path].open(
            "w",
            encoding="utf-8",
        ) as train50_handle,
        temporary_paths[validation_path].open(
            "w",
            encoding="utf-8",
        ) as validation_handle,
    ):
        for position, allocation in enumerate(
            allocations,
            start=1,
        ):
            task_id = allocation["task_id"]

            if task_id not in manifest:
                raise KeyError(
                    f"Task absent from manifest: {task_id}"
                )

            manifest_row = manifest[task_id]
            source_file = Path(
                manifest_row["source_file"]
            )

            task_dir = resolve_stage2_directory(
                stage2_root=stage2_root,
                task_index=allocation["task_index"],
                task_id=task_id,
            )

            metadata_path = (
                task_dir / "metadata.json"
            )
            selection_path = (
                task_dir
                / "selected_source_indices.npy"
            )

            if not metadata_path.is_file():
                raise FileNotFoundError(
                    metadata_path
                )

            if not selection_path.is_file():
                raise FileNotFoundError(
                    selection_path
                )

            metadata = json.loads(
                metadata_path.read_text(
                    encoding="utf-8"
                )
            )

            selected_sources = np.load(
                selection_path
            ).astype(
                np.int64,
                copy=False,
            )

            k25 = allocation[
                "allocation_25000"
            ]
            k50 = allocation[
                "allocation_50000"
            ]

            if selected_sources.shape != (k50,):
                raise RuntimeError(
                    f"{task_id}: Stage 2 selection shape "
                    f"{selected_sources.shape}; expected "
                    f"({k50},)."
                )

            if np.unique(
                selected_sources
            ).size != k50:
                raise RuntimeError(
                    f"{task_id}: duplicate selected "
                    "source indices."
                )

            method = metadata[
                "method"
            ]["name"]
            method_counts[method] += 1

            needed_sources = {
                int(index)
                for index in selected_sources
            }

            selected_records: dict[
                int,
                dict[str, Any],
            ] = {}

            for source_index, row in stream_split(
                source_file,
                "train",
            ):
                if source_index not in needed_sources:
                    continue

                if not is_valid_row(row):
                    raise RuntimeError(
                        f"{task_id}: selected invalid train "
                        f"row at source index {source_index}."
                    )

                selected_records[
                    source_index
                ] = {
                    "prompt": row["inputs"],
                    "response": row["targets"],
                    "task_id": task_id,
                    "corpus": allocation["corpus"],
                    "task_name": allocation["task_name"],
                    "source_file": str(source_file),
                    "source_index": source_index,
                    "graph_cut_rank": allocation[
                        "graph_cut_rank"
                    ],
                    "stage2_method": method,
                    "candidate_restricted": (
                        method
                        == "candidate_restricted"
                    ),
                    "split": "train",
                }

            missing_sources = (
                needed_sources
                - set(selected_records)
            )

            if missing_sources:
                raise RuntimeError(
                    f"{task_id}: selected source rows "
                    f"not found: "
                    f"{sorted(missing_sources)[:20]}"
                )

            # Preserve Facility Location ordering within
            # every task, matching ordering-prefix semantics.
            for rank, source_index_np in enumerate(
                selected_sources,
                start=1,
            ):
                source_index = int(
                    source_index_np
                )

                base_record = dict(
                    selected_records[source_index]
                )

                key = (
                    task_id,
                    source_index,
                )

                if key in train50_keys:
                    raise RuntimeError(
                        f"Duplicate 50K key: {key}"
                    )

                record50 = {
                    **base_record,
                    "selection_rank": rank,
                    "budget": 50_000,
                    "included_in_25000": (
                        rank <= k25
                    ),
                }

                write_jsonl(
                    train50_handle,
                    record50,
                )

                train50_keys.add(key)
                train50_count += 1

                if rank <= k25:
                    if key in train25_keys:
                        raise RuntimeError(
                            f"Duplicate 25K key: {key}"
                        )

                    record25 = {
                        **base_record,
                        "selection_rank": rank,
                        "budget": 25_000,
                        "included_in_25000": True,
                    }

                    write_jsonl(
                        train25_handle,
                        record25,
                    )

                    train25_keys.add(key)
                    train25_count += 1

            task_validation_count = 0

            for source_index, row in stream_split(
                source_file,
                "validation",
            ):
                if not is_valid_row(row):
                    continue

                key = (
                    task_id,
                    source_index,
                )

                if key in validation_keys:
                    raise RuntimeError(
                        f"Duplicate validation key: {key}"
                    )

                validation_record = {
                    "prompt": row["inputs"],
                    "response": row["targets"],
                    "task_id": task_id,
                    "corpus": allocation["corpus"],
                    "task_name": allocation["task_name"],
                    "source_file": str(source_file),
                    "source_index": source_index,
                    "graph_cut_rank": allocation[
                        "graph_cut_rank"
                    ],
                    "stage2_method": (
                        "all_valid_validation"
                    ),
                    "candidate_restricted": False,
                    "split": "validation",
                    "selection_rank": -1,
                    "budget": 0,
                    "included_in_25000": False,
                }

                write_jsonl(
                    validation_handle,
                    validation_record,
                )

                validation_keys.add(key)
                validation_count += 1
                task_validation_count += 1

            print(
                f"[{position:3d}/{EXPECTED_TASK_COUNT}] "
                f"{task_id} "
                f"train={k25}/{k50} "
                f"validation={task_validation_count} "
                f"method={method}",
                flush=True,
            )

    if train25_count != 25_000:
        raise RuntimeError(
            f"Materialized 25K count is "
            f"{train25_count:,}."
        )

    if train50_count != 50_000:
        raise RuntimeError(
            f"Materialized 50K count is "
            f"{train50_count:,}."
        )

    if validation_count != (
        args.expected_validation_count
    ):
        raise RuntimeError(
            f"Materialized validation count is "
            f"{validation_count:,}; expected "
            f"{args.expected_validation_count:,}."
        )

    if not train25_keys.issubset(
        train50_keys
    ):
        raise RuntimeError(
            "25K mixture is not a subset of 50K."
        )

    for final_path, temporary_path in (
        temporary_paths.items()
    ):
        temporary_path.replace(
            final_path
        )

    print()
    print("Loading audit JSONL into Arrow datasets...")

    train25_audit = load_jsonl_dataset(
        train25_path,
        cache_root,
    ).shuffle(
        seed=args.seed
    )

    train50_audit = load_jsonl_dataset(
        train50_path,
        cache_root,
    ).shuffle(
        seed=args.seed
    )

    validation_audit = load_jsonl_dataset(
        validation_path,
        cache_root,
    )

    if len(train25_audit) != 25_000:
        raise RuntimeError(
            "Arrow 25K count mismatch."
        )

    if len(train50_audit) != 50_000:
        raise RuntimeError(
            "Arrow 50K count mismatch."
        )

    if len(validation_audit) != (
        args.expected_validation_count
    ):
        raise RuntimeError(
            "Arrow validation count mismatch."
        )

    audit25 = DatasetDict(
        {
            "train": train25_audit,
            "validation": validation_audit,
        }
    )

    audit50 = DatasetDict(
        {
            "train": train50_audit,
            "validation": validation_audit,
        }
    )

    trainer25 = DatasetDict(
        {
            "train": train25_audit.select_columns(
                ["prompt", "response"]
            ),
            "validation": (
                validation_audit.select_columns(
                    ["prompt", "response"]
                )
            ),
        }
    )

    trainer50 = DatasetDict(
        {
            "train": train50_audit.select_columns(
                ["prompt", "response"]
            ),
            "validation": (
                validation_audit.select_columns(
                    ["prompt", "response"]
                )
            ),
        }
    )

    audit25_path = (
        output_root
        / "audit"
        / "smart_25000"
    )
    audit50_path = (
        output_root
        / "audit"
        / "smart_50000"
    )
    trainer25_path = (
        output_root
        / "trainer"
        / "smart_25000"
    )
    trainer50_path = (
        output_root
        / "trainer"
        / "smart_50000"
    )

    save_dataset_atomic(
        audit25,
        audit25_path,
        args.overwrite,
    )
    save_dataset_atomic(
        audit50,
        audit50_path,
        args.overwrite,
    )
    save_dataset_atomic(
        trainer25,
        trainer25_path,
        args.overwrite,
    )
    save_dataset_atomic(
        trainer50,
        trainer50_path,
        args.overwrite,
    )

    summary = {
        "format_version": 1,
        "stage": "SMART_final_materialization",
        "status": "complete",
        "seed": args.seed,
        "stage2_configuration": (
            stage2_catalog["configuration"]
        ),
        "stage2_method_counts": dict(
            sorted(method_counts.items())
        ),
        "counts": {
            "smart_25000_train": len(
                trainer25["train"]
            ),
            "smart_50000_train": len(
                trainer50["train"]
            ),
            "validation": len(
                trainer25["validation"]
            ),
        },
        "columns": {
            "trainer_train": (
                trainer25["train"].column_names
            ),
            "trainer_validation": (
                trainer25[
                    "validation"
                ].column_names
            ),
            "audit_train": (
                audit25["train"].column_names
            ),
        },
        "nesting": {
            "smart_25000_is_subset_of_50000": True,
            "smart_25000_membership_sha256": (
                membership_hash(
                    train25_keys
                )
            ),
            "smart_50000_membership_sha256": (
                membership_hash(
                    train50_keys
                )
            ),
        },
        "ordered_after_shuffle": {
            "smart_25000_sha256": (
                ordered_dataset_hash(
                    train25_audit
                )
            ),
            "smart_50000_sha256": (
                ordered_dataset_hash(
                    train50_audit
                )
            ),
        },
        "inputs": {
            "manifest": str(manifest_path),
            "manifest_sha256": (
                sha256_file(
                    manifest_path
                )
            ),
            "allocations": str(
                allocations_path
            ),
            "allocations_sha256": (
                sha256_file(
                    allocations_path
                )
            ),
            "stage2_catalog": str(
                catalog_path
            ),
            "stage2_catalog_sha256": (
                sha256_file(
                    catalog_path
                )
            ),
        },
        "outputs": {
            "trainer_smart_25000": str(
                trainer25_path
            ),
            "trainer_smart_50000": str(
                trainer50_path
            ),
            "audit_smart_25000": str(
                audit25_path
            ),
            "audit_smart_50000": str(
                audit50_path
            ),
            "raw_train_25000": str(
                train25_path
            ),
            "raw_train_50000": str(
                train50_path
            ),
            "raw_validation": str(
                validation_path
            ),
        },
    }

    final_summary_path = (
        output_root
        / "materialization_summary.json"
    )

    atomic_write_json(
        summary,
        final_summary_path,
    )

    print()
    print("=== Final mixture summary ===")
    print(
        f"SMART-25K train:  "
        f"{len(trainer25['train']):,}"
    )
    print(
        f"SMART-50K train:  "
        f"{len(trainer50['train']):,}"
    )
    print(
        f"Validation:       "
        f"{len(trainer25['validation']):,}"
    )
    print(
        f"Methods:          "
        f"{dict(method_counts)}"
    )
    print(
        f"25K subset 50K:   "
        f"{train25_keys.issubset(train50_keys)}"
    )
    print(
        f"Trainer columns:  "
        f"{trainer25['train'].column_names}"
    )
    print(
        f"25K dataset:      "
        f"{trainer25_path}"
    )
    print(
        f"50K dataset:      "
        f"{trainer50_path}"
    )
    print(
        f"Summary:          "
        f"{final_summary_path}"
    )
    print("SMART mixture materialization passed.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
