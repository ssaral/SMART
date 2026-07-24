from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

import ijson


EXPECTED_CORPUS_COUNTS = {
    "cot": 18,
    "flan2021": 63,
    "sglue": 17,
    "t0": 193,
    "tulu": 18,
}

FILENAME_PREFIXES = {
    "cot": "cot-task-",
    "flan2021": "flan2021-task-",
    "sglue": "sglue-",
    "t0": "t0-task-",
    "tulu": "tulu-",
}


@dataclass
class SplitStats:
    count: int = 0
    invalid_rows: int = 0
    missing_inputs: int = 0
    missing_targets: int = 0
    empty_inputs: int = 0
    empty_targets: int = 0
    max_input_chars: int = 0
    max_target_chars: int = 0
    total_input_chars: int = 0
    total_target_chars: int = 0


@dataclass
class FileStats:
    corpus: str
    task_id: str
    task_name: str
    source_file: str
    file_size_bytes: int
    top_level_keys: str

    train_count: int
    validation_count: int

    train_invalid_rows: int
    validation_invalid_rows: int

    train_missing_inputs: int
    validation_missing_inputs: int

    train_missing_targets: int
    validation_missing_targets: int

    train_empty_inputs: int
    validation_empty_inputs: int

    train_empty_targets: int
    validation_empty_targets: int

    train_max_input_chars: int
    validation_max_input_chars: int

    train_max_target_chars: int
    validation_max_target_chars: int

    train_avg_input_chars: float
    validation_avg_input_chars: float

    train_avg_target_chars: float
    validation_avg_target_chars: float

    input_types: str
    target_types: str

    errors: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect SMART local JSON task files without modifying them."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Directory containing cot, flan2021, sglue, t0 and tulu.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/mnt/warm_storage/saral/smart/dataset_inventory"),
        help="Directory for the generated reports.",
    )
    return parser.parse_args()


def task_name_from_path(path: Path, corpus: str) -> str:
    stem = path.stem
    prefix = FILENAME_PREFIXES[corpus]

    if not stem.startswith(prefix):
        raise ValueError(
            f"Unexpected filename for corpus '{corpus}': {path.name}. "
            f"Expected prefix '{prefix}'."
        )

    task_name = stem[len(prefix):]

    if not task_name:
        raise ValueError(f"Empty task name derived from {path}")

    return task_name


def read_top_level_keys(path: Path) -> set[str]:
    keys: set[str] = set()

    with path.open("rb") as handle:
        for prefix, event, value in ijson.parse(handle):
            if prefix == "" and event == "map_key":
                keys.add(str(value))

    return keys


def iter_split_rows(path: Path, split: str) -> Iterator[Any]:
    with path.open("rb") as handle:
        yield from ijson.items(handle, f"{split}.item")


def value_type_name(value: Any) -> str:
    if value is None:
        return "null"
    return type(value).__name__


def inspect_split(
    path: Path,
    split: str,
    input_types: Counter[str],
    target_types: Counter[str],
) -> SplitStats:
    stats = SplitStats()

    for row in iter_split_rows(path, split):
        stats.count += 1

        if not isinstance(row, dict):
            stats.invalid_rows += 1
            continue

        has_input = "inputs" in row
        has_target = "targets" in row

        if not has_input:
            stats.missing_inputs += 1

        if not has_target:
            stats.missing_targets += 1

        input_value = row.get("inputs")
        target_value = row.get("targets")

        input_types[value_type_name(input_value)] += 1
        target_types[value_type_name(target_value)] += 1

        if isinstance(input_value, str):
            length = len(input_value)
            stats.total_input_chars += length
            stats.max_input_chars = max(stats.max_input_chars, length)

            if not input_value.strip():
                stats.empty_inputs += 1
        elif input_value is None:
            stats.empty_inputs += 1

        if isinstance(target_value, str):
            length = len(target_value)
            stats.total_target_chars += length
            stats.max_target_chars = max(stats.max_target_chars, length)

            if not target_value.strip():
                stats.empty_targets += 1
        elif target_value is None:
            stats.empty_targets += 1

    return stats


def average(total: int, count: int) -> float:
    return round(total / count, 2) if count else 0.0


def inspect_file(path: Path, corpus: str) -> FileStats:
    errors: list[str] = []
    input_types: Counter[str] = Counter()
    target_types: Counter[str] = Counter()

    try:
        task_name = task_name_from_path(path, corpus)
    except Exception as exc:
        task_name = path.stem
        errors.append(str(exc))

    task_id = f"{corpus}::{task_name}"

    try:
        top_level_keys = read_top_level_keys(path)
    except Exception as exc:
        top_level_keys = set()
        errors.append(f"Unable to read top-level keys: {exc}")

    for required_key in ("train", "validation"):
        if required_key not in top_level_keys:
            errors.append(f"Missing top-level key: {required_key}")

    train_stats = SplitStats()
    validation_stats = SplitStats()

    if "train" in top_level_keys:
        try:
            train_stats = inspect_split(
                path,
                "train",
                input_types,
                target_types,
            )
        except Exception as exc:
            errors.append(f"Unable to inspect train split: {exc}")

    if "validation" in top_level_keys:
        try:
            validation_stats = inspect_split(
                path,
                "validation",
                input_types,
                target_types,
            )
        except Exception as exc:
            errors.append(f"Unable to inspect validation split: {exc}")

    return FileStats(
        corpus=corpus,
        task_id=task_id,
        task_name=task_name,
        source_file=str(path.resolve()),
        file_size_bytes=path.stat().st_size,
        top_level_keys="|".join(sorted(top_level_keys)),

        train_count=train_stats.count,
        validation_count=validation_stats.count,

        train_invalid_rows=train_stats.invalid_rows,
        validation_invalid_rows=validation_stats.invalid_rows,

        train_missing_inputs=train_stats.missing_inputs,
        validation_missing_inputs=validation_stats.missing_inputs,

        train_missing_targets=train_stats.missing_targets,
        validation_missing_targets=validation_stats.missing_targets,

        train_empty_inputs=train_stats.empty_inputs,
        validation_empty_inputs=validation_stats.empty_inputs,

        train_empty_targets=train_stats.empty_targets,
        validation_empty_targets=validation_stats.empty_targets,

        train_max_input_chars=train_stats.max_input_chars,
        validation_max_input_chars=validation_stats.max_input_chars,

        train_max_target_chars=train_stats.max_target_chars,
        validation_max_target_chars=validation_stats.max_target_chars,

        train_avg_input_chars=average(
            train_stats.total_input_chars,
            train_stats.count,
        ),
        validation_avg_input_chars=average(
            validation_stats.total_input_chars,
            validation_stats.count,
        ),

        train_avg_target_chars=average(
            train_stats.total_target_chars,
            train_stats.count,
        ),
        validation_avg_target_chars=average(
            validation_stats.total_target_chars,
            validation_stats.count,
        ),

        input_types=json.dumps(dict(sorted(input_types.items()))),
        target_types=json.dumps(dict(sorted(target_types.items()))),

        errors=" | ".join(errors),
    )


def write_csv(path: Path, rows: list[FileStats]) -> None:
    if not rows:
        raise RuntimeError("No rows available to write.")

    fieldnames = list(asdict(rows[0]).keys())

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(asdict(row))


def main() -> int:
    args = parse_args()
    data_root = args.data_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if not data_root.is_dir():
        print(f"ERROR: Data root does not exist: {data_root}", file=sys.stderr)
        return 1

    all_stats: list[FileStats] = []
    discovered_counts: dict[str, int] = {}

    for corpus, expected_count in EXPECTED_CORPUS_COUNTS.items():
        corpus_dir = data_root / corpus

        if not corpus_dir.is_dir():
            print(
                f"ERROR: Missing corpus directory: {corpus_dir}",
                file=sys.stderr,
            )
            return 1

        files = sorted(corpus_dir.glob("*.json"))
        discovered_counts[corpus] = len(files)

        print(
            f"{corpus:12s}: {len(files):4d} files "
            f"(expected {expected_count})",
            flush=True,
        )

        for index, path in enumerate(files, start=1):
            print(
                f"  [{index:03d}/{len(files):03d}] {path.name}",
                flush=True,
            )
            all_stats.append(inspect_file(path, corpus))

    task_ids = [row.task_id for row in all_stats]
    duplicate_task_ids = sorted(
        task_id
        for task_id, count in Counter(task_ids).items()
        if count > 1
    )

    corpus_summaries: dict[str, dict[str, Any]] = {}

    for corpus in EXPECTED_CORPUS_COUNTS:
        rows = [row for row in all_stats if row.corpus == corpus]

        corpus_summaries[corpus] = {
            "task_count": len(rows),
            "train_examples": sum(row.train_count for row in rows),
            "validation_examples": sum(
                row.validation_count for row in rows
            ),
            "total_examples": sum(
                row.train_count + row.validation_count for row in rows
            ),
            "files_with_errors": sum(bool(row.errors) for row in rows),
            "largest_train_task": (
                max(rows, key=lambda row: row.train_count).task_id
                if rows else None
            ),
            "largest_train_task_count": (
                max((row.train_count for row in rows), default=0)
            ),
        }

    total_train = sum(row.train_count for row in all_stats)
    total_validation = sum(row.validation_count for row in all_stats)

    files_with_errors = [
        {
            "task_id": row.task_id,
            "source_file": row.source_file,
            "errors": row.errors,
        }
        for row in all_stats
        if row.errors
    ]

    largest_tasks = sorted(
        all_stats,
        key=lambda row: row.train_count,
        reverse=True,
    )[:30]

    summary = {
        "data_root": str(data_root),
        "expected_task_count": sum(EXPECTED_CORPUS_COUNTS.values()),
        "discovered_task_count": len(all_stats),
        "expected_corpus_counts": EXPECTED_CORPUS_COUNTS,
        "discovered_corpus_counts": discovered_counts,
        "unique_task_ids": len(set(task_ids)),
        "duplicate_task_ids": duplicate_task_ids,
        "total_train_examples": total_train,
        "total_validation_examples": total_validation,
        "total_examples": total_train + total_validation,
        "files_with_errors": len(files_with_errors),
        "corpora": corpus_summaries,
        "largest_training_tasks": [
            {
                "rank": rank,
                "task_id": row.task_id,
                "corpus": row.corpus,
                "train_count": row.train_count,
                "validation_count": row.validation_count,
                "file_size_bytes": row.file_size_bytes,
                "train_max_input_chars": row.train_max_input_chars,
                "train_max_target_chars": row.train_max_target_chars,
            }
            for rank, row in enumerate(largest_tasks, start=1)
        ],
    }

    write_csv(output_root / "task_manifest.csv", all_stats)

    with (output_root / "summary.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(summary, handle, indent=2)

    with (output_root / "schema_errors.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(files_with_errors, handle, indent=2)

    print()
    print("=== Dataset summary ===")
    print(f"Tasks discovered:       {len(all_stats)}")
    print(f"Unique task IDs:        {len(set(task_ids))}")
    print(f"Training examples:      {total_train:,}")
    print(f"Validation examples:    {total_validation:,}")
    print(f"Files with errors:      {len(files_with_errors)}")
    print(f"Duplicate task IDs:     {len(duplicate_task_ids)}")

    print()
    print("=== Largest training tasks ===")

    for rank, row in enumerate(largest_tasks[:20], start=1):
        print(
            f"{rank:2d}. {row.task_id:70s} "
            f"{row.train_count:12,d}"
        )

    print()
    print(f"Reports written to: {output_root}")

    expected_total = sum(EXPECTED_CORPUS_COUNTS.values())

    checks_passed = (
        len(all_stats) == expected_total
        and len(set(task_ids)) == expected_total
        and not files_with_errors
        and discovered_counts == EXPECTED_CORPUS_COUNTS
    )

    if not checks_passed:
        print()
        print(
            "Dataset inspection completed, but one or more checks failed.",
            file=sys.stderr,
        )
        return 2

    print()
    print("All structural checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

chmod +x inspect_dataset.py
