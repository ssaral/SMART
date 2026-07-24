from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

import ijson


EXPECTED_COUNTS = {
    "cot": 18,
    "flan2021": 63,
    "sglue": 17,
    "t0": 193,
    "tulu": 18,
}

PREFIXES = {
    "cot": "cot-task-",
    "flan2021": "flan2021-task-",
    "sglue": "sglue-",
    "t0": "t0-task-",
    "tulu": "tulu-",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create reproducible valid-row statistics and exclusion "
            "records for the local SMART dataset."
        )
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "/mnt/warm_storage/saral/smart/prepared_data"
        ),
    )
    parser.add_argument(
        "--preview-chars",
        type=int,
        default=300,
    )
    return parser.parse_args()


def task_name(path: Path, corpus: str) -> str:
    prefix = PREFIXES[corpus]

    if not path.stem.startswith(prefix):
        raise ValueError(
            f"Unexpected filename {path.name}; "
            f"expected prefix {prefix!r}"
        )

    value = path.stem[len(prefix):]

    if not value:
        raise ValueError(f"Empty task name for {path}")

    return value


def task_id(path: Path, corpus: str) -> str:
    return f"{corpus}::{task_name(path, corpus)}"


def iter_rows(path: Path, split: str) -> Iterator[Any]:
    with path.open("rb") as handle:
        yield from ijson.items(handle, f"{split}.item")


def invalid_reasons(row: Any) -> list[str]:
    if not isinstance(row, dict):
        return ["row_not_object"]

    reasons: list[str] = []

    input_value = row.get("inputs")
    target_value = row.get("targets")

    if not isinstance(input_value, str):
        reasons.append("inputs_not_string")
    elif not input_value.strip():
        reasons.append("inputs_empty")

    if not isinstance(target_value, str):
        reasons.append("targets_not_string")
    elif not target_value.strip():
        reasons.append("targets_empty")

    return reasons


def preview(value: Any, limit: int) -> Any:
    if not isinstance(value, str):
        return {
            "type": type(value).__name__,
            "repr": repr(value)[:limit],
        }

    escaped = value.replace("\n", "\\n")

    if len(escaped) <= limit:
        return escaped

    return escaped[:limit] + "...[truncated]"


def main() -> int:
    args = parse_args()

    data_root = args.data_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    manifest_path = output_root / "clean_task_manifest.csv"
    exclusions_path = output_root / "excluded_rows.jsonl"
    summary_path = output_root / "cleaning_summary.json"

    manifest_rows: list[dict[str, Any]] = []
    global_reason_counts: Counter[str] = Counter()

    total_source_train = 0
    total_source_validation = 0
    total_valid_train = 0
    total_valid_validation = 0
    total_excluded_train = 0
    total_excluded_validation = 0

    task_count = 0
    task_ids: set[str] = set()

    with exclusions_path.open(
        "w",
        encoding="utf-8",
    ) as exclusions_handle:

        for corpus, expected_count in EXPECTED_COUNTS.items():
            corpus_dir = data_root / corpus

            if not corpus_dir.is_dir():
                print(
                    f"ERROR: Missing corpus directory: {corpus_dir}",
                    file=sys.stderr,
                )
                return 1

            paths = sorted(corpus_dir.glob("*.json"))

            if len(paths) != expected_count:
                print(
                    f"ERROR: {corpus} contains {len(paths)} files; "
                    f"expected {expected_count}.",
                    file=sys.stderr,
                )
                return 1

            for path in paths:
                current_task_id = task_id(path, corpus)

                if current_task_id in task_ids:
                    print(
                        f"ERROR: Duplicate task ID: {current_task_id}",
                        file=sys.stderr,
                    )
                    return 1

                task_ids.add(current_task_id)
                task_count += 1

                task_stats: dict[str, Any] = {
                    "corpus": corpus,
                    "task_id": current_task_id,
                    "task_name": task_name(path, corpus),
                    "source_file": str(path.resolve()),
                }

                for split in ("train", "validation"):
                    source_count = 0
                    valid_count = 0
                    excluded_count = 0
                    reason_counts: Counter[str] = Counter()

                    for source_index, row in enumerate(
                        iter_rows(path, split)
                    ):
                        source_count += 1
                        reasons = invalid_reasons(row)

                        if not reasons:
                            valid_count += 1
                            continue

                        excluded_count += 1
                        reason_counts.update(reasons)
                        global_reason_counts.update(reasons)

                        record = {
                            "corpus": corpus,
                            "task_id": current_task_id,
                            "source_file": str(path.resolve()),
                            "split": split,
                            "source_index": source_index,
                            "reasons": reasons,
                            "inputs_preview": preview(
                                row.get("inputs")
                                if isinstance(row, dict)
                                else None,
                                args.preview_chars,
                            ),
                            "targets_preview": preview(
                                row.get("targets")
                                if isinstance(row, dict)
                                else None,
                                args.preview_chars,
                            ),
                        }

                        exclusions_handle.write(
                            json.dumps(
                                record,
                                ensure_ascii=False,
                            )
                            + "\n"
                        )

                    task_stats[f"source_{split}_count"] = source_count
                    task_stats[f"valid_{split}_count"] = valid_count
                    task_stats[f"excluded_{split}_count"] = (
                        excluded_count
                    )

                    for reason in (
                        "row_not_object",
                        "inputs_not_string",
                        "inputs_empty",
                        "targets_not_string",
                        "targets_empty",
                    ):
                        task_stats[
                            f"{split}_{reason}_count"
                        ] = reason_counts[reason]

                    if split == "train":
                        total_source_train += source_count
                        total_valid_train += valid_count
                        total_excluded_train += excluded_count
                    else:
                        total_source_validation += source_count
                        total_valid_validation += valid_count
                        total_excluded_validation += excluded_count

                manifest_rows.append(task_stats)

    if not manifest_rows:
        print("ERROR: No task files were processed.", file=sys.stderr)
        return 1

    fieldnames = list(manifest_rows[0].keys())

    with manifest_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    empty_train_tasks = sorted(
        row["task_id"]
        for row in manifest_rows
        if row["valid_train_count"] == 0
    )

    empty_validation_tasks = sorted(
        row["task_id"]
        for row in manifest_rows
        if row["valid_validation_count"] == 0
    )

    affected_tasks = sorted(
        row["task_id"]
        for row in manifest_rows
        if (
            row["excluded_train_count"] > 0
            or row["excluded_validation_count"] > 0
        )
    )

    summary = {
        "data_root": str(data_root),
        "policy": {
            "valid_inputs": (
                "inputs must be a non-empty string after strip()"
            ),
            "valid_targets": (
                "targets must be a non-empty string after strip()"
            ),
            "source_files_modified": False,
        },
        "task_count": task_count,
        "unique_task_ids": len(task_ids),
        "affected_task_count": len(affected_tasks),
        "affected_tasks": affected_tasks,
        "train": {
            "source_rows": total_source_train,
            "valid_rows": total_valid_train,
            "excluded_rows": total_excluded_train,
        },
        "validation": {
            "source_rows": total_source_validation,
            "valid_rows": total_valid_validation,
            "excluded_rows": total_excluded_validation,
        },
        "exclusion_reason_counts": dict(
            sorted(global_reason_counts.items())
        ),
        "tasks_with_zero_valid_train_rows": empty_train_tasks,
        "tasks_with_zero_valid_validation_rows": (
            empty_validation_tasks
        ),
        "outputs": {
            "manifest": str(manifest_path),
            "excluded_rows": str(exclusions_path),
        },
    }

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            summary,
            handle,
            indent=2,
            ensure_ascii=False,
        )

    print("=== Cleaning summary ===")
    print(f"Tasks:                    {task_count}")
    print(f"Affected tasks:           {len(affected_tasks)}")
    print()
    print(f"Source train rows:        {total_source_train:,}")
    print(f"Valid train rows:         {total_valid_train:,}")
    print(f"Excluded train rows:      {total_excluded_train:,}")
    print()
    print(
        f"Source validation rows:   "
        f"{total_source_validation:,}"
    )
    print(
        f"Valid validation rows:    "
        f"{total_valid_validation:,}"
    )
    print(
        f"Excluded validation rows: "
        f"{total_excluded_validation:,}"
    )
    print()
    print(
        "Tasks with no valid train rows:      "
        f"{len(empty_train_tasks)}"
    )
    print(
        "Tasks with no valid validation rows: "
        f"{len(empty_validation_tasks)}"
    )
    print()
    print(f"Outputs written to: {output_root}")

    checks_passed = (
        task_count == 309
        and len(task_ids) == 309
        and total_source_train
        == total_valid_train + total_excluded_train
        and total_source_validation
        == total_valid_validation
        + total_excluded_validation
        and not empty_train_tasks
    )

    if not checks_passed:
        print(
            "ERROR: One or more preparation checks failed.",
            file=sys.stderr,
        )
        return 2

    print("All preparation checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
