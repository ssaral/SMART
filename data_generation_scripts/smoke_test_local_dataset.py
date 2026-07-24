from __future__ import annotations

import argparse
from pathlib import Path

from local_dataset import (
    count_valid_rows,
    iter_valid_rows,
    load_task_catalog,
    task_lookup,
    write_catalog_json,
)


EXPECTED_VALID_TRAIN = 6_266_471
EXPECTED_VALID_VALIDATION = 183_870

DEFAULT_SAMPLE_TASKS = [
    "sglue::axg",
    "tulu::wildguardmix",
    "sglue::qqp",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke-test the local SMART dataset adapter."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--catalog-output",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--sample-task-id",
        action="append",
        default=None,
        help=(
            "Task ID to sample. May be supplied multiple times. "
            "Defaults to three representative tasks."
        ),
    )
    parser.add_argument(
        "--samples-per-task",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--preview-chars",
        type=int,
        default=160,
    )
    parser.add_argument(
        "--verify-count-task-id",
        default="sglue::axg",
        help=(
            "One small task whose train and validation counts will "
            "be rescanned and compared with the manifest."
        ),
    )
    return parser.parse_args()


def text_preview(value: str, limit: int) -> str:
    value = value.replace("\n", "\\n")

    if len(value) <= limit:
        return value

    return value[:limit] + "...[truncated]"


def main() -> int:
    args = parse_args()

    tasks = load_task_catalog(args.manifest)
    lookup = task_lookup(tasks)

    total_train = sum(
        task.valid_train_count for task in tasks
    )
    total_validation = sum(
        task.valid_validation_count for task in tasks
    )

    print("=== Local SMART task catalog ===")
    print(f"Tasks:                    {len(tasks)}")
    print(f"Valid training rows:      {total_train:,}")
    print(f"Valid validation rows:    {total_validation:,}")

    if total_train != EXPECTED_VALID_TRAIN:
        raise RuntimeError(
            f"Expected {EXPECTED_VALID_TRAIN:,} valid training "
            f"rows but catalog contains {total_train:,}."
        )

    if total_validation != EXPECTED_VALID_VALIDATION:
        raise RuntimeError(
            f"Expected {EXPECTED_VALID_VALIDATION:,} valid "
            f"validation rows but catalog contains "
            f"{total_validation:,}."
        )

    zero_validation_tasks = [
        task.task_id
        for task in tasks
        if task.valid_validation_count == 0
    ]

    print()
    print("Tasks with zero valid validation rows:")

    for current_task_id in zero_validation_tasks:
        print(f"  - {current_task_id}")

    verify_task_id = args.verify_count_task_id

    if verify_task_id not in lookup:
        raise KeyError(
            f"Verification task not found: {verify_task_id}"
        )

    verify_task = lookup[verify_task_id]

    print()
    print(f"=== Full count verification: {verify_task_id} ===")

    scanned_train = count_valid_rows(
        verify_task,
        "train",
    )
    scanned_validation = count_valid_rows(
        verify_task,
        "validation",
    )

    print(
        f"Train:      manifest={verify_task.valid_train_count:,}, "
        f"scanned={scanned_train:,}"
    )
    print(
        "Validation: "
        f"manifest={verify_task.valid_validation_count:,}, "
        f"scanned={scanned_validation:,}"
    )

    if scanned_train != verify_task.valid_train_count:
        raise RuntimeError(
            f"Training count mismatch for {verify_task_id}"
        )

    if (
        scanned_validation
        != verify_task.valid_validation_count
    ):
        raise RuntimeError(
            f"Validation count mismatch for {verify_task_id}"
        )

    sample_task_ids = (
        args.sample_task_id
        if args.sample_task_id
        else DEFAULT_SAMPLE_TASKS
    )

    for current_task_id in sample_task_ids:
        if current_task_id not in lookup:
            raise KeyError(
                f"Sample task not found: {current_task_id}"
            )

        task = lookup[current_task_id]

        print()
        print(f"=== Sample: {current_task_id} ===")
        print(f"Source: {task.source_file}")
        print(f"Valid train rows: {task.valid_train_count:,}")
        print(
            f"Valid validation rows: "
            f"{task.valid_validation_count:,}"
        )

        sample_count = 0

        for example in iter_valid_rows(task, "train"):
            print()
            print(
                f"source_index={example['source_index']}"
            )
            print(
                "inputs="
                + text_preview(
                    example["inputs"],
                    args.preview_chars,
                )
            )
            print(
                "targets="
                + text_preview(
                    example["targets"],
                    args.preview_chars,
                )
            )

            sample_count += 1

            if sample_count >= args.samples_per_task:
                break

        if sample_count == 0:
            raise RuntimeError(
                f"No valid training samples found for "
                f"{current_task_id}"
            )

    write_catalog_json(tasks, args.catalog_output)

    print()
    print(f"Catalog written to: {args.catalog_output}")
    print("Local dataset adapter smoke test passed.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
