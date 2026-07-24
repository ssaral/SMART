"""Local JSON dataset adapter for the SMART replication.

This module does not modify source JSON files. It uses the cleaned task
manifest to expose the local collection as 309 SMART tasks.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import ijson


EXPECTED_TASK_COUNT = 309

CORPUS_ORDER = {
    "cot": 0,
    "flan2021": 1,
    "sglue": 2,
    "t0": 3,
    "tulu": 4,
}


@dataclass(frozen=True)
class TaskSpec:
    """Metadata for one local SMART task."""

    task_index: int
    corpus: str
    task_id: str
    task_name: str
    source_file: str
    valid_train_count: int
    valid_validation_count: int
    template_type: str = "default"

    @property
    def path(self) -> Path:
        return Path(self.source_file)


def is_valid_example(row: Any) -> bool:
    """Return True only for usable prompt-response examples."""

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


def _require_int(row: dict[str, str], key: str) -> int:
    try:
        value = int(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid integer value for column {key!r}: {row.get(key)!r}"
        ) from exc

    if value < 0:
        raise ValueError(f"Negative value for {key!r}: {value}")

    return value


def load_task_catalog(manifest_path: str | Path) -> list[TaskSpec]:
    """Load and validate the cleaned 309-task manifest."""

    manifest_path = Path(manifest_path).resolve()

    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Clean task manifest does not exist: {manifest_path}"
        )

    raw_rows: list[dict[str, str]] = []

    with manifest_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle)

        required_columns = {
            "corpus",
            "task_id",
            "task_name",
            "source_file",
            "valid_train_count",
            "valid_validation_count",
        }

        missing_columns = required_columns - set(
            reader.fieldnames or []
        )

        if missing_columns:
            raise ValueError(
                "Clean manifest is missing columns: "
                + ", ".join(sorted(missing_columns))
            )

        raw_rows.extend(reader)

    raw_rows.sort(
        key=lambda row: (
            CORPUS_ORDER.get(row["corpus"], 999),
            row["task_id"],
        )
    )

    tasks: list[TaskSpec] = []
    seen_task_ids: set[str] = set()

    for task_index, row in enumerate(raw_rows):
        corpus = row["corpus"]
        current_task_id = row["task_id"]
        source_file = str(Path(row["source_file"]).resolve())

        if corpus not in CORPUS_ORDER:
            raise ValueError(
                f"Unexpected corpus {corpus!r} for task "
                f"{current_task_id!r}"
            )

        if current_task_id in seen_task_ids:
            raise ValueError(
                f"Duplicate task ID in clean manifest: "
                f"{current_task_id}"
            )

        seen_task_ids.add(current_task_id)

        if not Path(source_file).is_file():
            raise FileNotFoundError(
                f"Source JSON file does not exist: {source_file}"
            )

        valid_train_count = _require_int(
            row,
            "valid_train_count",
        )
        valid_validation_count = _require_int(
            row,
            "valid_validation_count",
        )

        if valid_train_count == 0:
            raise ValueError(
                f"Task has no valid training examples: "
                f"{current_task_id}"
            )

        tasks.append(
            TaskSpec(
                task_index=task_index,
                corpus=corpus,
                task_id=current_task_id,
                task_name=row["task_name"],
                source_file=source_file,
                valid_train_count=valid_train_count,
                valid_validation_count=valid_validation_count,
            )
        )

    if len(tasks) != EXPECTED_TASK_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_TASK_COUNT} tasks but loaded "
            f"{len(tasks)}"
        )

    if len(seen_task_ids) != EXPECTED_TASK_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_TASK_COUNT} unique task IDs but "
            f"loaded {len(seen_task_ids)}"
        )

    return tasks


def task_lookup(
    tasks: Sequence[TaskSpec],
) -> dict[str, TaskSpec]:
    """Create a task_id -> TaskSpec mapping."""

    lookup = {task.task_id: task for task in tasks}

    if len(lookup) != len(tasks):
        raise ValueError("Task catalog contains duplicate task IDs.")

    return lookup


def iter_source_rows(
    task: TaskSpec,
    split: str,
) -> Iterator[tuple[int, Any]]:
    """Stream source rows from one task and split."""

    if split not in {"train", "validation"}:
        raise ValueError(
            f"Unsupported split {split!r}; "
            "expected 'train' or 'validation'."
        )

    with task.path.open("rb") as handle:
        for source_index, row in enumerate(
            ijson.items(handle, f"{split}.item")
        ):
            yield source_index, row


def iter_valid_rows(
    task: TaskSpec,
    split: str,
) -> Iterator[dict[str, Any]]:
    """Stream canonical valid rows for one task."""

    for source_index, row in iter_source_rows(task, split):
        if not is_valid_example(row):
            continue

        yield {
            "task_index": task.task_index,
            "task_id": task.task_id,
            "task_name": task.task_name,
            "corpus": task.corpus,
            "template_type": task.template_type,
            "split": split,
            "source_file": task.source_file,
            "source_index": source_index,
            "inputs": row["inputs"],
            "targets": row["targets"],
        }


def count_valid_rows(task: TaskSpec, split: str) -> int:
    """Rescan and count valid rows in one task split."""

    return sum(1 for _ in iter_valid_rows(task, split))


def write_catalog_json(
    tasks: Sequence[TaskSpec],
    output_path: str | Path,
) -> None:
    """Write the immutable local task catalog."""

    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "format_version": 1,
        "task_count": len(tasks),
        "template_policy": (
            "Each JSON file is one SMART task with one default "
            "template pool."
        ),
        "row_policy": (
            "inputs and targets must both be non-empty strings "
            "after strip()."
        ),
        "total_valid_train_rows": sum(
            task.valid_train_count for task in tasks
        ),
        "total_valid_validation_rows": sum(
            task.valid_validation_count for task in tasks
        ),
        "tasks": [asdict(task) for task in tasks],
    }

    temporary_path = output_path.with_suffix(
        output_path.suffix + ".tmp"
    )

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

    temporary_path.replace(output_path)
