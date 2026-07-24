from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import ijson


CORPORA = ("cot", "flan2021", "sglue", "t0", "tulu")

PREFIXES = {
    "cot": "cot-task-",
    "flan2021": "flan2021-task-",
    "sglue": "sglue-",
    "t0": "t0-task-",
    "tulu": "tulu-",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "/mnt/warm_storage/saral/smart/"
            "dataset_inventory/empty_row_audit.json"
        ),
    )
    parser.add_argument(
        "--samples-per-category",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--preview-chars",
        type=int,
        default=500,
    )
    return parser.parse_args()


def get_task_id(path: Path, corpus: str) -> str:
    prefix = PREFIXES[corpus]
    stem = path.stem

    if not stem.startswith(prefix):
        raise ValueError(
            f"Unexpected filename {path.name}; expected prefix {prefix}"
        )

    return f"{corpus}::{stem[len(prefix):]}"


def preview(value: Any, limit: int) -> Any:
    if not isinstance(value, str):
        return {
            "type": type(value).__name__,
            "value": repr(value)[:limit],
        }

    if len(value) <= limit:
        return value

    return value[:limit] + "...[truncated]"


def is_empty(value: Any) -> bool:
    return not isinstance(value, str) or not value.strip()


def iter_rows(path: Path, split: str):
    with path.open("rb") as handle:
        yield from ijson.items(handle, f"{split}.item")


def main() -> int:
    args = parse_args()
    data_root = args.data_root.resolve()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "data_root": str(data_root),
        "policy": (
            "A row is considered empty when inputs or targets is not "
            "a string, or when its stripped value is empty."
        ),
        "totals": {
            "train_empty_inputs": 0,
            "train_empty_targets": 0,
            "validation_empty_inputs": 0,
            "validation_empty_targets": 0,
        },
        "affected_tasks": {},
    }

    for corpus in CORPORA:
        corpus_dir = data_root / corpus

        for path in sorted(corpus_dir.glob("*.json")):
            task_id = get_task_id(path, corpus)

            counts = defaultdict(int)
            samples = defaultdict(list)

            for split in ("train", "validation"):
                for source_index, row in enumerate(
                    iter_rows(path, split)
                ):
                    input_value = (
                        row.get("inputs")
                        if isinstance(row, dict)
                        else None
                    )
                    target_value = (
                        row.get("targets")
                        if isinstance(row, dict)
                        else None
                    )

                    empty_input = is_empty(input_value)
                    empty_target = is_empty(target_value)

                    if empty_input:
                        key = f"{split}_empty_inputs"
                        counts[key] += 1

                        if (
                            len(samples[key])
                            < args.samples_per_category
                        ):
                            samples[key].append(
                                {
                                    "source_index": source_index,
                                    "inputs": preview(
                                        input_value,
                                        args.preview_chars,
                                    ),
                                    "targets": preview(
                                        target_value,
                                        args.preview_chars,
                                    ),
                                }
                            )

                    if empty_target:
                        key = f"{split}_empty_targets"
                        counts[key] += 1

                        if (
                            len(samples[key])
                            < args.samples_per_category
                        ):
                            samples[key].append(
                                {
                                    "source_index": source_index,
                                    "inputs": preview(
                                        input_value,
                                        args.preview_chars,
                                    ),
                                    "targets": preview(
                                        target_value,
                                        args.preview_chars,
                                    ),
                                }
                            )

            if counts:
                report["affected_tasks"][task_id] = {
                    "source_file": str(path.resolve()),
                    "counts": dict(counts),
                    "samples": dict(samples),
                }

                for key, value in counts.items():
                    report["totals"][key] += value

    report["affected_task_count"] = len(
        report["affected_tasks"]
    )

    with output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    print("=== Empty-row audit ===")
    print(f"Affected tasks: {report['affected_task_count']}")

    for key, value in report["totals"].items():
        print(f"{key:30s} {value:,}")

    print()
    print(f"Report written to: {output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
