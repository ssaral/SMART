We’ll start with **Step 1 only: initialize `smart_v2`, inventory the source data, and produce a provisional four-template audit**.

The template labels are not frozen yet. This pass assigns deterministic provisional labels and records ambiguous cases for inspection.

## 1. Create the clean directory layout

Inside the container:

```bash
cd /data/saral/wdir/smart_v2 || exit 1

mkdir -p \
  configs \
  environment \
  src/data \
  src/embeddings \
  src/stage1 \
  src/stage2 \
  src/mixtures \
  src/training \
  src/evaluation

touch \
  src/__init__.py \
  src/data/__init__.py \
  src/embeddings/__init__.py \
  src/stage1/__init__.py \
  src/stage2/__init__.py \
  src/mixtures/__init__.py \
  src/training/__init__.py \
  src/evaluation/__init__.py

mkdir -p \
  /mnt/warm_storage/saral/smart_v2/cache \
  /mnt/warm_storage/saral/smart_v2/manifests \
  /mnt/warm_storage/saral/smart_v2/author_format \
  /mnt/warm_storage/saral/smart_v2/embeddings \
  /mnt/warm_storage/saral/smart_v2/stage1 \
  /mnt/warm_storage/saral/smart_v2/stage2 \
  /mnt/warm_storage/saral/smart_v2/mixtures \
  /mnt/warm_storage/saral/smart_v2/checkpoints \
  /mnt/warm_storage/saral/smart_v2/evaluations \
  /mnt/warm_storage/saral/smart_v2/logs \
  /mnt/warm_storage/saral/smart_v2/tmp
```

Create the environment file:

```bash
cat > smart_v2_env.sh <<'SH'
#!/usr/bin/env bash

unset TRANSFORMERS_CACHE

export SMART_PROJECT_ROOT=/data/saral/wdir/smart_v2
export SMART_OUTPUT_ROOT=/mnt/warm_storage/saral/smart_v2

export HF_HOME=$SMART_OUTPUT_ROOT/cache/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$SMART_OUTPUT_ROOT/cache/datasets
export SENTENCE_TRANSFORMERS_HOME=$SMART_OUTPUT_ROOT/cache/sentence_transformers
export TORCH_HOME=$SMART_OUTPUT_ROOT/cache/torch
export XDG_CACHE_HOME=$SMART_OUTPUT_ROOT/cache/xdg
export TMPDIR=$SMART_OUTPUT_ROOT/tmp

export TOKENIZERS_PARALLELISM=false

mkdir -p \
  "$HF_HOME" \
  "$HF_HUB_CACHE" \
  "$HF_DATASETS_CACHE" \
  "$SENTENCE_TRANSFORMERS_HOME" \
  "$TORCH_HOME" \
  "$XDG_CACHE_HOME" \
  "$TMPDIR"
SH

chmod +x smart_v2_env.sh
source ./smart_v2_env.sh
```

---

## 2. Create the provisional template classifier

Create `/data/saral/wdir/smart_v2/src/data/template_classifier.py`:

```bash
cat > src/data/template_classifier.py <<'PY'
"""Deterministic provisional template classification.

The classifier maps each rendered prompt to one of the four SMART
template categories:

    zs_opt
    zs_noopt
    fs_opt
    fs_noopt

It does not rewrite prompts or create synthetic templates.

The result is provisional. Ambiguous classifications are surfaced for
manual inspection before the author-format datasets are materialized.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any


CLASSIFIER_VERSION = "smart-v2-template-heuristic-v1"

TEMPLATE_TYPES = (
    "zs_opt",
    "zs_noopt",
    "fs_opt",
    "fs_noopt",
)


# Strong option evidence such as:
#   (A) first option
#   B. second option
#   C) third option
LETTERED_OPTION_RE = re.compile(
    r"(?im)^\s*(?:\(([A-H])\)|([A-H])[.)])\s+\S"
)

# Same markers when several options appear on one line.
INLINE_OPTION_RE = re.compile(
    r"(?i)(?:^|\s)(?:\([A-H]\)|[A-H][.)])\s+\S"
)

OPTION_HEADER_RE = re.compile(
    r"(?im)^\s*(?:"
    r"answer\s+choices?"
    r"|choices?"
    r"|options?"
    r"|possible\s+answers?"
    r")\s*[:\-]"
)

NUMBERED_ITEM_RE = re.compile(
    r"(?m)^\s*\d{1,2}[.)]\s+\S"
)

TRUE_FALSE_RE = re.compile(
    r"(?i)\b(?:true\s*(?:/|or)\s*false|false\s*(?:/|or)\s*true)\b"
)

WEAK_OPTION_RE = re.compile(
    r"(?i)\b(?:"
    r"choose\s+(?:one|the\s+best|from)"
    r"|select\s+(?:one|the\s+best)"
    r"|which\s+of\s+the\s+following"
    r"|multiple[\s-]?choice"
    r"|answer\s+choice"
    r")\b"
)


# Few-shot evidence. A completed answer marker must contain text after
# the marker. A trailing unresolved "Answer:" is therefore not counted.
QUESTION_MARKER_RE = re.compile(
    r"(?im)^\s*(?:"
    r"Q(?:uestion)?"
    r"|Input"
    r"|Problem"
    r"|Prompt"
    r")\s*[:\-]"
)

COMPLETED_ANSWER_RE = re.compile(
    r"(?im)^\s*(?:"
    r"A(?:nswer)?"
    r"|Output"
    r"|Response"
    r"|Label"
    r")\s*[:\-]\s*\S"
)

EXAMPLE_MARKER_RE = re.compile(
    r"(?im)^\s*(?:"
    r"example"
    r"|demonstration"
    r"|demo"
    r")\s*(?:\d+)?\s*[:\-]"
)

WEAK_FEWSHOT_RE = re.compile(
    r"(?i)\b(?:"
    r"few[\s-]?shot"
    r"|worked\s+example"
    r"|here\s+are\s+some\s+examples"
    r"|following\s+examples"
    r"|use\s+the\s+examples"
    r")\b"
)


@dataclass(frozen=True)
class TemplateClassification:
    template_type: str
    shot_type: str
    option_type: str
    ambiguous: bool
    fewshot_strong: bool
    fewshot_weak: bool
    options_strong: bool
    options_weak: bool
    question_marker_count: int
    completed_answer_count: int
    example_marker_count: int
    lettered_option_count: int
    inline_option_count: int
    numbered_item_count: int
    option_header_count: int
    true_false_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def classify_prompt(prompt: str) -> TemplateClassification:
    """Assign one provisional SMART template type to a prompt."""

    if not isinstance(prompt, str):
        raise TypeError(
            f"Prompt must be a string, received {type(prompt).__name__}."
        )

    question_marker_count = len(
        QUESTION_MARKER_RE.findall(prompt)
    )
    completed_answer_count = len(
        COMPLETED_ANSWER_RE.findall(prompt)
    )
    example_marker_count = len(
        EXAMPLE_MARKER_RE.findall(prompt)
    )

    lettered_option_count = len(
        LETTERED_OPTION_RE.findall(prompt)
    )
    inline_option_count = len(
        INLINE_OPTION_RE.findall(prompt)
    )
    numbered_item_count = len(
        NUMBERED_ITEM_RE.findall(prompt)
    )
    option_header_count = len(
        OPTION_HEADER_RE.findall(prompt)
    )
    true_false_count = len(
        TRUE_FALSE_RE.findall(prompt)
    )

    weak_fewshot_phrase = bool(
        WEAK_FEWSHOT_RE.search(prompt)
    )
    weak_option_phrase = bool(
        WEAK_OPTION_RE.search(prompt)
    )

    # One completed demonstration followed by a second question is
    # sufficient for few-shot classification.
    fewshot_strong = (
        completed_answer_count >= 2
        or example_marker_count >= 2
        or (
            completed_answer_count >= 1
            and question_marker_count >= 2
        )
        or (
            example_marker_count >= 1
            and completed_answer_count >= 1
        )
    )

    fewshot_weak = (
        not fewshot_strong
        and (
            weak_fewshot_phrase
            or example_marker_count == 1
            or (
                completed_answer_count >= 1
                and question_marker_count >= 1
            )
        )
    )

    maximum_letter_option_count = max(
        lettered_option_count,
        inline_option_count,
    )

    options_strong = (
        maximum_letter_option_count >= 2
        or true_false_count >= 1
        or (
            option_header_count >= 1
            and numbered_item_count >= 2
        )
        or (
            option_header_count >= 1
            and maximum_letter_option_count >= 1
        )
    )

    options_weak = (
        not options_strong
        and (
            weak_option_phrase
            or option_header_count >= 1
            or maximum_letter_option_count == 1
        )
    )

    shot_type = "fs" if fewshot_strong else "zs"
    option_type = "opt" if options_strong else "noopt"

    template_type = f"{shot_type}_{option_type}"

    if template_type not in TEMPLATE_TYPES:
        raise RuntimeError(
            f"Unexpected template type: {template_type}"
        )

    return TemplateClassification(
        template_type=template_type,
        shot_type=shot_type,
        option_type=option_type,
        ambiguous=bool(fewshot_weak or options_weak),
        fewshot_strong=fewshot_strong,
        fewshot_weak=fewshot_weak,
        options_strong=options_strong,
        options_weak=options_weak,
        question_marker_count=question_marker_count,
        completed_answer_count=completed_answer_count,
        example_marker_count=example_marker_count,
        lettered_option_count=lettered_option_count,
        inline_option_count=inline_option_count,
        numbered_item_count=numbered_item_count,
        option_header_count=option_header_count,
        true_false_count=true_false_count,
    )
PY
```

---

## 3. Create the inventory and template-audit script

Create `/data/saral/wdir/smart_v2/src/data/audit_dataset.py`:

```bash
cat > src/data/audit_dataset.py <<'PY'
"""Inventory the SMART-v2 source collection and audit template types."""

from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

import ijson

from src.data.template_classifier import (
    CLASSIFIER_VERSION,
    TEMPLATE_TYPES,
    classify_prompt,
)


EXPECTED_CORPUS_TASKS = {
    "cot": 18,
    "flan2021": 63,
    "sglue": 17,
    "t0": 193,
    "tulu": 18,
}

EXPECTED_TOTAL_TASKS = 309

EXPECTED_COUNTS = {
    "source_train": 6_317_088,
    "valid_train": 6_266_471,
    "excluded_train": 50_617,
    "source_validation": 185_623,
    "valid_validation": 183_870,
    "excluded_validation": 1_753,
}

FILENAME_PREFIXES = {
    "cot": "cot-task-",
    "flan2021": "flan2021-task-",
    "sglue": "sglue-",
    "t0": "t0-task-",
    "tulu": "tulu-",
}

SAMPLE_COUNT_PER_BUCKET = 3
PREVIEW_CHARS = 500


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inventory the local SMART-v2 JSON collection and "
            "produce a provisional template audit."
        )
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=SAMPLE_COUNT_PER_BUCKET,
    )
    parser.add_argument(
        "--preview-chars",
        type=int,
        default=PREVIEW_CHARS,
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


def task_name_from_path(
    path: Path,
    corpus: str,
) -> str:
    prefix = FILENAME_PREFIXES[corpus]

    if not path.stem.startswith(prefix):
        raise ValueError(
            f"Unexpected filename {path.name!r} for corpus "
            f"{corpus!r}; expected prefix {prefix!r}."
        )

    task_name = path.stem[len(prefix):]

    if not task_name:
        raise ValueError(
            f"Empty task name derived from {path}."
        )

    return task_name


def iter_rows(
    path: Path,
    split: str,
) -> Iterator[Any]:
    with path.open("rb") as handle:
        yield from ijson.items(
            handle,
            f"{split}.item",
        )


def invalid_reasons(row: Any) -> list[str]:
    if not isinstance(row, dict):
        return ["row_not_object"]

    reasons: list[str] = []

    inputs = row.get("inputs")
    targets = row.get("targets")

    if not isinstance(inputs, str):
        reasons.append("inputs_not_string")
    elif not inputs.strip():
        reasons.append("inputs_empty")

    if not isinstance(targets, str):
        reasons.append("targets_not_string")
    elif not targets.strip():
        reasons.append("targets_empty")

    return reasons


def preview(
    value: Any,
    maximum_chars: int,
) -> Any:
    if not isinstance(value, str):
        return {
            "type": type(value).__name__,
            "repr": repr(value)[:maximum_chars],
        }

    escaped = value.replace("\n", "\\n")

    if len(escaped) <= maximum_chars:
        return escaped

    return (
        escaped[:maximum_chars]
        + "...[truncated]"
    )


def stable_sample_score(
    task_id: str,
    split: str,
    source_index: int,
    template_type: str,
    ambiguous: bool,
) -> int:
    payload = (
        f"23|{task_id}|{split}|{source_index}|"
        f"{template_type}|{int(ambiguous)}"
    ).encode("utf-8")

    digest = hashlib.sha256(payload).digest()

    return int.from_bytes(
        digest[:8],
        byteorder="big",
        signed=False,
    )


def add_minhash_sample(
    buckets: dict[
        tuple[str, str, str, bool],
        list[tuple[int, dict[str, Any]]],
    ],
    bucket: tuple[str, str, str, bool],
    score: int,
    record: dict[str, Any],
    maximum_samples: int,
) -> None:
    """Keep records with the lowest deterministic hash scores."""

    heap = buckets.setdefault(bucket, [])

    item = (-score, record)

    if len(heap) < maximum_samples:
        heapq.heappush(heap, item)
        return

    current_largest_score = -heap[0][0]

    if score < current_largest_score:
        heapq.heapreplace(heap, item)


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


def main() -> int:
    args = parse_args()

    if args.sample_count <= 0:
        raise ValueError(
            "--sample-count must be positive."
        )

    data_root = args.data_root.resolve()
    output_root = args.output_root.resolve()

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    source_files_path = (
        output_root / "source_files.csv"
    )
    task_manifest_path = (
        output_root / "task_manifest.csv"
    )
    invalid_rows_path = (
        output_root / "invalid_rows.jsonl"
    )
    template_samples_path = (
        output_root / "template_samples.jsonl"
    )
    summary_path = (
        output_root / "dataset_summary.json"
    )
    classifier_path = (
        output_root / "template_classifier.json"
    )

    source_file_rows: list[dict[str, Any]] = []
    task_manifest_rows: list[dict[str, Any]] = []

    task_ids: set[str] = set()
    global_template_counts: dict[
        str,
        Counter[str],
    ] = {
        "train": Counter(),
        "validation": Counter(),
    }

    global_ambiguous_counts = Counter()
    global_reason_counts = Counter()

    totals = Counter()

    sample_buckets: dict[
        tuple[str, str, str, bool],
        list[tuple[int, dict[str, Any]]],
    ] = {}

    start_time = time.perf_counter()

    with invalid_rows_path.open(
        "w",
        encoding="utf-8",
    ) as invalid_handle:

        for corpus, expected_task_count in (
            EXPECTED_CORPUS_TASKS.items()
        ):
            corpus_dir = data_root / corpus

            if not corpus_dir.is_dir():
                raise FileNotFoundError(
                    f"Missing corpus directory: {corpus_dir}"
                )

            paths = sorted(
                corpus_dir.glob("*.json")
            )

            if len(paths) != expected_task_count:
                raise RuntimeError(
                    f"Corpus {corpus!r} contains "
                    f"{len(paths)} JSON files; expected "
                    f"{expected_task_count}."
                )

            print(
                f"Processing {corpus}: "
                f"{len(paths)} task files",
                flush=True,
            )

            for file_position, path in enumerate(
                paths,
                start=1,
            ):
                task_name = task_name_from_path(
                    path,
                    corpus,
                )
                task_id = f"{corpus}::{task_name}"

                if task_id in task_ids:
                    raise RuntimeError(
                        f"Duplicate task ID: {task_id}"
                    )

                task_ids.add(task_id)

                source_file_rows.append(
                    {
                        "corpus": corpus,
                        "task_id": task_id,
                        "task_name": task_name,
                        "source_file": str(
                            path.resolve()
                        ),
                        "size_bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                    }
                )

                task_row: dict[str, Any] = {
                    "corpus": corpus,
                    "task_id": task_id,
                    "task_name": task_name,
                    "source_file": str(
                        path.resolve()
                    ),
                }

                for split in (
                    "train",
                    "validation",
                ):
                    source_count = 0
                    valid_count = 0
                    excluded_count = 0
                    ambiguous_count = 0

                    template_counts = Counter()
                    reason_counts = Counter()

                    for source_index, row in enumerate(
                        iter_rows(path, split)
                    ):
                        source_count += 1

                        reasons = invalid_reasons(row)

                        if reasons:
                            excluded_count += 1
                            reason_counts.update(reasons)
                            global_reason_counts.update(
                                reasons
                            )

                            invalid_record = {
                                "corpus": corpus,
                                "task_id": task_id,
                                "task_name": task_name,
                                "source_file": str(
                                    path.resolve()
                                ),
                                "split": split,
                                "source_index": (
                                    source_index
                                ),
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

                            invalid_handle.write(
                                json.dumps(
                                    invalid_record,
                                    ensure_ascii=False,
                                )
                                + "\n"
                            )
                            continue

                        valid_count += 1

                        classification = (
                            classify_prompt(
                                row["inputs"]
                            )
                        )

                        template_counts[
                            classification.template_type
                        ] += 1

                        global_template_counts[
                            split
                        ][
                            classification.template_type
                        ] += 1

                        if classification.ambiguous:
                            ambiguous_count += 1
                            global_ambiguous_counts[
                                split
                            ] += 1

                        bucket = (
                            task_id,
                            split,
                            classification.template_type,
                            classification.ambiguous,
                        )

                        sample_score = (
                            stable_sample_score(
                                task_id=task_id,
                                split=split,
                                source_index=source_index,
                                template_type=(
                                    classification.template_type
                                ),
                                ambiguous=(
                                    classification.ambiguous
                                ),
                            )
                        )

                        sample_record = {
                            "corpus": corpus,
                            "task_id": task_id,
                            "task_name": task_name,
                            "source_file": str(
                                path.resolve()
                            ),
                            "split": split,
                            "source_index": source_index,
                            "template_type": (
                                classification.template_type
                            ),
                            "ambiguous": (
                                classification.ambiguous
                            ),
                            "classification": (
                                classification.to_dict()
                            ),
                            "inputs_preview": preview(
                                row["inputs"],
                                args.preview_chars,
                            ),
                            "targets_preview": preview(
                                row["targets"],
                                args.preview_chars,
                            ),
                        }

                        add_minhash_sample(
                            buckets=sample_buckets,
                            bucket=bucket,
                            score=sample_score,
                            record=sample_record,
                            maximum_samples=(
                                args.sample_count
                            ),
                        )

                    task_row[
                        f"source_{split}_count"
                    ] = source_count
                    task_row[
                        f"valid_{split}_count"
                    ] = valid_count
                    task_row[
                        f"excluded_{split}_count"
                    ] = excluded_count
                    task_row[
                        f"{split}_ambiguous_count"
                    ] = ambiguous_count

                    for template_type in TEMPLATE_TYPES:
                        task_row[
                            f"{split}_{template_type}_count"
                        ] = template_counts[
                            template_type
                        ]

                    for reason in (
                        "row_not_object",
                        "inputs_not_string",
                        "inputs_empty",
                        "targets_not_string",
                        "targets_empty",
                    ):
                        task_row[
                            f"{split}_{reason}_count"
                        ] = reason_counts[reason]

                    totals[
                        f"source_{split}"
                    ] += source_count
                    totals[
                        f"valid_{split}"
                    ] += valid_count
                    totals[
                        f"excluded_{split}"
                    ] += excluded_count

                task_manifest_rows.append(task_row)

                if (
                    file_position % 10 == 0
                    or file_position == len(paths)
                ):
                    print(
                        f"  {corpus}: "
                        f"{file_position}/{len(paths)}",
                        flush=True,
                    )

    if len(task_ids) != EXPECTED_TOTAL_TASKS:
        raise RuntimeError(
            f"Expected {EXPECTED_TOTAL_TASKS} unique tasks; "
            f"found {len(task_ids)}."
        )

    source_file_fieldnames = [
        "corpus",
        "task_id",
        "task_name",
        "source_file",
        "size_bytes",
        "sha256",
    ]

    with source_files_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=source_file_fieldnames,
        )
        writer.writeheader()
        writer.writerows(source_file_rows)

    task_manifest_fieldnames = [
        "corpus",
        "task_id",
        "task_name",
        "source_file",
    ]

    for split in (
        "train",
        "validation",
    ):
        task_manifest_fieldnames.extend(
            [
                f"source_{split}_count",
                f"valid_{split}_count",
                f"excluded_{split}_count",
                f"{split}_ambiguous_count",
            ]
        )

        task_manifest_fieldnames.extend(
            [
                f"{split}_{template_type}_count"
                for template_type in TEMPLATE_TYPES
            ]
        )

        task_manifest_fieldnames.extend(
            [
                f"{split}_row_not_object_count",
                f"{split}_inputs_not_string_count",
                f"{split}_inputs_empty_count",
                f"{split}_targets_not_string_count",
                f"{split}_targets_empty_count",
            ]
        )

    with task_manifest_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=task_manifest_fieldnames,
        )
        writer.writeheader()
        writer.writerows(task_manifest_rows)

    flattened_samples: list[
        tuple[int, dict[str, Any]]
    ] = []

    for heap in sample_buckets.values():
        for negative_score, record in heap:
            flattened_samples.append(
                (-negative_score, record)
            )

    flattened_samples.sort(
        key=lambda item: (
            item[1]["corpus"],
            item[1]["task_id"],
            item[1]["split"],
            item[1]["template_type"],
            item[1]["ambiguous"],
            item[0],
        )
    )

    with template_samples_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        for score, record in flattened_samples:
            record = {
                "sample_hash_score": score,
                **record,
            }

            handle.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

    elapsed_seconds = (
        time.perf_counter() - start_time
    )

    mismatches: list[dict[str, Any]] = []

    for key, expected_value in (
        EXPECTED_COUNTS.items()
    ):
        actual_value = int(totals[key])

        if actual_value != expected_value:
            mismatches.append(
                {
                    "field": key,
                    "expected": expected_value,
                    "actual": actual_value,
                }
            )

    summary = {
        "format_version": 1,
        "stage": "smart_v2_source_inventory_and_template_audit",
        "status": (
            "complete"
            if not mismatches
            else "count_mismatch"
        ),
        "data_root": str(data_root),
        "task_count": len(task_ids),
        "corpus_task_counts": (
            EXPECTED_CORPUS_TASKS
        ),
        "row_validity_policy": (
            "inputs and targets must both be strings "
            "and non-empty after strip()."
        ),
        "counts": {
            key: int(value)
            for key, value in sorted(
                totals.items()
            )
        },
        "expected_counts": EXPECTED_COUNTS,
        "count_mismatches": mismatches,
        "invalid_reason_counts": dict(
            sorted(
                global_reason_counts.items()
            )
        ),
        "template_classifier": {
            "version": CLASSIFIER_VERSION,
            "status": "provisional",
            "official_template_types": list(
                TEMPLATE_TYPES
            ),
            "policy": (
                "Classify existing rendered prompts only. "
                "Do not duplicate rows, rewrite prompts, "
                "or synthesize options/demonstrations."
            ),
            "ambiguous_policy": (
                "Weak evidence does not change the provisional "
                "zs/noopt label, but the row is marked ambiguous "
                "for manual audit."
            ),
        },
        "template_counts": {
            split: {
                template_type: int(
                    global_template_counts[
                        split
                    ][template_type]
                )
                for template_type in TEMPLATE_TYPES
            }
            for split in (
                "train",
                "validation",
            )
        },
        "ambiguous_counts": {
            split: int(
                global_ambiguous_counts[split]
            )
            for split in (
                "train",
                "validation",
            )
        },
        "elapsed_seconds": elapsed_seconds,
        "outputs": {
            "source_files": str(
                source_files_path
            ),
            "task_manifest": str(
                task_manifest_path
            ),
            "invalid_rows": str(
                invalid_rows_path
            ),
            "template_samples": str(
                template_samples_path
            ),
            "template_classifier": str(
                classifier_path
            ),
        },
    }

    classifier_metadata = {
        "classifier_version": (
            CLASSIFIER_VERSION
        ),
        "template_types": list(
            TEMPLATE_TYPES
        ),
        "status": "provisional",
        "fewshot_rule": (
            "Strong few-shot evidence requires multiple completed "
            "answer markers, multiple explicit examples, or at "
            "least one completed answer plus at least two question "
            "markers."
        ),
        "option_rule": (
            "Strong option evidence requires at least two explicit "
            "lettered options, a True/False choice, or an option "
            "header with multiple enumerated choices."
        ),
        "ambiguity_rule": (
            "Weak textual evidence is recorded as ambiguous but "
            "does not independently trigger fs or opt."
        ),
    }

    atomic_write_json(
        classifier_metadata,
        classifier_path,
    )
    atomic_write_json(
        summary,
        summary_path,
    )

    print()
    print("=== SMART-v2 dataset audit ===")
    print(f"Tasks:                 {len(task_ids)}")
    print(
        f"Source train rows:     "
        f"{totals['source_train']:,}"
    )
    print(
        f"Valid train rows:      "
        f"{totals['valid_train']:,}"
    )
    print(
        f"Excluded train rows:   "
        f"{totals['excluded_train']:,}"
    )
    print(
        f"Source validation:     "
        f"{totals['source_validation']:,}"
    )
    print(
        f"Valid validation:      "
        f"{totals['valid_validation']:,}"
    )
    print(
        f"Excluded validation:   "
        f"{totals['excluded_validation']:,}"
    )

    print()
    print("Provisional train template counts:")

    for template_type in TEMPLATE_TYPES:
        print(
            f"  {template_type:10s} "
            f"{global_template_counts['train'][template_type]:,}"
        )

    print(
        f"  ambiguous  "
        f"{global_ambiguous_counts['train']:,}"
    )

    print()
    print(f"Summary:  {summary_path}")
    print(f"Manifest: {task_manifest_path}")
    print(f"Samples:  {template_samples_path}")

    if mismatches:
        print()
        print(
            "ERROR: Source counts differ from the frozen "
            "smart_v1 inventory.",
            file=sys.stderr,
        )

        for mismatch in mismatches:
            print(
                f"  {mismatch['field']}: "
                f"expected={mismatch['expected']:,}, "
                f"actual={mismatch['actual']:,}",
                file=sys.stderr,
            )

        return 2

    print()
    print("Source inventory checks passed.")
    print(
        "Template labels remain provisional pending sample audit."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
PY
```

Compile both files:

```bash
cd /data/saral/wdir/smart_v2 || exit 1

python3 -m py_compile \
  src/data/template_classifier.py \
  src/data/audit_dataset.py
```

---

## 4. Run the audit

```bash
cd /data/saral/wdir/smart_v2 || exit 1
source ./smart_v2_env.sh

python3 -m src.data.audit_dataset \
  --data-root /data/saral/wdir/smart_v2/data \
  --output-root /mnt/warm_storage/saral/smart_v2/manifests \
  --sample-count 3 \
  --preview-chars 500 \
  2>&1 | tee \
  /mnt/warm_storage/saral/smart_v2/logs/dataset_template_audit.log
```

Expected source counts:

```text
Tasks:                 309
Source train rows:     6,317,088
Valid train rows:      6,266,471
Excluded train rows:   50,617
Source validation:     185,623
Valid validation:      183,870
Excluded validation:   1,753
```

---

## 5. Print the most important template-audit results

```bash
python3 - <<'PY'
import csv
import json
from pathlib import Path

root = Path(
    "/mnt/warm_storage/saral/smart_v2/manifests"
)

with (root / "dataset_summary.json").open(
    encoding="utf-8"
) as handle:
    summary = json.load(handle)

print("Status:", summary["status"])
print("Tasks:", summary["task_count"])
print()
print("Train template counts:")

for key, value in summary["template_counts"]["train"].items():
    print(f"  {key:10s} {value:,}")

print(
    "  ambiguous ",
    f"{summary['ambiguous_counts']['train']:,}",
)

with (root / "task_manifest.csv").open(
    encoding="utf-8",
    newline="",
) as handle:
    rows = list(csv.DictReader(handle))

rows.sort(
    key=lambda row: int(
        row["train_ambiguous_count"]
    ),
    reverse=True,
)

print("\nTasks with most ambiguous train rows:")

for row in rows[:20]:
    print(
        f"{int(row['train_ambiguous_count']):8,d}  "
        f"{row['task_id']}"
    )
PY
```

Generated artifacts:

```text
/mnt/warm_storage/saral/smart_v2/manifests/
├── dataset_summary.json
├── invalid_rows.jsonl
├── source_files.csv
├── task_manifest.csv
├── template_classifier.json
└── template_samples.jsonl
```
