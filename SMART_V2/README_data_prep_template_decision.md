The source inventory is clean. The template classifier is **not ready to freeze** yet: 925,798 of 6,266,471 valid training rows are ambiguous, about **14.8%**.

That does not mean the provisional labels are wrong. It means the weak-signal rules are firing heavily on tasks such as MNLI, HellaSwag, QQP, QNLI, and RACE. These often use answer-choice formats that our first regular expressions may not recognize strongly enough.

We should not build the author-format datasets yet. The authors’ code divides each task budget among the four template pools, so incorrect template labels would directly change the Stage 2 candidate pools and selected mixture. 

# Step 2 — Template stability and review packet

This step does not change any label. It determines:

* whether each JSON task is mostly one template category;
* which tasks genuinely contain mixed template types;
* why rows are marked ambiguous;
* whether we should use row-level labels or task-level/template-family rules.

## 2.1 Create the review script

````bash
cd /data/saral/wdir/smart_v2 || exit 1

cat > src/data/build_template_review.py <<'PY'
"""Build a compact review packet from the provisional template audit."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


TEMPLATE_TYPES = (
    "zs_opt",
    "zs_noopt",
    "fs_opt",
    "fs_noopt",
)

REPRESENTATIVE_TASKS = (
    "flan2021::glue_mnli_2.0.0",
    "cot::cot_esnli",
    "flan2021::hellaswag_1.1.0",
    "flan2021::glue_qqp_2.0.0",
    "flan2021::glue_qnli_2.0.0",
    "tulu::flanv2",
    "t0::race_middle_Select_the_best_answer_generate_span_",
    "t0::wiqa_which_of_the_following_is_the_supposed_perturbation",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--top-ambiguous",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--top-mixed",
        type=int,
        default=15,
    )
    parser.add_argument(
        "--samples-per-task",
        type=int,
        default=8,
    )
    return parser.parse_args()


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        return list(csv.DictReader(handle))


def atomic_write_json(
    payload: Any,
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


def percentage(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def main() -> int:
    args = parse_args()
    root = args.manifest_root.resolve()

    manifest_path = root / "task_manifest.csv"
    samples_path = root / "template_samples.jsonl"

    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)

    if not samples_path.is_file():
        raise FileNotFoundError(samples_path)

    manifest_rows = load_csv(manifest_path)
    stability_rows: list[dict[str, Any]] = []

    for row in manifest_rows:
        valid_count = int(row["valid_train_count"])
        ambiguous_count = int(
            row["train_ambiguous_count"]
        )

        template_counts = {
            template_type: int(
                row[f"train_{template_type}_count"]
            )
            for template_type in TEMPLATE_TYPES
        }

        dominant_type = max(
            TEMPLATE_TYPES,
            key=lambda value: template_counts[value],
        )
        dominant_count = template_counts[
            dominant_type
        ]

        active_types = sum(
            count > 0
            for count in template_counts.values()
        )

        stability_rows.append(
            {
                "corpus": row["corpus"],
                "task_id": row["task_id"],
                "task_name": row["task_name"],
                "valid_train_count": valid_count,
                **{
                    f"{template_type}_count": (
                        template_counts[template_type]
                    )
                    for template_type in TEMPLATE_TYPES
                },
                "dominant_template_type": dominant_type,
                "dominant_template_count": dominant_count,
                "dominant_template_fraction": (
                    dominant_count / valid_count
                    if valid_count
                    else 0.0
                ),
                "active_template_type_count": active_types,
                "ambiguous_count": ambiguous_count,
                "ambiguous_fraction": (
                    ambiguous_count / valid_count
                    if valid_count
                    else 0.0
                ),
            }
        )

    stability_path = root / "template_stability.csv"

    with stability_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                stability_rows[0].keys()
            ),
        )
        writer.writeheader()
        writer.writerows(stability_rows)

    homogeneous_99 = sum(
        row["dominant_template_fraction"] >= 0.99
        for row in stability_rows
    )
    homogeneous_95 = sum(
        row["dominant_template_fraction"] >= 0.95
        for row in stability_rows
    )
    homogeneous_90 = sum(
        row["dominant_template_fraction"] >= 0.90
        for row in stability_rows
    )

    highly_ambiguous = sum(
        row["ambiguous_fraction"] >= 0.25
        for row in stability_rows
    )

    samples_by_task: dict[
        str,
        list[dict[str, Any]],
    ] = defaultdict(list)

    with samples_path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        for line in handle:
            if not line.strip():
                continue

            record = json.loads(line)

            if (
                record.get("split") == "train"
                and record.get("ambiguous") is True
            ):
                samples_by_task[
                    record["task_id"]
                ].append(record)

    top_ambiguous_rows = sorted(
        stability_rows,
        key=lambda row: (
            row["ambiguous_count"],
            row["ambiguous_fraction"],
        ),
        reverse=True,
    )[: args.top_ambiguous]

    # Ignore tiny tasks when identifying genuinely mixed tasks.
    mixed_candidates = [
        row
        for row in stability_rows
        if row["valid_train_count"] >= 100
    ]

    top_mixed_rows = sorted(
        mixed_candidates,
        key=lambda row: (
            row["dominant_template_fraction"],
            -row["valid_train_count"],
        ),
    )[: args.top_mixed]

    review_task_ids: list[str] = []

    for task_id in REPRESENTATIVE_TASKS:
        if task_id not in review_task_ids:
            review_task_ids.append(task_id)

    for row in top_ambiguous_rows + top_mixed_rows:
        if row["task_id"] not in review_task_ids:
            review_task_ids.append(row["task_id"])

    stability_lookup = {
        row["task_id"]: row
        for row in stability_rows
    }

    review_path = root / "template_review.md"

    with review_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        handle.write(
            "# SMART-v2 provisional template review\n\n"
        )
        handle.write(
            "These labels are provisional. This report does not "
            "modify the dataset or freeze template assignments.\n\n"
        )

        handle.write("## Collection-level stability\n\n")
        handle.write(
            f"- Tasks with ≥99% one provisional type: "
            f"{homogeneous_99}/309\n"
        )
        handle.write(
            f"- Tasks with ≥95% one provisional type: "
            f"{homogeneous_95}/309\n"
        )
        handle.write(
            f"- Tasks with ≥90% one provisional type: "
            f"{homogeneous_90}/309\n"
        )
        handle.write(
            f"- Tasks with ≥25% ambiguous rows: "
            f"{highly_ambiguous}/309\n\n"
        )

        for task_id in review_task_ids:
            if task_id not in stability_lookup:
                continue

            row = stability_lookup[task_id]

            handle.write(f"## `{task_id}`\n\n")
            handle.write(
                f"- Valid train rows: "
                f"{row['valid_train_count']:,}\n"
            )
            handle.write(
                f"- Dominant provisional type: "
                f"`{row['dominant_template_type']}` "
                f"({percentage(row['dominant_template_fraction'])})\n"
            )
            handle.write(
                f"- Ambiguous rows: "
                f"{row['ambiguous_count']:,} "
                f"({percentage(row['ambiguous_fraction'])})\n"
            )
            handle.write(
                "- Template counts: "
                + ", ".join(
                    f"`{template_type}`="
                    f"{row[f'{template_type}_count']:,}"
                    for template_type in TEMPLATE_TYPES
                )
                + "\n\n"
            )

            task_samples = sorted(
                samples_by_task.get(task_id, []),
                key=lambda record: (
                    record["template_type"],
                    record["source_index"],
                ),
            )[: args.samples_per_task]

            if not task_samples:
                handle.write(
                    "_No ambiguous samples retained for this task._\n\n"
                )
                continue

            for sample_number, sample in enumerate(
                task_samples,
                start=1,
            ):
                classification = sample[
                    "classification"
                ]

                handle.write(
                    f"### Sample {sample_number}: "
                    f"`{sample['template_type']}`\n\n"
                )
                handle.write(
                    f"- Source index: "
                    f"{sample['source_index']}\n"
                )
                handle.write(
                    "- Signals: "
                    f"fewshot_strong="
                    f"{classification['fewshot_strong']}, "
                    f"fewshot_weak="
                    f"{classification['fewshot_weak']}, "
                    f"options_strong="
                    f"{classification['options_strong']}, "
                    f"options_weak="
                    f"{classification['options_weak']}\n"
                )
                handle.write(
                    "- Marker counts: "
                    f"questions="
                    f"{classification['question_marker_count']}, "
                    f"completed_answers="
                    f"{classification['completed_answer_count']}, "
                    f"examples="
                    f"{classification['example_marker_count']}, "
                    f"lettered_options="
                    f"{classification['lettered_option_count']}, "
                    f"inline_options="
                    f"{classification['inline_option_count']}, "
                    f"numbered_items="
                    f"{classification['numbered_item_count']}, "
                    f"option_headers="
                    f"{classification['option_header_count']}, "
                    f"true_false="
                    f"{classification['true_false_count']}\n\n"
                )

                handle.write("**Prompt preview**\n\n")
                handle.write("```text\n")
                handle.write(
                    sample["inputs_preview"]
                )
                handle.write("\n```\n\n")

                handle.write("**Target preview**\n\n")
                handle.write("```text\n")
                handle.write(
                    sample["targets_preview"]
                )
                handle.write("\n```\n\n")

    summary = {
        "status": "complete",
        "task_count": len(stability_rows),
        "tasks_with_dominant_fraction_at_least_0_99": (
            homogeneous_99
        ),
        "tasks_with_dominant_fraction_at_least_0_95": (
            homogeneous_95
        ),
        "tasks_with_dominant_fraction_at_least_0_90": (
            homogeneous_90
        ),
        "tasks_with_ambiguous_fraction_at_least_0_25": (
            highly_ambiguous
        ),
        "top_ambiguous_tasks": [
            {
                "task_id": row["task_id"],
                "ambiguous_count": row["ambiguous_count"],
                "ambiguous_fraction": (
                    row["ambiguous_fraction"]
                ),
            }
            for row in top_ambiguous_rows
        ],
        "top_mixed_tasks": [
            {
                "task_id": row["task_id"],
                "dominant_template_type": (
                    row["dominant_template_type"]
                ),
                "dominant_template_fraction": (
                    row["dominant_template_fraction"]
                ),
                "valid_train_count": (
                    row["valid_train_count"]
                ),
            }
            for row in top_mixed_rows
        ],
        "review_task_ids": review_task_ids,
        "outputs": {
            "stability_csv": str(stability_path),
            "review_markdown": str(review_path),
        },
    }

    summary_path = (
        root / "template_stability_summary.json"
    )
    atomic_write_json(summary, summary_path)

    print("=== Template stability report ===")
    print(
        "Tasks with >=99% one type:",
        f"{homogeneous_99}/309",
    )
    print(
        "Tasks with >=95% one type:",
        f"{homogeneous_95}/309",
    )
    print(
        "Tasks with >=90% one type:",
        f"{homogeneous_90}/309",
    )
    print(
        "Tasks with >=25% ambiguous rows:",
        f"{highly_ambiguous}/309",
    )
    print()
    print("Stability CSV:", stability_path)
    print("Review packet:", review_path)
    print("Summary JSON:", summary_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
PY

python3 -m py_compile \
  src/data/build_template_review.py
````

## 2.2 Run it

```bash
cd /data/saral/wdir/smart_v2 || exit 1

python3 -m src.data.build_template_review \
  --manifest-root /mnt/warm_storage/saral/smart_v2/manifests \
  --top-ambiguous 20 \
  --top-mixed 15 \
  --samples-per-task 8 \
  2>&1 | tee \
  /mnt/warm_storage/saral/smart_v2/logs/template_stability.log
```

## 2.3 Print the summary

```bash
cat \
  /mnt/warm_storage/saral/smart_v2/manifests/template_stability_summary.json
```

Inspect the first review sections:

```bash
sed -n '1,260p' \
  /mnt/warm_storage/saral/smart_v2/manifests/template_review.md
```

Generated files:

```text
/mnt/warm_storage/saral/smart_v2/manifests/
├── template_stability.csv
├── template_stability_summary.json
└── template_review.md
```

The key decision after this review will be whether to use:

1. row-level template classification;
2. one template label per source JSON task;
3. task-specific parsing rules for mixed datasets such as `tulu::flanv2`.

Do not regenerate embeddings or author-format datasets until this decision is frozen.
