Stage 1 Graph Cut is complete and verified.

## Step 8 — Generate exact 25K/50K task and template budgets

This reproduces the authors’ sequence:

1. Apply second-order Taylor softmax to Graph Cut gains.
2. Split the global budget using floor plus cyclic remainder.
3. Cap by task capacity and cyclically redistribute overflow.
4. Split every task budget across template pools.

The released implementation uses:

```python
1 + x + x**2 / 2
```

followed by cyclic remainder allocation.  Task-capacity redistribution is also cyclic in selected-task insertion order. 

Because our primary baseline has only `zs_noopt`, the authors’ template-budget logic should produce:

```text
[zs_opt, zs_noopt, fs_opt, fs_noopt]
[0,      task_budget, 0,      0]
```

for every task. 

### 8.1 Create the budget script

```bash
cd /data/saral/wdir/smart_v2 || exit 1

cat > src/stage1/build_allocations.py <<'PY'
"""Generate exact SMART task and task-template budgets.

This ports the released SMART behavior for:

- Taylor softmax over Graph Cut marginal gains
- floor plus cyclic remainder integerization
- task-capacity capping and cyclic redistribution
- task-to-template budget splitting

The primary SMART-v2 baseline exposes one template bucket:
zs_noopt.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch


EXPECTED_TASK_COUNT = 309

TEMPLATE_TYPES = (
    "zs_opt",
    "zs_noopt",
    "fs_opt",
    "fs_noopt",
)

EXPECTED_TEMPLATE_TYPE = "zs_noopt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--graph-cut-order",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--submixture-config",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--task-indices-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--budgets",
        type=int,
        nargs="+",
        default=[25_000, 50_000],
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
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


def atomic_write_pickle(
    payload: Any,
    path: Path,
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")

    with temporary.open("wb") as handle:
        pickle.dump(
            payload,
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    temporary.replace(path)


def load_submixtures(
    path: Path,
) -> list[str]:
    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        payload = json.load(handle)

    submixtures = payload.get("submixtures")

    if not isinstance(submixtures, list):
        raise ValueError(
            "submixture config must contain a list."
        )

    if not submixtures:
        raise ValueError(
            "submixture list cannot be empty."
        )

    if len(submixtures) != len(set(submixtures)):
        raise ValueError(
            "submixture list contains duplicates."
        )

    return submixtures


def load_task_indices(
    submixtures: list[str],
    root: Path,
) -> tuple[
    dict[str, dict[str, dict[str, list[int]]]],
    list[str],
    dict[str, int],
    dict[str, str],
]:
    all_task_indices: dict[
        str,
        dict[str, dict[str, list[int]]],
    ] = {}

    all_tasks: list[str] = []
    task_totals: dict[str, int] = {}
    task_corpus: dict[str, str] = {}

    for corpus in submixtures:
        path = root / f"{corpus}.pkl"

        if not path.is_file():
            raise FileNotFoundError(path)

        with path.open("rb") as handle:
            mapping = pickle.load(handle)

        if not isinstance(mapping, dict):
            raise TypeError(
                f"{path}: expected dictionary."
            )

        all_task_indices[corpus] = mapping

        for task_id, template_mapping in mapping.items():
            if task_id in task_totals:
                raise RuntimeError(
                    f"Duplicate task ID: {task_id}"
                )

            if not isinstance(
                template_mapping,
                dict,
            ):
                raise TypeError(
                    f"{task_id}: template mapping "
                    "is not a dictionary."
                )

            available_templates = set(
                template_mapping
            )

            if available_templates != {
                EXPECTED_TEMPLATE_TYPE
            }:
                raise RuntimeError(
                    f"{task_id}: expected only "
                    f"{EXPECTED_TEMPLATE_TYPE}; found "
                    f"{sorted(available_templates)}."
                )

            total = 0

            for template_type in TEMPLATE_TYPES:
                total += len(
                    template_mapping.get(
                        template_type,
                        [],
                    )
                )

            if total <= 0:
                raise RuntimeError(
                    f"{task_id}: task has no examples."
                )

            all_tasks.append(task_id)
            task_totals[task_id] = total
            task_corpus[task_id] = corpus

    if len(all_tasks) != EXPECTED_TASK_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_TASK_COUNT} tasks; "
            f"found {len(all_tasks)}."
        )

    if len(set(all_tasks)) != EXPECTED_TASK_COUNT:
        raise RuntimeError(
            "Task list contains duplicate task IDs."
        )

    return (
        all_task_indices,
        all_tasks,
        task_totals,
        task_corpus,
    )


def load_graph_cut_order(
    path: Path,
) -> list[dict[str, Any]]:
    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        raw_rows = list(csv.DictReader(handle))

    required = {
        "graph_cut_rank",
        "task_index",
        "task_id",
        "corpus",
        "valid_train_count",
        "marginal_gain",
    }

    if not raw_rows:
        raise RuntimeError(
            "Graph Cut order is empty."
        )

    missing = required - set(raw_rows[0])

    if missing:
        raise ValueError(
            "Graph Cut CSV is missing columns: "
            + ", ".join(sorted(missing))
        )

    rows: list[dict[str, Any]] = []

    for raw in raw_rows:
        rows.append(
            {
                "graph_cut_rank": int(
                    raw["graph_cut_rank"]
                ),
                "task_index": int(
                    raw["task_index"]
                ),
                "task_id": raw["task_id"],
                "corpus": raw["corpus"],
                "valid_train_count": int(
                    raw["valid_train_count"]
                ),
                "marginal_gain": float(
                    raw["marginal_gain"]
                ),
            }
        )

    rows.sort(
        key=lambda row: row["graph_cut_rank"]
    )

    if len(rows) != EXPECTED_TASK_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_TASK_COUNT} Graph Cut rows; "
            f"found {len(rows)}."
        )

    expected_ranks = list(
        range(1, EXPECTED_TASK_COUNT + 1)
    )

    actual_ranks = [
        row["graph_cut_rank"]
        for row in rows
    ]

    if actual_ranks != expected_ranks:
        raise RuntimeError(
            "Graph Cut ranks are not contiguous."
        )

    if len(
        {
            row["task_id"]
            for row in rows
        }
    ) != EXPECTED_TASK_COUNT:
        raise RuntimeError(
            "Graph Cut order contains duplicate tasks."
        )

    gains = np.asarray(
        [
            row["marginal_gain"]
            for row in rows
        ],
        dtype=np.float64,
    )

    if not np.isfinite(gains).all():
        raise RuntimeError(
            "Graph Cut gains contain non-finite values."
        )

    return rows


def budget_split(
    budget: int,
    ratio: list[float],
) -> list[int]:
    """Exact released SMART floor plus cyclic remainder."""

    ratio_sum = sum(ratio)

    if ratio_sum <= 0:
        raise ValueError(
            "Ratio sum must be positive."
        )

    parts = [
        math.floor(
            budget * value / ratio_sum
        )
        for value in ratio
    ]

    remainder = budget - sum(parts)

    for index in range(remainder):
        parts[index % len(ratio)] += 1

    return parts


def taylor_softmax_v1(
    x: torch.Tensor,
    dim: int = 1,
    n: int = 2,
    use_log: bool = False,
) -> torch.Tensor:
    """Exact released SMART second-order Taylor softmax."""

    assert n % 2 == 0 and n > 0

    function_value = torch.ones_like(x)
    denominator = 1.0

    for power in range(1, n + 1):
        denominator *= power

        function_value = (
            function_value
            + x.pow(power) / denominator
        )

    output = function_value / function_value.sum(
        dim=dim,
        keepdims=True,
    )

    if use_log:
        output = output.log()

    return output


def get_task_budgets(
    selected_tasks: list[str],
    gains: list[float],
    all_tasks: list[str],
    task_totals: dict[str, int],
    num_instances: int,
    temperature: float,
) -> tuple[
    dict[str, int],
    np.ndarray,
    list[int],
    int,
]:
    if len(selected_tasks) != len(gains):
        raise RuntimeError(
            "Selected tasks and gains differ in length."
        )

    selected_capacity = sum(
        task_totals[task_id]
        for task_id in selected_tasks
    )

    if selected_capacity < num_instances:
        raise RuntimeError(
            f"Selected task capacity {selected_capacity:,} "
            f"is below budget {num_instances:,}."
        )

    gain_array = np.array(
        [gains],
    ) / temperature

    gain_tensor = torch.from_numpy(
        gain_array
    )

    probabilities = taylor_softmax_v1(
        gain_tensor
    ).numpy()[0]

    initial_budgets = budget_split(
        num_instances,
        probabilities.tolist(),
    )

    task_budgets: dict[str, int] = {}

    for index, budget in enumerate(
        initial_budgets
    ):
        task_budgets[
            selected_tasks[index]
        ] = budget

    to_redistribute = 0

    for task_id in task_budgets.keys():
        if (
            task_totals[task_id]
            < task_budgets[task_id]
        ):
            to_redistribute += (
                task_budgets[task_id]
                - task_totals[task_id]
            )

            task_budgets[task_id] = (
                task_totals[task_id]
            )

    overflow_total = to_redistribute

    while to_redistribute > 0:
        made_progress = False

        for task_id in task_budgets.keys():
            if (
                task_totals[task_id]
                > task_budgets[task_id]
            ):
                task_budgets[task_id] += 1
                to_redistribute -= 1
                made_progress = True

                if to_redistribute == 0:
                    break

        if not made_progress:
            raise RuntimeError(
                "Unable to redistribute task overflow."
            )

    for task_id in all_tasks:
        if task_id not in task_budgets:
            task_budgets[task_id] = 0

    if sum(task_budgets.values()) != num_instances:
        raise RuntimeError(
            f"Task budgets sum to "
            f"{sum(task_budgets.values()):,}; "
            f"expected {num_instances:,}."
        )

    for task_id, budget in task_budgets.items():
        if budget < 0:
            raise RuntimeError(
                f"{task_id}: negative task budget."
            )

        if budget > task_totals[task_id]:
            raise RuntimeError(
                f"{task_id}: budget {budget:,} exceeds "
                f"capacity {task_totals[task_id]:,}."
            )

    return (
        task_budgets,
        probabilities,
        initial_budgets,
        overflow_total,
    )


def get_task_template_budgets(
    submixtures: list[str],
    task_indices: dict[
        str,
        dict[str, dict[str, list[int]]],
    ],
    task_budgets: dict[str, int],
) -> dict[str, dict[str, list[int]]]:
    """Port of the released SMART template-budget logic."""

    counted_budget = 0
    total_budget = 0

    output: dict[
        str,
        dict[str, list[int]],
    ] = {}

    for corpus in submixtures:
        output[corpus] = {}

        for task_id, template_mapping in (
            task_indices[corpus].items()
        ):
            task_budget = task_budgets[
                task_id
            ]

            total_budget += task_budget

            template_counts = [
                len(
                    template_mapping.get(
                        template_type,
                        [],
                    )
                )
                for template_type in TEMPLATE_TYPES
            ]

            if sum(template_counts) < task_budget:
                raise RuntimeError(
                    f"{task_id}: task budget exceeds "
                    "available template instances."
                )

            is_fewshot_present = (
                template_counts[2]
                + template_counts[3]
                > 0
            )

            template_budgets = [0, 0, 0, 0]

            if not is_fewshot_present:
                if template_counts[0] == 0:
                    if template_counts[1] == 0:
                        raise RuntimeError(
                            f"{task_id}: no zero-shot "
                            "template is available."
                        )

                    template_budgets[1] = (
                        task_budget
                    )

                elif template_counts[1] == 0:
                    template_budgets[0] = (
                        task_budget
                    )

                else:
                    (
                        zs_opt,
                        zs_noopt,
                    ) = budget_split(
                        task_budget,
                        [
                            template_counts[0],
                            template_counts[1],
                        ],
                    )

                    template_budgets[0] = (
                        zs_opt
                    )
                    template_budgets[1] = (
                        zs_noopt
                    )

            else:
                (
                    zero_shot_budget,
                    few_shot_budget,
                ) = budget_split(
                    task_budget,
                    [
                        template_counts[0]
                        + template_counts[1],
                        template_counts[2]
                        + template_counts[3],
                    ],
                )

                if template_counts[0] == 0:
                    if template_counts[1] == 0:
                        raise RuntimeError(
                            f"{task_id}: no zero-shot "
                            "template is available."
                        )

                    template_budgets[1] = (
                        zero_shot_budget
                    )

                elif template_counts[1] == 0:
                    template_budgets[0] = (
                        zero_shot_budget
                    )

                else:
                    (
                        zs_opt,
                        zs_noopt,
                    ) = budget_split(
                        zero_shot_budget,
                        [
                            template_counts[0],
                            template_counts[1],
                        ],
                    )

                    template_budgets[0] = (
                        zs_opt
                    )
                    template_budgets[1] = (
                        zs_noopt
                    )

                if template_counts[2] == 0:
                    if template_counts[3] == 0:
                        raise RuntimeError(
                            f"{task_id}: no few-shot "
                            "template is available."
                        )

                    template_budgets[3] = (
                        few_shot_budget
                    )

                elif template_counts[3] == 0:
                    template_budgets[2] = (
                        few_shot_budget
                    )

                else:
                    (
                        fs_opt,
                        fs_noopt,
                    ) = budget_split(
                        few_shot_budget,
                        [
                            template_counts[2],
                            template_counts[3],
                        ],
                    )

                    template_budgets[2] = (
                        fs_opt
                    )
                    template_budgets[3] = (
                        fs_noopt
                    )

            output[corpus][task_id] = (
                template_budgets
            )

            counted_budget += sum(
                template_budgets
            )

    if counted_budget != total_budget:
        raise RuntimeError(
            f"Template budgets sum to "
            f"{counted_budget:,}; expected "
            f"{total_budget:,}."
        )

    return output


def main() -> int:
    args = parse_args()

    if args.temperature <= 0:
        raise ValueError(
            "--temperature must be positive."
        )

    budgets = list(
        dict.fromkeys(args.budgets)
    )

    if not budgets:
        raise ValueError(
            "At least one budget is required."
        )

    if any(
        budget <= 0
        for budget in budgets
    ):
        raise ValueError(
            "Every budget must be positive."
        )

    graph_cut_path = (
        args.graph_cut_order.resolve()
    )
    config_path = (
        args.submixture_config.resolve()
    )
    indices_root = (
        args.task_indices_root.resolve()
    )
    output_root = args.output_root.resolve()

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    submixtures = load_submixtures(
        config_path
    )

    (
        task_indices,
        all_tasks,
        task_totals,
        task_corpus,
    ) = load_task_indices(
        submixtures,
        indices_root,
    )

    graph_cut_rows = load_graph_cut_order(
        graph_cut_path
    )

    selected_tasks = [
        row["task_id"]
        for row in graph_cut_rows
    ]

    if set(selected_tasks) != set(all_tasks):
        missing = sorted(
            set(all_tasks) - set(selected_tasks)
        )
        unexpected = sorted(
            set(selected_tasks) - set(all_tasks)
        )

        raise RuntimeError(
            "Graph Cut and task-index task sets differ. "
            f"Missing={missing}, unexpected={unexpected}"
        )

    for row in graph_cut_rows:
        task_id = row["task_id"]

        if (
            row["valid_train_count"]
            != task_totals[task_id]
        ):
            raise RuntimeError(
                f"{task_id}: Graph Cut capacity "
                f"{row['valid_train_count']:,} differs "
                f"from task-index capacity "
                f"{task_totals[task_id]:,}."
            )

        if row["corpus"] != task_corpus[
            task_id
        ]:
            raise RuntimeError(
                f"{task_id}: corpus mismatch."
            )

    gains = [
        row["marginal_gain"]
        for row in graph_cut_rows
    ]

    allocation_results: dict[
        int,
        dict[str, Any],
    ] = {}

    for budget in budgets:
        (
            task_budgets,
            probabilities,
            initial_budgets,
            overflow_total,
        ) = get_task_budgets(
            selected_tasks=selected_tasks,
            gains=gains,
            all_tasks=all_tasks,
            task_totals=task_totals,
            num_instances=budget,
            temperature=args.temperature,
        )

        template_budgets = (
            get_task_template_budgets(
                submixtures=submixtures,
                task_indices=task_indices,
                task_budgets=task_budgets,
            )
        )

        for corpus in submixtures:
            for task_id, values in (
                template_budgets[
                    corpus
                ].items()
            ):
                expected = [
                    0,
                    task_budgets[task_id],
                    0,
                    0,
                ]

                if values != expected:
                    raise RuntimeError(
                        f"{task_id}: expected single-template "
                        f"budget {expected}; found {values}."
                    )

        allocation_results[budget] = {
            "task_budgets": task_budgets,
            "template_budgets": (
                template_budgets
            ),
            "probabilities": probabilities,
            "initial_budgets": (
                initial_budgets
            ),
            "overflow_total": (
                overflow_total
            ),
        }

        atomic_write_pickle(
            task_budgets,
            output_root
            / f"task_budgets_{budget}.pkl",
        )

        atomic_write_pickle(
            template_budgets,
            output_root
            / (
                "task_template_budgets_"
                f"{budget}.pkl"
            ),
        )

        atomic_write_json(
            {
                "budget": budget,
                "task_budgets": (
                    task_budgets
                ),
            },
            output_root
            / f"task_budgets_{budget}.json",
        )

        atomic_write_json(
            {
                "budget": budget,
                "template_order": list(
                    TEMPLATE_TYPES
                ),
                "task_template_budgets": (
                    template_budgets
                ),
            },
            output_root
            / (
                "task_template_budgets_"
                f"{budget}.json"
            ),
        )

    output_rows: list[
        dict[str, Any]
    ] = []

    for row_index, graph_row in enumerate(
        graph_cut_rows
    ):
        task_id = graph_row["task_id"]

        output_row: dict[str, Any] = {
            **graph_row,
            "taylor_score": (
                1.0
                + graph_row["marginal_gain"]
                / args.temperature
                + 0.5
                * (
                    graph_row[
                        "marginal_gain"
                    ]
                    / args.temperature
                )
                ** 2
            ),
        }

        for budget in budgets:
            result = allocation_results[
                budget
            ]

            probability = float(
                result["probabilities"][
                    row_index
                ]
            )

            output_row.update(
                {
                    f"probability_{budget}": (
                        probability
                    ),
                    f"raw_budget_{budget}": (
                        probability * budget
                    ),
                    f"initial_budget_{budget}": int(
                        result[
                            "initial_budgets"
                        ][row_index]
                    ),
                    f"final_task_budget_{budget}": (
                        result[
                            "task_budgets"
                        ][task_id]
                    ),
                    f"zs_opt_budget_{budget}": 0,
                    f"zs_noopt_budget_{budget}": (
                        result[
                            "task_budgets"
                        ][task_id]
                    ),
                    f"fs_opt_budget_{budget}": 0,
                    f"fs_noopt_budget_{budget}": 0,
                }
            )

        output_row[
            "required_stage2_prefix"
        ] = max(
            allocation_results[budget][
                "task_budgets"
            ][task_id]
            for budget in budgets
        )

        output_rows.append(
            output_row
        )

    allocation_path = (
        output_root / "task_allocations.csv"
    )

    with allocation_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                output_rows[0].keys()
            ),
        )
        writer.writeheader()
        writer.writerows(output_rows)

    monotonicity: dict[str, Any] = {}

    sorted_budgets = sorted(budgets)

    for smaller, larger in zip(
        sorted_budgets,
        sorted_budgets[1:],
    ):
        violations = [
            task_id
            for task_id in all_tasks
            if (
                allocation_results[
                    larger
                ]["task_budgets"][task_id]
                <
                allocation_results[
                    smaller
                ]["task_budgets"][task_id]
            )
        ]

        monotonicity[
            f"{smaller}_to_{larger}"
        ] = {
            "violation_count": len(
                violations
            ),
            "violating_tasks": violations,
        }

        if violations:
            raise RuntimeError(
                f"Cross-budget monotonicity failed: "
                f"{violations}"
            )

    budget_summaries: dict[str, Any] = {}

    for budget in budgets:
        result = allocation_results[
            budget
        ]

        values = np.asarray(
            [
                result["task_budgets"][
                    task_id
                ]
                for task_id in selected_tasks
            ],
            dtype=np.int64,
        )

        template_total = sum(
            sum(template_values)
            for corpus in submixtures
            for template_values in (
                result[
                    "template_budgets"
                ][corpus].values()
            )
        )

        probability_sum = float(
            np.sum(
                result["probabilities"],
                dtype=np.float64,
            )
        )

        if not np.isclose(
            probability_sum,
            1.0,
            rtol=0.0,
            atol=1e-12,
        ):
            raise RuntimeError(
                f"Budget {budget}: probabilities "
                f"sum to {probability_sum}."
            )

        if int(values.sum()) != budget:
            raise RuntimeError(
                f"Budget {budget}: task sum mismatch."
            )

        if template_total != budget:
            raise RuntimeError(
                f"Budget {budget}: template sum mismatch."
            )

        budget_summaries[str(budget)] = {
            "requested_budget": budget,
            "task_budget_sum": int(
                values.sum()
            ),
            "template_budget_sum": (
                template_total
            ),
            "probability_sum": (
                probability_sum
            ),
            "overflow_redistributed": int(
                result["overflow_total"]
            ),
            "tasks_with_nonzero_budget": int(
                np.sum(values > 0)
            ),
            "tasks_with_zero_budget": int(
                np.sum(values == 0)
            ),
            "minimum_task_budget": int(
                values.min()
            ),
            "maximum_task_budget": int(
                values.max()
            ),
            "tasks_at_capacity": int(
                sum(
                    result["task_budgets"][
                        task_id
                    ]
                    == task_totals[task_id]
                    for task_id in all_tasks
                )
            ),
        }

    summary_path = (
        output_root / "allocation_summary.json"
    )

    atomic_write_json(
        {
            "format_version": 1,
            "stage": (
                "smart_v2_task_and_template_allocation"
            ),
            "status": "complete",
            "configuration": {
                "task_count": (
                    EXPECTED_TASK_COUNT
                ),
                "budgets": budgets,
                "temperature": (
                    args.temperature
                ),
                "score_function": (
                    "1 + x + x^2/2"
                ),
                "integerization": (
                    "floor then cyclic remainder "
                    "in Graph Cut order"
                ),
                "capacity_handling": (
                    "cap then cyclic redistribution "
                    "in selected-task insertion order"
                ),
                "template_order": list(
                    TEMPLATE_TYPES
                ),
                "available_template_type": (
                    EXPECTED_TEMPLATE_TYPE
                ),
                "approximation": False,
            },
            "inputs": {
                "graph_cut_order": str(
                    graph_cut_path
                ),
                "graph_cut_order_sha256": (
                    sha256_file(
                        graph_cut_path
                    )
                ),
                "submixture_config": str(
                    config_path
                ),
                "submixture_config_sha256": (
                    sha256_file(
                        config_path
                    )
                ),
                "task_indices_root": str(
                    indices_root
                ),
            },
            "budgets": budget_summaries,
            "cross_budget_monotonicity": (
                monotonicity
            ),
            "maximum_required_stage2_prefix": (
                max(
                    row[
                        "required_stage2_prefix"
                    ]
                    for row in output_rows
                )
            ),
            "outputs": {
                "task_allocations_csv": str(
                    allocation_path
                ),
            },
        },
        summary_path,
    )

    print(
        "=== SMART-v2 task/template allocations ==="
    )
    print(f"Tasks:       {len(all_tasks)}")
    print(f"Temperature: {args.temperature}")

    for budget in budgets:
        info = budget_summaries[
            str(budget)
        ]

        print()
        print(f"Budget {budget:,}:")
        print(
            f"  Task budget sum:       "
            f"{info['task_budget_sum']:,}"
        )
        print(
            f"  Template budget sum:   "
            f"{info['template_budget_sum']:,}"
        )
        print(
            f"  Tasks with allocation: "
            f"{info['tasks_with_nonzero_budget']}"
        )
        print(
            f"  Overflow redistributed:"
            f" {info['overflow_redistributed']:,}"
        )
        print(
            f"  Allocation range:      "
            f"{info['minimum_task_budget']} to "
            f"{info['maximum_task_budget']}"
        )
        print(
            f"  Tasks at capacity:     "
            f"{info['tasks_at_capacity']}"
        )

    print()
    print(
        "Maximum Stage 2 prefix:",
        max(
            row["required_stage2_prefix"]
            for row in output_rows
        ),
    )
    print(f"Allocation CSV: {allocation_path}")
    print(f"Summary JSON:   {summary_path}")
    print()
    print(
        "SMART-v2 task/template allocation passed."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
PY

python3 -m py_compile \
  src/stage1/build_allocations.py
```

### 8.2 Run allocation generation

```bash
cd /data/saral/wdir/smart_v2 || exit 1
source ./smart_v2_env.sh

mkdir -p \
  /mnt/warm_storage/saral/smart_v2/stage1/allocations

python3 -m src.stage1.build_allocations \
  --graph-cut-order /mnt/warm_storage/saral/smart_v2/stage1/graph_cut/graph_cut_order.csv \
  --submixture-config /data/saral/wdir/smart_v2/configs/submixtures.json \
  --task-indices-root /mnt/warm_storage/saral/smart_v2/author_format/task_indices \
  --output-root /mnt/warm_storage/saral/smart_v2/stage1/allocations \
  --budgets 25000 50000 \
  --temperature 1.0 \
  2>&1 | tee \
  /mnt/warm_storage/saral/smart_v2/logs/stage1_allocations.log
```

### 8.3 Inspect the summary

```bash
cat \
  /mnt/warm_storage/saral/smart_v2/stage1/allocations/allocation_summary.json
```

Print the largest and smallest allocations:

```bash
python3 - <<'PY'
import csv
from pathlib import Path

path = Path(
    "/mnt/warm_storage/saral/smart_v2/"
    "stage1/allocations/task_allocations.csv"
)

with path.open(
    encoding="utf-8",
    newline="",
) as handle:
    rows = list(csv.DictReader(handle))

for budget in (25000, 50000):
    column = f"final_task_budget_{budget}"

    ordered = sorted(
        rows,
        key=lambda row: int(row[column]),
        reverse=True,
    )

    print(f"\nTop 10 for {budget:,}")

    for row in ordered[:10]:
        print(
            row["graph_cut_rank"],
            row["task_id"],
            row[column],
        )

    print(f"\nBottom 10 for {budget:,}")

    for row in ordered[-10:]:
        print(
            row["graph_cut_rank"],
            row["task_id"],
            row[column],
        )
PY
```

Expected outputs:

```text
/mnt/warm_storage/saral/smart_v2/stage1/allocations/
├── allocation_summary.json
├── task_allocations.csv
├── task_budgets_25000.pkl
├── task_budgets_25000.json
├── task_budgets_50000.pkl
├── task_budgets_50000.json
├── task_template_budgets_25000.pkl
├── task_template_budgets_25000.json
├── task_template_budgets_50000.pkl
└── task_template_budgets_50000.json
```

Acceptance conditions:

```text
status                         = complete
25K task budget sum            = 25,000
25K template budget sum        = 25,000
50K task budget sum            = 50,000
50K template budget sum        = 50,000
cross-budget violations        = 0
every template allocation      = [0, task_budget, 0, 0]
every task budget              <= task capacity
```
