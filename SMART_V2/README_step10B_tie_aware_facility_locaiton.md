The exact-tie diagnosis once accepted, we can proceed. We will now replace strict index parity with a **tie-aware, common-reference audit**.

This audit will check three representative pools:

```text
tiny:   sglue::copa
small:  cot::stream_qed_ii
medium: t0::wiki_qa_Decide_good_answer
```

For each pool it will:

* run dense Submodlib Facility Location;
* run blockwise Facility Location on two GPUs;
* independently recompute both objective trajectories in CPU float64;
* classify the first ordering divergence;
* require zero true mismatches;
* require objective parity;
* test cross-GPU reproducibility.

The authors’ mixture code only consumes the required prefix of each stored ordering, so the audit uses the actual maximum prefix needed for each task. 

# Step 10B — Tie-aware Facility Location audit

## 1. Create the audit configuration

```bash
cd /data/saral/wdir/smart_v2 || exit 1

cat > configs/stage2_fl_audit.json <<'JSON'
{
  "format_version": 1,
  "tasks": [
    "sglue::copa",
    "cot::stream_qed_ii",
    "t0::wiki_qa_Decide_good_answer"
  ],
  "primary_device": "cuda:0",
  "secondary_device": "cuda:1",
  "primary_singleton_candidate_batch": 256,
  "secondary_singleton_candidate_batch": 512,
  "tie_absolute_tolerance": 1e-10,
  "tie_relative_tolerance": 1e-10,
  "gain_absolute_tolerance": 0.002,
  "gain_relative_tolerance": 1e-5,
  "maximum_relative_objective_difference": 1e-7,
  "maximum_mean_coverage_difference": 1e-7
}
JSON
```

## 2. Create the tie-aware audit script

```bash
cat > src/stage2/audit_fl_engines.py <<'PY'
"""Tie-aware audit of dense and matrix-free Facility Location.

This script does not require identical selected indices when two
candidates have identical or numerically equivalent marginal gains.

Acceptance requires:

1. Dense and blockwise saved gains agree with a common CPU float64
   reference.
2. The first ordering divergence is an exact or numerical tie.
3. Dense and blockwise prefix objectives agree within tolerance.
4. Repeated blockwise runs on two GPUs return the same ordering.
5. No true mismatch is observed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import submodlib.functions as submod_fn

from src.stage2.validate_blockwise_fl import (
    blockwise_lazy_greedy,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--pool-manifest",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--embedding-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
    )

    return parser.parse_args()


def atomic_write_json(
    payload: Any,
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

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


def load_config(path: Path) -> dict[str, Any]:
    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        config = json.load(handle)

    required = {
        "tasks",
        "primary_device",
        "secondary_device",
        "primary_singleton_candidate_batch",
        "secondary_singleton_candidate_batch",
        "tie_absolute_tolerance",
        "tie_relative_tolerance",
        "gain_absolute_tolerance",
        "gain_relative_tolerance",
        "maximum_relative_objective_difference",
        "maximum_mean_coverage_difference",
    }

    missing = required - set(config)

    if missing:
        raise ValueError(
            "Audit config is missing fields: "
            + ", ".join(sorted(missing))
        )

    tasks = config["tasks"]

    if not isinstance(tasks, list) or not tasks:
        raise ValueError(
            "Audit config must contain a non-empty task list."
        )

    if len(tasks) != len(set(tasks)):
        raise ValueError(
            "Audit task list contains duplicates."
        )

    return config


def load_pool_manifest(
    path: Path,
) -> dict[str, dict[str, Any]]:
    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise RuntimeError(
            "Pool manifest is empty."
        )

    result: dict[str, dict[str, Any]] = {}

    for raw in rows:
        task_id = raw["task_id"]

        if task_id in result:
            raise RuntimeError(
                f"Duplicate pool task: {task_id}"
            )

        result[task_id] = {
            "task_id": task_id,
            "corpus": raw["corpus"],
            "template_type": raw[
                "template_type"
            ],
            "pool_size": int(
                raw["pool_size"]
            ),
            "required_stage2_prefix": int(
                raw[
                    "required_stage2_prefix"
                ]
            ),
            "first_dataset_index": int(
                raw["first_dataset_index"]
            ),
            "last_dataset_index": int(
                raw["last_dataset_index"]
            ),
            "dense_kernel_float32_gib": float(
                raw[
                    "dense_kernel_float32_gib"
                ]
            ),
        }

    return result


def normalize_float64(
    embeddings: np.ndarray,
) -> np.ndarray:
    matrix = np.asarray(
        embeddings,
        dtype=np.float64,
    )

    norms = np.linalg.norm(
        matrix,
        axis=1,
        keepdims=True,
    )

    if np.any(norms <= 0.0):
        raise RuntimeError(
            "Embedding pool contains a zero vector."
        )

    matrix = matrix / norms

    if not np.isfinite(matrix).all():
        raise RuntimeError(
            "Normalized embeddings contain "
            "non-finite values."
        )

    return matrix


def evaluate_order_float64(
    normalized: np.ndarray,
    order: list[int],
) -> dict[str, np.ndarray]:
    """Evaluate one ordering using a common CPU float64 reference."""

    n = normalized.shape[0]

    coverage = np.zeros(
        n,
        dtype=np.float64,
    )

    gains = np.empty(
        len(order),
        dtype=np.float64,
    )
    objectives = np.empty(
        len(order),
        dtype=np.float64,
    )
    mean_coverages = np.empty(
        len(order),
        dtype=np.float64,
    )
    minimum_coverages = np.empty(
        len(order),
        dtype=np.float64,
    )

    seen: set[int] = set()

    for rank, candidate in enumerate(order):
        if candidate < 0 or candidate >= n:
            raise RuntimeError(
                f"Candidate {candidate} is outside "
                f"the pool of size {n}."
            )

        if candidate in seen:
            raise RuntimeError(
                f"Ordering contains duplicate "
                f"candidate {candidate}."
            )

        seen.add(candidate)

        similarities = (
            normalized
            @ normalized[candidate]
        )

        updated = np.maximum(
            coverage,
            similarities,
        )

        previous_objective = float(
            coverage.sum(
                dtype=np.float64
            )
        )

        objective = float(
            updated.sum(
                dtype=np.float64
            )
        )

        gains[rank] = (
            objective
            - previous_objective
        )
        objectives[rank] = objective
        mean_coverages[rank] = (
            objective / n
        )
        minimum_coverages[rank] = float(
            updated.min()
        )

        coverage = updated

    return {
        "gains": gains,
        "objectives": objectives,
        "mean_coverages": (
            mean_coverages
        ),
        "minimum_coverages": (
            minimum_coverages
        ),
    }


def marginal_gain_float64(
    normalized: np.ndarray,
    coverage: np.ndarray,
    candidate: int,
) -> float:
    similarities = (
        normalized
        @ normalized[candidate]
    )

    return float(
        np.maximum(
            similarities - coverage,
            0.0,
        ).sum(
            dtype=np.float64
        )
    )


def first_mismatch(
    first: list[int],
    second: list[int],
) -> int | None:
    if len(first) != len(second):
        raise ValueError(
            "Orderings have different lengths."
        )

    for rank, (
        first_candidate,
        second_candidate,
    ) in enumerate(
        zip(
            first,
            second,
            strict=True,
        )
    ):
        if (
            first_candidate
            != second_candidate
        ):
            return rank

    return None


def classify_first_divergence(
    normalized: np.ndarray,
    dense_order: list[int],
    blockwise_order: list[int],
    tie_atol: float,
    tie_rtol: float,
) -> dict[str, Any]:
    mismatch_rank = first_mismatch(
        dense_order,
        blockwise_order,
    )

    if mismatch_rank is None:
        return {
            "classification": (
                "index_match"
            ),
            "rank_1based": None,
            "dense_candidate": None,
            "blockwise_candidate": None,
            "dense_candidate_gain": None,
            "blockwise_candidate_gain": None,
            "absolute_gain_gap": 0.0,
            "tie_threshold": None,
        }

    coverage = np.zeros(
        normalized.shape[0],
        dtype=np.float64,
    )

    for candidate in dense_order[
        :mismatch_rank
    ]:
        similarities = (
            normalized
            @ normalized[candidate]
        )

        coverage = np.maximum(
            coverage,
            similarities,
        )

    dense_candidate = dense_order[
        mismatch_rank
    ]
    blockwise_candidate = (
        blockwise_order[
            mismatch_rank
        ]
    )

    dense_gain = marginal_gain_float64(
        normalized=normalized,
        coverage=coverage,
        candidate=dense_candidate,
    )

    blockwise_gain = (
        marginal_gain_float64(
            normalized=normalized,
            coverage=coverage,
            candidate=blockwise_candidate,
        )
    )

    gap = abs(
        dense_gain
        - blockwise_gain
    )

    scale = max(
        1.0,
        abs(dense_gain),
        abs(blockwise_gain),
    )

    threshold = (
        tie_atol
        + tie_rtol * scale
    )

    if gap == 0.0:
        classification = "exact_tie"
    elif gap <= threshold:
        classification = (
            "numerical_tie"
        )
    else:
        classification = (
            "true_mismatch"
        )

    return {
        "classification": (
            classification
        ),
        "rank_1based": (
            mismatch_rank + 1
        ),
        "dense_candidate": (
            dense_candidate
        ),
        "blockwise_candidate": (
            blockwise_candidate
        ),
        "dense_candidate_gain": (
            dense_gain
        ),
        "blockwise_candidate_gain": (
            blockwise_gain
        ),
        "absolute_gain_gap": gap,
        "tie_threshold": (
            threshold
        ),
    }


def checkpoint_ranks(
    prefix: int,
) -> list[int]:
    values = {
        1,
        10,
        25,
        50,
        100,
        prefix,
    }

    return sorted(
        value
        for value in values
        if 1 <= value <= prefix
    )


def compare_objectives(
    dense_reference: dict[
        str,
        np.ndarray,
    ],
    blockwise_reference: dict[
        str,
        np.ndarray,
    ],
    prefix: int,
) -> dict[str, Any]:
    checkpoints: list[
        dict[str, Any]
    ] = []

    maximum_relative_difference = (
        0.0
    )
    maximum_mean_coverage_difference = (
        0.0
    )

    for rank_1based in checkpoint_ranks(
        prefix
    ):
        index = rank_1based - 1

        dense_objective = float(
            dense_reference[
                "objectives"
            ][index]
        )
        blockwise_objective = float(
            blockwise_reference[
                "objectives"
            ][index]
        )

        absolute_difference = abs(
            dense_objective
            - blockwise_objective
        )

        relative_difference = (
            absolute_difference
            / max(
                1.0,
                abs(dense_objective),
            )
        )

        dense_mean = float(
            dense_reference[
                "mean_coverages"
            ][index]
        )
        blockwise_mean = float(
            blockwise_reference[
                "mean_coverages"
            ][index]
        )

        mean_difference = abs(
            dense_mean
            - blockwise_mean
        )

        maximum_relative_difference = max(
            maximum_relative_difference,
            relative_difference,
        )

        maximum_mean_coverage_difference = max(
            maximum_mean_coverage_difference,
            mean_difference,
        )

        checkpoints.append(
            {
                "rank": rank_1based,
                "dense_objective": (
                    dense_objective
                ),
                "blockwise_objective": (
                    blockwise_objective
                ),
                "absolute_difference": (
                    absolute_difference
                ),
                "relative_difference": (
                    relative_difference
                ),
                "dense_mean_coverage": (
                    dense_mean
                ),
                "blockwise_mean_coverage": (
                    blockwise_mean
                ),
                "mean_coverage_difference": (
                    mean_difference
                ),
                "dense_minimum_coverage": float(
                    dense_reference[
                        "minimum_coverages"
                    ][index]
                ),
                "blockwise_minimum_coverage": float(
                    blockwise_reference[
                        "minimum_coverages"
                    ][index]
                ),
            }
        )

    return {
        "checkpoints": checkpoints,
        "maximum_relative_objective_difference": (
            maximum_relative_difference
        ),
        "maximum_mean_coverage_difference": (
            maximum_mean_coverage_difference
        ),
    }


def gain_validation(
    saved_gains: np.ndarray,
    reference_gains: np.ndarray,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    errors = np.abs(
        saved_gains
        - reference_gains
    )

    return {
        "passed": bool(
            np.allclose(
                saved_gains,
                reference_gains,
                atol=atol,
                rtol=rtol,
            )
        ),
        "maximum_absolute_error": float(
            errors.max()
        ),
        "mean_absolute_error": float(
            errors.mean()
        ),
    }


def write_order_csv(
    path: Path,
    dense_order: list[int],
    dense_gains: np.ndarray,
    blockwise_order: list[int],
    blockwise_gains: np.ndarray,
    secondary_order: list[int],
    secondary_gains: np.ndarray,
    first_dataset_index: int,
) -> None:
    rows: list[dict[str, Any]] = []

    for rank in range(
        len(dense_order)
    ):
        rows.append(
            {
                "rank": rank + 1,
                "dense_local_index": (
                    dense_order[rank]
                ),
                "blockwise_primary_local_index": (
                    blockwise_order[
                        rank
                    ]
                ),
                "blockwise_secondary_local_index": (
                    secondary_order[
                        rank
                    ]
                ),
                "dense_dataset_index": (
                    first_dataset_index
                    + dense_order[rank]
                ),
                "blockwise_primary_dataset_index": (
                    first_dataset_index
                    + blockwise_order[
                        rank
                    ]
                ),
                "blockwise_secondary_dataset_index": (
                    first_dataset_index
                    + secondary_order[
                        rank
                    ]
                ),
                "dense_gain": float(
                    dense_gains[rank]
                ),
                "blockwise_primary_gain": float(
                    blockwise_gains[
                        rank
                    ]
                ),
                "blockwise_secondary_gain": float(
                    secondary_gains[
                        rank
                    ]
                ),
            }
        )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                rows[0].keys()
            ),
        )
        writer.writeheader()
        writer.writerows(rows)


def audit_task(
    *,
    pool: dict[str, Any],
    embedding_root: Path,
    output_root: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    task_id = pool["task_id"]
    corpus = pool["corpus"]
    n = pool["pool_size"]
    prefix = pool[
        "required_stage2_prefix"
    ]

    if not 0 < prefix < n:
        raise RuntimeError(
            f"{task_id}: expected "
            f"0 < prefix < n; "
            f"prefix={prefix}, n={n}."
        )

    matrix_path = (
        embedding_root
        / corpus
        / "train_prompt_embeddings.npy"
    )

    full_matrix = np.load(
        matrix_path,
        mmap_mode="r",
    )

    start = pool[
        "first_dataset_index"
    ]
    end = (
        pool[
            "last_dataset_index"
        ]
        + 1
    )

    if end - start != n:
        raise RuntimeError(
            f"{task_id}: pool interval "
            "does not match pool size."
        )

    embeddings = np.asarray(
        full_matrix[start:end],
        dtype=np.float32,
    ).copy()

    del full_matrix

    task_root = (
        output_root
        / task_id.replace(
            "::",
            "__",
        )
    )

    task_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print(
        "========================================"
    )
    print(f"Task:         {task_id}")
    print(f"Pool size:    {n:,}")
    print(f"Prefix:       {prefix}")
    print(
        f"Dense kernel: "
        f"{pool['dense_kernel_float32_gib']:.3f} GiB"
    )
    print(
        "========================================"
    )

    dense_start = time.perf_counter()

    dense_objective = (
        submod_fn.facilityLocation
        .FacilityLocationFunction(
            n=n,
            separate_rep=False,
            mode="dense",
            data=embeddings,
            create_dense_cpp_kernel_in_python=False,
        )
    )

    dense_result = (
        dense_objective.maximize(
            budget=prefix,
            optimizer="LazyGreedy",
            stopIfZeroGain=False,
            stopIfNegativeGain=False,
            verbose=False,
            show_progress=True,
        )
    )

    dense_seconds = (
        time.perf_counter()
        - dense_start
    )

    dense_order = [
        int(index)
        for index, _ in dense_result
    ]

    dense_saved_gains = np.asarray(
        [
            float(gain)
            for _, gain in dense_result
        ],
        dtype=np.float64,
    )

    primary_start = (
        time.perf_counter()
    )

    (
        primary_result,
        primary_info,
    ) = blockwise_lazy_greedy(
        embeddings=embeddings,
        budget=prefix,
        device_name=config[
            "primary_device"
        ],
        singleton_candidate_batch=int(
            config[
                "primary_singleton_candidate_batch"
            ]
        ),
    )

    primary_seconds = (
        time.perf_counter()
        - primary_start
    )

    primary_order = [
        int(index)
        for index, _ in primary_result
    ]

    primary_saved_gains = np.asarray(
        [
            float(gain)
            for _, gain
            in primary_result
        ],
        dtype=np.float64,
    )

    secondary_start = (
        time.perf_counter()
    )

    (
        secondary_result,
        secondary_info,
    ) = blockwise_lazy_greedy(
        embeddings=embeddings,
        budget=prefix,
        device_name=config[
            "secondary_device"
        ],
        singleton_candidate_batch=int(
            config[
                "secondary_singleton_candidate_batch"
            ]
        ),
    )

    secondary_seconds = (
        time.perf_counter()
        - secondary_start
    )

    secondary_order = [
        int(index)
        for index, _ in secondary_result
    ]

    secondary_saved_gains = np.asarray(
        [
            float(gain)
            for _, gain
            in secondary_result
        ],
        dtype=np.float64,
    )

    normalized = normalize_float64(
        embeddings
    )

    dense_reference = (
        evaluate_order_float64(
            normalized,
            dense_order,
        )
    )

    primary_reference = (
        evaluate_order_float64(
            normalized,
            primary_order,
        )
    )

    secondary_reference = (
        evaluate_order_float64(
            normalized,
            secondary_order,
        )
    )

    dense_gain_check = gain_validation(
        saved_gains=dense_saved_gains,
        reference_gains=(
            dense_reference[
                "gains"
            ]
        ),
        atol=float(
            config[
                "gain_absolute_tolerance"
            ]
        ),
        rtol=float(
            config[
                "gain_relative_tolerance"
            ]
        ),
    )

    primary_gain_check = (
        gain_validation(
            saved_gains=(
                primary_saved_gains
            ),
            reference_gains=(
                primary_reference[
                    "gains"
                ]
            ),
            atol=float(
                config[
                    "gain_absolute_tolerance"
                ]
            ),
            rtol=float(
                config[
                    "gain_relative_tolerance"
                ]
            ),
        )
    )

    secondary_gain_check = (
        gain_validation(
            saved_gains=(
                secondary_saved_gains
            ),
            reference_gains=(
                secondary_reference[
                    "gains"
                ]
            ),
            atol=float(
                config[
                    "gain_absolute_tolerance"
                ]
            ),
            rtol=float(
                config[
                    "gain_relative_tolerance"
                ]
            ),
        )
    )

    divergence = (
        classify_first_divergence(
            normalized=normalized,
            dense_order=dense_order,
            blockwise_order=(
                primary_order
            ),
            tie_atol=float(
                config[
                    "tie_absolute_tolerance"
                ]
            ),
            tie_rtol=float(
                config[
                    "tie_relative_tolerance"
                ]
            ),
        )
    )

    objective_comparison = (
        compare_objectives(
            dense_reference=(
                dense_reference
            ),
            blockwise_reference=(
                primary_reference
            ),
            prefix=prefix,
        )
    )

    cross_device_order_match = (
        primary_order
        == secondary_order
    )

    cross_device_gain_match = bool(
        np.allclose(
            primary_saved_gains,
            secondary_saved_gains,
            atol=float(
                config[
                    "gain_absolute_tolerance"
                ]
            ),
            rtol=float(
                config[
                    "gain_relative_tolerance"
                ]
            ),
        )
    )

    secondary_objective_comparison = (
        compare_objectives(
            dense_reference=(
                primary_reference
            ),
            blockwise_reference=(
                secondary_reference
            ),
            prefix=prefix,
        )
    )

    true_mismatch = (
        divergence[
            "classification"
        ]
        == "true_mismatch"
    )

    objective_passed = (
        objective_comparison[
            "maximum_relative_objective_difference"
        ]
        <= float(
            config[
                "maximum_relative_objective_difference"
            ]
        )
        and
        objective_comparison[
            "maximum_mean_coverage_difference"
        ]
        <= float(
            config[
                "maximum_mean_coverage_difference"
            ]
        )
    )

    reproducibility_passed = (
        cross_device_order_match
        and cross_device_gain_match
        and
        secondary_objective_comparison[
            "maximum_relative_objective_difference"
        ]
        <= float(
            config[
                "maximum_relative_objective_difference"
            ]
        )
    )

    passed = (
        dense_gain_check["passed"]
        and primary_gain_check[
            "passed"
        ]
        and secondary_gain_check[
            "passed"
        ]
        and not true_mismatch
        and objective_passed
        and reproducibility_passed
    )

    order_path = (
        task_root / "audit_order.csv"
    )

    write_order_csv(
        path=order_path,
        dense_order=dense_order,
        dense_gains=(
            dense_saved_gains
        ),
        blockwise_order=(
            primary_order
        ),
        blockwise_gains=(
            primary_saved_gains
        ),
        secondary_order=(
            secondary_order
        ),
        secondary_gains=(
            secondary_saved_gains
        ),
        first_dataset_index=start,
    )

    report = {
        "status": (
            "verified"
            if passed
            else "failed"
        ),
        "task": pool,
        "dense": {
            "elapsed_seconds": (
                dense_seconds
            ),
            "gain_validation": (
                dense_gain_check
            ),
        },
        "blockwise_primary": {
            "device": config[
                "primary_device"
            ],
            "elapsed_seconds": (
                primary_seconds
            ),
            "gain_validation": (
                primary_gain_check
            ),
            **primary_info,
        },
        "blockwise_secondary": {
            "device": config[
                "secondary_device"
            ],
            "elapsed_seconds": (
                secondary_seconds
            ),
            "gain_validation": (
                secondary_gain_check
            ),
            **secondary_info,
        },
        "first_divergence": (
            divergence
        ),
        "dense_blockwise_objectives": (
            objective_comparison
        ),
        "cross_device_reproducibility": {
            "order_match": (
                cross_device_order_match
            ),
            "gain_match": (
                cross_device_gain_match
            ),
            "objective_comparison": (
                secondary_objective_comparison
            ),
            "passed": (
                reproducibility_passed
            ),
        },
        "acceptance": {
            "true_mismatch": (
                true_mismatch
            ),
            "objective_passed": (
                objective_passed
            ),
            "reproducibility_passed": (
                reproducibility_passed
            ),
            "overall_passed": passed,
        },
        "outputs": {
            "order_csv": str(
                order_path
            ),
        },
    }

    report_path = (
        task_root / "audit_report.json"
    )

    atomic_write_json(
        report,
        report_path,
    )

    print()
    print(f"Audit status:       {report['status']}")
    print(
        "First divergence:   "
        f"{divergence['classification']}"
    )
    print(
        "Divergence rank:    "
        f"{divergence['rank_1based']}"
    )
    print(
        "Objective rel diff: "
        f"{objective_comparison['maximum_relative_objective_difference']:.3e}"
    )
    print(
        "Mean coverage diff: "
        f"{objective_comparison['maximum_mean_coverage_difference']:.3e}"
    )
    print(
        "Cross-GPU order:    "
        f"{cross_device_order_match}"
    )
    print(
        "Cross-GPU gains:    "
        f"{cross_device_gain_match}"
    )
    print(f"Report:             {report_path}")

    return report


def main() -> int:
    args = parse_args()

    config_path = args.config.resolve()
    pool_manifest_path = (
        args.pool_manifest.resolve()
    )
    embedding_root = (
        args.embedding_root.resolve()
    )
    output_root = (
        args.output_root.resolve()
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    config = load_config(
        config_path
    )
    pools = load_pool_manifest(
        pool_manifest_path
    )

    reports: list[dict[str, Any]] = []

    total_start = time.perf_counter()

    for task_id in config["tasks"]:
        if task_id not in pools:
            raise RuntimeError(
                f"Audit task is not present "
                f"in the pool manifest: {task_id}"
            )

        report = audit_task(
            pool=pools[task_id],
            embedding_root=(
                embedding_root
            ),
            output_root=(
                output_root
            ),
            config=config,
        )

        reports.append(report)

    elapsed = (
        time.perf_counter()
        - total_start
    )

    failed_tasks = [
        report["task"]["task_id"]
        for report in reports
        if report["status"]
        != "verified"
    ]

    divergence_counts: dict[
        str,
        int,
    ] = {}

    for report in reports:
        classification = report[
            "first_divergence"
        ]["classification"]

        divergence_counts[
            classification
        ] = (
            divergence_counts.get(
                classification,
                0,
            )
            + 1
        )

    summary = {
        "format_version": 1,
        "stage": (
            "smart_v2_tie_aware_fl_audit"
        ),
        "status": (
            "verified"
            if not failed_tasks
            else "failed"
        ),
        "configuration": config,
        "task_count": len(reports),
        "verified_task_count": (
            len(reports)
            - len(failed_tasks)
        ),
        "failed_task_count": (
            len(failed_tasks)
        ),
        "failed_tasks": (
            failed_tasks
        ),
        "first_divergence_counts": (
            divergence_counts
        ),
        "true_mismatch_count": sum(
            report[
                "first_divergence"
            ]["classification"]
            == "true_mismatch"
            for report in reports
        ),
        "cross_device_order_match_count": sum(
            report[
                "cross_device_reproducibility"
            ]["order_match"]
            for report in reports
        ),
        "elapsed_seconds": elapsed,
        "task_reports": [
            {
                "task_id": (
                    report["task"][
                        "task_id"
                    ]
                ),
                "status": (
                    report["status"]
                ),
                "first_divergence": (
                    report[
                        "first_divergence"
                    ]
                ),
                "maximum_relative_objective_difference": (
                    report[
                        "dense_blockwise_objectives"
                    ][
                        "maximum_relative_objective_difference"
                    ]
                ),
                "cross_device_order_match": (
                    report[
                        "cross_device_reproducibility"
                    ]["order_match"]
                ),
            }
            for report in reports
        ],
    }

    summary_path = (
        output_root
        / "audit_summary.json"
    )

    atomic_write_json(
        summary,
        summary_path,
    )

    print()
    print(
        "========================================"
    )
    print(
        "=== Facility Location audit summary ==="
    )
    print(
        "========================================"
    )
    print(
        f"Status:             "
        f"{summary['status']}"
    )
    print(
        f"Tasks:              "
        f"{summary['task_count']}"
    )
    print(
        f"Verified:           "
        f"{summary['verified_task_count']}"
    )
    print(
        f"Failed:             "
        f"{summary['failed_task_count']}"
    )
    print(
        f"True mismatches:    "
        f"{summary['true_mismatch_count']}"
    )
    print(
        f"Cross-GPU matches:  "
        f"{summary['cross_device_order_match_count']}/"
        f"{summary['task_count']}"
    )
    print(
        f"Elapsed:            "
        f"{elapsed / 60:.2f} minutes"
    )
    print(f"Summary:            {summary_path}")

    if failed_tasks:
        raise RuntimeError(
            "Tie-aware Facility Location audit failed "
            f"for: {failed_tasks}"
        )

    print()
    print(
        "Tie-aware Facility Location audit passed."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
PY

python3 -m py_compile \
  src/stage2/audit_fl_engines.py
```

## 3. Run the audit

This can take noticeably longer for the medium task because the dense oracle creates a roughly 0.58 GiB raw kernel and both blockwise GPU runs evaluate the full candidate pool.

```bash
cd /data/saral/wdir/smart_v2 || exit 1
source ./smart_v2_env.sh

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

mkdir -p \
  /mnt/warm_storage/saral/smart_v2/stage2/audit

python3 -m src.stage2.audit_fl_engines \
  --config /data/saral/wdir/smart_v2/configs/stage2_fl_audit.json \
  --pool-manifest /mnt/warm_storage/saral/smart_v2/stage2/pools/pool_manifest.csv \
  --embedding-root /mnt/warm_storage/saral/smart_v2/embeddings/gte-large \
  --output-root /mnt/warm_storage/saral/smart_v2/stage2/audit \
  2>&1 | tee \
  /mnt/warm_storage/saral/smart_v2/logs/stage2_fl_audit.log
```

## 4. Inspect the compact result

```bash
cat \
  /mnt/warm_storage/saral/smart_v2/stage2/audit/audit_summary.json
```

Expected layout:

```text
/mnt/warm_storage/saral/smart_v2/stage2/audit/
├── sglue__copa/
│   ├── audit_order.csv
│   └── audit_report.json
├── cot__stream_qed_ii/
│   ├── audit_order.csv
│   └── audit_report.json
├── t0__wiki_qa_Decide_good_answer/
│   ├── audit_order.csv
│   └── audit_report.json
└── audit_summary.json
```

Acceptance conditions:

```text
status                         = verified
verified task count            = 3
failed task count              = 0
true mismatch count            = 0
cross-GPU order matches        = 3/3
dense saved gains valid        = true
blockwise saved gains valid    = true
objective relative difference  <= 1e-7
mean coverage difference       <= 1e-7
```

An `exact_tie` or `numerical_tie` is acceptable. A `true_mismatch`, cross-GPU ordering difference, or objective-threshold failure must be diagnosed before any large-pool ordering is generated.
