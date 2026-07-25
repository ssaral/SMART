The global audit certifies the singleton shortcut for all 309 pools. No fallback, candidate sampling, or sparse approximation is needed.

Before launching every ordering, verify the **final float64 production engine itself** against dense Submodlib and across two GPUs. The earlier audit exercised the prototype blockwise engine; this step closes that verification gap.

# Step 13 — Audit the final production engine

We will rerun the final `run_blockwise_ordering.py` on the same tiny, small, and medium pools using `cuda:0` and `cuda:1`, then compare:

* production gains against an independent float64 replay;
* dense Submodlib versus production objectives;
* first divergence classification;
* exact cross-GPU ordering reproducibility.

## 13.1 Run the final engine on both GPUs

```bash
cd /data/saral/wdir/smart_v2 || exit 1
source ./smart_v2_env.sh

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

AUDIT_ROOT=/mnt/warm_storage/saral/smart_v2/stage2/production_audit

TASKS=(
  "sglue::copa"
  "cot::stream_qed_ii"
  "t0::wiki_qa_Decide_good_answer"
)

for DEVICE in 0 1; do
  for TASK in "${TASKS[@]}"; do
    SLUG="${TASK//::/__}"
    OUT="$AUDIT_ROOT/cuda${DEVICE}/$SLUG"

    mkdir -p "$OUT"

    python3 -m src.stage2.run_blockwise_ordering \
      --pool-manifest /mnt/warm_storage/saral/smart_v2/stage2/pools/pool_manifest.csv \
      --embedding-root /mnt/warm_storage/saral/smart_v2/embeddings/gte-large \
      --task-id "$TASK" \
      --output-root "$OUT" \
      --device "cuda:${DEVICE}" \
      --tie-atol 1e-10 \
      --tie-rtol 1e-10 \
      --replay-atol 1e-8 \
      --replay-rtol 1e-10 \
      --progress-every 25 \
      2>&1 | tee \
      "/mnt/warm_storage/saral/smart_v2/logs/production_audit_cuda${DEVICE}_${SLUG}.log"
  done
done
```

Each of the six runs must finish with:

```text
Status: verified
Replay gain error: 0 or <= 1e-8
Exact blockwise ordering passed.
```

## 13.2 Create the production-audit verifier

```bash
cat > src/stage2/verify_production_engine.py <<'PY'
"""Verify the final float64 Stage 2 production engine.

Compares:
- dense Submodlib orderings from the earlier audit;
- final production orderings on cuda:0;
- final production orderings on cuda:1;
- independent CPU float64 objective trajectories.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.stage2.audit_fl_engines import (
    classify_first_divergence,
    compare_objectives,
    evaluate_order_float64,
    normalize_float64,
)


TASKS = (
    "sglue::copa",
    "cot::stream_qed_ii",
    "t0::wiki_qa_Decide_good_answer",
)

OBJECTIVE_RELATIVE_TOLERANCE = 1e-7
MEAN_COVERAGE_TOLERANCE = 1e-7
PRODUCTION_GAIN_ATOL = 1e-8
PRODUCTION_GAIN_RTOL = 1e-10
CROSS_GPU_GAIN_ATOL = 1e-10
CROSS_GPU_GAIN_RTOL = 1e-12
TIE_ATOL = 1e-10
TIE_RTOL = 1e-10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

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
        "--dense-audit-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--production-audit-root",
        type=Path,
        required=True,
    )

    return parser.parse_args()


def atomic_write_json(
    value: Any,
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
            value,
            handle,
            indent=2,
            ensure_ascii=False,
        )
        handle.write("\n")

    temporary.replace(path)


def slug(task_id: str) -> str:
    return task_id.replace(
        "::",
        "__",
    )


def load_manifest(
    path: Path,
) -> dict[str, dict[str, Any]]:
    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        raw_rows = list(
            csv.DictReader(handle)
        )

    result: dict[str, dict[str, Any]] = {}

    for raw in raw_rows:
        task_id = raw["task_id"]

        result[task_id] = {
            "task_id": task_id,
            "corpus": raw["corpus"],
            "pool_size": int(
                raw["pool_size"]
            ),
            "prefix": int(
                raw[
                    "required_stage2_prefix"
                ]
            ),
            "first_dataset_index": int(
                raw[
                    "first_dataset_index"
                ]
            ),
            "last_dataset_index": int(
                raw[
                    "last_dataset_index"
                ]
            ),
        }

    return result


def load_dense_order(
    path: Path,
) -> tuple[list[int], np.ndarray]:
    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        rows = list(
            csv.DictReader(handle)
        )

    order = [
        int(row["dense_local_index"])
        for row in rows
    ]

    gains = np.asarray(
        [
            float(row["dense_gain"])
            for row in rows
        ],
        dtype=np.float64,
    )

    return order, gains


def load_production_order(
    path: Path,
) -> tuple[list[int], np.ndarray]:
    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        rows = list(
            csv.DictReader(handle)
        )

    order = [
        int(row["local_index"])
        for row in rows
    ]

    gains = np.asarray(
        [
            float(row["marginal_gain"])
            for row in rows
        ],
        dtype=np.float64,
    )

    return order, gains


def validate_saved_gains(
    saved: np.ndarray,
    reference: np.ndarray,
) -> dict[str, Any]:
    errors = np.abs(
        saved - reference
    )

    return {
        "passed": bool(
            np.allclose(
                saved,
                reference,
                atol=PRODUCTION_GAIN_ATOL,
                rtol=PRODUCTION_GAIN_RTOL,
            )
        ),
        "maximum_absolute_error": float(
            errors.max()
        ),
        "mean_absolute_error": float(
            errors.mean()
        ),
    }


def main() -> int:
    args = parse_args()

    manifest_path = (
        args.pool_manifest.resolve()
    )
    embedding_root = (
        args.embedding_root.resolve()
    )
    dense_root = (
        args.dense_audit_root.resolve()
    )
    production_root = (
        args.production_audit_root.resolve()
    )

    pools = load_manifest(
        manifest_path
    )

    task_reports: list[
        dict[str, Any]
    ] = []

    for task_id in TASKS:
        if task_id not in pools:
            raise RuntimeError(
                f"Missing manifest entry: {task_id}"
            )

        pool = pools[task_id]
        task_slug = slug(task_id)

        dense_path = (
            dense_root
            / task_slug
            / "audit_order.csv"
        )

        cuda0_path = (
            production_root
            / "cuda0"
            / task_slug
            / "ordering.csv"
        )

        cuda1_path = (
            production_root
            / "cuda1"
            / task_slug
            / "ordering.csv"
        )

        for path in (
            dense_path,
            cuda0_path,
            cuda1_path,
        ):
            if not path.is_file():
                raise FileNotFoundError(path)

        (
            dense_order,
            dense_saved_gains,
        ) = load_dense_order(
            dense_path
        )

        (
            cuda0_order,
            cuda0_saved_gains,
        ) = load_production_order(
            cuda0_path
        )

        (
            cuda1_order,
            cuda1_saved_gains,
        ) = load_production_order(
            cuda1_path
        )

        prefix = pool["prefix"]

        if not (
            len(dense_order)
            == len(cuda0_order)
            == len(cuda1_order)
            == prefix
        ):
            raise RuntimeError(
                f"{task_id}: ordering-length mismatch."
            )

        matrix_path = (
            embedding_root
            / pool["corpus"]
            / "train_prompt_embeddings.npy"
        )

        matrix = np.load(
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

        if end - start != pool["pool_size"]:
            raise RuntimeError(
                f"{task_id}: pool interval mismatch."
            )

        embeddings = np.asarray(
            matrix[start:end],
            dtype=np.float32,
        ).copy()

        del matrix

        normalized = normalize_float64(
            embeddings
        )

        dense_reference = (
            evaluate_order_float64(
                normalized,
                dense_order,
            )
        )

        cuda0_reference = (
            evaluate_order_float64(
                normalized,
                cuda0_order,
            )
        )

        cuda1_reference = (
            evaluate_order_float64(
                normalized,
                cuda1_order,
            )
        )

        cuda0_gain_check = (
            validate_saved_gains(
                cuda0_saved_gains,
                cuda0_reference["gains"],
            )
        )

        cuda1_gain_check = (
            validate_saved_gains(
                cuda1_saved_gains,
                cuda1_reference["gains"],
            )
        )

        divergence = (
            classify_first_divergence(
                normalized=normalized,
                dense_order=dense_order,
                blockwise_order=(
                    cuda0_order
                ),
                tie_atol=TIE_ATOL,
                tie_rtol=TIE_RTOL,
            )
        )

        dense_production_objectives = (
            compare_objectives(
                dense_reference=(
                    dense_reference
                ),
                blockwise_reference=(
                    cuda0_reference
                ),
                prefix=prefix,
            )
        )

        cross_gpu_objectives = (
            compare_objectives(
                dense_reference=(
                    cuda0_reference
                ),
                blockwise_reference=(
                    cuda1_reference
                ),
                prefix=prefix,
            )
        )

        cross_gpu_order_match = (
            cuda0_order
            == cuda1_order
        )

        cross_gpu_gain_match = bool(
            np.allclose(
                cuda0_saved_gains,
                cuda1_saved_gains,
                atol=CROSS_GPU_GAIN_ATOL,
                rtol=CROSS_GPU_GAIN_RTOL,
            )
        )

        cross_gpu_max_gain_error = float(
            np.max(
                np.abs(
                    cuda0_saved_gains
                    - cuda1_saved_gains
                )
            )
        )

        no_true_mismatch = (
            divergence[
                "classification"
            ]
            != "true_mismatch"
        )

        objective_passed = (
            dense_production_objectives[
                "maximum_relative_objective_difference"
            ]
            <= OBJECTIVE_RELATIVE_TOLERANCE
            and
            dense_production_objectives[
                "maximum_mean_coverage_difference"
            ]
            <= MEAN_COVERAGE_TOLERANCE
        )

        cross_gpu_passed = (
            cross_gpu_order_match
            and cross_gpu_gain_match
            and
            cross_gpu_objectives[
                "maximum_relative_objective_difference"
            ]
            <= OBJECTIVE_RELATIVE_TOLERANCE
        )

        passed = (
            cuda0_gain_check["passed"]
            and cuda1_gain_check["passed"]
            and no_true_mismatch
            and objective_passed
            and cross_gpu_passed
        )

        report = {
            "task_id": task_id,
            "status": (
                "verified"
                if passed
                else "failed"
            ),
            "pool_size": (
                pool["pool_size"]
            ),
            "prefix": prefix,
            "first_dense_production_divergence": (
                divergence
            ),
            "cuda0_gain_validation": (
                cuda0_gain_check
            ),
            "cuda1_gain_validation": (
                cuda1_gain_check
            ),
            "dense_production_objectives": (
                dense_production_objectives
            ),
            "cross_gpu": {
                "order_match": (
                    cross_gpu_order_match
                ),
                "gain_match": (
                    cross_gpu_gain_match
                ),
                "maximum_gain_error": (
                    cross_gpu_max_gain_error
                ),
                "objective_comparison": (
                    cross_gpu_objectives
                ),
                "passed": (
                    cross_gpu_passed
                ),
            },
            "acceptance": {
                "no_true_mismatch": (
                    no_true_mismatch
                ),
                "objective_passed": (
                    objective_passed
                ),
                "overall_passed": passed,
            },
        }

        task_reports.append(report)

        print()
        print(
            "========================================"
        )
        print(f"Task:                 {task_id}")
        print(
            f"Status:               "
            f"{report['status']}"
        )
        print(
            "Dense divergence:     "
            f"{divergence['classification']}"
        )
        print(
            "Divergence rank:      "
            f"{divergence['rank_1based']}"
        )
        print(
            "Objective rel diff:   "
            f"{dense_production_objectives['maximum_relative_objective_difference']:.3e}"
        )
        print(
            "Mean coverage diff:   "
            f"{dense_production_objectives['maximum_mean_coverage_difference']:.3e}"
        )
        print(
            "CUDA 0 replay error:  "
            f"{cuda0_gain_check['maximum_absolute_error']:.3e}"
        )
        print(
            "CUDA 1 replay error:  "
            f"{cuda1_gain_check['maximum_absolute_error']:.3e}"
        )
        print(
            "Cross-GPU order:      "
            f"{cross_gpu_order_match}"
        )
        print(
            "Cross-GPU gain error: "
            f"{cross_gpu_max_gain_error:.3e}"
        )

    failed_tasks = [
        report["task_id"]
        for report in task_reports
        if report["status"] != "verified"
    ]

    summary = {
        "format_version": 1,
        "stage": (
            "smart_v2_final_production_engine_audit"
        ),
        "status": (
            "verified"
            if not failed_tasks
            else "failed"
        ),
        "task_count": len(task_reports),
        "verified_task_count": (
            len(task_reports)
            - len(failed_tasks)
        ),
        "failed_task_count": (
            len(failed_tasks)
        ),
        "failed_tasks": failed_tasks,
        "true_mismatch_count": sum(
            report[
                "first_dense_production_divergence"
            ]["classification"]
            == "true_mismatch"
            for report in task_reports
        ),
        "cross_gpu_order_match_count": sum(
            report["cross_gpu"][
                "order_match"
            ]
            for report in task_reports
        ),
        "task_reports": (
            task_reports
        ),
    }

    summary_path = (
        production_root
        / "production_audit_summary.json"
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
        "=== Final production-engine audit ==="
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
        f"{summary['cross_gpu_order_match_count']}/"
        f"{summary['task_count']}"
    )
    print(f"Summary:            {summary_path}")

    if failed_tasks:
        raise RuntimeError(
            "Final production-engine audit failed "
            f"for: {failed_tasks}"
        )

    print()
    print(
        "Final float64 production engine passed."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
PY

python3 -m py_compile \
  src/stage2/verify_production_engine.py
```

## 13.3 Run the verifier

```bash
cd /data/saral/wdir/smart_v2 || exit 1
source ./smart_v2_env.sh

python3 -m src.stage2.verify_production_engine \
  --pool-manifest /mnt/warm_storage/saral/smart_v2/stage2/pools/pool_manifest.csv \
  --embedding-root /mnt/warm_storage/saral/smart_v2/embeddings/gte-large \
  --dense-audit-root /mnt/warm_storage/saral/smart_v2/stage2/audit \
  --production-audit-root /mnt/warm_storage/saral/smart_v2/stage2/production_audit \
  2>&1 | tee \
  /mnt/warm_storage/saral/smart_v2/logs/verify_production_engine.log
```

Acceptance conditions:

```text
status                    = verified
verified tasks            = 3
failed tasks              = 0
true mismatches           = 0
cross-GPU order matches   = 3/3
production replay errors  <= 1e-8
objective relative diff   <= 1e-7
mean coverage diff        <= 1e-7
```

Exact or numerical ties against dense Submodlib remain acceptable. A cross-GPU ordering difference or true mismatch is not acceptable.
