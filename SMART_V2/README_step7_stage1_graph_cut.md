Once ,the task means are correct. Their norms are below 1 because they are averages of prompt embeddings pointing in different directions; that is expected.

## Step 7 — Run exact Stage 1 Graph Cut

The authors construct a cosine kernel over task embeddings, instantiate dense Graph Cut with `lambdaVal=0.4`, and run `LazyGreedy`. 

We will preserve that behavior, including the exact `budget == 309` compatibility workaround.

### 7.1 Confirm Submodlib

```bash
cd /data/saral/wdir/smart_v2 || exit 1
source ./smart_v2_env.sh

python3 - <<'PY'
import importlib.metadata
import submodlib

print(
    "submodlib-py:",
    importlib.metadata.version("submodlib-py"),
)
PY
```

Expected:

```text
submodlib-py: 0.0.3
```

### 7.2 Create the Graph Cut script

```bash
cat > src/stage1/run_graph_cut.py <<'PY'
"""Run SMART Stage 1 task-level Graph Cut."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import platform
import time
from pathlib import Path
from typing import Any

import importlib.metadata
import numpy as np
import submodlib
import submodlib.functions as submod_fn


EXPECTED_TASK_COUNT = 309
EXPECTED_DIMENSION = 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--task-embedding-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=309,
    )
    parser.add_argument(
        "--lambda-value",
        type=float,
        default=0.4,
    )
    parser.add_argument(
        "--determinism-runs",
        type=int,
        default=2,
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


def load_catalog(
    path: Path,
) -> list[dict[str, Any]]:
    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        raw_rows = list(csv.DictReader(handle))

    rows: list[dict[str, Any]] = []

    for raw in raw_rows:
        rows.append(
            {
                "task_index": int(raw["task_index"]),
                "task_rank_1based": int(
                    raw["task_rank_1based"]
                ),
                "corpus": raw["corpus"],
                "task_id": raw["task_id"],
                "valid_train_count": int(
                    raw["valid_train_count"]
                ),
                "mean_embedding_norm": float(
                    raw["mean_embedding_norm"]
                ),
            }
        )

    rows.sort(
        key=lambda row: row["task_index"]
    )

    expected_indices = list(
        range(len(rows))
    )

    actual_indices = [
        row["task_index"]
        for row in rows
    ]

    if actual_indices != expected_indices:
        raise RuntimeError(
            "Task catalog indices are not contiguous."
        )

    return rows


def make_graph_cut(
    kernel: np.ndarray,
    lambda_value: float,
) -> submod_fn.graphCut.GraphCutFunction:
    return submod_fn.graphCut.GraphCutFunction(
        n=int(kernel.shape[0]),
        mode="dense",
        ggsijs=kernel,
        lambdaVal=lambda_value,
        separate_rep=False,
    )


def full_lazy_greedy(
    kernel: np.ndarray,
    budget: int,
    lambda_value: float,
    show_progress: bool,
) -> list[tuple[int, float]]:
    """Run exact LazyGreedy including submodlib's budget==n case."""

    n = int(kernel.shape[0])

    if budget <= 0:
        raise ValueError(
            "Budget must be positive."
        )

    if budget > n:
        raise ValueError(
            f"Budget {budget} exceeds ground-set size {n}."
        )

    objective = make_graph_cut(
        kernel=kernel,
        lambda_value=lambda_value,
    )

    if budget < n:
        result = objective.maximize(
            budget=budget,
            optimizer="LazyGreedy",
            stopIfZeroGain=False,
            stopIfNegativeGain=False,
            verbose=False,
            show_progress=show_progress,
        )

        return [
            (int(index), float(gain))
            for index, gain in result
        ]

    # submodlib-py 0.0.3 requires budget < n.
    # LazyGreedy selects the first n-1 elements. The only remaining
    # element is necessarily the final greedy element.
    if n == 1:
        gain = float(
            objective.marginalGain(
                set(),
                0,
            )
        )
        return [(0, gain)]

    result = objective.maximize(
        budget=n - 1,
        optimizer="LazyGreedy",
        stopIfZeroGain=False,
        stopIfNegativeGain=False,
        verbose=False,
        show_progress=show_progress,
    )

    result = [
        (int(index), float(gain))
        for index, gain in result
    ]

    selected = {
        index
        for index, _ in result
    }

    remaining = sorted(
        set(range(n)) - selected
    )

    if len(remaining) != 1:
        raise RuntimeError(
            "Expected exactly one remaining task after "
            f"n-1 selections; found {remaining}."
        )

    final_index = remaining[0]

    final_gain = float(
        objective.marginalGain(
            selected,
            final_index,
        )
    )

    previous_value = float(
        objective.evaluate(selected)
    )
    complete_value = float(
        objective.evaluate(
            selected | {final_index}
        )
    )

    evaluation_gain = (
        complete_value - previous_value
    )

    if not np.isclose(
        final_gain,
        evaluation_gain,
        rtol=1e-6,
        atol=1e-5,
    ):
        raise RuntimeError(
            "Final marginal gain validation failed: "
            f"marginalGain={final_gain:.12g}, "
            f"evaluate difference={evaluation_gain:.12g}."
        )

    result.append(
        (final_index, final_gain)
    )

    return result


def compare_results(
    reference: list[tuple[int, float]],
    candidate: list[tuple[int, float]],
) -> tuple[bool, bool, float]:
    reference_indices = [
        index
        for index, _ in reference
    ]
    candidate_indices = [
        index
        for index, _ in candidate
    ]

    order_equal = (
        reference_indices
        == candidate_indices
    )

    reference_gains = np.asarray(
        [
            gain
            for _, gain in reference
        ],
        dtype=np.float64,
    )

    candidate_gains = np.asarray(
        [
            gain
            for _, gain in candidate
        ],
        dtype=np.float64,
    )

    maximum_gain_difference = float(
        np.max(
            np.abs(
                reference_gains
                - candidate_gains
            )
        )
    )

    gains_equal = bool(
        np.array_equal(
            reference_gains,
            candidate_gains,
        )
    )

    return (
        order_equal,
        gains_equal,
        maximum_gain_difference,
    )


def main() -> int:
    args = parse_args()

    if args.determinism_runs < 2:
        raise ValueError(
            "--determinism-runs must be at least 2."
        )

    task_root = (
        args.task_embedding_root.resolve()
    )
    output_root = args.output_root.resolve()

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    tasks_path = task_root / "tasks.pkl"
    embeddings_path = (
        task_root / "tasks_embeddings.npy"
    )
    catalog_path = (
        task_root / "task_catalog.csv"
    )
    verification_path = (
        task_root / "verification_report.json"
    )

    for path in (
        tasks_path,
        embeddings_path,
        catalog_path,
        verification_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    with verification_path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        verification = json.load(handle)

    if verification.get("status") != "verified":
        raise RuntimeError(
            "Task embeddings are not verified."
        )

    with tasks_path.open("rb") as handle:
        tasks = pickle.load(handle)

    embeddings = np.load(
        embeddings_path,
    )

    catalog = load_catalog(
        catalog_path
    )

    if len(tasks) != EXPECTED_TASK_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_TASK_COUNT} tasks; "
            f"found {len(tasks)}."
        )

    if embeddings.shape != (
        EXPECTED_TASK_COUNT,
        EXPECTED_DIMENSION,
    ):
        raise RuntimeError(
            f"Unexpected task embedding shape: "
            f"{embeddings.shape}"
        )

    if embeddings.dtype != np.float32:
        raise RuntimeError(
            f"Unexpected task embedding dtype: "
            f"{embeddings.dtype}"
        )

    if len(catalog) != EXPECTED_TASK_COUNT:
        raise RuntimeError(
            "Task catalog has the wrong number of rows."
        )

    for index, task_id in enumerate(tasks):
        if catalog[index]["task_id"] != task_id:
            raise RuntimeError(
                f"Task order mismatch at index {index}."
            )

    if args.budget != EXPECTED_TASK_COUNT:
        raise RuntimeError(
            "This primary baseline requires budget=309."
        )

    print("=== SMART-v2 Stage 1 Graph Cut ===")
    print(f"Tasks:             {len(tasks)}")
    print(f"Embedding shape:   {embeddings.shape}")
    print(f"Budget:            {args.budget}")
    print(f"Lambda:            {args.lambda_value}")
    print("Optimizer:         LazyGreedy")
    print(
        "Kernel:            dense cosine via "
        "submodlib/sklearn"
    )
    print(f"Output root:       {output_root}")
    print()

    kernel_start = time.perf_counter()

    kernel = submodlib.helper.create_kernel(
        X=embeddings,
        metric="cosine",
        method="sklearn",
    )

    kernel = np.asarray(
        kernel,
        dtype=np.float32,
    )

    kernel_elapsed = (
        time.perf_counter()
        - kernel_start
    )

    expected_kernel_shape = (
        EXPECTED_TASK_COUNT,
        EXPECTED_TASK_COUNT,
    )

    if kernel.shape != expected_kernel_shape:
        raise RuntimeError(
            f"Kernel shape {kernel.shape} "
            f"!= {expected_kernel_shape}."
        )

    if not np.isfinite(kernel).all():
        raise RuntimeError(
            "Kernel contains non-finite values."
        )

    symmetry_error = float(
        np.max(
            np.abs(
                kernel - kernel.T
            )
        )
    )

    diagonal_error = float(
        np.max(
            np.abs(
                np.diag(kernel) - 1.0
            )
        )
    )

    norms = np.linalg.norm(
        embeddings,
        axis=1,
        keepdims=True,
    )

    normalized = (
        embeddings
        / norms
    ).astype(
        np.float32,
        copy=False,
    )

    manual_kernel = (
        normalized @ normalized.T
    ).astype(
        np.float32,
        copy=False,
    )

    manual_cosine_error = float(
        np.max(
            np.abs(
                kernel - manual_kernel
            )
        )
    )

    if symmetry_error > 1e-6:
        raise RuntimeError(
            f"Kernel symmetry error is too large: "
            f"{symmetry_error:.9g}"
        )

    if diagonal_error > 1e-5:
        raise RuntimeError(
            f"Kernel diagonal error is too large: "
            f"{diagonal_error:.9g}"
        )

    if manual_cosine_error > 1e-5:
        raise RuntimeError(
            "Submodlib kernel does not match manual "
            f"cosine computation: {manual_cosine_error:.9g}"
        )

    kernel_path = (
        output_root
        / "task_similarity_cosine.npy"
    )

    np.save(
        kernel_path,
        kernel,
        allow_pickle=False,
    )

    print(
        f"Kernel construction: {kernel_elapsed:.3f} seconds"
    )
    print(f"Kernel dtype:        {kernel.dtype}")
    print(
        f"Kernel range:        "
        f"{kernel.min():.8f} to "
        f"{kernel.max():.8f}"
    )
    print(
        f"Symmetry error:      {symmetry_error:.3e}"
    )
    print(
        f"Diagonal error:      {diagonal_error:.3e}"
    )
    print(
        f"Manual cosine error: {manual_cosine_error:.3e}"
    )
    print()

    runs: list[
        list[tuple[int, float]]
    ] = []

    run_times: list[float] = []

    for run_number in range(
        1,
        args.determinism_runs + 1,
    ):
        run_start = time.perf_counter()

        result = full_lazy_greedy(
            kernel=kernel,
            budget=args.budget,
            lambda_value=args.lambda_value,
            show_progress=(
                run_number == 1
            ),
        )

        elapsed = (
            time.perf_counter()
            - run_start
        )

        runs.append(result)
        run_times.append(elapsed)

        print(
            f"Graph Cut run {run_number}/"
            f"{args.determinism_runs}: "
            f"{len(result)} selections in "
            f"{elapsed:.3f} seconds"
        )

    reference = runs[0]

    deterministic_order = True
    deterministic_gains = True
    maximum_rerun_gain_difference = 0.0

    for candidate in runs[1:]:
        (
            order_equal,
            gains_equal,
            gain_difference,
        ) = compare_results(
            reference,
            candidate,
        )

        deterministic_order &= order_equal
        deterministic_gains &= gains_equal
        maximum_rerun_gain_difference = max(
            maximum_rerun_gain_difference,
            gain_difference,
        )

    if not deterministic_order:
        raise RuntimeError(
            "Graph Cut order changed across repeated runs."
        )

    if not deterministic_gains:
        raise RuntimeError(
            "Graph Cut gains changed across repeated runs."
        )

    selected_indices = [
        index
        for index, _ in reference
    ]

    gains = np.asarray(
        [
            gain
            for _, gain in reference
        ],
        dtype=np.float64,
    )

    if len(reference) != args.budget:
        raise RuntimeError(
            f"Expected {args.budget} selections; "
            f"received {len(reference)}."
        )

    if len(set(selected_indices)) != args.budget:
        raise RuntimeError(
            "Graph Cut result contains duplicate task indices."
        )

    if set(selected_indices) != set(
        range(EXPECTED_TASK_COUNT)
    ):
        raise RuntimeError(
            "Graph Cut result does not contain all tasks."
        )

    if not np.isfinite(gains).all():
        raise RuntimeError(
            "Graph Cut gains contain non-finite values."
        )

    cumulative_gains = np.cumsum(
        gains,
        dtype=np.float64,
    )

    output_rows: list[
        dict[str, Any]
    ] = []

    for rank_zero_based, (
        task_index,
        marginal_gain,
    ) in enumerate(reference):
        task = catalog[task_index]

        output_rows.append(
            {
                "graph_cut_rank": (
                    rank_zero_based + 1
                ),
                "task_index": task_index,
                "task_id": task["task_id"],
                "corpus": task["corpus"],
                "valid_train_count": (
                    task["valid_train_count"]
                ),
                "task_embedding_norm": (
                    task["mean_embedding_norm"]
                ),
                "marginal_gain": marginal_gain,
                "cumulative_gain": float(
                    cumulative_gains[
                        rank_zero_based
                    ]
                ),
            }
        )

    order_path = (
        output_root
        / "graph_cut_order.csv"
    )

    with order_path.open(
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

    nonpositive_count = int(
        np.sum(gains <= 0.0)
    )
    negative_count = int(
        np.sum(gains < 0.0)
    )

    result_path = (
        output_root
        / "graph_cut_result.json"
    )

    atomic_write_json(
        {
            "format_version": 1,
            "stage": (
                "smart_v2_stage1_graph_cut"
            ),
            "status": "complete",
            "configuration": {
                "task_count": (
                    EXPECTED_TASK_COUNT
                ),
                "budget": args.budget,
                "lambda_value": (
                    args.lambda_value
                ),
                "optimizer": "LazyGreedy",
                "kernel": (
                    "submodlib.helper.create_kernel "
                    "metric=cosine method=sklearn"
                ),
                "mode": "dense",
                "separate_rep": False,
                "stop_if_zero_gain": False,
                "stop_if_negative_gain": False,
                "determinism_runs": (
                    args.determinism_runs
                ),
                "full_budget_compatibility": {
                    "library": (
                        "submodlib-py 0.0.3"
                    ),
                    "reason": (
                        "maximize requires budget to be "
                        "strictly less than effective "
                        "ground-set size"
                    ),
                    "library_budget": (
                        EXPECTED_TASK_COUNT - 1
                    ),
                    "completion": (
                        "Append the unique remaining task "
                        "with exact marginalGain, cross-checked "
                        "using evaluate differences."
                    ),
                    "approximation": False,
                },
            },
            "software": {
                "submodlib_py_version": (
                    importlib.metadata.version(
                        "submodlib-py"
                    )
                ),
                "numpy_version": np.__version__,
                "python_version": (
                    platform.python_version()
                ),
            },
            "inputs": {
                "tasks_pickle": str(
                    tasks_path
                ),
                "tasks_pickle_sha256": (
                    sha256_file(tasks_path)
                ),
                "task_embeddings": str(
                    embeddings_path
                ),
                "task_embeddings_sha256": (
                    sha256_file(
                        embeddings_path
                    )
                ),
                "task_catalog": str(
                    catalog_path
                ),
                "task_catalog_sha256": (
                    sha256_file(catalog_path)
                ),
                "verification_report": str(
                    verification_path
                ),
                "verification_report_sha256": (
                    sha256_file(
                        verification_path
                    )
                ),
            },
            "kernel": {
                "shape": list(kernel.shape),
                "dtype": str(kernel.dtype),
                "minimum": float(
                    kernel.min()
                ),
                "maximum": float(
                    kernel.max()
                ),
                "symmetry_error": (
                    symmetry_error
                ),
                "diagonal_error": (
                    diagonal_error
                ),
                "manual_cosine_error": (
                    manual_cosine_error
                ),
                "construction_seconds": (
                    kernel_elapsed
                ),
                "path": str(kernel_path),
                "sha256": sha256_file(
                    kernel_path
                ),
            },
            "selection": {
                "selection_count": (
                    len(reference)
                ),
                "unique_selection_count": (
                    len(set(selected_indices))
                ),
                "deterministic_order": (
                    deterministic_order
                ),
                "deterministic_gains": (
                    deterministic_gains
                ),
                "maximum_rerun_gain_difference": (
                    maximum_rerun_gain_difference
                ),
                "run_times_seconds": (
                    run_times
                ),
                "minimum_gain": float(
                    gains.min()
                ),
                "maximum_gain": float(
                    gains.max()
                ),
                "nonpositive_gain_count": (
                    nonpositive_count
                ),
                "negative_gain_count": (
                    negative_count
                ),
                "final_cumulative_gain": float(
                    cumulative_gains[-1]
                ),
            },
            "outputs": {
                "graph_cut_order": str(
                    order_path
                ),
                "graph_cut_order_sha256": (
                    sha256_file(order_path)
                ),
            },
        },
        result_path,
    )

    print()
    print("=== Graph Cut summary ===")
    print(
        f"Selected tasks:       "
        f"{len(reference)}/{EXPECTED_TASK_COUNT}"
    )
    print(
        f"Gain range:           "
        f"{gains.min():.6f} to "
        f"{gains.max():.6f}"
    )
    print(
        f"Nonpositive gains:    "
        f"{nonpositive_count}"
    )
    print(
        f"Negative gains:       "
        f"{negative_count}"
    )
    print(
        f"Final cumulative gain:"
        f"{cumulative_gains[-1]:.4f}"
    )

    print()
    print("First 10 selected tasks:")

    for row in output_rows[:10]:
        print(
            f"  {row['graph_cut_rank']:3d}. "
            f"{row['task_id']} "
            f"gain={row['marginal_gain']:.6f}"
        )

    print()
    print(f"Order CSV:   {order_path}")
    print(f"Result JSON: {result_path}")
    print()
    print(
        "SMART-v2 Stage 1 Graph Cut passed."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
PY

python3 -m py_compile \
  src/stage1/run_graph_cut.py
```

### 7.3 Run Stage 1

```bash
cd /data/saral/wdir/smart_v2 || exit 1
source ./smart_v2_env.sh

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

mkdir -p \
  /mnt/warm_storage/saral/smart_v2/stage1/graph_cut

python3 -m src.stage1.run_graph_cut \
  --task-embedding-root /mnt/warm_storage/saral/smart_v2/embeddings/gte-large/task_embeddings \
  --output-root /mnt/warm_storage/saral/smart_v2/stage1/graph_cut \
  --budget 309 \
  --lambda-value 0.4 \
  --determinism-runs 2 \
  2>&1 | tee \
  /mnt/warm_storage/saral/smart_v2/logs/stage1_graph_cut.log
```

Expected artifacts:

```text
/mnt/warm_storage/saral/smart_v2/stage1/graph_cut/
├── graph_cut_order.csv
├── graph_cut_result.json
└── task_similarity_cosine.npy
```

Acceptance conditions:

```text
status                 = complete
selection_count        = 309
unique selections      = 309
deterministic order    = true
deterministic gains    = true
kernel shape           = 309 × 309
kernel finite          = true
symmetry error         <= 1e-6
manual cosine error    <= 1e-5
```

Do not reject the result merely because a marginal gain is zero or negative. The authors request the complete task budget, and all gains must be retained for the Taylor allocation stage.
