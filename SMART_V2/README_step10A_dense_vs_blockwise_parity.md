Once the pool manifest passed. It confirms why we need the matrix-free implementation: 31 pools require more than 16 GiB for the raw dense float32 kernel, and QQP alone would require about 493 GiB.

The next step is **one exact parity smoke test** on `sglue::copa`:

* run the authors’ dense `FacilityLocationFunction` over the complete 500-example pool;
* generate its full (n-1) ordering;
* run our matrix-free exact LazyGreedy implementation to the required prefix of 166;
* require identical selected indices;
* independently recompute every marginal gain from explicit cosine similarities.

The final mixture code only consumes prefixes from the saved orderings, so proving prefix parity is the relevant correctness test.  Submodlib provides Facility Location and LazyGreedy as separate objective and optimizer components. ([GitHub][1])

# Step 10A — Dense versus exact blockwise parity

## 1. Create the parity script

```bash
cd /data/saral/wdir/smart_v2 || exit 1

cat > src/stage2/validate_blockwise_fl.py <<'PY'
"""Compare the authors' dense Facility Location ordering with an
exact matrix-free implementation on one task/template pool.
"""

from __future__ import annotations

import argparse
import csv
import heapq
import inspect
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import submodlib.functions as submod_fn
import torch


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
        "--task-id",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
    )
    parser.add_argument(
        "--singleton-candidate-batch",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--gain-atol",
        type=float,
        default=2e-3,
    )
    parser.add_argument(
        "--gain-rtol",
        type=float,
        default=1e-5,
    )

    return parser.parse_args()


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


def load_pool(
    path: Path,
    task_id: str,
) -> dict[str, Any]:
    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        rows = list(csv.DictReader(handle))

    matches = [
        row
        for row in rows
        if row["task_id"] == task_id
    ]

    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one pool row for {task_id!r}; "
            f"found {len(matches)}."
        )

    raw = matches[0]

    return {
        "task_id": raw["task_id"],
        "corpus": raw["corpus"],
        "template_type": raw["template_type"],
        "pool_size": int(raw["pool_size"]),
        "required_stage2_prefix": int(
            raw["required_stage2_prefix"]
        ),
        "first_dataset_index": int(
            raw["first_dataset_index"]
        ),
        "last_dataset_index": int(
            raw["last_dataset_index"]
        ),
    }


def normalize_numpy(
    embeddings: np.ndarray,
) -> np.ndarray:
    norms = np.linalg.norm(
        embeddings,
        axis=1,
        keepdims=True,
    )

    if np.any(norms <= 0.0):
        raise RuntimeError(
            "Pool contains a zero-norm embedding."
        )

    return (
        embeddings / norms
    ).astype(
        np.float32,
        copy=False,
    )


def explicit_order_gains(
    normalized: np.ndarray,
    order: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Recompute FL gains directly from cosine similarities."""

    coverage = np.zeros(
        normalized.shape[0],
        dtype=np.float32,
    )

    gains: list[float] = []
    objectives: list[float] = []

    for candidate in order:
        similarities = (
            normalized
            @ normalized[candidate]
        )

        updated = np.maximum(
            coverage,
            similarities,
        )

        gain = float(
            np.sum(
                updated,
                dtype=np.float64,
            )
            - np.sum(
                coverage,
                dtype=np.float64,
            )
        )

        coverage = updated

        gains.append(gain)
        objectives.append(
            float(
                np.sum(
                    coverage,
                    dtype=np.float64,
                )
            )
        )

    return (
        np.asarray(
            gains,
            dtype=np.float64,
        ),
        np.asarray(
            objectives,
            dtype=np.float64,
        ),
    )


def exact_singleton_gains(
    normalized: torch.Tensor,
    candidate_batch: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute all singleton FL gains without storing an n by n matrix.

    When the spherical-cap bound proves every pairwise cosine is
    positive, singleton gains equal:

        normalized @ normalized.sum(axis=0)

    Otherwise similarities are computed in exact candidate blocks.
    """

    centroid = normalized.sum(
        dim=0,
        dtype=torch.float64,
    )

    centroid_norm = torch.linalg.vector_norm(
        centroid
    )

    if float(centroid_norm.item()) <= 0.0:
        raise RuntimeError(
            "Pool centroid has zero norm."
        )

    centroid = centroid / centroid_norm

    centroid_cosines = torch.mv(
        normalized.to(
            dtype=torch.float64
        ),
        centroid,
    )

    minimum_centroid_cosine = float(
        centroid_cosines.min().item()
    )

    # If every point is within angle theta of the centroid,
    # every pair is within 2*theta. Therefore:
    #
    # pairwise cosine >= 2*m^2 - 1,
    #
    # where m is the minimum centroid cosine.
    pairwise_lower_bound = (
        2.0
        * minimum_centroid_cosine**2
        - 1.0
    )

    if pairwise_lower_bound > 1e-6:
        vector_sum = normalized.sum(
            dim=0
        )

        gains = torch.mv(
            normalized,
            vector_sum,
        )

        return gains, {
            "singleton_method": (
                "positive_spherical_cap_column_sum"
            ),
            "minimum_centroid_cosine": (
                minimum_centroid_cosine
            ),
            "proven_pairwise_cosine_lower_bound": (
                pairwise_lower_bound
            ),
        }

    n = int(normalized.shape[0])

    gains = torch.empty(
        n,
        dtype=torch.float32,
        device=normalized.device,
    )

    for start in range(
        0,
        n,
        candidate_batch,
    ):
        end = min(
            start + candidate_batch,
            n,
        )

        similarities = (
            normalized
            @ normalized[start:end].T
        )

        gains[start:end] = torch.clamp_min(
            similarities,
            0.0,
        ).sum(
            dim=0
        )

    return gains, {
        "singleton_method": (
            "exact_blockwise_positive_part"
        ),
        "minimum_centroid_cosine": (
            minimum_centroid_cosine
        ),
        "proven_pairwise_cosine_lower_bound": (
            pairwise_lower_bound
        ),
    }


def blockwise_lazy_greedy(
    embeddings: np.ndarray,
    budget: int,
    device_name: str,
    singleton_candidate_batch: int,
) -> tuple[
    list[tuple[int, float]],
    dict[str, Any],
]:
    """Run exact lazy greedy without storing a dense kernel."""

    n = embeddings.shape[0]

    if not 0 < budget < n:
        raise ValueError(
            "Budget must satisfy 0 < budget < pool size."
        )

    device = torch.device(
        device_name
    )

    if (
        device.type == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA was requested but is unavailable."
        )

    torch.set_grad_enabled(False)
    torch.set_float32_matmul_precision(
        "highest"
    )

    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(
            device
        )

        # Avoid TF32 changing cosine comparisons.
        torch.backends.cuda.matmul.allow_tf32 = (
            False
        )

    matrix = torch.from_numpy(
        np.ascontiguousarray(
            embeddings
        )
    ).to(
        device=device
    )

    norms = torch.linalg.vector_norm(
        matrix,
        dim=1,
        keepdim=True,
    )

    if bool(
        torch.any(norms <= 0.0).item()
    ):
        raise RuntimeError(
            "Pool contains a zero-norm embedding."
        )

    normalized = matrix / norms

    del matrix
    del norms

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    singleton_start = (
        time.perf_counter()
    )

    singleton_gains, singleton_info = (
        exact_singleton_gains(
            normalized=normalized,
            candidate_batch=(
                singleton_candidate_batch
            ),
        )
    )

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    singleton_seconds = (
        time.perf_counter()
        - singleton_start
    )

    singleton_cpu = (
        singleton_gains
        .detach()
        .cpu()
        .numpy()
        .astype(
            np.float64,
            copy=False,
        )
    )

    if not np.isfinite(
        singleton_cpu
    ).all():
        raise RuntimeError(
            "Singleton gains contain "
            "non-finite values."
        )

    # Entries contain:
    #
    #   (-upper_bound, candidate_index, state_version)
    #
    # At state k, an entry stamped k is an exact gain for
    # the current selected set. Older entries remain valid
    # upper bounds by submodularity.
    heap: list[
        tuple[float, int, int]
    ] = [
        (
            -float(
                singleton_cpu[index]
            ),
            index,
            0,
        )
        for index in range(n)
    ]

    heapq.heapify(heap)

    selected_mask = np.zeros(
        n,
        dtype=bool,
    )

    coverage = torch.zeros(
        n,
        dtype=torch.float32,
        device=device,
    )

    result: list[
        tuple[int, float]
    ] = []

    exact_gain_evaluations = 0

    selection_start = (
        time.perf_counter()
    )

    for state in range(budget):
        while True:
            (
                negative_gain,
                candidate,
                stamp,
            ) = heapq.heappop(heap)

            if selected_mask[candidate]:
                continue

            if stamp == state:
                selected_gain = (
                    -negative_gain
                )

                similarities = torch.mv(
                    normalized,
                    normalized[candidate],
                )

                check_gain = float(
                    torch.clamp_min(
                        similarities
                        - coverage,
                        0.0,
                    ).sum().item()
                )

                exact_gain_evaluations += 1

                if not math.isclose(
                    selected_gain,
                    check_gain,
                    rel_tol=1e-6,
                    abs_tol=2e-3,
                ):
                    raise RuntimeError(
                        "Candidate gain changed within "
                        "the same lazy-greedy state: "
                        f"heap={selected_gain}, "
                        f"recomputed={check_gain}."
                    )

                coverage = torch.maximum(
                    coverage,
                    similarities,
                )

                selected_mask[
                    candidate
                ] = True

                result.append(
                    (
                        candidate,
                        selected_gain,
                    )
                )

                break

            similarities = torch.mv(
                normalized,
                normalized[candidate],
            )

            gain = float(
                torch.clamp_min(
                    similarities - coverage,
                    0.0,
                ).sum().item()
            )

            exact_gain_evaluations += 1

            heapq.heappush(
                heap,
                (
                    -gain,
                    candidate,
                    state,
                ),
            )

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    selection_seconds = (
        time.perf_counter()
        - selection_start
    )

    peak_device_bytes = (
        int(
            torch.cuda.max_memory_allocated(
                device
            )
        )
        if device.type == "cuda"
        else 0
    )

    return result, {
        **singleton_info,
        "singleton_seconds": (
            singleton_seconds
        ),
        "selection_seconds": (
            selection_seconds
        ),
        "exact_gain_evaluations": (
            exact_gain_evaluations
        ),
        "peak_device_bytes": (
            peak_device_bytes
        ),
    }


def main() -> int:
    args = parse_args()

    pool_manifest = (
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

    pool = load_pool(
        pool_manifest,
        args.task_id,
    )

    n = pool["pool_size"]
    prefix = pool[
        "required_stage2_prefix"
    ]

    if not 0 < prefix < n:
        raise RuntimeError(
            "Parity task must have "
            "0 < prefix < pool size; "
            f"prefix={prefix}, n={n}."
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
        pool["last_dataset_index"]
        + 1
    )

    if end - start != n:
        raise RuntimeError(
            "Pool interval does not match "
            "the recorded pool size."
        )

    embeddings = np.asarray(
        matrix[start:end],
        dtype=np.float32,
    ).copy()

    del matrix

    print(
        "=== Facility Location parity smoke test ==="
    )
    print(f"Task:          {pool['task_id']}")
    print(f"Corpus:        {pool['corpus']}")
    print(f"Pool size:     {n:,}")
    print(f"Prefix:        {prefix}")
    print(
        f"Embedding:     "
        f"{embeddings.shape} "
        f"{embeddings.dtype}"
    )
    print(f"Device:        {args.device}")
    print()

    # Exact authors-code oracle:
    # construct the dense Facility Location object from data
    # and request its complete n-1 ordering.
    dense_start = (
        time.perf_counter()
    )

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

    dense_full = (
        dense_objective.maximize(
            budget=n - 1,
            optimizer="LazyGreedy",
            show_progress=True,
        )
    )

    dense_seconds = (
        time.perf_counter()
        - dense_start
    )

    if len(dense_full) != n - 1:
        raise RuntimeError(
            f"Dense oracle returned "
            f"{len(dense_full)} elements; "
            f"expected {n - 1}."
        )

    dense_prefix = [
        (
            int(index),
            float(gain),
        )
        for index, gain
        in dense_full[:prefix]
    ]

    blockwise_start = (
        time.perf_counter()
    )

    (
        blockwise_prefix,
        blockwise_info,
    ) = blockwise_lazy_greedy(
        embeddings=embeddings,
        budget=prefix,
        device_name=args.device,
        singleton_candidate_batch=(
            args.singleton_candidate_batch
        ),
    )

    blockwise_seconds = (
        time.perf_counter()
        - blockwise_start
    )

    dense_indices = [
        index
        for index, _ in dense_prefix
    ]

    blockwise_indices = [
        index
        for index, _ in blockwise_prefix
    ]

    dense_gains = np.asarray(
        [
            gain
            for _, gain in dense_prefix
        ],
        dtype=np.float64,
    )

    blockwise_gains = np.asarray(
        [
            gain
            for _, gain
            in blockwise_prefix
        ],
        dtype=np.float64,
    )

    selection_match = (
        dense_indices
        == blockwise_indices
    )

    matching_prefix_length = 0

    for dense_index, block_index in zip(
        dense_indices,
        blockwise_indices,
        strict=True,
    ):
        if dense_index != block_index:
            break

        matching_prefix_length += 1

    normalized = normalize_numpy(
        embeddings
    )

    (
        explicit_dense_gains,
        explicit_dense_objectives,
    ) = explicit_order_gains(
        normalized,
        dense_indices,
    )

    (
        explicit_blockwise_gains,
        explicit_blockwise_objectives,
    ) = explicit_order_gains(
        normalized,
        blockwise_indices,
    )

    dense_gain_errors = np.abs(
        dense_gains
        - explicit_dense_gains
    )

    blockwise_gain_errors = np.abs(
        blockwise_gains
        - explicit_blockwise_gains
    )

    cross_gain_errors = np.abs(
        dense_gains
        - blockwise_gains
    )

    dense_gains_valid = bool(
        np.allclose(
            dense_gains,
            explicit_dense_gains,
            rtol=args.gain_rtol,
            atol=args.gain_atol,
        )
    )

    blockwise_gains_valid = bool(
        np.allclose(
            blockwise_gains,
            explicit_blockwise_gains,
            rtol=args.gain_rtol,
            atol=args.gain_atol,
        )
    )

    cross_gains_match = bool(
        np.allclose(
            dense_gains,
            blockwise_gains,
            rtol=args.gain_rtol,
            atol=args.gain_atol,
        )
    )

    # Safe here because the parity pool contains only 500 rows.
    full_kernel = (
        normalized
        @ normalized.T
    )

    minimum_pairwise_cosine = float(
        full_kernel.min()
    )

    maximum_pairwise_cosine = float(
        full_kernel.max()
    )

    singleton_exact = np.maximum(
        full_kernel,
        0.0,
    ).sum(
        axis=0,
        dtype=np.float64,
    )

    vector_sum_singleton = (
        normalized
        @ normalized.sum(axis=0)
    ).astype(
        np.float64
    )

    singleton_shortcut_error = float(
        np.max(
            np.abs(
                singleton_exact
                - vector_sum_singleton
            )
        )
    )

    status = (
        "verified"
        if (
            selection_match
            and dense_gains_valid
            and blockwise_gains_valid
            and cross_gains_match
        )
        else "failed"
    )

    output_rows: list[
        dict[str, Any]
    ] = []

    for rank in range(prefix):
        output_rows.append(
            {
                "rank": rank + 1,
                "dense_local_index": (
                    dense_indices[rank]
                ),
                "blockwise_local_index": (
                    blockwise_indices[rank]
                ),
                "dense_dataset_index": (
                    start
                    + dense_indices[rank]
                ),
                "blockwise_dataset_index": (
                    start
                    + blockwise_indices[
                        rank
                    ]
                ),
                "dense_gain": (
                    dense_gains[rank]
                ),
                "blockwise_gain": (
                    blockwise_gains[rank]
                ),
                "explicit_dense_gain": (
                    explicit_dense_gains[
                        rank
                    ]
                ),
                "explicit_blockwise_gain": (
                    explicit_blockwise_gains[
                        rank
                    ]
                ),
            }
        )

    order_path = (
        output_root / "parity_order.csv"
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
        writer.writerows(
            output_rows
        )

    report = {
        "format_version": 1,
        "stage": (
            "smart_v2_dense_blockwise_fl_parity"
        ),
        "status": status,
        "task": pool,
        "dense_author_code": {
            "constructor_signature": str(
                inspect.signature(
                    submod_fn
                    .facilityLocation
                    .FacilityLocationFunction
                )
            ),
            "ordering_budget": n - 1,
            "compared_prefix": prefix,
            "elapsed_seconds": (
                dense_seconds
            ),
        },
        "blockwise": {
            "compared_prefix": prefix,
            "elapsed_seconds": (
                blockwise_seconds
            ),
            **blockwise_info,
        },
        "similarity_diagnostics": {
            "minimum_pairwise_cosine": (
                minimum_pairwise_cosine
            ),
            "maximum_pairwise_cosine": (
                maximum_pairwise_cosine
            ),
            "singleton_shortcut_max_absolute_error": (
                singleton_shortcut_error
            ),
        },
        "parity": {
            "selection_match": (
                selection_match
            ),
            "matching_prefix_length": (
                matching_prefix_length
            ),
            "dense_gain_matches_explicit_cosine": (
                dense_gains_valid
            ),
            "blockwise_gain_matches_explicit_cosine": (
                blockwise_gains_valid
            ),
            "dense_and_blockwise_gains_match": (
                cross_gains_match
            ),
            "maximum_dense_explicit_gain_error": float(
                dense_gain_errors.max()
            ),
            "maximum_blockwise_explicit_gain_error": float(
                blockwise_gain_errors.max()
            ),
            "maximum_cross_gain_error": float(
                cross_gain_errors.max()
            ),
            "dense_final_explicit_objective": float(
                explicit_dense_objectives[
                    -1
                ]
            ),
            "blockwise_final_explicit_objective": float(
                explicit_blockwise_objectives[
                    -1
                ]
            ),
        },
        "outputs": {
            "order_csv": str(
                order_path
            ),
        },
    }

    report_path = (
        output_root / "parity_report.json"
    )

    atomic_write_json(
        report,
        report_path,
    )

    print()
    print("=== Parity summary ===")
    print(
        f"Status:                         "
        f"{status}"
    )
    print(
        f"Selection match:                "
        f"{selection_match}"
    )
    print(
        f"Matching prefix:                "
        f"{matching_prefix_length}/{prefix}"
    )
    print(
        f"Dense gain vs explicit cosine:  "
        f"{dense_gains_valid}"
    )
    print(
        f"Blockwise vs explicit cosine:   "
        f"{blockwise_gains_valid}"
    )
    print(
        f"Dense/blockwise gains match:    "
        f"{cross_gains_match}"
    )
    print(
        f"Maximum cross gain error:       "
        f"{cross_gain_errors.max():.6g}"
    )
    print(
        f"Minimum pairwise cosine:        "
        f"{minimum_pairwise_cosine:.8f}"
    )
    print(
        f"Dense elapsed:                  "
        f"{dense_seconds:.3f}s"
    )
    print(
        f"Blockwise elapsed:              "
        f"{blockwise_seconds:.3f}s"
    )
    print(
        f"Blockwise gain evaluations:     "
        f"{blockwise_info['exact_gain_evaluations']:,}"
    )
    print(
        f"Report:                         "
        f"{report_path}"
    )

    if status != "verified":
        raise RuntimeError(
            "Facility Location parity failed. "
            "Inspect parity_report.json and "
            "parity_order.csv."
        )

    print()
    print(
        "Dense and exact blockwise "
        "Facility Location agree."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
PY

python3 -m py_compile \
  src/stage2/validate_blockwise_fl.py
```

## 2. Run the tiny-pool parity test

```bash
cd /data/saral/wdir/smart_v2 || exit 1
source ./smart_v2_env.sh

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

mkdir -p \
  /mnt/warm_storage/saral/smart_v2/stage2/parity/sglue__copa

python3 -m src.stage2.validate_blockwise_fl \
  --pool-manifest /mnt/warm_storage/saral/smart_v2/stage2/pools/pool_manifest.csv \
  --embedding-root /mnt/warm_storage/saral/smart_v2/embeddings/gte-large \
  --task-id sglue::copa \
  --output-root /mnt/warm_storage/saral/smart_v2/stage2/parity/sglue__copa \
  --device cuda:0 \
  --singleton-candidate-batch 256 \
  --gain-atol 0.002 \
  --gain-rtol 0.00001 \
  2>&1 | tee \
  /mnt/warm_storage/saral/smart_v2/logs/stage2_parity_sglue_copa.log
```

## Acceptance conditions

```text
Status                           = verified
Selection match                  = True
Matching prefix                  = 166/166
Dense gain vs explicit cosine    = True
Blockwise vs explicit cosine     = True
Dense/blockwise gains match      = True
```

Outputs:

```text
/mnt/warm_storage/saral/smart_v2/stage2/parity/sglue__copa/
├── parity_order.csv
└── parity_report.json
```

A failure is not grounds to loosen tolerances or accept a different ordering. It must be diagnosed as one of:

* cosine-kernel semantic mismatch;
* floating-point accumulation difference;
* LazyGreedy tie-breaking difference;
* an implementation error.

[1]: https://github.com/decile-team/submodlib "GitHub - decile-team/submodlib: Summarize Massive Datasets using Submodular Optimization · GitHub"
