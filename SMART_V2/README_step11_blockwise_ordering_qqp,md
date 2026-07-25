The audit passed exactly as required:

* 3/3 pools verified.
* Zero true mismatches.
* Dense and blockwise objective differences were effectively zero.
* Cross-GPU order and gains matched.
* Every divergence was an objective-equivalent tie.

We can now test the production engine on the hardest pool, `sglue::qqp`. We should not launch all 309 pools until this completes successfully.

The saved ordering only needs the required prefix because the authors’ mixture script takes prefixes from the precomputed instance orderings. 

# Step 11 — Production blockwise ordering for QQP

This implementation uses:

* all 363,846 QQP examples;
* GPU float64 normalization and marginal gains;
* no dense similarity matrix;
* exact LazyGreedy upper-bound certification;
* deterministic tie arbitration;
* independent replay verification;
* author-format dataset indices.

## 11.1 Create the production selector

```bash
cd /data/saral/wdir/smart_v2 || exit 1

cat > src/stage2/run_blockwise_ordering.py <<'PY'
"""Generate an exact full-pool Facility Location prefix without n^2 storage.

The implementation uses:

- the complete task/template candidate pool;
- cosine similarities from GTE embeddings;
- GPU float64 arithmetic;
- exact LazyGreedy upper bounds;
- deterministic numerical-tie arbitration;
- independent replay verification.

It does not sample candidates or approximate the Facility Location
objective.
"""

from __future__ import annotations

import argparse
import csv
import heapq
import json
import math
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np
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
        "--tie-atol",
        type=float,
        default=1e-10,
    )
    parser.add_argument(
        "--tie-rtol",
        type=float,
        default=1e-10,
    )
    parser.add_argument(
        "--replay-atol",
        type=float,
        default=1e-8,
    )
    parser.add_argument(
        "--replay-rtol",
        type=float,
        default=1e-10,
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
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


def atomic_write_pickle(
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

    with temporary.open("wb") as handle:
        pickle.dump(
            payload,
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

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
            f"Expected one manifest row for {task_id!r}; "
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
        "dense_kernel_float32_gib": float(
            raw["dense_kernel_float32_gib"]
        ),
    }


def clean_heap_top(
    heap: list[tuple[float, int, int]],
    selected: np.ndarray,
) -> None:
    while heap and selected[heap[0][1]]:
        heapq.heappop(heap)


def numerical_tie_threshold(
    gain: float,
    atol: float,
    rtol: float,
) -> float:
    return (
        atol
        + rtol * max(1.0, abs(gain))
    )


def load_normalized_embeddings(
    *,
    matrix_path: Path,
    start: int,
    end: int,
    device: torch.device,
) -> torch.Tensor:
    full_matrix = np.load(
        matrix_path,
        mmap_mode="r",
    )

    if start < 0 or end > full_matrix.shape[0]:
        raise RuntimeError(
            "Pool interval exceeds the embedding matrix."
        )

    pool_view = np.asarray(
        full_matrix[start:end],
        dtype=np.float32,
    )

    if pool_view.ndim != 2:
        raise RuntimeError(
            f"Unexpected pool shape: {pool_view.shape}"
        )

    # Copy directly to the GPU, converting to float64.
    embeddings = torch.as_tensor(
        pool_view,
        dtype=torch.float64,
        device=device,
    )

    del full_matrix
    del pool_view

    norms = torch.linalg.vector_norm(
        embeddings,
        dim=1,
        keepdim=True,
    )

    if bool(
        torch.any(norms <= 0.0).item()
    ):
        raise RuntimeError(
            "Pool contains a zero-norm embedding."
        )

    embeddings = embeddings / norms

    if not bool(
        torch.isfinite(embeddings).all().item()
    ):
        raise RuntimeError(
            "Normalized embeddings contain "
            "non-finite values."
        )

    return embeddings.contiguous()


def prove_positive_pairwise_cosines(
    embeddings: torch.Tensor,
) -> dict[str, float]:
    """Prove a positive lower bound using a common spherical cap."""

    centroid = embeddings.sum(
        dim=0,
        dtype=torch.float64,
    )

    centroid_norm = torch.linalg.vector_norm(
        centroid
    )

    if float(centroid_norm.item()) <= 0.0:
        raise RuntimeError(
            "Embedding centroid has zero norm."
        )

    centroid = centroid / centroid_norm

    centroid_cosines = torch.mv(
        embeddings,
        centroid,
    )

    minimum_centroid_cosine = float(
        centroid_cosines.min().item()
    )

    # If x and y are each within angle theta of the centroid,
    # their angle is at most 2 theta:
    #
    # cos(x, y) >= cos(2 theta) = 2 cos(theta)^2 - 1.
    pairwise_lower_bound = (
        2.0
        * minimum_centroid_cosine**2
        - 1.0
    )

    return {
        "minimum_centroid_cosine": (
            minimum_centroid_cosine
        ),
        "proven_pairwise_cosine_lower_bound": (
            pairwise_lower_bound
        ),
    }


def exact_gain(
    embeddings: torch.Tensor,
    coverage: torch.Tensor,
    candidate: int,
) -> tuple[float, torch.Tensor]:
    similarities = torch.mv(
        embeddings,
        embeddings[candidate],
    )

    gain = float(
        torch.clamp_min(
            similarities - coverage,
            0.0,
        ).sum(
            dtype=torch.float64
        ).item()
    )

    return gain, similarities


def replay_ordering(
    *,
    embeddings: torch.Tensor,
    ordering: list[tuple[int, float]],
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    coverage = torch.zeros(
        embeddings.shape[0],
        dtype=torch.float64,
        device=embeddings.device,
    )

    maximum_gain_error = 0.0
    objective_values: list[float] = []
    replay_gains: list[float] = []

    for rank, (
        candidate,
        saved_gain,
    ) in enumerate(
        ordering,
        start=1,
    ):
        replay_gain, similarities = exact_gain(
            embeddings,
            coverage,
            candidate,
        )

        gain_error = abs(
            replay_gain - saved_gain
        )

        maximum_gain_error = max(
            maximum_gain_error,
            gain_error,
        )

        if not math.isclose(
            replay_gain,
            saved_gain,
            abs_tol=atol,
            rel_tol=rtol,
        ):
            raise RuntimeError(
                f"Replay gain mismatch at rank {rank}: "
                f"saved={saved_gain:.15g}, "
                f"replayed={replay_gain:.15g}, "
                f"error={gain_error:.6g}."
            )

        coverage = torch.maximum(
            coverage,
            similarities,
        )

        replay_gains.append(
            replay_gain
        )
        objective_values.append(
            float(
                coverage.sum(
                    dtype=torch.float64
                ).item()
            )
        )

    return {
        "maximum_gain_error": (
            maximum_gain_error
        ),
        "final_objective": (
            objective_values[-1]
            if objective_values
            else 0.0
        ),
        "objective_values": (
            objective_values
        ),
        "replay_gains": replay_gains,
    }


def main() -> int:
    args = parse_args()

    if args.tie_atol < 0 or args.tie_rtol < 0:
        raise ValueError(
            "Tie tolerances cannot be negative."
        )

    if args.progress_every <= 0:
        raise ValueError(
            "--progress-every must be positive."
        )

    manifest_path = (
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
        manifest_path,
        args.task_id,
    )

    n = pool["pool_size"]
    budget = pool[
        "required_stage2_prefix"
    ]

    if not 0 < budget < n:
        raise RuntimeError(
            f"Expected 0 < prefix < pool size; "
            f"prefix={budget}, n={n}."
        )

    device = torch.device(
        args.device
    )

    if (
        device.type == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA was requested but is unavailable."
        )

    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(
            device
        )

        torch.backends.cuda.matmul.allow_tf32 = (
            False
        )

    torch.set_grad_enabled(False)
    torch.set_float32_matmul_precision(
        "highest"
    )

    matrix_path = (
        embedding_root
        / pool["corpus"]
        / "train_prompt_embeddings.npy"
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
            "Pool interval does not match pool size."
        )

    print(
        "=== Exact blockwise Facility Location ==="
    )
    print(f"Task:             {pool['task_id']}")
    print(f"Corpus:           {pool['corpus']}")
    print(f"Template:         {pool['template_type']}")
    print(f"Pool size:        {n:,}")
    print(f"Required prefix:  {budget}")
    print(
        f"Dense kernel:     "
        f"{pool['dense_kernel_float32_gib']:.2f} GiB"
    )
    print(f"Device:           {device}")
    print("Arithmetic:       float64")
    print(
        "Tie policy:       smallest dataset index "
        "within numerical tie"
    )
    print()

    total_start = time.perf_counter()

    load_start = time.perf_counter()

    embeddings = load_normalized_embeddings(
        matrix_path=matrix_path,
        start=start,
        end=end,
        device=device,
    )

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    load_seconds = (
        time.perf_counter()
        - load_start
    )

    positivity = (
        prove_positive_pairwise_cosines(
            embeddings
        )
    )

    pairwise_lower_bound = positivity[
        "proven_pairwise_cosine_lower_bound"
    ]

    if pairwise_lower_bound <= 0.0:
        raise RuntimeError(
            "Could not prove that all pairwise cosine "
            "similarities are positive. Exact singleton "
            "gains would require a quadratic computation. "
            f"Proven lower bound={pairwise_lower_bound:.9g}."
        )

    singleton_start = time.perf_counter()

    embedding_sum = embeddings.sum(
        dim=0,
        dtype=torch.float64,
    )

    singleton_gains = torch.mv(
        embeddings,
        embedding_sum,
    )

    if not bool(
        torch.isfinite(
            singleton_gains
        ).all().item()
    ):
        raise RuntimeError(
            "Singleton gains contain non-finite values."
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

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    singleton_seconds = (
        time.perf_counter()
        - singleton_start
    )

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

    selected = np.zeros(
        n,
        dtype=bool,
    )

    coverage = torch.zeros(
        n,
        dtype=torch.float64,
        device=device,
    )

    ordering: list[
        tuple[int, float]
    ] = []

    rank_records: list[
        dict[str, Any]
    ] = []

    total_exact_gain_evaluations = 0
    maximum_tie_count = 0
    total_tied_ranks = 0
    maximum_numerical_tie_gap = 0.0

    selection_start = time.perf_counter()

    for state in range(budget):
        evaluated: dict[int, float] = {}

        # Obtain at least one exact current-state candidate.
        while True:
            clean_heap_top(
                heap,
                selected,
            )

            if not heap:
                raise RuntimeError(
                    "LazyGreedy heap became empty."
                )

            (
                negative_bound,
                candidate,
                stamp,
            ) = heapq.heappop(heap)

            if stamp == state:
                candidate_gain = (
                    -negative_bound
                )
                evaluated[candidate] = (
                    candidate_gain
                )
                best_gain = candidate_gain
                break

            candidate_gain, _ = exact_gain(
                embeddings,
                coverage,
                candidate,
            )

            total_exact_gain_evaluations += 1

            heapq.heappush(
                heap,
                (
                    -candidate_gain,
                    candidate,
                    state,
                ),
            )

        # Evaluate every candidate whose valid upper bound can
        # still beat or numerically tie the current best.
        while True:
            clean_heap_top(
                heap,
                selected,
            )

            threshold = (
                numerical_tie_threshold(
                    best_gain,
                    args.tie_atol,
                    args.tie_rtol,
                )
            )

            if not heap:
                break

            next_upper_bound = (
                -heap[0][0]
            )

            if (
                next_upper_bound
                < best_gain - threshold
            ):
                break

            (
                negative_bound,
                candidate,
                stamp,
            ) = heapq.heappop(heap)

            if candidate in evaluated:
                continue

            if stamp == state:
                candidate_gain = (
                    -negative_bound
                )
            else:
                (
                    candidate_gain,
                    _,
                ) = exact_gain(
                    embeddings,
                    coverage,
                    candidate,
                )

                total_exact_gain_evaluations += 1

            evaluated[candidate] = (
                candidate_gain
            )

            if candidate_gain > best_gain:
                best_gain = candidate_gain

        threshold = numerical_tie_threshold(
            best_gain,
            args.tie_atol,
            args.tie_rtol,
        )

        tied_candidates = sorted(
            candidate
            for candidate, gain
            in evaluated.items()
            if best_gain - gain <= threshold
        )

        if not tied_candidates:
            raise RuntimeError(
                "Tie arbitration produced no candidate."
            )

        # Since task pools occupy contiguous dataset intervals,
        # smallest local index equals smallest dataset index.
        selected_candidate = (
            tied_candidates[0]
        )

        selected_gain = evaluated[
            selected_candidate
        ]

        tie_count = len(
            tied_candidates
        )

        maximum_tie_count = max(
            maximum_tie_count,
            tie_count,
        )

        if tie_count > 1:
            total_tied_ranks += 1

            tied_gains = [
                evaluated[candidate]
                for candidate
                in tied_candidates
            ]

            maximum_numerical_tie_gap = max(
                maximum_numerical_tie_gap,
                max(tied_gains)
                - min(tied_gains),
            )

        # Return all evaluated but unselected candidates to the
        # heap with exact current-state gains.
        for candidate, gain in (
            evaluated.items()
        ):
            if (
                candidate
                == selected_candidate
            ):
                continue

            heapq.heappush(
                heap,
                (
                    -gain,
                    candidate,
                    state,
                ),
            )

        # Independent gain recomputation immediately before
        # updating the coverage vector.
        (
            verified_gain,
            selected_similarities,
        ) = exact_gain(
            embeddings,
            coverage,
            selected_candidate,
        )

        total_exact_gain_evaluations += 1

        if not math.isclose(
            selected_gain,
            verified_gain,
            abs_tol=args.replay_atol,
            rel_tol=args.replay_rtol,
        ):
            raise RuntimeError(
                f"Selected gain changed at rank "
                f"{state + 1}: "
                f"arbitrated={selected_gain:.15g}, "
                f"verified={verified_gain:.15g}."
            )

        coverage = torch.maximum(
            coverage,
            selected_similarities,
        )

        selected[
            selected_candidate
        ] = True

        ordering.append(
            (
                selected_candidate,
                verified_gain,
            )
        )

        clean_heap_top(
            heap,
            selected,
        )

        next_upper_bound = (
            -heap[0][0]
            if heap
            else float("-inf")
        )

        certificate_threshold = (
            numerical_tie_threshold(
                verified_gain,
                args.tie_atol,
                args.tie_rtol,
            )
        )

        if (
            next_upper_bound
            > verified_gain
            + certificate_threshold
        ):
            raise RuntimeError(
                f"LazyGreedy certification failed at "
                f"rank {state + 1}: selected gain "
                f"{verified_gain:.15g}, next upper bound "
                f"{next_upper_bound:.15g}."
            )

        objective = float(
            coverage.sum(
                dtype=torch.float64
            ).item()
        )

        rank_records.append(
            {
                "rank": state + 1,
                "local_index": (
                    selected_candidate
                ),
                "dataset_index": (
                    start
                    + selected_candidate
                ),
                "marginal_gain": (
                    verified_gain
                ),
                "objective": objective,
                "tie_count": tie_count,
                "tie_candidates_local": (
                    tied_candidates[:100]
                ),
                "tie_candidates_truncated": (
                    tie_count > 100
                ),
                "tie_threshold": (
                    certificate_threshold
                ),
                "next_upper_bound": (
                    next_upper_bound
                ),
                "exact_gain_evaluations_total": (
                    total_exact_gain_evaluations
                ),
            }
        )

        if (
            state == 0
            or (state + 1)
            % args.progress_every
            == 0
            or state + 1 == budget
        ):
            elapsed = (
                time.perf_counter()
                - selection_start
            )

            print(
                f"Rank {state + 1:4d}/{budget}: "
                f"candidate={selected_candidate:7d} "
                f"gain={verified_gain:.9f} "
                f"ties={tie_count:3d} "
                f"evals={total_exact_gain_evaluations:,} "
                f"objective={objective:.3f} "
                f"elapsed={elapsed / 60:.2f} min",
                flush=True,
            )

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    selection_seconds = (
        time.perf_counter()
        - selection_start
    )

    if len(ordering) != budget:
        raise RuntimeError(
            "Ordering length does not equal "
            "the requested prefix."
        )

    if len(
        {
            candidate
            for candidate, _
            in ordering
        }
    ) != budget:
        raise RuntimeError(
            "Ordering contains duplicate candidates."
        )

    replay_start = time.perf_counter()

    replay = replay_ordering(
        embeddings=embeddings,
        ordering=ordering,
        atol=args.replay_atol,
        rtol=args.replay_rtol,
    )

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    replay_seconds = (
        time.perf_counter()
        - replay_start
    )

    final_objective = (
        rank_records[-1][
            "objective"
        ]
    )

    if not math.isclose(
        replay["final_objective"],
        final_objective,
        abs_tol=args.replay_atol,
        rel_tol=args.replay_rtol,
    ):
        raise RuntimeError(
            "Replay final objective does not match "
            "the selection objective."
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

    total_seconds = (
        time.perf_counter()
        - total_start
    )

    ordering_csv_path = (
        output_root / "ordering.csv"
    )

    with ordering_csv_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                rank_records[0].keys()
            ),
        )
        writer.writeheader()
        writer.writerows(
            rank_records
        )

    author_ordering = [
        (
            start + candidate,
            gain,
        )
        for candidate, gain
        in ordering
    ]

    ordering_pickle_path = (
        output_root / "ordering.pkl"
    )

    atomic_write_pickle(
        author_ordering,
        ordering_pickle_path,
    )

    report_path = (
        output_root / "run_report.json"
    )

    atomic_write_json(
        {
            "format_version": 1,
            "stage": (
                "smart_v2_exact_blockwise_facility_location"
            ),
            "status": "verified",
            "task": pool,
            "configuration": {
                "device": str(device),
                "arithmetic": "float64",
                "similarity": "cosine",
                "objective": (
                    "Facility Location over complete "
                    "task/template pool"
                ),
                "optimizer": (
                    "exact LazyGreedy"
                ),
                "tie_policy": (
                    "Choose the smallest dataset index "
                    "among candidates within the frozen "
                    "float64 numerical-tie tolerance."
                ),
                "tie_atol": args.tie_atol,
                "tie_rtol": args.tie_rtol,
                "candidate_sampling": False,
                "sparse_similarity": False,
                "dense_kernel_materialized": False,
                "approximation": False,
            },
            "positivity_proof": (
                positivity
            ),
            "selection": {
                "pool_size": n,
                "prefix_length": budget,
                "unique_selected_count": (
                    len(
                        {
                            candidate
                            for candidate, _
                            in ordering
                        }
                    )
                ),
                "total_exact_gain_evaluations": (
                    total_exact_gain_evaluations
                ),
                "tied_rank_count": (
                    total_tied_ranks
                ),
                "maximum_tie_count": (
                    maximum_tie_count
                ),
                "maximum_numerical_tie_gap": (
                    maximum_numerical_tie_gap
                ),
                "final_objective": (
                    final_objective
                ),
            },
            "verification": {
                "replay_passed": True,
                "maximum_replay_gain_error": (
                    replay[
                        "maximum_gain_error"
                    ]
                ),
                "replay_final_objective": (
                    replay[
                        "final_objective"
                    ]
                ),
            },
            "timing": {
                "embedding_load_seconds": (
                    load_seconds
                ),
                "singleton_seconds": (
                    singleton_seconds
                ),
                "selection_seconds": (
                    selection_seconds
                ),
                "replay_seconds": (
                    replay_seconds
                ),
                "total_seconds": (
                    total_seconds
                ),
            },
            "memory": {
                "peak_device_bytes": (
                    peak_device_bytes
                ),
                "peak_device_gib": (
                    peak_device_bytes
                    / (1024**3)
                ),
            },
            "outputs": {
                "ordering_csv": str(
                    ordering_csv_path
                ),
                "ordering_pickle": str(
                    ordering_pickle_path
                ),
            },
        },
        report_path,
    )

    print()
    print("=== Ordering summary ===")
    print("Status:                 verified")
    print(f"Task:                   {pool['task_id']}")
    print(f"Pool size:              {n:,}")
    print(f"Selected prefix:        {budget}")
    print(
        f"Gain evaluations:       "
        f"{total_exact_gain_evaluations:,}"
    )
    print(
        f"Ranks with ties:        "
        f"{total_tied_ranks}"
    )
    print(
        f"Maximum tie count:      "
        f"{maximum_tie_count}"
    )
    print(
        f"Final objective:        "
        f"{final_objective:.9f}"
    )
    print(
        f"Replay gain error:      "
        f"{replay['maximum_gain_error']:.3e}"
    )
    print(
        f"Peak device memory:     "
        f"{peak_device_bytes / (1024**3):.3f} GiB"
    )
    print(
        f"Selection time:         "
        f"{selection_seconds / 60:.2f} minutes"
    )
    print(
        f"Total time:             "
        f"{total_seconds / 60:.2f} minutes"
    )
    print(f"Ordering CSV:           {ordering_csv_path}")
    print(f"Ordering pickle:        {ordering_pickle_path}")
    print(f"Report:                 {report_path}")
    print()
    print(
        "Exact blockwise ordering passed."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
PY

python3 -m py_compile \
  src/stage2/run_blockwise_ordering.py
```

## 11.2 Run it on `sglue::qqp`

```bash
cd /data/saral/wdir/smart_v2 || exit 1
source ./smart_v2_env.sh

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

QQP_OUT=/mnt/warm_storage/saral/smart_v2/stage2/blockwise/sglue__qqp

mkdir -p "$QQP_OUT"

python3 -m src.stage2.run_blockwise_ordering \
  --pool-manifest /mnt/warm_storage/saral/smart_v2/stage2/pools/pool_manifest.csv \
  --embedding-root /mnt/warm_storage/saral/smart_v2/embeddings/gte-large \
  --task-id sglue::qqp \
  --output-root "$QQP_OUT" \
  --device cuda:0 \
  --tie-atol 1e-10 \
  --tie-rtol 1e-10 \
  --replay-atol 1e-8 \
  --replay-rtol 1e-10 \
  --progress-every 10 \
  2>&1 | tee \
  /mnt/warm_storage/saral/smart_v2/logs/stage2_blockwise_sglue_qqp.log
```

This run will first verify a strictly positive pairwise-cosine lower bound. It will stop rather than approximate if that proof fails.

## 11.3 Inspect the report

```bash
python3 - <<'PY'
import json
from pathlib import Path

path = Path(
    "/mnt/warm_storage/saral/smart_v2/"
    "stage2/blockwise/sglue__qqp/"
    "run_report.json"
)

with path.open(encoding="utf-8") as handle:
    report = json.load(handle)

print("Status:", report["status"])
print("Task:", report["task"]["task_id"])
print("Pool size:", f"{report['selection']['pool_size']:,}")
print("Prefix:", report["selection"]["prefix_length"])
print(
    "Pairwise lower bound:",
    report["positivity_proof"][
        "proven_pairwise_cosine_lower_bound"
    ],
)
print(
    "Gain evaluations:",
    f"{report['selection']['total_exact_gain_evaluations']:,}",
)
print(
    "Tied ranks:",
    report["selection"]["tied_rank_count"],
)
print(
    "Maximum tie count:",
    report["selection"]["maximum_tie_count"],
)
print(
    "Replay error:",
    report["verification"]["maximum_replay_gain_error"],
)
print(
    "Peak GPU GiB:",
    report["memory"]["peak_device_gib"],
)
print(
    "Selection minutes:",
    report["timing"]["selection_seconds"] / 60,
)
PY
```

Acceptance conditions:

```text
status                            = verified
pool size                         = 363,846
prefix length                     = 187
proven pairwise lower bound       > 0
unique selected count             = 187
replay passed                     = true
maximum replay gain error         <= 1e-8
dense kernel materialized         = false
candidate sampling                = false
approximation                     = false
```

The resulting `ordering.pkl` already contains the author-format dataset indices and can be reused when the full per-corpus ordering files are assembled.
