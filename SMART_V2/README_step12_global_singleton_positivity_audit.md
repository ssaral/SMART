The hardest pool passed with no approximation:

```text
full pool             363,846 candidates
selected prefix       187
dense matrix avoided  ~493 GiB
peak GPU memory       6.60 GiB
replay error          0
runtime               12.79 minutes
```

Before scheduling all 309 pools, verify that the positive-similarity singleton shortcut used by the production engine is valid for **every** pool.

# Step 12 — Global singleton-positivity audit

## 12.1 Create the audit script

```bash
cd /data/saral/wdir/smart_v2 || exit 1

cat > src/stage2/audit_all_pool_positivity.py <<'PY'
"""Audit the spherical-cap singleton shortcut for every Stage 2 pool.

For normalized embeddings x_i and normalized centroid c, let

    m = min_i <x_i, c>.

Every pair of examples then satisfies the lower bound

    <x_i, x_j> >= 2*m^2 - 1.

When this bound is safely positive, every pairwise cosine is positive,
and Facility Location singleton gains can be computed exactly as

    gain(j | empty) = x_j dot sum_i x_i

without constructing an n by n kernel.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


EXPECTED_POOL_COUNT = 309
EXPECTED_TOTAL_ROWS = 6_266_471


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
        "--safety-margin",
        type=float,
        default=1e-6,
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=20,
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


def load_manifest(
    path: Path,
) -> list[dict[str, Any]]:
    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        raw_rows = list(
            csv.DictReader(handle)
        )

    if len(raw_rows) != EXPECTED_POOL_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_POOL_COUNT} pools; "
            f"found {len(raw_rows)}."
        )

    rows: list[dict[str, Any]] = []

    for raw in raw_rows:
        rows.append(
            {
                "pool_rank": int(
                    raw["pool_rank"]
                ),
                "corpus": raw["corpus"],
                "task_id": raw["task_id"],
                "template_type": (
                    raw["template_type"]
                ),
                "pool_size": int(
                    raw["pool_size"]
                ),
                "required_stage2_prefix": int(
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
                "dense_kernel_float32_gib": float(
                    raw[
                        "dense_kernel_float32_gib"
                    ]
                ),
            }
        )

    rows.sort(
        key=lambda row: row["pool_rank"]
    )

    return rows


def inspect_pool(
    *,
    pool: dict[str, Any],
    embedding_root: Path,
    device: torch.device,
    safety_margin: float,
) -> dict[str, Any]:
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

    if end - start != pool["pool_size"]:
        raise RuntimeError(
            f"{pool['task_id']}: interval size "
            f"{end - start:,} differs from pool size "
            f"{pool['pool_size']:,}."
        )

    if start < 0 or end > matrix.shape[0]:
        raise RuntimeError(
            f"{pool['task_id']}: pool interval exceeds "
            "the corpus embedding matrix."
        )

    host_view = np.asarray(
        matrix[start:end],
        dtype=np.float32,
    )

    embeddings = torch.as_tensor(
        host_view,
        dtype=torch.float64,
        device=device,
    )

    del matrix
    del host_view

    norms = torch.linalg.vector_norm(
        embeddings,
        dim=1,
        keepdim=True,
    )

    minimum_input_norm = float(
        norms.min().item()
    )
    maximum_input_norm = float(
        norms.max().item()
    )

    if minimum_input_norm <= 0.0:
        raise RuntimeError(
            f"{pool['task_id']}: zero-norm embedding."
        )

    embeddings = embeddings / norms

    centroid = embeddings.sum(
        dim=0,
        dtype=torch.float64,
    )

    centroid_norm = torch.linalg.vector_norm(
        centroid
    )

    if float(centroid_norm.item()) <= 0.0:
        raise RuntimeError(
            f"{pool['task_id']}: zero-norm centroid."
        )

    centroid = centroid / centroid_norm

    centroid_cosines = torch.mv(
        embeddings,
        centroid,
    )

    minimum_centroid_cosine = float(
        centroid_cosines.min().item()
    )
    maximum_centroid_cosine = float(
        centroid_cosines.max().item()
    )

    pairwise_lower_bound = (
        2.0
        * minimum_centroid_cosine**2
        - 1.0
    )

    certified = (
        pairwise_lower_bound
        > safety_margin
    )

    del embeddings
    del norms
    del centroid
    del centroid_cosines

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()

    return {
        **pool,
        "minimum_input_norm": (
            minimum_input_norm
        ),
        "maximum_input_norm": (
            maximum_input_norm
        ),
        "minimum_centroid_cosine": (
            minimum_centroid_cosine
        ),
        "maximum_centroid_cosine": (
            maximum_centroid_cosine
        ),
        "proven_pairwise_cosine_lower_bound": (
            pairwise_lower_bound
        ),
        "safety_margin": safety_margin,
        "singleton_shortcut_certified": (
            certified
        ),
    }


def main() -> int:
    args = parse_args()

    if args.safety_margin < 0.0:
        raise ValueError(
            "--safety-margin cannot be negative."
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

    pools = load_manifest(
        manifest_path
    )

    total_rows = sum(
        pool["pool_size"]
        for pool in pools
    )

    if total_rows != EXPECTED_TOTAL_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_TOTAL_ROWS:,} rows; "
            f"manifest contains {total_rows:,}."
        )

    print(
        "=== Stage 2 singleton-positivity audit ==="
    )
    print(f"Pools:          {len(pools)}")
    print(f"Candidate rows: {total_rows:,}")
    print(f"Device:         {device}")
    print(
        f"Safety margin:  "
        f"{args.safety_margin:.3e}"
    )
    print()

    start_time = time.perf_counter()

    reports: list[dict[str, Any]] = []

    for position, pool in enumerate(
        pools,
        start=1,
    ):
        pool_start = time.perf_counter()

        report = inspect_pool(
            pool=pool,
            embedding_root=embedding_root,
            device=device,
            safety_margin=(
                args.safety_margin
            ),
        )

        report["elapsed_seconds"] = (
            time.perf_counter()
            - pool_start
        )

        reports.append(report)

        if (
            not report[
                "singleton_shortcut_certified"
            ]
            or position == 1
            or position % args.progress_every == 0
            or position == len(pools)
        ):
            print(
                f"{position:3d}/{len(pools)} "
                f"n={pool['pool_size']:7,d} "
                f"bound="
                f"{report['proven_pairwise_cosine_lower_bound']:.9f} "
                f"certified="
                f"{report['singleton_shortcut_certified']} "
                f"{pool['task_id']}",
                flush=True,
            )

    elapsed = (
        time.perf_counter()
        - start_time
    )

    failed = [
        report
        for report in reports
        if not report[
            "singleton_shortcut_certified"
        ]
    ]

    ordered_by_bound = sorted(
        reports,
        key=lambda report: (
            report[
                "proven_pairwise_cosine_lower_bound"
            ]
        ),
    )

    output_csv = (
        output_root
        / "pool_positivity.csv"
    )

    with output_csv.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                reports[0].keys()
            ),
        )
        writer.writeheader()
        writer.writerows(reports)

    peak_device_bytes = (
        int(
            torch.cuda.max_memory_allocated(
                device
            )
        )
        if device.type == "cuda"
        else 0
    )

    status = (
        "verified"
        if not failed
        else "failed"
    )

    summary_path = (
        output_root
        / "positivity_summary.json"
    )

    atomic_write_json(
        {
            "format_version": 1,
            "stage": (
                "smart_v2_all_pool_singleton_positivity"
            ),
            "status": status,
            "configuration": {
                "device": str(device),
                "arithmetic": "float64",
                "safety_margin": (
                    args.safety_margin
                ),
                "bound": (
                    "pairwise cosine >= "
                    "2 * minimum_centroid_cosine^2 - 1"
                ),
                "purpose": (
                    "Certify exact singleton Facility "
                    "Location gains using embedding "
                    "column sums without an n-by-n matrix."
                ),
            },
            "pool_count": len(reports),
            "candidate_rows": total_rows,
            "certified_pool_count": (
                len(reports) - len(failed)
            ),
            "failed_pool_count": len(failed),
            "failed_tasks": [
                {
                    "task_id": report[
                        "task_id"
                    ],
                    "pool_size": report[
                        "pool_size"
                    ],
                    "lower_bound": report[
                        "proven_pairwise_cosine_lower_bound"
                    ],
                }
                for report in failed
            ],
            "minimum_bound": (
                ordered_by_bound[0][
                    "proven_pairwise_cosine_lower_bound"
                ]
            ),
            "minimum_bound_task": (
                ordered_by_bound[0][
                    "task_id"
                ]
            ),
            "maximum_bound": (
                ordered_by_bound[-1][
                    "proven_pairwise_cosine_lower_bound"
                ]
            ),
            "five_smallest_bounds": [
                {
                    "task_id": report[
                        "task_id"
                    ],
                    "pool_size": report[
                        "pool_size"
                    ],
                    "lower_bound": report[
                        "proven_pairwise_cosine_lower_bound"
                    ],
                }
                for report
                in ordered_by_bound[:5]
            ],
            "elapsed_seconds": elapsed,
            "peak_device_bytes": (
                peak_device_bytes
            ),
            "peak_device_gib": (
                peak_device_bytes
                / (1024**3)
            ),
            "outputs": {
                "pool_positivity_csv": str(
                    output_csv
                ),
            },
        },
        summary_path,
    )

    print()
    print(
        "=== Singleton-positivity summary ==="
    )
    print(f"Status:             {status}")
    print(f"Pools checked:      {len(reports)}")
    print(
        f"Certified pools:    "
        f"{len(reports) - len(failed)}"
    )
    print(
        f"Failed pools:       "
        f"{len(failed)}"
    )
    print(
        f"Minimum bound:      "
        f"{ordered_by_bound[0]['proven_pairwise_cosine_lower_bound']:.9f}"
    )
    print(
        f"Minimum-bound task: "
        f"{ordered_by_bound[0]['task_id']}"
    )
    print(
        f"Elapsed:            "
        f"{elapsed / 60:.2f} minutes"
    )
    print(
        f"Peak GPU memory:    "
        f"{peak_device_bytes / (1024**3):.3f} GiB"
    )
    print(f"CSV:                {output_csv}")
    print(f"Summary:            {summary_path}")

    if failed:
        raise RuntimeError(
            "The singleton shortcut is not certified "
            f"for {len(failed)} pools. Do not start the "
            "full ordering run."
        )

    print()
    print(
        "All Stage 2 pools passed the singleton "
        "positivity audit."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
PY

python3 -m py_compile \
  src/stage2/audit_all_pool_positivity.py
```

## 12.2 Run the audit

```bash
cd /data/saral/wdir/smart_v2 || exit 1
source ./smart_v2_env.sh

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

mkdir -p \
  /mnt/warm_storage/saral/smart_v2/stage2/positivity

python3 -m src.stage2.audit_all_pool_positivity \
  --pool-manifest /mnt/warm_storage/saral/smart_v2/stage2/pools/pool_manifest.csv \
  --embedding-root /mnt/warm_storage/saral/smart_v2/embeddings/gte-large \
  --output-root /mnt/warm_storage/saral/smart_v2/stage2/positivity \
  --device cuda:0 \
  --safety-margin 1e-6 \
  --progress-every 20 \
  2>&1 | tee \
  /mnt/warm_storage/saral/smart_v2/logs/stage2_positivity_audit.log
```

## 12.3 Inspect the result

```bash
python3 - <<'PY'
import json
from pathlib import Path

path = Path(
    "/mnt/warm_storage/saral/smart_v2/"
    "stage2/positivity/positivity_summary.json"
)

with path.open(encoding="utf-8") as handle:
    report = json.load(handle)

print("Status:", report["status"])
print("Pools:", report["pool_count"])
print(
    "Certified:",
    report["certified_pool_count"],
)
print(
    "Failed:",
    report["failed_pool_count"],
)
print(
    "Minimum bound:",
    report["minimum_bound"],
)
print(
    "Minimum-bound task:",
    report["minimum_bound_task"],
)

print("\nFive smallest bounds:")
for row in report["five_smallest_bounds"]:
    print(
        f"{row['lower_bound']:.9f}  "
        f"n={row['pool_size']:7,d}  "
        f"{row['task_id']}"
    )
PY
```

Acceptance conditions:

```text
status                 = verified
pool count             = 309
certified pool count   = 309
failed pool count      = 0
minimum bound          > 1e-6
candidate rows         = 6,266,471
```

A failed pool must not be bypassed or silently approximated.
