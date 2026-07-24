"""Analyze dense Facility Location feasibility for all SMART tasks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


EXPECTED_TASK_COUNT = 309
GIB = 1024 ** 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate memory requirements for the authors' dense "
            "Facility Location Stage 2 implementation."
        )
    )

    parser.add_argument(
        "--allocations",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--embedding-dimension",
        type=int,
        default=1024,
    )
    parser.add_argument(
        "--embedding-dtype-bytes",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--kernel-dtype-bytes",
        type=int,
        default=4,
        help="Raw float32 cosine-kernel estimate.",
    )
    parser.add_argument(
        "--working-set-multiplier",
        type=float,
        default=3.0,
        help=(
            "Conservative multiplier over one raw float32 kernel "
            "to allow for native copies, temporary buffers and "
            "submodular state."
        ),
    )
    parser.add_argument(
        "--usable-memory-fraction",
        type=float,
        default=0.60,
        help=(
            "Maximum fraction of currently available CPU RAM "
            "that a single task may consume."
        ),
    )

    return parser.parse_args()


def gibibytes(number_of_bytes: int | float) -> float:
    return float(number_of_bytes) / GIB


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
    payload: dict[str, Any],
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


def read_proc_meminfo() -> dict[str, int]:
    values: dict[str, int] = {}
    path = Path("/proc/meminfo")

    if not path.is_file():
        return values

    for line in path.read_text(
        encoding="utf-8",
    ).splitlines():
        if ":" not in line:
            continue

        key, raw_value = line.split(":", 1)
        fields = raw_value.strip().split()

        if not fields:
            continue

        value = int(fields[0])

        # /proc/meminfo reports values in KiB.
        if len(fields) > 1 and fields[1].lower() == "kb":
            value *= 1024

        values[key] = value

    return values


def read_integer_file(path: Path) -> int | None:
    if not path.is_file():
        return None

    value = path.read_text(
        encoding="utf-8",
    ).strip()

    if value in {"", "max"}:
        return None

    try:
        return int(value)
    except ValueError:
        return None


def detect_effective_available_memory() -> dict[str, Any]:
    meminfo = read_proc_meminfo()

    host_total = meminfo.get("MemTotal")
    host_available = meminfo.get("MemAvailable")

    # Docker/cgroup v2.
    cgroup_limit = read_integer_file(
        Path("/sys/fs/cgroup/memory.max")
    )
    cgroup_current = read_integer_file(
        Path("/sys/fs/cgroup/memory.current")
    )

    # Docker/cgroup v1 fallback.
    if cgroup_limit is None:
        cgroup_limit = read_integer_file(
            Path(
                "/sys/fs/cgroup/memory/"
                "memory.limit_in_bytes"
            )
        )
        cgroup_current = read_integer_file(
            Path(
                "/sys/fs/cgroup/memory/"
                "memory.usage_in_bytes"
            )
        )

    cgroup_headroom = None

    if (
        cgroup_limit is not None
        and cgroup_current is not None
        and cgroup_limit > cgroup_current
        and cgroup_limit < 2 ** 60
    ):
        cgroup_headroom = (
            cgroup_limit - cgroup_current
        )

    candidates = [
        value
        for value in (
            host_available,
            cgroup_headroom,
        )
        if value is not None and value > 0
    ]

    effective_available = (
        min(candidates)
        if candidates
        else None
    )

    return {
        "host_total_bytes": host_total,
        "host_available_bytes": host_available,
        "cgroup_limit_bytes": cgroup_limit,
        "cgroup_current_bytes": cgroup_current,
        "cgroup_headroom_bytes": cgroup_headroom,
        "effective_available_bytes": effective_available,
    }


def load_allocations(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        raw_rows = list(csv.DictReader(handle))

    if len(raw_rows) != EXPECTED_TASK_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_TASK_COUNT} task allocations; "
            f"found {len(raw_rows)}."
        )

    required_columns = {
        "graph_cut_rank",
        "task_index",
        "task_id",
        "corpus",
        "valid_train_count",
        "final_allocation_25000",
        "final_allocation_50000",
    }

    missing = required_columns - set(
        raw_rows[0].keys()
    )

    if missing:
        raise ValueError(
            "Allocation CSV is missing columns: "
            + ", ".join(sorted(missing))
        )

    rows: list[dict[str, Any]] = []

    for raw in raw_rows:
        n = int(raw["valid_train_count"])
        allocation_25k = int(
            raw["final_allocation_25000"]
        )
        allocation_50k = int(
            raw["final_allocation_50000"]
        )

        if not (
            0 <= allocation_25k
            <= allocation_50k
            <= n
        ):
            raise RuntimeError(
                f"Invalid allocation for {raw['task_id']}: "
                f"n={n}, 25K={allocation_25k}, "
                f"50K={allocation_50k}."
            )

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
                "valid_train_count": n,
                "allocation_25000": allocation_25k,
                "allocation_50000": allocation_50k,
                "required_ordering_length": allocation_50k,
            }
        )

    return rows


def main() -> int:
    args = parse_args()

    if args.embedding_dimension <= 0:
        raise ValueError(
            "--embedding-dimension must be positive."
        )

    if args.embedding_dtype_bytes <= 0:
        raise ValueError(
            "--embedding-dtype-bytes must be positive."
        )

    if args.kernel_dtype_bytes <= 0:
        raise ValueError(
            "--kernel-dtype-bytes must be positive."
        )

    if args.working_set_multiplier < 1:
        raise ValueError(
            "--working-set-multiplier must be >= 1."
        )

    if not 0 < args.usable_memory_fraction <= 1:
        raise ValueError(
            "--usable-memory-fraction must be in (0, 1]."
        )

    allocation_path = args.allocations.resolve()
    output_root = args.output_root.resolve()

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = load_allocations(allocation_path)
    memory = detect_effective_available_memory()

    effective_available = memory[
        "effective_available_bytes"
    ]

    if effective_available is None:
        raise RuntimeError(
            "Unable to determine available CPU memory."
        )

    planning_memory_budget = int(
        effective_available
        * args.usable_memory_fraction
    )

    output_rows: list[dict[str, Any]] = []

    for row in rows:
        n = row["valid_train_count"]

        embedding_bytes = (
            n
            * args.embedding_dimension
            * args.embedding_dtype_bytes
        )

        pairwise_entries = n * n

        kernel_f32_bytes = (
            pairwise_entries
            * args.kernel_dtype_bytes
        )

        kernel_f64_bytes = (
            pairwise_entries * 8
        )

        conservative_working_set_bytes = int(
            embedding_bytes
            + (
                kernel_f32_bytes
                * args.working_set_multiplier
            )
        )

        if (
            conservative_working_set_bytes
            <= planning_memory_budget
        ):
            feasibility = "exact_dense_candidate"
        elif kernel_f32_bytes <= planning_memory_budget:
            feasibility = "benchmark_required"
        else:
            feasibility = "exact_dense_infeasible"

        output_rows.append(
            {
                **row,
                "selection_fraction_25000": (
                    row["allocation_25000"] / n
                ),
                "selection_fraction_50000": (
                    row["allocation_50000"] / n
                ),
                "pairwise_similarity_entries": (
                    pairwise_entries
                ),
                "embedding_storage_gib": gibibytes(
                    embedding_bytes
                ),
                "dense_kernel_float32_gib": gibibytes(
                    kernel_f32_bytes
                ),
                "dense_kernel_float64_gib": gibibytes(
                    kernel_f64_bytes
                ),
                "conservative_working_set_gib": (
                    gibibytes(
                        conservative_working_set_bytes
                    )
                ),
                "planning_memory_budget_gib": (
                    gibibytes(
                        planning_memory_budget
                    )
                ),
                "feasibility_class": feasibility,
            }
        )

    output_rows.sort(
        key=lambda row: (
            -row["valid_train_count"],
            row["task_id"],
        )
    )

    csv_path = (
        output_root / "stage2_feasibility.csv"
    )
    summary_path = (
        output_root / "stage2_feasibility_summary.json"
    )

    with csv_path.open(
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

    class_counts = Counter(
        row["feasibility_class"]
        for row in output_rows
    )

    largest_tasks = [
        {
            "rank_by_size": rank,
            "task_id": row["task_id"],
            "corpus": row["corpus"],
            "valid_train_count": (
                row["valid_train_count"]
            ),
            "allocation_25000": (
                row["allocation_25000"]
            ),
            "allocation_50000": (
                row["allocation_50000"]
            ),
            "dense_kernel_float32_gib": (
                row["dense_kernel_float32_gib"]
            ),
            "conservative_working_set_gib": (
                row["conservative_working_set_gib"]
            ),
            "feasibility_class": (
                row["feasibility_class"]
            ),
        }
        for rank, row in enumerate(
            output_rows[:30],
            start=1,
        )
    ]

    summary = {
        "format_version": 1,
        "stage": "SMART_stage2_feasibility",
        "status": "complete",
        "task_count": len(output_rows),
        "allocation_checks": {
            "sum_25000": sum(
                row["allocation_25000"]
                for row in output_rows
            ),
            "sum_50000": sum(
                row["allocation_50000"]
                for row in output_rows
            ),
            "maximum_required_ordering_length": max(
                row["required_ordering_length"]
                for row in output_rows
            ),
            "cross_budget_monotonic": all(
                row["allocation_50000"]
                >= row["allocation_25000"]
                for row in output_rows
            ),
        },
        "memory": {
            **memory,
            "host_total_gib": (
                gibibytes(
                    memory["host_total_bytes"]
                )
                if memory["host_total_bytes"]
                is not None
                else None
            ),
            "host_available_gib": (
                gibibytes(
                    memory["host_available_bytes"]
                )
                if memory["host_available_bytes"]
                is not None
                else None
            ),
            "effective_available_gib": (
                gibibytes(effective_available)
            ),
            "usable_memory_fraction": (
                args.usable_memory_fraction
            ),
            "planning_memory_budget_gib": (
                gibibytes(
                    planning_memory_budget
                )
            ),
        },
        "estimation_assumptions": {
            "embedding_dimension": (
                args.embedding_dimension
            ),
            "embedding_dtype_bytes": (
                args.embedding_dtype_bytes
            ),
            "raw_kernel_dtype_bytes": (
                args.kernel_dtype_bytes
            ),
            "working_set_multiplier": (
                args.working_set_multiplier
            ),
            "conservative_estimate": (
                "embedding storage + multiplier times one "
                "float32 n-by-n similarity kernel"
            ),
            "classification_is_final": False,
            "next_requirement": (
                "Empirical peak-RAM benchmarks on representative "
                "tasks before exact/scalable classification is frozen."
            ),
        },
        "feasibility_class_counts": dict(
            sorted(class_counts.items())
        ),
        "largest_tasks": largest_tasks,
        "input": {
            "task_allocations_csv": str(
                allocation_path
            ),
            "sha256": sha256_file(
                allocation_path
            ),
        },
        "output": {
            "stage2_feasibility_csv": str(
                csv_path
            ),
        },
    }

    atomic_write_json(
        summary,
        summary_path,
    )

    print("=== SMART Stage 2 feasibility ===")
    print(f"Tasks:                       {len(output_rows)}")
    print(
        f"Detected effective RAM:      "
        f"{gibibytes(effective_available):.2f} GiB"
    )
    print(
        f"Planning RAM budget:         "
        f"{gibibytes(planning_memory_budget):.2f} GiB"
    )
    print(
        f"Working-set multiplier:      "
        f"{args.working_set_multiplier:.2f}"
    )
    print(
        f"Maximum required FL budget:  "
        f"{summary['allocation_checks']['maximum_required_ordering_length']}"
    )

    print("\nFeasibility classes:")

    for name, count in sorted(
        class_counts.items()
    ):
        print(f"  {name:28s} {count}")

    print("\nLargest 15 tasks:")

    for row in output_rows[:15]:
        print(
            f"  {row['task_id']:55s} "
            f"n={row['valid_train_count']:8,d} "
            f"k={row['required_ordering_length']:3d} "
            f"kernel={row['dense_kernel_float32_gib']:8.2f} GiB "
            f"estimate={row['conservative_working_set_gib']:8.2f} GiB "
            f"{row['feasibility_class']}"
        )

    print()
    print(f"CSV:     {csv_path}")
    print(f"Summary: {summary_path}")
    print("Stage 2 feasibility analysis passed.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
