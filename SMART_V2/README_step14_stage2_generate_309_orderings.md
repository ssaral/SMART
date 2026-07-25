The final production engine is now frozen:

* exact full-pool Facility Location objective;
* deterministic float64 tie arbitration;
* zero true mismatches against dense Submodlib;
* identical results across GPUs;
* no sampling, sparsification, or objective approximation.

To avoid engine-dependent tie behavior, use this same production engine for all 309 pools. Dense Submodlib remains the verification oracle only. The authors’ mixture builder consumes prefixes from saved orderings, so computing each required 50K-prefix allocation is sufficient. 

# Step 14 — Generate all 309 Stage 2 orderings

## 14.1 Create the resumable multi-GPU scheduler

```bash
cd /data/saral/wdir/smart_v2 || exit 1

cat > src/stage2/run_all_orderings.py <<'PY'
"""Run the verified Stage 2 Facility Location engine on all pools.

Properties:

- one sequential worker per CUDA device;
- dynamically scheduled largest estimated jobs first;
- resumable at task granularity;
- existing verified task outputs are skipped;
- failed or incomplete task outputs are rerun;
- every output is independently validated;
- no candidate sampling or sparse approximation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
import queue
import re
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPECTED_POOL_COUNT = 309
EXPECTED_TOTAL_ROWS = 6_266_471
EXPECTED_TOTAL_PREFIX = 50_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--project-root",
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
        "--positivity-summary",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--production-audit-summary",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--logs-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--devices",
        type=int,
        nargs="+",
        required=True,
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
        default=25,
    )

    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


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


def task_slug(task_id: str) -> str:
    value = task_id.replace(
        "::",
        "__",
    )

    value = re.sub(
        r"[^A-Za-z0-9._-]+",
        "_",
        value,
    )

    return value.strip("_")


def load_json(path: Path) -> dict[str, Any]:
    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        payload = json.load(handle)

    if not isinstance(payload, dict):
        raise TypeError(
            f"{path}: expected a JSON object."
        )

    return payload


def verify_prerequisites(
    positivity_path: Path,
    production_audit_path: Path,
) -> None:
    positivity = load_json(
        positivity_path
    )

    if positivity.get("status") != "verified":
        raise RuntimeError(
            "Singleton-positivity audit is not verified."
        )

    if (
        int(
            positivity.get(
                "certified_pool_count",
                -1,
            )
        )
        != EXPECTED_POOL_COUNT
    ):
        raise RuntimeError(
            "Singleton-positivity audit does not "
            "certify all 309 pools."
        )

    if int(
        positivity.get(
            "failed_pool_count",
            -1,
        )
    ) != 0:
        raise RuntimeError(
            "Singleton-positivity audit contains failures."
        )

    production = load_json(
        production_audit_path
    )

    if production.get("status") != "verified":
        raise RuntimeError(
            "Production-engine audit is not verified."
        )

    if int(
        production.get(
            "true_mismatch_count",
            -1,
        )
    ) != 0:
        raise RuntimeError(
            "Production-engine audit contains a "
            "true dense/blockwise mismatch."
        )

    if int(
        production.get(
            "cross_gpu_order_match_count",
            -1,
        )
    ) != int(
        production.get(
            "task_count",
            -2,
        )
    ):
        raise RuntimeError(
            "Production-engine audit did not reproduce "
            "every ordering across GPUs."
        )


def load_pools(
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

    pools: list[dict[str, Any]] = []
    seen_tasks: set[str] = set()
    seen_slugs: set[str] = set()

    for raw in raw_rows:
        task_id = raw["task_id"]
        slug = task_slug(task_id)

        if task_id in seen_tasks:
            raise RuntimeError(
                f"Duplicate task ID: {task_id}"
            )

        if slug in seen_slugs:
            raise RuntimeError(
                f"Task slug collision: {slug}"
            )

        seen_tasks.add(task_id)
        seen_slugs.add(slug)

        pool = {
            "pool_rank": int(
                raw["pool_rank"]
            ),
            "task_id": task_id,
            "slug": slug,
            "corpus": raw["corpus"],
            "template_type": (
                raw["template_type"]
            ),
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

        if pool["pool_size"] <= 0:
            raise RuntimeError(
                f"{task_id}: empty candidate pool."
            )

        if not (
            0
            < pool["prefix"]
            <= pool["pool_size"]
        ):
            raise RuntimeError(
                f"{task_id}: invalid prefix "
                f"{pool['prefix']} for pool size "
                f"{pool['pool_size']}."
            )

        if (
            pool["last_dataset_index"]
            - pool["first_dataset_index"]
            + 1
            != pool["pool_size"]
        ):
            raise RuntimeError(
                f"{task_id}: manifest interval mismatch."
            )

        pools.append(pool)

    total_rows = sum(
        pool["pool_size"]
        for pool in pools
    )

    if total_rows != EXPECTED_TOTAL_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_TOTAL_ROWS:,} candidate "
            f"rows; found {total_rows:,}."
        )

    total_prefix = sum(
        pool["prefix"]
        for pool in pools
    )

    if total_prefix != EXPECTED_TOTAL_PREFIX:
        raise RuntimeError(
            f"Expected Stage 2 prefixes to sum to "
            f"{EXPECTED_TOTAL_PREFIX:,}; found "
            f"{total_prefix:,}."
        )

    full_pool_prefixes = [
        pool["task_id"]
        for pool in pools
        if pool["prefix"]
        == pool["pool_size"]
    ]

    if full_pool_prefixes:
        raise RuntimeError(
            "The audited production engine currently requires "
            "prefix < pool_size. The following pools require "
            "a full-pool prefix and must be handled explicitly "
            "before launching the batch: "
            f"{full_pool_prefixes}"
        )

    return pools


def validate_task_output(
    *,
    pool: dict[str, Any],
    output_dir: Path,
    tie_atol: float,
    tie_rtol: float,
    replay_atol: float,
    replay_rtol: float,
) -> tuple[bool, str]:
    report_path = (
        output_dir / "run_report.json"
    )
    csv_path = (
        output_dir / "ordering.csv"
    )
    pickle_path = (
        output_dir / "ordering.pkl"
    )

    for path in (
        report_path,
        csv_path,
        pickle_path,
    ):
        if not path.is_file():
            return (
                False,
                f"missing {path.name}",
            )

    try:
        report = load_json(
            report_path
        )
    except Exception as error:
        return (
            False,
            f"invalid report: {error}",
        )

    try:
        if report.get("status") != "verified":
            return (
                False,
                "report status is not verified",
            )

        task = report["task"]
        configuration = report[
            "configuration"
        ]
        selection = report["selection"]
        verification = report[
            "verification"
        ]

        if task["task_id"] != pool[
            "task_id"
        ]:
            return (
                False,
                "task ID mismatch",
            )

        if int(
            selection["pool_size"]
        ) != pool["pool_size"]:
            return (
                False,
                "pool-size mismatch",
            )

        if int(
            selection["prefix_length"]
        ) != pool["prefix"]:
            return (
                False,
                "prefix-length mismatch",
            )

        if int(
            selection[
                "unique_selected_count"
            ]
        ) != pool["prefix"]:
            return (
                False,
                "unique-selection count mismatch",
            )

        if (
            configuration.get(
                "approximation"
            )
            is not False
        ):
            return (
                False,
                "output is marked approximate",
            )

        if (
            configuration.get(
                "candidate_sampling"
            )
            is not False
        ):
            return (
                False,
                "candidate sampling was enabled",
            )

        if (
            configuration.get(
                "sparse_similarity"
            )
            is not False
        ):
            return (
                False,
                "sparse similarity was enabled",
            )

        if not math.isclose(
            float(
                configuration[
                    "tie_atol"
                ]
            ),
            tie_atol,
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            return (
                False,
                "tie-atol mismatch",
            )

        if not math.isclose(
            float(
                configuration[
                    "tie_rtol"
                ]
            ),
            tie_rtol,
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            return (
                False,
                "tie-rtol mismatch",
            )

        if (
            verification.get(
                "replay_passed"
            )
            is not True
        ):
            return (
                False,
                "replay is not marked passed",
            )

        if float(
            verification[
                "maximum_replay_gain_error"
            ]
        ) > replay_atol:
            return (
                False,
                "replay gain error exceeds tolerance",
            )

        if float(
            report[
                "positivity_proof"
            ][
                "proven_pairwise_cosine_lower_bound"
            ]
        ) <= 0.0:
            return (
                False,
                "positive-pairwise proof failed",
            )

        with pickle_path.open(
            "rb",
        ) as handle:
            ordering = pickle.load(
                handle
            )

        if len(ordering) != pool[
            "prefix"
        ]:
            return (
                False,
                "ordering pickle length mismatch",
            )

        dataset_indices: list[int] = []
        gains: list[float] = []

        for item in ordering:
            if (
                not isinstance(
                    item,
                    (tuple, list),
                )
                or len(item) < 2
            ):
                return (
                    False,
                    "invalid ordering item",
                )

            dataset_index = int(
                item[0]
            )
            gain = float(
                item[1]
            )

            dataset_indices.append(
                dataset_index
            )
            gains.append(gain)

        if len(
            set(dataset_indices)
        ) != pool["prefix"]:
            return (
                False,
                "ordering contains duplicate indices",
            )

        if not all(
            pool[
                "first_dataset_index"
            ]
            <= index
            <= pool[
                "last_dataset_index"
            ]
            for index in dataset_indices
        ):
            return (
                False,
                "ordering index outside task pool",
            )

        if not all(
            math.isfinite(gain)
            for gain in gains
        ):
            return (
                False,
                "ordering contains non-finite gains",
            )

        with csv_path.open(
            "r",
            encoding="utf-8",
            newline="",
        ) as handle:
            csv_rows = list(
                csv.DictReader(handle)
            )

        if len(csv_rows) != pool[
            "prefix"
        ]:
            return (
                False,
                "ordering CSV length mismatch",
            )

        # replay_rtol is recorded indirectly through the
        # successful replay. Keep it in this signature so a
        # changed batch configuration invalidates the state.
        _ = replay_rtol

    except Exception as error:
        return (
            False,
            f"validation exception: {error}",
        )

    return True, "verified"


def main() -> int:
    args = parse_args()

    if not args.devices:
        raise ValueError(
            "At least one CUDA device is required."
        )

    if len(args.devices) != len(
        set(args.devices)
    ):
        raise ValueError(
            "CUDA device IDs must be unique."
        )

    if args.progress_every <= 0:
        raise ValueError(
            "--progress-every must be positive."
        )

    project_root = (
        args.project_root.resolve()
    )
    manifest_path = (
        args.pool_manifest.resolve()
    )
    embedding_root = (
        args.embedding_root.resolve()
    )
    positivity_path = (
        args.positivity_summary.resolve()
    )
    production_audit_path = (
        args.production_audit_summary.resolve()
    )
    output_root = (
        args.output_root.resolve()
    )
    logs_root = (
        args.logs_root.resolve()
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )
    logs_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    verify_prerequisites(
        positivity_path,
        production_audit_path,
    )

    pools = load_pools(
        manifest_path
    )

    state_path = (
        output_root
        / "batch_state.json"
    )
    summary_path = (
        output_root
        / "batch_summary.json"
    )

    print_lock = threading.Lock()
    state_lock = threading.Lock()
    stop_event = threading.Event()

    task_states: dict[
        str,
        dict[str, Any],
    ] = {}

    pending_pools: list[
        dict[str, Any]
    ] = []

    for pool in pools:
        output_dir = (
            output_root / pool["slug"]
        )

        valid, reason = validate_task_output(
            pool=pool,
            output_dir=output_dir,
            tie_atol=args.tie_atol,
            tie_rtol=args.tie_rtol,
            replay_atol=(
                args.replay_atol
            ),
            replay_rtol=(
                args.replay_rtol
            ),
        )

        if valid:
            task_states[
                pool["task_id"]
            ] = {
                "status": "verified",
                "mode": "preexisting",
                "device": None,
                "output_dir": str(
                    output_dir
                ),
                "reason": reason,
            }
        else:
            task_states[
                pool["task_id"]
            ] = {
                "status": "pending",
                "mode": "scheduled",
                "device": None,
                "output_dir": str(
                    output_dir
                ),
                "reason": reason,
            }

            pending_pools.append(
                pool
            )

    batch_started = utc_now()
    wall_start = time.perf_counter()

    def write_state(
        status: str,
    ) -> None:
        with state_lock:
            counts: dict[str, int] = {}

            for value in (
                task_states.values()
            ):
                name = value["status"]
                counts[name] = (
                    counts.get(name, 0)
                    + 1
                )

            atomic_write_json(
                {
                    "format_version": 1,
                    "stage": (
                        "smart_v2_all_stage2_orderings"
                    ),
                    "status": status,
                    "started_at": (
                        batch_started
                    ),
                    "updated_at": utc_now(),
                    "configuration": {
                        "devices": (
                            args.devices
                        ),
                        "tie_atol": (
                            args.tie_atol
                        ),
                        "tie_rtol": (
                            args.tie_rtol
                        ),
                        "replay_atol": (
                            args.replay_atol
                        ),
                        "replay_rtol": (
                            args.replay_rtol
                        ),
                        "progress_every": (
                            args.progress_every
                        ),
                        "engine": (
                            "exact float64 "
                            "full-pool LazyGreedy"
                        ),
                        "approximation": False,
                    },
                    "counts": counts,
                    "tasks": task_states,
                },
                state_path,
            )

    print(
        "=== SMART-v2 Stage 2 batch ==="
    )
    print(f"Pools:               {len(pools)}")
    print(
        f"Required prefixes:   "
        f"{sum(pool['prefix'] for pool in pools):,}"
    )
    print(
        f"Preexisting outputs: "
        f"{len(pools) - len(pending_pools)}"
    )
    print(
        f"Pending outputs:     "
        f"{len(pending_pools)}"
    )
    print(f"Devices:             {args.devices}")
    print(f"Output root:         {output_root}")
    print(f"Logs root:           {logs_root}")
    print()

    work_queue: queue.PriorityQueue[
        tuple[int, int, dict[str, Any]]
    ] = queue.PriorityQueue()

    for pool in pending_pools:
        # n * prefix is a conservative scheduling proxy.
        estimated_cost = (
            pool["pool_size"]
            * pool["prefix"]
        )

        work_queue.put(
            (
                -estimated_cost,
                pool["pool_rank"],
                pool,
            )
        )

    write_state("running")

    def worker(device_id: int) -> None:
        device_name = (
            f"cuda:{device_id}"
        )

        while not stop_event.is_set():
            try:
                (
                    _,
                    _,
                    pool,
                ) = work_queue.get_nowait()
            except queue.Empty:
                return

            task_id = pool["task_id"]
            slug = pool["slug"]
            output_dir = (
                output_root / slug
            )
            log_path = (
                logs_root
                / f"{slug}.log"
            )

            output_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            with state_lock:
                task_states[
                    task_id
                ].update(
                    {
                        "status": "running",
                        "device": (
                            device_name
                        ),
                        "started_at": (
                            utc_now()
                        ),
                        "log_path": str(
                            log_path
                        ),
                    }
                )

            write_state("running")

            with print_lock:
                print(
                    f"[{device_name}] START "
                    f"{task_id} "
                    f"n={pool['pool_size']:,} "
                    f"prefix={pool['prefix']}",
                    flush=True,
                )

            command = [
                sys.executable,
                "-m",
                (
                    "src.stage2."
                    "run_blockwise_ordering"
                ),
                "--pool-manifest",
                str(manifest_path),
                "--embedding-root",
                str(embedding_root),
                "--task-id",
                task_id,
                "--output-root",
                str(output_dir),
                "--device",
                device_name,
                "--tie-atol",
                str(args.tie_atol),
                "--tie-rtol",
                str(args.tie_rtol),
                "--replay-atol",
                str(args.replay_atol),
                "--replay-rtol",
                str(args.replay_rtol),
                "--progress-every",
                str(args.progress_every),
            ]

            environment = dict(
                os.environ
            )
            environment[
                "OMP_NUM_THREADS"
            ] = "1"
            environment[
                "MKL_NUM_THREADS"
            ] = "1"
            environment[
                "OPENBLAS_NUM_THREADS"
            ] = "1"

            task_start = (
                time.perf_counter()
            )

            try:
                with log_path.open(
                    "w",
                    encoding="utf-8",
                ) as log_handle:
                    process = subprocess.run(
                        command,
                        cwd=str(
                            project_root
                        ),
                        env=environment,
                        stdout=log_handle,
                        stderr=(
                            subprocess.STDOUT
                        ),
                        check=False,
                    )

                elapsed = (
                    time.perf_counter()
                    - task_start
                )

                valid, reason = (
                    validate_task_output(
                        pool=pool,
                        output_dir=(
                            output_dir
                        ),
                        tie_atol=(
                            args.tie_atol
                        ),
                        tie_rtol=(
                            args.tie_rtol
                        ),
                        replay_atol=(
                            args.replay_atol
                        ),
                        replay_rtol=(
                            args.replay_rtol
                        ),
                    )
                )

                if (
                    process.returncode != 0
                    or not valid
                ):
                    raise RuntimeError(
                        f"returncode="
                        f"{process.returncode}; "
                        f"validation={reason}"
                    )

                report = load_json(
                    output_dir
                    / "run_report.json"
                )

                with state_lock:
                    task_states[
                        task_id
                    ].update(
                        {
                            "status": (
                                "verified"
                            ),
                            "mode": (
                                "generated"
                            ),
                            "finished_at": (
                                utc_now()
                            ),
                            "elapsed_seconds": (
                                elapsed
                            ),
                            "selection_seconds": (
                                report[
                                    "timing"
                                ][
                                    "selection_seconds"
                                ]
                            ),
                            "gain_evaluations": (
                                report[
                                    "selection"
                                ][
                                    "total_exact_gain_evaluations"
                                ]
                            ),
                            "peak_device_gib": (
                                report[
                                    "memory"
                                ][
                                    "peak_device_gib"
                                ]
                            ),
                            "reason": (
                                "verified"
                            ),
                        }
                    )

                write_state("running")

                with print_lock:
                    print(
                        f"[{device_name}] DONE  "
                        f"{task_id} "
                        f"in {elapsed / 60:.2f} min",
                        flush=True,
                    )

            except Exception as error:
                with state_lock:
                    task_states[
                        task_id
                    ].update(
                        {
                            "status": "failed",
                            "finished_at": (
                                utc_now()
                            ),
                            "error": str(
                                error
                            ),
                        }
                    )

                stop_event.set()
                write_state("failed")

                with print_lock:
                    print(
                        f"[{device_name}] FAILED "
                        f"{task_id}: {error}",
                        file=sys.stderr,
                        flush=True,
                    )

            finally:
                work_queue.task_done()

    threads = [
        threading.Thread(
            target=worker,
            args=(device_id,),
            name=(
                f"stage2-cuda-{device_id}"
            ),
            daemon=False,
        )
        for device_id in args.devices
    ]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    failed_tasks = [
        task_id
        for task_id, value
        in task_states.items()
        if value["status"] == "failed"
    ]

    if failed_tasks:
        write_state("failed")

        raise RuntimeError(
            "Stage 2 batch failed. No additional "
            "tasks were scheduled after the first "
            f"failure. Failed tasks: {failed_tasks}"
        )

    # Final independent validation of every task output.
    final_failures: list[
        dict[str, str]
    ] = []

    reports: list[
        dict[str, Any]
    ] = []

    for pool in pools:
        output_dir = (
            output_root / pool["slug"]
        )

        valid, reason = validate_task_output(
            pool=pool,
            output_dir=output_dir,
            tie_atol=args.tie_atol,
            tie_rtol=args.tie_rtol,
            replay_atol=(
                args.replay_atol
            ),
            replay_rtol=(
                args.replay_rtol
            ),
        )

        if not valid:
            final_failures.append(
                {
                    "task_id": (
                        pool["task_id"]
                    ),
                    "reason": reason,
                }
            )
            continue

        reports.append(
            load_json(
                output_dir
                / "run_report.json"
            )
        )

    if final_failures:
        write_state("failed")

        raise RuntimeError(
            "Final batch validation failed: "
            f"{final_failures}"
        )

    wall_seconds = (
        time.perf_counter()
        - wall_start
    )

    total_gain_evaluations = sum(
        int(
            report["selection"][
                "total_exact_gain_evaluations"
            ]
        )
        for report in reports
    )

    total_selection_seconds = sum(
        float(
            report["timing"][
                "selection_seconds"
            ]
        )
        for report in reports
    )

    maximum_peak_device_gib = max(
        float(
            report["memory"][
                "peak_device_gib"
            ]
        )
        for report in reports
    )

    tied_task_count = sum(
        int(
            report["selection"][
                "tied_rank_count"
            ]
        )
        > 0
        for report in reports
    )

    total_tied_ranks = sum(
        int(
            report["selection"][
                "tied_rank_count"
            ]
        )
        for report in reports
    )

    generated_count = sum(
        value.get("mode")
        == "generated"
        for value in (
            task_states.values()
        )
    )

    preexisting_count = sum(
        value.get("mode")
        == "preexisting"
        for value in (
            task_states.values()
        )
    )

    summary = {
        "format_version": 1,
        "stage": (
            "smart_v2_all_stage2_orderings"
        ),
        "status": "verified",
        "started_at": batch_started,
        "finished_at": utc_now(),
        "configuration": {
            "devices": args.devices,
            "engine": (
                "exact float64 full-pool "
                "Facility Location LazyGreedy"
            ),
            "tie_policy": (
                "smallest dataset index within "
                "frozen numerical-tie tolerance"
            ),
            "tie_atol": (
                args.tie_atol
            ),
            "tie_rtol": (
                args.tie_rtol
            ),
            "replay_atol": (
                args.replay_atol
            ),
            "replay_rtol": (
                args.replay_rtol
            ),
            "candidate_sampling": False,
            "sparse_similarity": False,
            "approximation": False,
        },
        "pool_count": len(pools),
        "verified_pool_count": (
            len(reports)
        ),
        "failed_pool_count": 0,
        "candidate_rows": sum(
            pool["pool_size"]
            for pool in pools
        ),
        "total_prefix_length": sum(
            pool["prefix"]
            for pool in pools
        ),
        "generated_this_run": (
            generated_count
        ),
        "preexisting_verified": (
            preexisting_count
        ),
        "total_exact_gain_evaluations": (
            total_gain_evaluations
        ),
        "tasks_with_ties": (
            tied_task_count
        ),
        "total_tied_ranks": (
            total_tied_ranks
        ),
        "wall_seconds": wall_seconds,
        "summed_selection_seconds": (
            total_selection_seconds
        ),
        "maximum_peak_device_gib": (
            maximum_peak_device_gib
        ),
        "outputs": {
            "task_output_root": str(
                output_root
            ),
            "batch_state": str(
                state_path
            ),
        },
    }

    atomic_write_json(
        summary,
        summary_path,
    )

    write_state("verified")

    print()
    print(
        "=== Stage 2 batch summary ==="
    )
    print("Status:                  verified")
    print(
        f"Pools verified:          "
        f"{len(reports)}/{len(pools)}"
    )
    print(
        f"Generated this run:      "
        f"{generated_count}"
    )
    print(
        f"Preexisting verified:    "
        f"{preexisting_count}"
    )
    print(
        f"Candidate rows:          "
        f"{summary['candidate_rows']:,}"
    )
    print(
        f"Total selected prefixes: "
        f"{summary['total_prefix_length']:,}"
    )
    print(
        f"Gain evaluations:        "
        f"{total_gain_evaluations:,}"
    )
    print(
        f"Tasks containing ties:   "
        f"{tied_task_count}"
    )
    print(
        f"Total tied ranks:        "
        f"{total_tied_ranks}"
    )
    print(
        f"Wall time:               "
        f"{wall_seconds / 3600:.2f} hours"
    )
    print(
        f"Maximum GPU memory:      "
        f"{maximum_peak_device_gib:.3f} GiB"
    )
    print(f"State:                   {state_path}")
    print(f"Summary:                 {summary_path}")
    print()
    print(
        "All Stage 2 orderings passed."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
PY

python3 -m py_compile \
  src/stage2/run_all_orderings.py
```

## 14.2 Launch on four GPUs

The existing verified QQP result under `stage2/blockwise/sglue__qqp` will be detected and skipped.

```bash
cd /data/saral/wdir/smart_v2 || exit 1
source ./smart_v2_env.sh

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

mkdir -p \
  /mnt/warm_storage/saral/smart_v2/stage2/blockwise \
  /mnt/warm_storage/saral/smart_v2/logs/stage2_orderings

python3 -m src.stage2.run_all_orderings \
  --project-root /data/saral/wdir/smart_v2 \
  --pool-manifest /mnt/warm_storage/saral/smart_v2/stage2/pools/pool_manifest.csv \
  --embedding-root /mnt/warm_storage/saral/smart_v2/embeddings/gte-large \
  --positivity-summary /mnt/warm_storage/saral/smart_v2/stage2/positivity/positivity_summary.json \
  --production-audit-summary /mnt/warm_storage/saral/smart_v2/stage2/production_audit/production_audit_summary.json \
  --output-root /mnt/warm_storage/saral/smart_v2/stage2/blockwise \
  --logs-root /mnt/warm_storage/saral/smart_v2/logs/stage2_orderings \
  --devices 0 1 2 3 \
  --tie-atol 1e-10 \
  --tie-rtol 1e-10 \
  --replay-atol 1e-8 \
  --replay-rtol 1e-10 \
  --progress-every 25 \
  2>&1 | tee \
  /mnt/warm_storage/saral/smart_v2/logs/stage2_all_orderings.log
```

The scheduler runs the largest estimated jobs first and assigns the next available job to whichever GPU finishes first.

It is resumable. On rerun, every completed task is validated and skipped.

## 14.3 Monitor progress

Compact status:

```bash
python3 - <<'PY'
import json
from collections import Counter
from pathlib import Path

path = Path(
    "/mnt/warm_storage/saral/smart_v2/"
    "stage2/blockwise/batch_state.json"
)

with path.open(encoding="utf-8") as handle:
    state = json.load(handle)

counts = Counter(
    task["status"]
    for task in state["tasks"].values()
)

print("Batch status:", state["status"])

for name in (
    "verified",
    "running",
    "pending",
    "failed",
):
    print(
        f"{name:10s}",
        counts.get(name, 0),
    )

print("\nRunning tasks:")

for task_id, task in state[
    "tasks"
].items():
    if task["status"] == "running":
        print(
            task["device"],
            task_id,
        )
PY
```

Inspect a worker log:

```bash
tail -n 30 \
  /mnt/warm_storage/saral/smart_v2/logs/stage2_orderings/flan2021__glue_mnli_2.0.0.log
```

## Acceptance conditions

```text
status                       = verified
pool count                   = 309
verified pool count          = 309
failed pool count            = 0
candidate rows               = 6,266,471
total prefix length          = 50,000
candidate sampling           = false
sparse similarity            = false
approximation                = false
every replay gain error      <= 1e-8
every ordering index         within its task pool
every ordering prefix        unique
```

Final outputs:

```text
/mnt/warm_storage/saral/smart_v2/stage2/blockwise/
├── <task-slug>/
│   ├── ordering.csv
│   ├── ordering.pkl
│   └── run_report.json
├── ...
├── batch_state.json
└── batch_summary.json
```
