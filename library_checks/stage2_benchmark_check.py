import json
from pathlib import Path

root = Path(
    "/mnt/warm_storage/saral/smart/"
    "artifacts/stage2_exact_benchmarks"
)

reports = []

for path in sorted(root.glob("*_report.json")):
    with path.open(encoding="utf-8") as handle:
        result = json.load(handle)

    reports.append(result)

print(
    f"{'task':38s} "
    f"{'n':>8s} "
    f"{'k':>5s} "
    f"{'construct_s':>12s} "
    f"{'greedy_s':>10s} "
    f"{'total_s':>10s} "
    f"{'peak_GiB':>10s}"
)

for result in reports:
    task = result["task"]["task_id"]
    benchmark = result["benchmark"]
    timing = result["timing_seconds"]
    memory = result["memory"]

    print(
        f"{task[:38]:38s} "
        f"{benchmark['sample_size']:8,d} "
        f"{benchmark['effective_budget']:5d} "
        f"{timing['construct_objective']:12.3f} "
        f"{timing['maximize']:10.3f} "
        f"{timing['facility_location_total']:10.3f} "
        f"{memory['peak_rss_after_maximize_gib']:10.3f}"
    )
