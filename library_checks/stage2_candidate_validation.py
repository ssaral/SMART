import json
from pathlib import Path

root = Path(
    "/mnt/warm_storage/saral/smart/"
    "artifacts/stage2_candidate_validation"
)

reports = []

for path in root.glob("*_report.json"):
    with path.open(encoding="utf-8") as handle:
        report = json.load(handle)

    reports.append(report)

reports.sort(
    key=lambda report: (
        report["task"]["task_id"],
        report["benchmark"]["represented_size"],
        report["benchmark"]["candidate_size"],
    )
)

print(
    f"{'task':35s} "
    f"{'rep_n':>7s} "
    f"{'cand_n':>7s} "
    f"{'construct':>10s} "
    f"{'greedy':>8s} "
    f"{'rss_GiB':>8s} "
    f"{'ratio25':>10s} "
    f"{'ratio50':>10s} "
    f"{'overlap50':>10s}"
)

for report in reports:
    benchmark = report["benchmark"]
    timing = report["timing_seconds"]
    memory = report["memory"]
    metrics = report["metrics"]

    print(
        f"{report['task']['task_id'][:35]:35s} "
        f"{benchmark['represented_size']:7,d} "
        f"{benchmark['candidate_size']:7,d} "
        f"{timing['construct_objective']:10.2f} "
        f"{timing['maximize']:8.2f} "
        f"{memory['peak_rss_after_maximize_gib']:8.2f} "
        f"{metrics['25000']['objective_ratio']:10.6f} "
        f"{metrics['50000']['objective_ratio']:10.6f} "
        f"{metrics['50000']['selection_overlap_fraction']:10.4f}"
    )

    equivalence = report.get(
        "full_candidate_equivalence"
    )

    if equivalence is not None:
        print(
            "    full-candidate equivalence:",
            equivalence,
        )
