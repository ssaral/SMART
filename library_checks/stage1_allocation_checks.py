import csv
from pathlib import Path

path = Path(
    "/mnt/warm_storage/saral/smart/"
    "artifacts/stage1_allocations/"
    "task_allocations.csv"
)

with path.open(
    encoding="utf-8",
    newline="",
) as handle:
    rows = list(csv.DictReader(handle))

for budget in (25000, 50000):
    column = f"final_allocation_{budget}"

    ordered = sorted(
        rows,
        key=lambda row: int(row[column]),
        reverse=True,
    )

    print(f"\nTop 10 allocations for {budget:,}:")
    for row in ordered[:10]:
        print(
            row["graph_cut_rank"],
            row["task_id"],
            row[column],
        )

    print(f"\nBottom 10 allocations for {budget:,}:")
    for row in ordered[-10:]:
        print(
            row["graph_cut_rank"],
            row["task_id"],
            row[column],
        )
