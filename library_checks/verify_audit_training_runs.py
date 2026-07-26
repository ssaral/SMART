import csv
from pathlib import Path

path = Path(
    "/mnt/warm_storage/saral/smart/"
    "artifacts/tokenization_audit/"
    "tokenization_summary.csv"
)

with path.open(
    encoding="utf-8",
    newline="",
) as handle:
    rows = list(csv.DictReader(handle))

print(
    f"{'model':12s} "
    f"{'split':24s} "
    f"{'rows':>8s} "
    f"{'mean':>8s} "
    f"{'p99':>8s} "
    f"{'max':>8s} "
    f"{'resp':>8s} "
    f"{'trunc':>8s} "
    f"{'prompt4k':>9s} "
    f"{'zero':>6s}"
)

for row in rows:
    print(
        f"{row['model']:12s} "
        f"{row['split']:24s} "
        f"{int(row['rows']):8,d} "
        f"{float(row['retained_mean']):8.1f} "
        f"{float(row['full_p99']):8.1f} "
        f"{int(row['full_max']):8,d} "
        f"{float(row['supervised_mean']):8.1f} "
        f"{int(row['sequence_truncated']):8,d} "
        f"{int(row['prompt_at_limit']):9,d} "
        f"{int(row['zero_supervised_tokens']):6,d}"
    )
