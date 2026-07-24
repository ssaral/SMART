import json
from pathlib import Path

path = Path(
    "/mnt/warm_storage/saral/smart/"
    "artifacts/stage1_graph_cut/"
    "graph_cut_result.json"
)

with path.open(encoding="utf-8") as handle:
    result = json.load(handle)

config = result["configuration"]
graph_cut = result["graph_cut"]
kernel = result["kernel"]

print("Status:", result["status"])
print("Task count:", config["task_count"])
print("Budget:", config["budget"])
print("Lambda:", config["lambda_value"])
print("Selections:", graph_cut["selection_count"])
print(
    "Deterministic order:",
    graph_cut["deterministic_order"],
)
print(
    "Deterministic gains:",
    graph_cut["deterministic_gains"],
)
print(
    "Kernel symmetry error:",
    kernel["symmetry_max_absolute_error"],
)
print(
    "Kernel diagonal error:",
    kernel["diagonal_max_absolute_error"],
)
print(
    "Gain range:",
    graph_cut["minimum_marginal_gain"],
    graph_cut["maximum_marginal_gain"],
)
print(
    "Negative gains:",
    graph_cut["negative_gain_count"],
)

print("\nFirst 10 tasks:")
for item in graph_cut["selections"][:10]:
    print(
        item["graph_cut_rank"],
        item["task_id"],
        item["marginal_gain"],
    )
