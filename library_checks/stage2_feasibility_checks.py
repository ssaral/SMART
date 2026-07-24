import json

path = (
    "/mnt/warm_storage/saral/smart/artifacts/"
    "stage2_feasibility/stage2_feasibility_summary.json"
)

with open(path, encoding="utf-8") as handle:
    result = json.load(handle)

print("Status:", result["status"])
print("Tasks:", result["task_count"])
print(
    "Effective RAM:",
    result["memory"]["effective_available_gib"],
    "GiB",
)
print(
    "Planning RAM budget:",
    result["memory"]["planning_memory_budget_gib"],
    "GiB",
)
print(
    "Maximum FL budget:",
    result["allocation_checks"][
        "maximum_required_ordering_length"
    ],
)
print(
    "Classes:",
    result["feasibility_class_counts"],
)

print("\nLargest tasks:")
for task in result["largest_tasks"][:15]:
    print(
        task["task_id"],
        task["valid_train_count"],
        task["allocation_50000"],
        round(
            task["dense_kernel_float32_gib"],
            2,
        ),
        round(
            task["conservative_working_set_gib"],
            2,
        ),
        task["feasibility_class"],
    )
