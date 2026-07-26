from datasets import load_from_disk

root = (
    "/mnt/warm_storage/saral/smart/"
    "datasets/trainer"
)

for budget in (25000, 50000):
    path = f"{root}/smart_{budget}"
    dataset = load_from_disk(path)

    print()
    print("Dataset:", path)
    print(dataset)
    print("Train rows:", len(dataset["train"]))
    print(
        "Validation rows:",
        len(dataset["validation"]),
    )
    print(
        "Train columns:",
        dataset["train"].column_names,
    )
    print(
        "Validation columns:",
        dataset["validation"].column_names,
    )

    assert len(dataset["train"]) == budget
    assert len(dataset["validation"]) == 183870

    assert dataset["train"].column_names == [
        "prompt",
        "response",
    ]
    assert dataset["validation"].column_names == [
        "prompt",
        "response",
    ]

    for split in ("train", "validation"):
        for row in dataset[split].select(
            range(min(1000, len(dataset[split])))
        ):
            assert isinstance(row["prompt"], str)
            assert row["prompt"].strip()
            assert isinstance(row["response"], str)
            assert row["response"].strip()

print()
print("Saved DatasetDict verification passed.")
