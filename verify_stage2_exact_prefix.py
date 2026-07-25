from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np
import submodlib.functions as submod_fn


ROOT = Path("/mnt/warm_storage/saral/smart")
TASK_ID = "cot::stream_qed_ii"


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", value)


with (
    ROOT
    / "artifacts/stage1_allocations/task_allocations.csv"
).open(encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle))

matches = [row for row in rows if row["task_id"] == TASK_ID]
assert len(matches) == 1

row = matches[0]
task_index = int(row["task_index"])
task_size = int(row["valid_train_count"])
k50 = int(row["final_allocation_50000"])

embedding_dir = (
    ROOT
    / "artifacts/prompt_embeddings/gte-large/tasks"
    / f"{task_index:04d}_{safe_name(TASK_ID)}"
)

production_dir = (
    ROOT
    / "artifacts/stage2_selection/tasks"
    / f"{task_index:04d}_{safe_name(TASK_ID)}"
)

embeddings = np.load(
    embedding_dir / "prompt_embeddings.npy",
    mmap_mode="r",
)

production = np.load(
    production_dir / "selected_full_local_indices.npy"
)

objective = (
    submod_fn.facilityLocation
    .FacilityLocationFunction(
        n=task_size,
        separate_rep=False,
        mode="dense",
        data=np.asarray(
            embeddings,
            dtype=np.float32,
            order="C",
        ),
        metric="cosine",
        create_dense_cpp_kernel_in_python=False,
    )
)

# Exact author-style ordering length.
author_result = objective.maximize(
    budget=task_size - 1,
    optimizer="LazyGreedy",
    stopIfZeroGain=False,
    stopIfNegativeGain=False,
    verbose=False,
    show_progress=True,
)

author_prefix = np.asarray(
    [int(index) for index, _ in author_result[:k50]],
    dtype=np.int64,
)

print("Task:", TASK_ID)
print("Task size:", task_size)
print("Required prefix:", k50)
print(
    "Author prefix equals production:",
    bool(np.array_equal(author_prefix, production)),
)

mismatches = np.flatnonzero(
    author_prefix != production
)

print("Position mismatches:", int(mismatches.size))

if mismatches.size:
    first = int(mismatches[0])
    print(
        "First mismatch:",
        first,
        int(author_prefix[first]),
        int(production[first]),
    )

assert np.array_equal(
    author_prefix,
    production,
), "Exact branch does not reproduce the author-style prefix."

print("Exact Stage 2 prefix equivalence passed.")
