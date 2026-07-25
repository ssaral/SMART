The prompt-embedding stage is now frozen as verified.

Next, reproduce the authors’ `get_task_embeddings()` step: for each task, gather all prompt embeddings across its available template buckets and compute their mean. The released code then saves the ordered task list and the resulting task-embedding matrix. 

# Step 6 — Build and verify 309 task-mean embeddings

## 6.1 Freeze the local submixture order

The authors use a fixed submixture loop order. We will use this adapted fixed order:

```text
flan2021 → t0 → sglue → cot → tulu
```

Create the configuration:

```bash
cd /data/saral/wdir/smart_v2 || exit 1

cat > configs/submixtures.json <<'JSON'
{
  "format_version": 1,
  "submixtures": [
    "flan2021",
    "t0",
    "sglue",
    "cot",
    "tulu"
  ],
  "policy": "Fixed authors-style submixture iteration order for task embedding construction and all downstream SMART stages.",
  "within_submixture_order": "Preserve the task order stored in the author-compatible task_indices pickle.",
  "task_namespaced": true
}
JSON
```

This order must remain unchanged because task indices are used for Graph Cut tie-breaking and artifact alignment.

---

## 6.2 Create the task-embedding builder

```bash
cat > src/embeddings/build_task_embeddings.py <<'PY'
"""Build SMART task embeddings by averaging prompt embeddings per task.

The implementation follows the authors' get_task_embeddings behavior:

1. Iterate submixtures in a fixed order.
2. Iterate tasks in task_indices insertion order.
3. Gather rows from all supported template buckets.
4. Compute the float32 NumPy mean of prompt embeddings.
5. Save tasks.pkl and tasks_embeddings.npy.

For the primary SMART-v2 baseline, every task has one zs_noopt bucket.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from datasets import load_from_disk


TEMPLATE_TYPES = (
    "zs_opt",
    "zs_noopt",
    "fs_opt",
    "fs_noopt",
)

EXPECTED_TEMPLATE_TYPE = "zs_noopt"
EXPECTED_TASK_COUNT = 309
EXPECTED_TOTAL_ROWS = 6_266_471
EXPECTED_DIMENSION = 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--author-format-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--task-indices-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--embedding-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
    )

    return parser.parse_args()


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
    payload: Any,
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


def atomic_write_pickle(
    payload: Any,
    path: Path,
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")

    with temporary.open("wb") as handle:
        pickle.dump(
            payload,
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    temporary.replace(path)


def atomic_write_numpy(
    array: np.ndarray,
    path: Path,
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")

    with temporary.open("wb") as handle:
        np.save(
            handle,
            array,
            allow_pickle=False,
        )

    temporary.replace(path)


def load_submixtures(
    path: Path,
) -> list[str]:
    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        config = json.load(handle)

    submixtures = config.get("submixtures")

    if not isinstance(submixtures, list):
        raise ValueError(
            "Configuration must contain a submixtures list."
        )

    if not submixtures:
        raise ValueError(
            "Submixture list cannot be empty."
        )

    if not all(
        isinstance(value, str) and value
        for value in submixtures
    ):
        raise ValueError(
            "Every submixture name must be a non-empty string."
        )

    if len(submixtures) != len(set(submixtures)):
        raise ValueError(
            "Submixture order contains duplicates."
        )

    return submixtures


def contiguous_bounds(
    indices: np.ndarray,
    *,
    task_id: str,
) -> tuple[int, int]:
    if indices.ndim != 1:
        raise RuntimeError(
            f"{task_id}: indices are not one-dimensional."
        )

    if indices.size == 0:
        raise RuntimeError(
            f"{task_id}: task has no prompt indices."
        )

    if indices[0] < 0:
        raise RuntimeError(
            f"{task_id}: task has a negative index."
        )

    if indices.size > 1:
        differences = np.diff(indices)

        if not np.all(differences == 1):
            raise RuntimeError(
                f"{task_id}: indices are not contiguous. "
                "The primary single-template baseline expects "
                "each task to occupy one contiguous dataset block."
            )

    start = int(indices[0])
    end = int(indices[-1]) + 1

    return start, end


def main() -> int:
    args = parse_args()

    config_path = args.config.resolve()
    author_root = args.author_format_root.resolve()
    indices_root = args.task_indices_root.resolve()
    embedding_root = args.embedding_root.resolve()
    output_root = args.output_root.resolve()

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    submixtures = load_submixtures(config_path)

    tasks: list[str] = []
    task_embeddings: list[np.ndarray] = []
    catalog_rows: list[dict[str, Any]] = []
    corpus_summaries: list[dict[str, Any]] = []

    total_rows = 0

    print("=== SMART-v2 task embeddings ===")
    print("Submixture order:")
    for position, corpus in enumerate(
        submixtures,
        start=1,
    ):
        print(f"  {position}. {corpus}")

    for corpus_position, corpus in enumerate(
        submixtures,
        start=1,
    ):
        dataset_path = author_root / corpus
        index_path = indices_root / f"{corpus}.pkl"
        matrix_path = (
            embedding_root
            / corpus
            / "train_prompt_embeddings.npy"
        )

        if not dataset_path.is_dir():
            raise FileNotFoundError(dataset_path)

        if not index_path.is_file():
            raise FileNotFoundError(index_path)

        if not matrix_path.is_file():
            raise FileNotFoundError(matrix_path)

        dataset = load_from_disk(
            str(dataset_path)
        )
        train = dataset["train"]

        task_feature = train.features["task_name"]

        if not hasattr(task_feature, "names"):
            raise RuntimeError(
                f"{corpus}: task_name is not a ClassLabel."
            )

        feature_task_names = list(
            task_feature.names
        )

        with index_path.open("rb") as handle:
            task_indices = pickle.load(handle)

        if not isinstance(task_indices, dict):
            raise TypeError(
                f"{corpus}: task-index pickle is not a dict."
            )

        indexed_task_names = list(
            task_indices.keys()
        )

        if indexed_task_names != feature_task_names:
            raise RuntimeError(
                f"{corpus}: task-index insertion order does "
                "not match the author-format ClassLabel order."
            )

        matrix = np.load(
            matrix_path,
            mmap_mode="r",
        )

        expected_matrix_shape = (
            len(train),
            EXPECTED_DIMENSION,
        )

        if matrix.shape != expected_matrix_shape:
            raise RuntimeError(
                f"{corpus}: prompt matrix shape "
                f"{matrix.shape} != {expected_matrix_shape}."
            )

        if matrix.dtype != np.float32:
            raise RuntimeError(
                f"{corpus}: prompt matrix dtype "
                f"{matrix.dtype} is not float32."
            )

        corpus_rows = 0
        corpus_task_start = len(tasks)

        print()
        print(f"Processing {corpus}")
        print(f"  Tasks: {len(indexed_task_names)}")
        print(f"  Rows:  {len(train):,}")

        for corpus_task_position, task_id in enumerate(
            indexed_task_names,
            start=1,
        ):
            template_mapping = task_indices[
                task_id
            ]

            if not isinstance(template_mapping, dict):
                raise TypeError(
                    f"{task_id}: template mapping is not a dict."
                )

            available_templates = set(
                template_mapping
            )

            if available_templates != {
                EXPECTED_TEMPLATE_TYPE
            }:
                raise RuntimeError(
                    f"{task_id}: primary baseline expected "
                    f"only {EXPECTED_TEMPLATE_TYPE!r}; found "
                    f"{sorted(available_templates)}."
                )

            ordered_indices: list[int] = []

            # This follows the authors' template-order loop.
            for template_type in TEMPLATE_TYPES:
                if template_type in template_mapping:
                    ordered_indices.extend(
                        template_mapping[
                            template_type
                        ]
                    )

            indices = np.asarray(
                ordered_indices,
                dtype=np.int64,
            )

            start, end = contiguous_bounds(
                indices,
                task_id=task_id,
            )

            if end > len(train):
                raise RuntimeError(
                    f"{task_id}: final index {end - 1:,} "
                    f"exceeds dataset size {len(train):,}."
                )

            row_count = end - start

            if row_count != indices.size:
                raise RuntimeError(
                    f"{task_id}: contiguous interval contains "
                    f"{row_count:,} rows but index mapping "
                    f"contains {indices.size:,}."
                )

            # Authors use np.mean on float32 prompt embeddings.
            # A contiguous slice preserves the exact source row order
            # without materializing a large fancy-indexed copy.
            mean_embedding = np.mean(
                matrix[start:end],
                axis=0,
                dtype=np.float32,
            )

            mean_embedding = np.asarray(
                mean_embedding,
                dtype=np.float32,
            )

            if mean_embedding.shape != (
                EXPECTED_DIMENSION,
            ):
                raise RuntimeError(
                    f"{task_id}: mean shape "
                    f"{mean_embedding.shape} is invalid."
                )

            if not np.isfinite(
                mean_embedding
            ).all():
                raise RuntimeError(
                    f"{task_id}: task mean contains "
                    "non-finite values."
                )

            mean_norm = float(
                np.linalg.norm(mean_embedding)
            )

            if mean_norm <= 0.0:
                raise RuntimeError(
                    f"{task_id}: task mean has zero norm."
                )

            global_task_index = len(tasks)

            tasks.append(task_id)
            task_embeddings.append(
                mean_embedding
            )

            catalog_rows.append(
                {
                    "task_index": global_task_index,
                    "task_rank_1based": (
                        global_task_index + 1
                    ),
                    "submixture_position": (
                        corpus_position
                    ),
                    "corpus": corpus,
                    "corpus_task_position": (
                        corpus_task_position
                    ),
                    "task_id": task_id,
                    "template_types": (
                        EXPECTED_TEMPLATE_TYPE
                    ),
                    "valid_train_count": (
                        row_count
                    ),
                    "first_dataset_index": start,
                    "last_dataset_index": (
                        end - 1
                    ),
                    "mean_embedding_norm": (
                        mean_norm
                    ),
                }
            )

            corpus_rows += row_count
            total_rows += row_count

            if (
                corpus_task_position % 20 == 0
                or corpus_task_position
                == len(indexed_task_names)
            ):
                print(
                    f"  {corpus_task_position}/"
                    f"{len(indexed_task_names)} tasks",
                    flush=True,
                )

        if corpus_rows != len(train):
            raise RuntimeError(
                f"{corpus}: task mappings cover "
                f"{corpus_rows:,} rows but the training "
                f"dataset contains {len(train):,}."
            )

        corpus_summaries.append(
            {
                "corpus": corpus,
                "submixture_position": (
                    corpus_position
                ),
                "task_start_index": (
                    corpus_task_start
                ),
                "task_end_index_exclusive": (
                    len(tasks)
                ),
                "task_count": (
                    len(tasks) - corpus_task_start
                ),
                "train_rows": corpus_rows,
                "task_indices_path": str(
                    index_path
                ),
                "task_indices_sha256": (
                    sha256_file(index_path)
                ),
                "prompt_embeddings_path": str(
                    matrix_path
                ),
            }
        )

    if len(tasks) != EXPECTED_TASK_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_TASK_COUNT} tasks; "
            f"constructed {len(tasks)}."
        )

    if len(set(tasks)) != EXPECTED_TASK_COUNT:
        raise RuntimeError(
            "Task list contains duplicate task IDs."
        )

    if total_rows != EXPECTED_TOTAL_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_TOTAL_ROWS:,} source rows; "
            f"task means used {total_rows:,}."
        )

    task_matrix = np.stack(
        task_embeddings,
        axis=0,
    ).astype(
        np.float32,
        copy=False,
    )

    expected_shape = (
        EXPECTED_TASK_COUNT,
        EXPECTED_DIMENSION,
    )

    if task_matrix.shape != expected_shape:
        raise RuntimeError(
            f"Task matrix shape {task_matrix.shape} "
            f"!= {expected_shape}."
        )

    if not np.isfinite(task_matrix).all():
        raise RuntimeError(
            "Task matrix contains non-finite values."
        )

    tasks_path = output_root / "tasks.pkl"
    matrix_path = (
        output_root / "tasks_embeddings.npy"
    )
    catalog_path = (
        output_root / "task_catalog.csv"
    )
    summary_path = (
        output_root / "build_summary.json"
    )

    atomic_write_pickle(
        tasks,
        tasks_path,
    )
    atomic_write_numpy(
        task_matrix,
        matrix_path,
    )

    with catalog_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                catalog_rows[0].keys()
            ),
        )
        writer.writeheader()
        writer.writerows(catalog_rows)

    task_norms = np.linalg.norm(
        task_matrix,
        axis=1,
    )

    atomic_write_json(
        {
            "format_version": 1,
            "stage": (
                "smart_v2_task_mean_embeddings"
            ),
            "status": "complete",
            "configuration": {
                "submixture_order": submixtures,
                "template_order": list(
                    TEMPLATE_TYPES
                ),
                "available_template_type": (
                    EXPECTED_TEMPLATE_TYPE
                ),
                "reduction": (
                    "numpy.mean(axis=0, dtype=float32)"
                ),
                "input_dimension": (
                    EXPECTED_DIMENSION
                ),
                "output_dtype": "float32",
                "task_order_policy": (
                    "Fixed submixture order followed by "
                    "task_indices insertion order."
                ),
            },
            "task_count": len(tasks),
            "source_prompt_rows": total_rows,
            "task_embedding_shape": list(
                task_matrix.shape
            ),
            "task_embedding_dtype": str(
                task_matrix.dtype
            ),
            "task_embedding_norm": {
                "minimum": float(
                    task_norms.min()
                ),
                "maximum": float(
                    task_norms.max()
                ),
                "mean": float(
                    task_norms.mean()
                ),
            },
            "inputs": {
                "submixture_config": str(
                    config_path
                ),
                "submixture_config_sha256": (
                    sha256_file(config_path)
                ),
            },
            "corpora": corpus_summaries,
            "outputs": {
                "tasks_pickle": str(
                    tasks_path
                ),
                "tasks_pickle_sha256": (
                    sha256_file(tasks_path)
                ),
                "task_embeddings": str(
                    matrix_path
                ),
                "task_embeddings_sha256": (
                    sha256_file(matrix_path)
                ),
                "task_catalog": str(
                    catalog_path
                ),
                "task_catalog_sha256": (
                    sha256_file(catalog_path)
                ),
            },
        },
        summary_path,
    )

    print()
    print("=== Task-embedding summary ===")
    print(f"Tasks:            {len(tasks)}")
    print(f"Source rows:      {total_rows:,}")
    print(f"Matrix shape:     {task_matrix.shape}")
    print(f"Matrix dtype:     {task_matrix.dtype}")
    print(
        "Task norm range: "
        f"{task_norms.min():.8f} to "
        f"{task_norms.max():.8f}"
    )
    print(f"Tasks pickle:     {tasks_path}")
    print(f"Task embeddings:  {matrix_path}")
    print(f"Task catalog:     {catalog_path}")
    print(f"Summary:          {summary_path}")
    print()
    print("Task-mean embedding construction passed.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
PY

python3 -m py_compile \
  src/embeddings/build_task_embeddings.py
```

---

## 6.3 Run task-mean construction

```bash
cd /data/saral/wdir/smart_v2 || exit 1
source ./smart_v2_env.sh

mkdir -p \
  /mnt/warm_storage/saral/smart_v2/embeddings/gte-large/task_embeddings

python3 -m src.embeddings.build_task_embeddings \
  --config /data/saral/wdir/smart_v2/configs/submixtures.json \
  --author-format-root /mnt/warm_storage/saral/smart_v2/author_format \
  --task-indices-root /mnt/warm_storage/saral/smart_v2/author_format/task_indices \
  --embedding-root /mnt/warm_storage/saral/smart_v2/embeddings/gte-large \
  --output-root /mnt/warm_storage/saral/smart_v2/embeddings/gte-large/task_embeddings \
  2>&1 | tee \
  /mnt/warm_storage/saral/smart_v2/logs/build_task_embeddings.log
```

Expected artifacts:

```text
/mnt/warm_storage/saral/smart_v2/embeddings/gte-large/task_embeddings/
├── tasks.pkl
├── tasks_embeddings.npy
├── task_catalog.csv
└── build_summary.json
```

---

## 6.4 Create the verifier

The primary integrity check recomputes the same float32 NumPy mean and requires exact equality. A chunked float64 mean is also computed as a numerical diagnostic.

```bash
cat > src/embeddings/verify_task_embeddings.py <<'PY'
"""Verify SMART-v2 task-mean embeddings against prompt matrices."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np


EXPECTED_TASK_COUNT = 309
EXPECTED_TOTAL_ROWS = 6_266_471
EXPECTED_DIMENSION = 1024
FLOAT64_DIAGNOSTIC_LIMIT = 5e-5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--task-embedding-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--prompt-embedding-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--float64-chunk-rows",
        type=int,
        default=8192,
    )

    return parser.parse_args()


def atomic_write_json(
    payload: Any,
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


def load_catalog(
    path: Path,
) -> list[dict[str, Any]]:
    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        raw_rows = list(csv.DictReader(handle))

    rows: list[dict[str, Any]] = []

    for raw in raw_rows:
        rows.append(
            {
                "task_index": int(
                    raw["task_index"]
                ),
                "corpus": raw["corpus"],
                "task_id": raw["task_id"],
                "valid_train_count": int(
                    raw["valid_train_count"]
                ),
                "first_dataset_index": int(
                    raw["first_dataset_index"]
                ),
                "last_dataset_index": int(
                    raw["last_dataset_index"]
                ),
            }
        )

    rows.sort(
        key=lambda row: row["task_index"]
    )

    return rows


def float64_chunked_mean(
    matrix: np.ndarray,
    start: int,
    end: int,
    chunk_rows: int,
) -> np.ndarray:
    accumulator = np.zeros(
        EXPECTED_DIMENSION,
        dtype=np.float64,
    )

    count = 0

    for chunk_start in range(
        start,
        end,
        chunk_rows,
    ):
        chunk_end = min(
            chunk_start + chunk_rows,
            end,
        )

        block = np.asarray(
            matrix[chunk_start:chunk_end],
            dtype=np.float64,
        )

        accumulator += block.sum(
            axis=0,
            dtype=np.float64,
        )
        count += block.shape[0]

    if count != end - start:
        raise RuntimeError(
            "Float64 diagnostic row count mismatch."
        )

    return accumulator / count


def main() -> int:
    args = parse_args()

    if args.float64_chunk_rows <= 0:
        raise ValueError(
            "--float64-chunk-rows must be positive."
        )

    task_root = args.task_embedding_root.resolve()
    prompt_root = args.prompt_embedding_root.resolve()

    tasks_path = task_root / "tasks.pkl"
    task_matrix_path = (
        task_root / "tasks_embeddings.npy"
    )
    catalog_path = task_root / "task_catalog.csv"

    with tasks_path.open("rb") as handle:
        tasks = pickle.load(handle)

    task_matrix = np.load(
        task_matrix_path,
        mmap_mode="r",
    )
    catalog = load_catalog(
        catalog_path
    )

    if len(tasks) != EXPECTED_TASK_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_TASK_COUNT} tasks; "
            f"found {len(tasks)}."
        )

    if task_matrix.shape != (
        EXPECTED_TASK_COUNT,
        EXPECTED_DIMENSION,
    ):
        raise RuntimeError(
            f"Unexpected task matrix shape: "
            f"{task_matrix.shape}"
        )

    if task_matrix.dtype != np.float32:
        raise RuntimeError(
            f"Unexpected task matrix dtype: "
            f"{task_matrix.dtype}"
        )

    if len(catalog) != EXPECTED_TASK_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_TASK_COUNT} catalog "
            f"rows; found {len(catalog)}."
        )

    exact_matches = 0
    maximum_float32_error = 0.0
    maximum_float64_drift = 0.0
    total_rows = 0
    failures: list[dict[str, Any]] = []
    corpus_intervals: dict[
        str,
        list[tuple[int, int, str]],
    ] = {}

    matrix_cache: dict[str, np.ndarray] = {}

    start_time = time.perf_counter()

    for expected_task_index, row in enumerate(
        catalog
    ):
        if row["task_index"] != expected_task_index:
            raise RuntimeError(
                "Catalog task indices are not contiguous."
            )

        task_id = row["task_id"]

        if tasks[expected_task_index] != task_id:
            raise RuntimeError(
                f"Task ordering mismatch at index "
                f"{expected_task_index}: "
                f"{tasks[expected_task_index]!r} != "
                f"{task_id!r}"
            )

        corpus = row["corpus"]

        if corpus not in matrix_cache:
            matrix_cache[corpus] = np.load(
                prompt_root
                / corpus
                / "train_prompt_embeddings.npy",
                mmap_mode="r",
            )

        prompt_matrix = matrix_cache[corpus]

        start = row["first_dataset_index"]
        end = row["last_dataset_index"] + 1

        expected_count = (
            row["valid_train_count"]
        )

        if end - start != expected_count:
            raise RuntimeError(
                f"{task_id}: catalog interval size "
                f"{end - start:,} != "
                f"{expected_count:,}."
            )

        if end > prompt_matrix.shape[0]:
            raise RuntimeError(
                f"{task_id}: interval exceeds "
                "prompt matrix size."
            )

        saved = np.asarray(
            task_matrix[expected_task_index],
            dtype=np.float32,
        )

        recomputed_float32 = np.mean(
            prompt_matrix[start:end],
            axis=0,
            dtype=np.float32,
        )

        float32_error = float(
            np.max(
                np.abs(
                    saved - recomputed_float32
                )
            )
        )

        maximum_float32_error = max(
            maximum_float32_error,
            float32_error,
        )

        if np.array_equal(
            saved,
            recomputed_float32,
        ):
            exact_matches += 1
        else:
            failures.append(
                {
                    "task_index": (
                        expected_task_index
                    ),
                    "task_id": task_id,
                    "reason": (
                        "float32_recompute_mismatch"
                    ),
                    "maximum_absolute_error": (
                        float32_error
                    ),
                }
            )

        reference_float64 = float64_chunked_mean(
            matrix=prompt_matrix,
            start=start,
            end=end,
            chunk_rows=args.float64_chunk_rows,
        )

        float64_drift = float(
            np.max(
                np.abs(
                    saved.astype(np.float64)
                    - reference_float64
                )
            )
        )

        maximum_float64_drift = max(
            maximum_float64_drift,
            float64_drift,
        )

        if (
            float64_drift
            > FLOAT64_DIAGNOSTIC_LIMIT
        ):
            failures.append(
                {
                    "task_index": (
                        expected_task_index
                    ),
                    "task_id": task_id,
                    "reason": (
                        "float64_diagnostic_drift"
                    ),
                    "maximum_absolute_error": (
                        float64_drift
                    ),
                    "limit": (
                        FLOAT64_DIAGNOSTIC_LIMIT
                    ),
                }
            )

        if not np.isfinite(saved).all():
            failures.append(
                {
                    "task_index": (
                        expected_task_index
                    ),
                    "task_id": task_id,
                    "reason": "nonfinite_task_mean",
                }
            )

        if float(
            np.linalg.norm(saved)
        ) <= 0.0:
            failures.append(
                {
                    "task_index": (
                        expected_task_index
                    ),
                    "task_id": task_id,
                    "reason": "zero_norm_task_mean",
                }
            )

        corpus_intervals.setdefault(
            corpus,
            [],
        ).append(
            (start, end, task_id)
        )

        total_rows += expected_count

        if (
            expected_task_index + 1
        ) % 25 == 0:
            print(
                f"Verified "
                f"{expected_task_index + 1}/"
                f"{EXPECTED_TASK_COUNT} tasks",
                flush=True,
            )

    for corpus, intervals in (
        corpus_intervals.items()
    ):
        intervals.sort()

        prompt_matrix = matrix_cache[corpus]

        expected_start = 0

        for start, end, task_id in intervals:
            if start != expected_start:
                failures.append(
                    {
                        "corpus": corpus,
                        "task_id": task_id,
                        "reason": (
                            "task_intervals_do_not_partition_corpus"
                        ),
                        "expected_start": (
                            expected_start
                        ),
                        "actual_start": start,
                    }
                )

            expected_start = end

        if expected_start != prompt_matrix.shape[0]:
            failures.append(
                {
                    "corpus": corpus,
                    "reason": (
                        "task_intervals_do_not_cover_corpus"
                    ),
                    "covered_rows": expected_start,
                    "matrix_rows": (
                        prompt_matrix.shape[0]
                    ),
                }
            )

    if total_rows != EXPECTED_TOTAL_ROWS:
        failures.append(
            {
                "reason": "global_row_count_mismatch",
                "expected": EXPECTED_TOTAL_ROWS,
                "actual": total_rows,
            }
        )

    task_norms = np.linalg.norm(
        np.asarray(task_matrix),
        axis=1,
    )

    elapsed = (
        time.perf_counter() - start_time
    )

    status = (
        "verified"
        if not failures
        else "failed"
    )

    report_path = (
        task_root / "verification_report.json"
    )

    atomic_write_json(
        {
            "format_version": 1,
            "stage": (
                "smart_v2_task_embedding_verification"
            ),
            "status": status,
            "task_count": len(tasks),
            "source_rows_checked": total_rows,
            "exact_float32_matches": (
                exact_matches
            ),
            "maximum_float32_recompute_error": (
                maximum_float32_error
            ),
            "maximum_float64_reference_drift": (
                maximum_float64_drift
            ),
            "float64_diagnostic_limit": (
                FLOAT64_DIAGNOSTIC_LIMIT
            ),
            "task_norm": {
                "minimum": float(
                    task_norms.min()
                ),
                "maximum": float(
                    task_norms.max()
                ),
                "mean": float(
                    task_norms.mean()
                ),
            },
            "failure_count": len(failures),
            "failures": failures,
            "elapsed_seconds": elapsed,
        },
        report_path,
    )

    print()
    print("=== Task-embedding verification ===")
    print(f"Status:                    {status}")
    print(f"Tasks checked:             {len(tasks)}")
    print(
        f"Source rows checked:       "
        f"{total_rows:,}"
    )
    print(
        f"Exact float32 matches:     "
        f"{exact_matches}/{len(tasks)}"
    )
    print(
        f"Maximum float32 error:     "
        f"{maximum_float32_error:.9g}"
    )
    print(
        f"Maximum float64 drift:     "
        f"{maximum_float64_drift:.9g}"
    )
    print(
        f"Task norm range:           "
        f"{task_norms.min():.8f} to "
        f"{task_norms.max():.8f}"
    )
    print(f"Failures:                  {len(failures)}")
    print(
        f"Elapsed:                   "
        f"{elapsed / 60:.2f} minutes"
    )
    print(f"Report:                    {report_path}")

    if failures:
        raise RuntimeError(
            "Task-embedding verification failed. "
            "Inspect verification_report.json."
        )

    print()
    print(
        "All task-embedding verification checks passed."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
PY

python3 -m py_compile \
  src/embeddings/verify_task_embeddings.py
```

---

## 6.5 Run verification

```bash
cd /data/saral/wdir/smart_v2 || exit 1
source ./smart_v2_env.sh

python3 -m src.embeddings.verify_task_embeddings \
  --task-embedding-root /mnt/warm_storage/saral/smart_v2/embeddings/gte-large/task_embeddings \
  --prompt-embedding-root /mnt/warm_storage/saral/smart_v2/embeddings/gte-large \
  --float64-chunk-rows 8192 \
  2>&1 | tee \
  /mnt/warm_storage/saral/smart_v2/logs/verify_task_embeddings.log
```

Acceptance conditions:

```text
Status                      = verified
Tasks checked               = 309
Source rows checked         = 6,266,471
Exact float32 matches       = 309/309
Maximum float32 error       = 0
Failures                    = 0
Task matrix shape           = (309, 1024)
Task matrix dtype           = float32
```

The float64 drift is diagnostic only, with a failure threshold of `5e-5`.
