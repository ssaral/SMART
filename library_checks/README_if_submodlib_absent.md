Proceed only after verification checks have passed. This document resolves the `submodlib` library absent from the ecosystem.

The current PyPI distribution is named **`submodlib-py`**, while the Python import remains `import submodlib`. Version `0.0.3` provides a prebuilt CPython 3.10 manylinux wheel, so your container should not need to compile the C++ extension. ([PyPI][1]) ([PyPI][1])

## Step 8.0 — Install and validate Submodlib

Inside the running Docker container:

```bash
cd /data/saral/wdir/smart || exit 1

ENV_ROOT=/mnt/warm_storage/saral/smart/environment
mkdir -p "$ENV_ROOT"

python3 -m pip freeze | sort > \
  "$ENV_ROOT/pip_freeze_before_submodlib.txt"

python3 -m pip install \
  --no-cache-dir \
  --only-binary=:all: \
  "submodlib-py==0.0.3"

python3 -m pip check

python3 -m pip freeze | sort > \
  "$ENV_ROOT/pip_freeze_after_submodlib.txt"
```

### Compatibility smoke test

Create:

```bash
cd /data/saral/wdir/smart || exit 1

cat > submod_check.py <<'PY'
from __future__ import annotations

import importlib.metadata

import numpy as np
import submodlib
import submodlib.functions as submod_fn


def make_objective(
    kernel: np.ndarray,
) -> submod_fn.graphCut.GraphCutFunction:
    return submod_fn.graphCut.GraphCutFunction(
        n=kernel.shape[0],
        mode="dense",
        ggsijs=kernel,
        lambdaVal=0.4,
        separate_rep=False,
    )


def full_lazy_greedy_order(
    kernel: np.ndarray,
) -> list[tuple[int, float]]:
    """Return all n elements despite submodlib's budget < n rule."""

    n = kernel.shape[0]
    objective = make_objective(kernel)

    if n == 1:
        gain = float(
            objective.marginalGain(set(), 0)
        )
        return [(0, gain)]

    result = objective.maximize(
        budget=n - 1,
        optimizer="LazyGreedy",
        stopIfZeroGain=False,
        stopIfNegativeGain=False,
        verbose=False,
        show_progress=False,
    )

    result = [
        (int(index), float(gain))
        for index, gain in result
    ]

    selected = {
        index
        for index, _ in result
    }

    remaining = sorted(
        set(range(n)) - selected
    )

    if len(remaining) != 1:
        raise RuntimeError(
            f"Expected one remaining element; found {remaining}"
        )

    final_index = remaining[0]

    final_gain = float(
        objective.marginalGain(
            selected,
            final_index,
        )
    )

    # Independent validation using objective values.
    previous_value = float(
        objective.evaluate(selected)
    )
    complete_set = selected | {final_index}
    complete_value = float(
        objective.evaluate(complete_set)
    )
    evaluation_gain = complete_value - previous_value

    if not np.isclose(
        final_gain,
        evaluation_gain,
        rtol=1e-6,
        atol=1e-6,
    ):
        raise RuntimeError(
            "Final marginal gain validation failed: "
            f"marginalGain={final_gain}, "
            f"evaluate difference={evaluation_gain}"
        )

    result.append(
        (final_index, final_gain)
    )

    return result


print(
    "submodlib-py:",
    importlib.metadata.version("submodlib-py"),
)
print("NumPy:", np.__version__)

points = np.asarray(
    [
        [1.0, 0.0],
        [0.9, 0.1],
        [0.0, 1.0],
        [0.1, 0.9],
    ],
    dtype=np.float32,
)

kernel = submodlib.helper.create_kernel(
    X=points,
    metric="cosine",
    method="sklearn",
)

kernel = np.asarray(kernel)

print("Kernel shape:", kernel.shape)
print("Kernel dtype:", kernel.dtype)
print("Kernel finite:", bool(np.isfinite(kernel).all()))
print("Kernel:")
print(kernel)

result_1 = full_lazy_greedy_order(kernel)
result_2 = full_lazy_greedy_order(kernel)

print("Graph Cut result 1:", result_1)
print("Graph Cut result 2:", result_2)

assert kernel.shape == (4, 4)
assert np.isfinite(kernel).all()
assert len(result_1) == 4
assert len({index for index, _ in result_1}) == 4
assert {index for index, _ in result_1} == set(range(4))
assert result_1 == result_2

print("Submodlib compatibility smoke test passed.")
PY

python3 submod_check.py
```

The documented Graph Cut API accepts a dense `ggsijs` similarity matrix, `lambdaVal`, and the `LazyGreedy` optimizer, matching the SMART repository call we inspected. ([submodlib.readthedocs.io][2]) ([submodlib.readthedocs.io][2])

Expected ending:

```text
Graph Cut result 1: [...]
Graph Cut result 2: [...]
Submodlib compatibility smoke test passed.
```

## Make the dependency persistent

Because your container uses `--rm`, this installation disappears when the container exits. Add a small derived image rather than reinstalling each time.

Create `/data/saral/wdir/smart/Dockerfile.smart`:

```dockerfile
FROM ssaral/lessexp:v3

RUN python3 -m pip install \
    --no-cache-dir \
    --only-binary=:all: \
    submodlib-py==0.0.3
```

Build it outside the container:

```bash
cd /data/saral/wdir/smart

docker build \
  -f Dockerfile.smart \
  -t ssaral/lessexp:v3-smart .
```

For the current running container, proceed immediately after the smoke test passes. Then run the previously created Stage 1 command:

```bash
cd /data/saral/wdir/smart || exit 1

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

mkdir -p \
  /mnt/warm_storage/saral/smart/artifacts/stage1_graph_cut

python3 data_generation_scripts/run_stage1_graph_cut.py \
  --manifest /mnt/warm_storage/saral/smart/prepared_data/clean_task_manifest.csv \
  --task-embeddings /mnt/warm_storage/saral/smart/artifacts/prompt_embeddings/gte-large/task_embeddings.npy \
  --output-root /mnt/warm_storage/saral/smart/artifacts/stage1_graph_cut \
  --lambda-value 0.4 \
  --budget 309 \
  --determinism-runs 2 \
  2>&1 | tee \
  /mnt/warm_storage/saral/smart/artifacts/stage1_graph_cut/stage1_graph_cut.log
```

[1]: https://pypi.org/project/submodlib-py/ "submodlib-py · PyPI"
[2]: https://submodlib.readthedocs.io/en/latest/functions/graphCut.html "Graph Cut — submodlib 1.1.5 documentation"
