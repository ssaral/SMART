# SMART Instruction-Tuning Replication

> End-to-end reconstruction of the SMART data-selection, fine-tuning, and leaderboard-evaluation pipeline used in this project.

---

## Table of contents

1. [Scope](#1-scope)
2. [SMART methodology](#2-smart-methodology)
3. [Reproduction-fidelity labels](#3-reproduction-fidelity-labels)
4. [Project layout](#4-project-layout)
5. [Environment](#5-environment)
6. [Source data and raw format](#6-source-data-and-raw-format)
7. [Source-data audit](#7-source-data-audit)
8. [Prompt embeddings](#8-prompt-embeddings)
9. [Task embeddings](#9-task-embeddings)
10. [Stage 1: Graph Cut task weighting](#10-stage-1-graph-cut-task-weighting)
11. [Stage 2: Facility Location](#11-stage-2-facility-location)
12. [Large-task and QQP memory handling](#12-large-task-and-qqp-memory-handling)
13. [Stage 2 production execution](#13-stage-2-production-execution)
14. [Materializing the selected datasets](#14-materializing-the-selected-datasets)
15. [Trainer-required data format](#15-trainer-required-data-format)
16. [Tokenization and supervision audit](#16-tokenization-and-supervision-audit)
17. [Reduced training-time validation](#17-reduced-training-time-validation)
18. [Fine-tuning configuration](#18-fine-tuning-configuration)
19. [Distributed scheduler correction](#19-distributed-scheduler-correction)
20. [LoRA extension](#20-lora-extension)
21. [Training smoke tests](#21-training-smoke-tests)
22. [Production fine-tuning](#22-production-fine-tuning)
23. [Checkpoint naming](#23-checkpoint-naming)
24. [Training-result verification](#24-training-result-verification)
25. [Leaderboard evaluation](#25-leaderboard-evaluation)
26. [Four-GPU evaluation scheduler](#26-four-gpu-evaluation-scheduler)
27. [Script inventory](#27-script-inventory)
28. [Reproducibility invariants](#28-reproducibility-invariants)
29. [Known deviations](#29-known-deviations)
30. [Frequently asked questions](#30-frequently-asked-questions)
31. [Clean rerun checklist](#31-clean-rerun-checklist)
32. [Reporting recommendations](#32-reporting-recommendations)
33. [Provenance to retain](#33-provenance-to-retain)
34. [References](#34-references)

---

# 1. Scope

This repository reproduces the **methodology** of SMART: a two-stage submodular data-mixture procedure for selecting a compact instruction-tuning set from a much larger multitask instruction collection.

The complete local pipeline:

1. audits locally available instruction-task files;
2. normalizes their prompt and response fields;
3. embeds every valid training prompt using GTE-large;
4. averages prompt embeddings into task embeddings;
5. applies Graph Cut over tasks to determine task importance;
6. converts Graph Cut gains into per-task data allocations;
7. applies Facility Location inside each task;
8. materializes fixed SMART-25K and SMART-50K datasets;
9. audits the materialized data with both model tokenizers;
10. fine-tunes Llama-2-7B and Qwen2-7B;
11. runs both full fine-tuning and LoRA experiments;
12. evaluates all saved checkpoints with `lm-evaluation-harness`.

The locally reconstructed ground set contains:

```text
Task files:             309
Valid training rows:    6,266,471
Original validation:      183,870
```

The local ground set comes from:

```text
cot
flan2021
sglue
t0
tulu
```

The SMART paper used the complete FLAN 2022 collection, reported as approximately:

```text
Tasks:       1,840
Examples:    17.5 million
```

Therefore, this work must be described as:

> **A method-level SMART replication on a locally reconstructed 309-task instruction collection, with Qwen2 and LoRA extensions.**

It is not a bit-for-bit reproduction of the authors' entire data release or reported benchmark table.

---

# 2. SMART methodology

Let the instruction collection be:

\[
D=\{T_1,\ldots,T_M\},
\]

where every task \(T_i\) is a collection of prompt-response pairs:

\[
T_i=\{(\text{prompt}_{ij},\text{response}_{ij})\}_{j=1}^{N_i}.
\]

SMART performs two cardinality-constrained submodular optimizations.

## 2.1 Stage 1: task weighting

Each training prompt is embedded using GTE-large.

The embedding for task \(T_i\) is the mean of its prompt embeddings:

\[
e(T_i)
=
\frac{1}{|T_i|}
\sum_{x\in T_i} e(x).
\]

Graph Cut is then optimized over the task embeddings:

\[
f_{\mathrm{GC}}(X)
=
\sum_{i\in V,j\in X}s_{ij}
-
\lambda
\sum_{i,j\in X}s_{ij}.
\]

The local Stage 1 configuration is:

```text
Objective:       Graph Cut
Lambda:          0.4
Similarity:      cosine
Optimizer:       LazyGreedy
Seed:            23
Task count:      309
```

The marginal gains returned by greedy selection are converted into probabilities using the authors' second-order Taylor-softmax transformation.

Those probabilities determine how much of the global 25K or 50K budget is assigned to each task.

## 2.2 Stage 2: instance selection

Within every task, SMART uses Facility Location:

\[
f_{\mathrm{FL}}(X)
=
\sum_{i\in V}\max_{j\in X}s_{ij}.
\]

Facility Location chooses examples that jointly represent the full task.

The local Stage 2 configuration is:

```text
Objective:       Facility Location
Similarity:      cosine
Optimizer:       LazyGreedy
Seed:            23
Input features:  GTE-large prompt embeddings
```

The 50K selection ordering is generated first. The 25K mixture uses the appropriate prefix of that ordering, making the two budgets nested.

---

# 3. Reproduction-fidelity labels

Not every local engineering decision has the same scientific status.

This README uses the following labels.

| Label | Meaning |
|---|---|
| **Author method** | Directly follows the SMART paper or released implementation. |
| **Infrastructure adaptation** | Changes storage, hardware, or scheduling mechanics without intentionally changing the mathematical experiment. |
| **Validated approximation** | Approximates an otherwise infeasible computation while preserving the represented data and objective as closely as measured. |
| **Correctness fix** | Repairs behavior that contradicted the intended experiment. |
| **Experimental extension** | An additional experiment not presented as the primary author-comparable result. |

## 3.1 Fidelity matrix

| Component | Local implementation | Classification |
|---|---|---|
| Prompt representation | GTE-large, 1024-dimensional embeddings | Author method |
| Task representation | Mean prompt embedding | Author method |
| Stage 1 objective | Graph Cut | Author method |
| Graph Cut coefficient | `lambda=0.4` | Author method |
| Stage 1 optimizer | LazyGreedy | Author method |
| Stage 2 objective | Facility Location | Author method |
| Stage 2 similarity | Prompt-level cosine similarity | Author method |
| Tasks with at most 5,000 rows | Exact dense Facility Location | Author method |
| Tasks with more than 5,000 rows | Full represented set with 2,048 selectable candidates | Validated approximation |
| 25K as prefix of 50K | Nested selection order | Reproducibility adaptation |
| Four H200 GPUs | Global batch preserved at 64 | Infrastructure adaptation |
| Manual scheduler stepping | One scheduler step per optimizer update | Correctness fix |
| Reduced 10K train-time validation | Full validation retained separately | Infrastructure adaptation |
| Qwen2-7B | Same SMART mixtures and training protocol | Experimental extension |
| LoRA | Controlled parameter-efficient fine-tuning | Experimental extension |

---

# 4. Project layout

The project uses two roots.

```text
Code and local base models:
/data/saral/wdir/smart

Large artifacts and outputs:
/mnt/warm_storage/saral/smart
```

## 4.1 Code tree

```text
/data/saral/wdir/smart/
├── README.md
├── config.yaml
├── local_dataset.py
│
├── llama2_7b/
├── qwen2_7b/
│
├── data/
│   ├── cot/
│   ├── flan2021/
│   ├── sglue/
│   ├── t0/
│   └── tulu/
│
├── data_generation_scripts/
│   ├── get_SMART_mixture.py
│   ├── get_SMART_mixture.sh
│   ├── run_stage2_selection.py
│   ├── build_reduced_validation_datasets.py
│   └── utils/
│       ├── gc_ordering.py
│       ├── fl_ordering.py
│       ├── logdet_ordering.py
│       └── random_ordering.py
│
├── instruction_tuner.py
├── instruction_tuner.sh
├── instruction_tuner_local.py
├── instruction_tuner_local.before_scheduler_fix.py
│
├── accelerate_4gpu_bf16.yaml
│
├── run_training_smokes.sh
├── run_training_smokes_v2.sh
├── run_one_production_training.sh
├── run_one_production_training_v2.sh
├── run_remaining_production_trainings.sh
├── run_all_finetuning_v2.sh
│
├── run_all_leaderboard_evals.sh
└── run_all_lm_eval_wait4.sh
```

## 4.2 Artifact tree

```text
/mnt/warm_storage/saral/smart/
├── artifacts/
│   ├── stage1_allocations/
│   │   └── task_allocations.csv
│   │
│   ├── prompt_embeddings/
│   │   └── gte-large/
│   │       └── tasks/
│   │           └── <task-index>_<task-id>/
│   │               ├── prompt_embeddings.npy
│   │               ├── source_indices.npy
│   │               └── metadata.json
│   │
│   └── stage2_selection/
│       ├── stage2_selection_catalog.json
│       ├── stage2_selection_summary.json
│       ├── stage2_production.log
│       └── tasks/
│           └── <task-index>_<task-id>/
│               ├── selected_full_local_indices.npy
│               ├── selected_source_indices.npy
│               ├── marginal_gains.npy
│               ├── candidate_full_local_indices.npy
│               ├── selection.csv
│               └── metadata.json
│
├── datasets/
│   ├── trainer_eval_safe/
│   │   ├── smart_25000/
│   │   └── smart_50000/
│   │
│   └── trainer_eval_safe_10k/
│       ├── smart_25000/
│       ├── smart_50000/
│       └── reduced_validation_summary.json
│
├── models/
│   └── <checkpoint-name>/
│
├── logs/
│   ├── production_training/
│   └── production_training_v2/
│
├── cache/
│   └── lm_eval_leaderboard_corrected/
│
└── evaluations/
    └── lm_eval_leaderboard_corrected/
        ├── scheduler/
        ├── results_manifest.tsv
        └── <checkpoint-name>/
            ├── mmlu_pro.json
            ├── mmlu_pro.log
            ├── bbh.json
            ├── bbh.log
            ├── musr.json
            ├── musr.log
            ├── math_hard.json
            ├── math_hard.log
            ├── ifeval.json
            ├── ifeval.log
            ├── gpqa.json
            └── gpqa.log
```

## 4.3 Script-inventory note

Some early data-audit, embedding, Stage 1, materialization, and tokenizer-audit steps were executed as one-off Python heredocs rather than saved under a stable filename.

This README does not invent filenames for code that was not verifiably persisted.

Before publishing the project, generate an exact inventory from the live checkout:

```bash
cd /data/saral/wdir/smart || exit 1

find . \
  -path './.git' -prune -o \
  -type f \
  \( -name '*.py' -o -name '*.sh' -o -name '*.yaml' \) \
  -print |
sort > replication_script_inventory.txt
```

Commit `replication_script_inventory.txt` with the repository.

---

# 5. Environment

## 5.1 Hardware

Production training was designed for:

```text
GPUs:                 4 × NVIDIA H200
Distributed backend:  NCCL
Precision:            bfloat16
Attention:            FlashAttention 2
Maximum sequence:     4096
Seed:                 23
```

## 5.2 Core dependencies

```text
torch
transformers
accelerate
datasets
sentence-transformers
numpy
submodlib
peft
flash-attn
```

Evaluation additionally requires:

```text
lm-evaluation-harness
math evaluation dependencies
IFEval dependencies
sentencepiece where required
```

## 5.3 Environment capture

Record exact versions from the production environment:

```bash
python3 -m pip freeze \
  > /mnt/warm_storage/saral/smart/environment_pip_freeze.txt
```

```bash
nvidia-smi -q \
  > /mnt/warm_storage/saral/smart/environment_nvidia_smi_q.txt
```

Record the current source revision:

```bash
cd /data/saral/wdir/smart

git rev-parse HEAD \
  > /mnt/warm_storage/saral/smart/project_git_commit.txt

git status --short \
  > /mnt/warm_storage/saral/smart/project_git_status.txt
```

---

# 6. Source data and raw format

## 6.1 Local collections

The local task files come from:

```text
/data/saral/wdir/smart/data/cot
/data/saral/wdir/smart/data/flan2021
/data/saral/wdir/smart/data/sglue
/data/saral/wdir/smart/data/t0
/data/saral/wdir/smart/data/tulu
```

## 6.2 Raw JSON format

Each source file is generally one JSON object containing split names such as `train` and `validation`.

Representative structure:

```json
{
  "train": [
    {
      "inputs": "Instruction or prompted input",
      "targets": "Expected response"
    }
  ],
  "validation": [
    {
      "inputs": "Validation prompt",
      "targets": "Validation response"
    }
  ]
}
```

The important raw fields are:

```text
inputs
targets
```

Some source records contained:

- empty strings;
- whitespace-only answers;
- missing values;
- non-string values;
- malformed rows;
- prompts whose tokenized length left no space for a supervised answer.

No new semantic instructions or answers were synthesized during normalization.

## 6.3 Desired trainer format

The released SMART materialization logic maps:

```text
inputs  -> prompt
targets -> response
```

The final trainer-facing record is:

```json
{
  "prompt": "Instruction or prompted input",
  "response": "Expected response"
}
```

The final dataset is a Hugging Face `DatasetDict`:

```text
DatasetDict
├── train
│   ├── prompt
│   └── response
└── validation
    ├── prompt
    └── response
```

---

# 7. Source-data audit

Before generating embeddings, every task file was audited.

The audit established:

- source path;
- corpus;
- task name;
- deterministic task ID;
- available split names;
- train and validation counts;
- prompt and target field types;
- empty prompt counts;
- empty target counts;
- malformed-row counts;
- stable source indices.

Audited totals:

```text
Tasks:                 309
Valid training rows:   6,266,471
Validation rows:         183,870
```

## 7.1 Validity policy

A training row must contain:

```text
a usable prompt
a usable supervised response
```

Rows with missing, empty, or unusable training targets were excluded from the valid selection ground set.

This is not an arbitrary benchmark-oriented subset. A row without a supervised answer cannot contribute a valid causal-language-model target.

## 7.2 Source-index preservation

The pipeline preserves the mapping:

```text
valid local row index
        ->
original source-file row index
```

This mapping is saved as:

```text
source_indices.npy
```

The selected Stage 2 rows can therefore be traced back to their exact source task and source position.

This makes the final 25K and 50K datasets auditable and reconstructible.

---

# 8. Prompt embeddings

## 8.1 Encoder

SMART represents prompts with GTE-large.

The local configuration is:

```text
Encoder:     thenlper/gte-large
Dimension:   1024
Input:       prompt text only
Output:      float32
```

Responses are not included in the selection embedding.

This follows SMART's use of prompt similarity for both task and instance selection.

## 8.2 Per-task storage

Embeddings are stored separately for each task:

```text
/mnt/warm_storage/saral/smart/artifacts/prompt_embeddings/gte-large/tasks/
└── <task-index>_<task-id>/
    ├── prompt_embeddings.npy
    ├── source_indices.npy
    └── metadata.json
```

Expected array shapes:

```text
prompt_embeddings.npy:
    [valid_train_count, 1024]

source_indices.npy:
    [valid_train_count]
```

## 8.3 Why embeddings were stored per task

Storing millions of 1,024-dimensional vectors in one Python object would create substantial memory and serialization overhead.

Instead:

- each task was processed independently;
- vectors were saved as contiguous float32 NPY arrays;
- source-row mappings were saved alongside embeddings;
- downstream code used memory mapping;
- task-sized objects were released between tasks.

This changes storage mechanics, not SMART's embeddings or objective.

## 8.4 Memory mapping

Stage 2 loads embeddings using:

```python
np.load(
    embeddings_path,
    mmap_mode="r",
)
```

This avoids copying every task's complete embedding matrix into a second in-memory representation.

---

# 9. Task embeddings

For each task \(T_i\), its task embedding is:

\[
e(T_i)
=
\frac{1}{|T_i|}
\sum_{x\in T_i}e(x).
\]

This follows the SMART paper.

The task-level similarity matrix is only:

```text
309 × 309
```

Therefore, Stage 1 does not require candidate restriction or approximate similarity storage.

It was run as an exact dense task-level optimization.

---

# 10. Stage 1: Graph Cut task weighting

## 10.1 Configuration

```text
Task count:     309
Objective:      Graph Cut
Lambda:         0.4
Similarity:     cosine
Optimizer:      LazyGreedy
Seed:           23
```

All 309 local task units received positive allocations.

For these experiments:

```text
M' = M = 309
```

Therefore, Stage 1 determines task ranking and task weights rather than pruning the task collection to a smaller task count.

This corresponds to the authors' full-task weighted-mixture setting.

## 10.2 Graph Cut gains

LazyGreedy produces an ordered task list and one marginal gain per selected task.

The gains act as task importance scores.

## 10.3 Taylor-softmax allocation

The marginal gains were converted into task probabilities using the authors' second-order Taylor-softmax transformation.

The probabilities were multiplied by the desired total example budget:

```text
25,000
50,000
```

Integer reconciliation ensured exact totals:

```text
sum(final_allocation_25000) = 25,000
sum(final_allocation_50000) = 50,000
```

## 10.4 Allocation artifact

```text
/mnt/warm_storage/saral/smart/artifacts/stage1_allocations/task_allocations.csv
```

Important columns:

```text
graph_cut_rank
task_index
task_id
corpus
task_name
valid_train_count
final_allocation_25000
final_allocation_50000
```

## 10.5 Allocation audit

```bash
python3 - <<'PY'
import csv
from pathlib import Path

path = Path(
    "/mnt/warm_storage/saral/smart/"
    "artifacts/stage1_allocations/task_allocations.csv"
)

with path.open(
    encoding="utf-8",
    newline="",
) as handle:
    rows = list(csv.DictReader(handle))

assert len(rows) == 309

total_25 = sum(
    int(row["final_allocation_25000"])
    for row in rows
)

total_50 = sum(
    int(row["final_allocation_50000"])
    for row in rows
)

assert total_25 == 25_000
assert total_50 == 50_000

for row in rows:
    n = int(row["valid_train_count"])
    k25 = int(row["final_allocation_25000"])
    k50 = int(row["final_allocation_50000"])

    assert 0 < k25 <= k50 < n

print("Tasks:", len(rows))
print("25K total:", total_25)
print("50K total:", total_50)
print("Stage 1 allocation audit passed.")
PY
```

---

# 11. Stage 2: Facility Location

## 11.1 Author computation

For one task containing \(n\) examples, exact dense Facility Location operates over an \(n \times n\) cosine-similarity kernel.

For small tasks, this is feasible and was used directly.

## 11.2 Frozen Stage 2 policy

```text
Seed:                    23

50K ordering length:     task's 50K allocation
25K selection:           prefix of the 50K ordering

Task size <= 5,000:      exact dense Facility Location
Task size > 5,000:       candidate-restricted Facility Location

Candidate pool:          2,048 examples
Represented set:         every valid example in the task

Optimizer:               LazyGreedy
Similarity:              cosine
```

## 11.3 Production method counts

```text
Exact dense tasks:            59
Candidate-restricted tasks:  250
Total tasks:                 309
```

The complete production Stage 2 selection required approximately:

```text
10.84 hours
```

## 11.4 Nested budgets

For each task:

```text
selected order length = allocation for SMART-50K
SMART-25K rows         = prefix of that order
```

This means SMART-25K is nested inside SMART-50K.

Nested budgets reduce random variation when comparing data-scale effects.

---

# 12. Large-task and QQP memory handling

This section is critical because the large memory figure can easily be misinterpreted.

## 12.1 QQP was not a 400+ GB text dataset

The local QQP source file was approximately:

```text
114,772,611 bytes
```

or roughly:

```text
109 MiB
```

Its valid training split contained:

```text
363,846 rows
```

The 400+ GiB issue did **not** refer to the size of the raw QQP JSON.

It referred to the dense pairwise float32 similarity kernel.

## 12.2 Why exact dense QQP is approximately 493 GiB

An exact dense similarity matrix for 363,846 examples contains:

\[
363,846^2
\]

entries.

At four bytes per float32 entry:

\[
363,846^2 \times 4
\approx 493\text{ GiB}.
\]

Therefore:

```text
Raw QQP text:              ~109 MiB
Dense QQP similarity:      ~493 GiB
```

The problem is quadratic similarity storage, not raw-data loading.

## 12.3 Was QQP reduced to 2,048 represented rows?

**No.**

For candidate-restricted Facility Location:

```text
Represented QQP examples:  363,846
Selectable candidates:       2,048
```

Every valid QQP prompt remains in the represented set and contributes to the Facility Location coverage objective.

Only the set of examples eligible to become selected facilities is restricted.

The cross-kernel becomes:

\[
363,846 \times 2,048 \times 4
\approx 2.78\text{ GiB}.
\]

Conceptually:

```text
all 363,846 valid QQP examples
              |
              | are represented by similarity to
              v
2,048 deterministic candidate facilities
              |
              | Facility Location + LazyGreedy
              v
the task's final allocated examples
```

This is not equivalent to discarding all but 2,048 QQP rows before selection.

## 12.4 Where subsetting actually occurs

There are two distinct operations that may both be called "subsetting."

### Intended SMART subset

SMART's purpose is to select:

```text
25,000 or 50,000 examples
```

from millions of available instruction examples.

That final reduction is not an engineering compromise. It is the method being evaluated.

### Candidate restriction

For tasks larger than 5,000 rows, the set of examples eligible to be selected is restricted to 2,048 deterministic candidates.

This is an additional scalability approximation.

The represented set is still the complete valid task.

## 12.5 Does candidate restriction necessarily lower benchmarks?

No deterministic claim can be made in either direction without running both training pipelines.

Candidate restriction can change the selected examples. Therefore, it may affect downstream benchmark scores.

However, it is not logically true that scores must decrease simply because candidate restriction was used.

Reasons:

- all valid task examples remain represented in the objective;
- the same GTE-large prompt embeddings are used;
- the same cosine similarity is used;
- the same Facility Location objective is used;
- the same LazyGreedy optimizer is used;
- the candidate set is deterministic;
- the candidate size was validated before production;
- objective retention was extremely high on validation tasks.

Measured objective ratios relative to exact selection:

```text
cot::stream_qed_ii
  SMART-25K ratio: 0.999221
  SMART-50K ratio: 0.999100

ANLI R3 with 10K represented examples
  SMART-25K ratio: 0.997669
  SMART-50K ratio: 0.996663
```

These measurements show that the candidate-restricted solution closely preserves the Facility Location objective on the tested tasks.

They do **not** prove equal downstream benchmark accuracy.

The scientifically defensible statement is:

> Candidate-restricted Facility Location is a validated scalability approximation. It retains the complete represented task and nearly preserves the Facility Location objective on tested tasks. It may produce a different coreset than exact dense Facility Location, but it is not equivalent to blindly training on a random 2,048-row task subset.

## 12.6 Why candidate size 2,048 was selected

Several candidate sizes were compared against tractable reference computations.

The production criterion was fixed before running all tasks.

A candidate pool of 2,048 was the smallest tested pool that passed the objective-retention gate on both validation tasks.

## 12.7 Why exact threshold 5,000 was selected

Measured exact-dense scaling included:

```text
5K task:   18.75 seconds
10K task:  63.44 seconds
```

The 5K threshold keeps exact author behavior where it is practical while avoiding rapid quadratic growth for larger tasks.

## 12.8 What exact large-task reproduction would require

Exact full-selectable Facility Location for QQP would require either:

- hundreds of GiB of host memory for direct dense construction;
- a mathematically equivalent distributed implementation;
- a streaming Facility Location implementation that does not materialize the full square kernel;
- blockwise gain updates proven equivalent to dense LazyGreedy.

The current implementation must therefore be labeled as:

```text
exact for small tasks
validated candidate-restricted approximation for large tasks
```

---

# 13. Stage 2 production execution

Production selector:

```text
/data/saral/wdir/smart/data_generation_scripts/run_stage2_selection.py
```

## 13.1 Responsibilities

The script:

- loads all 309 Stage 1 allocations;
- validates 25K and 50K totals;
- memory-maps prompt embeddings;
- memory-maps source indices;
- chooses exact or candidate-restricted mode;
- generates deterministic candidate indices;
- runs Facility Location with LazyGreedy;
- checks selected-index uniqueness;
- checks finite marginal gains;
- maps local selected indices to source indices;
- manually recomputes Facility Location coverage;
- verifies cumulative greedy gains against manual coverage;
- saves atomic per-task results;
- skips already completed and valid task outputs;
- writes global catalogs and summaries.

## 13.2 Production command

```bash
cd /data/saral/wdir/smart || exit 1

ROOT=/mnt/warm_storage/saral/smart
STAGE2="$ROOT/artifacts/stage2_selection"

mkdir -p "$STAGE2"

THREADS=$(nproc)

if [ "$THREADS" -gt 32 ]; then
  THREADS=32
fi

export OMP_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS="$THREADS"

python3 \
  data_generation_scripts/run_stage2_selection.py \
  --allocations "$ROOT/artifacts/stage1_allocations/task_allocations.csv" \
  --embedding-root "$ROOT/artifacts/prompt_embeddings/gte-large" \
  --output-root "$STAGE2" \
  --exact-threshold 5000 \
  --candidate-size 2048 \
  --seed 23 \
  2>&1 | tee "$STAGE2/stage2_production.log"
```

## 13.3 Per-task outputs

```text
selected_full_local_indices.npy
selected_source_indices.npy
marginal_gains.npy
selection.csv
metadata.json
```

Candidate-restricted tasks additionally contain:

```text
candidate_full_local_indices.npy
```

## 13.4 Global outputs

```text
stage2_selection_catalog.json
stage2_selection_summary.json
```

## 13.5 Resumability

The production script verifies saved metadata and array shapes before skipping a task.

Rerunning the same command:

- does not recompute valid completed tasks;
- recomputes missing or inconsistent outputs;
- preserves deterministic candidate membership and selection.

---

# 14. Materializing the selected datasets

Stage 2 produces selected source indices. The trainer requires actual prompt-response records.

Materialization therefore performs:

1. read `task_allocations.csv`;
2. read each task's `selected_source_indices.npy`;
3. open the original source task JSON;
4. recover the selected source records;
5. map `inputs` to `prompt`;
6. map `targets` to `response`;
7. concatenate records across tasks;
8. shuffle the training split with seed 23;
9. create a Hugging Face `DatasetDict`;
10. save the dataset with `save_to_disk`.

## 14.1 Canonical outputs

```text
/mnt/warm_storage/saral/smart/datasets/trainer_eval_safe/smart_25000

/mnt/warm_storage/saral/smart/datasets/trainer_eval_safe/smart_50000
```

## 14.2 Expected training sizes

```text
SMART-25K train rows: 25,000
SMART-50K train rows: 50,000
```

## 14.3 Dataset audit

```bash
python3 - <<'PY'
from datasets import load_from_disk

root = (
    "/mnt/warm_storage/saral/smart/"
    "datasets/trainer_eval_safe"
)

for budget in (25000, 50000):
    dataset = load_from_disk(
        f"{root}/smart_{budget}"
    )

    print()
    print(f"SMART-{budget}")
    print("Train:", len(dataset["train"]))
    print(
        "Validation:",
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

    assert set(
        dataset["train"].column_names
    ) == {
        "prompt",
        "response",
    }

print()
print("Dataset audit passed.")
PY
```

## 14.4 No chat reformatting

Materialization does not convert examples into:

```text
messages
conversations
chat turns
ChatML
Llama chat format
Qwen chat format
```

The retained fields are plain:

```text
prompt
response
```

This follows the released SMART trainer's instruction format.

---

# 15. Trainer-required data format

Local trainer:

```text
/data/saral/wdir/smart/instruction_tuner_local.py
```

For every record, the trainer creates:

```python
full_text = prompt + " " + response
```

It then:

1. tokenizes the complete text;
2. tokenizes the prompt separately;
3. creates labels from the complete token sequence;
4. replaces prompt label positions with `-100`;
5. computes loss only over response tokens.

Conceptually:

```text
tokens:
[prompt tokens] [separator] [response tokens]

labels:
[-100 ... -100] [response token IDs]
```

Maximum sequence length:

```text
4096
```

This is supervised response-only causal-language-model loss.

---

# 16. Tokenization and supervision audit

## 16.1 Why an audit was necessary

A raw row can have a non-empty response but still contribute no supervised token after tokenization and truncation.

For example:

```text
prompt token length >= 4096
```

leaves no space for the response.

The resulting label sequence may contain only:

```text
-100
```

Such a row cannot produce a meaningful supervised validation loss.

## 16.2 Model-specific tokenization

The audit was run with both:

```text
/data/saral/wdir/smart/llama2_7b
/data/saral/wdir/smart/qwen2_7b
```

A row valid for one tokenizer is not automatically valid for the other.

## 16.3 Common eval-safe validation

The union of validation rows with zero supervised tokens under either tokenizer was removed.

Counts:

```text
Original validation:    183,870
Rows removed:               119
Eval-safe validation:   183,751
```

The training split was not changed.

Both SMART-25K and SMART-50K use the same eval-safe validation split.

## 16.4 Why this does not constitute benchmark cherry-picking

The removed rows have no valid supervised target under the actual trainer's tokenization and truncation rules.

They cannot provide a meaningful validation loss.

The filtering criterion does not use:

- model correctness;
- validation loss magnitude;
- benchmark labels;
- downstream benchmark performance;
- task difficulty.

It is purely a training-objective validity check.

---

# 17. Reduced training-time validation

## 17.1 Motivation

The full eval-safe validation split contains:

```text
183,751 examples
```

One representative Llama2 full-fine-tuning run spent approximately:

```text
Model loading:        ~7 minutes
Training:             22 minutes 29 seconds
Full validation:      44 minutes 21 seconds
Model saving:         ~5 minutes
```

The final progress bar included validation and checkpoint saving, which made the run appear much longer than the optimization phase.

## 17.2 Fixed 10K validation split

A deterministic 10,000-example subset was created for training-time validation.

Output root:

```text
/mnt/warm_storage/saral/smart/datasets/trainer_eval_safe_10k
```

Contents:

```text
trainer_eval_safe_10k/
├── smart_25000/
├── smart_50000/
└── reduced_validation_summary.json
```

Policy:

```text
Source validation:     183,751 eval-safe examples
Selected validation:    10,000 examples
Seed:                        23
Shared across budgets:      yes
Shared across models:       yes
Training split changed:     no
```

## 17.3 Creation script

```text
data_generation_scripts/build_reduced_validation_datasets.py
```

Run:

```bash
cd /data/saral/wdir/smart || exit 1

python3 \
  data_generation_scripts/build_reduced_validation_datasets.py
```

## 17.4 Why reduced validation does not change model weights

Validation is executed only after the last optimizer update.

It is not used for:

- early stopping;
- best-checkpoint selection;
- scheduler control;
- hyperparameter adaptation;
- gradient computation.

Therefore:

```text
same training records
same record order
same optimizer updates
same resulting weights
```

Only the reported validation-loss estimate and validation runtime change.

The complete 183,751-row validation split remains retained for optional final loss evaluation.

---

# 18. Fine-tuning configuration

## 18.1 Author-comparable full fine-tuning

```text
Epochs:                      1
Learning rate:               2e-5
Weight decay:                0.1
Scheduler:                   cosine
Warmup ratio:                0.01
Maximum sequence length:     4096
Precision:                   bf16
Attention:                   FlashAttention 2
Seed:                        23

Per-device train batch:      1
Number of GPUs:              4
Gradient accumulation:       16
Global training batch:       64
```

## 18.2 Four-GPU adaptation

The paper used eight A100-80GB GPUs.

The local environment uses four H200 GPUs.

Global batch is preserved:

\[
1
\times
4
\times
16
=
64.
\]

Preserving global batch is important because it preserves the number of optimizer updates per epoch.

## 18.3 Optimizer-step counts

For SMART-25K:

\[
\left\lceil
\frac{25,000}{64}
\right\rceil
=
391.
\]

For SMART-50K:

\[
\left\lceil
\frac{50,000}{64}
\right\rceil
=
782.
\]

Explicit values:

```text
SMART-25K: 391 optimizer steps
SMART-50K: 782 optimizer steps
```

## 18.4 Warmup steps

With a 1% warmup ratio:

```text
SMART-25K:
floor(0.01 × 391) = 3

SMART-50K:
floor(0.01 × 782) = 7
```

## 18.5 Accelerate configuration

```text
/data/saral/wdir/smart/accelerate_4gpu_bf16.yaml
```

It configures:

```text
distributed type: MULTI_GPU
processes:        4
mixed precision: bf16
GPU IDs:          0,1,2,3
```

---

# 19. Distributed scheduler correction

## 19.1 Original problem

The first production trainer constructed a cosine scheduler for:

```text
391 or 782 scheduler steps
```

However, Accelerate's scheduler wrapper advanced the underlying scheduler once per distributed process.

With four processes, the scheduler advanced approximately four times for every optimizer update.

## 19.2 Evidence from the log

A one-epoch cosine schedule should warm up and then decrease monotonically.

The pilot log instead showed:

```text
optimizer step 10:   lr ≈ 1.96e-5
optimizer step 100:  lr ≈ 2.65e-8
optimizer step 190:  lr ≈ 1.99e-5
optimizer step 290:  lr ≈ 1.61e-8
optimizer step 390:  lr ≈ 2.00e-5
```

The schedule repeatedly decayed and rose again.

That behavior contradicted the intended SMART training configuration.

## 19.3 Corrected construction

The corrected trainer uses:

```python
accelerator = Accelerator(
    gradient_accumulation_steps=(
        args.gradient_accumulation_steps
    ),
    step_scheduler_with_optimizer=False,
    **accelerator_log_kwargs,
)
```

The scheduler is stepped manually after a synchronized optimizer update:

```python
if accelerator.sync_gradients:
    if not accelerator.optimizer_step_was_skipped:
        lr_scheduler.step()

    progress_bar.update(1)
    completed_steps += 1
```

## 19.4 Final invariant

The corrected trainer verifies:

```text
scheduler last_epoch == completed optimizer steps
```

This ensures one scheduler tick per actual optimizer update.

## 19.5 Checkpoint consequence

Checkpoints produced before the scheduler fix are pilot artifacts.

They should not be used for final result reporting.

Corrected run names contain:

```text
scheduler_fixed
```

Backup of the pre-fix trainer:

```text
instruction_tuner_local.before_scheduler_fix.py
```

---

# 20. LoRA extension

LoRA is treated as an experimental extension rather than the primary author-comparable SMART result.

## 20.1 Configuration

```text
LORA_R:        64
LORA_ALPHA:    32
LORA_DROPOUT:  0.05
Bias:          none
```

## 20.2 Target modules

```text
q_proj
k_proj
v_proj
o_proj
gate_proj
up_proj
down_proj
```

Compact form:

```text
q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
```

## 20.3 Why `lm_head` was excluded

The corrected LoRA experiment excludes:

```text
lm_head
```

Reasons:

- the seven projection modules cover transformer attention and MLP projections;
- they exist in both model families;
- Qwen2's larger vocabulary makes an `lm_head` adapter substantially larger;
- excluding `lm_head` makes cross-model parameter-efficient comparisons cleaner.

## 20.4 Expected trainable parameters

```text
Llama2-7B LoRA: 159,907,840
Qwen2-7B LoRA:  161,480,704
```

## 20.5 Learning rate

LoRA uses:

```text
2e-5
```

This keeps the learning rate controlled across full and LoRA runs.

It should not be presented as a separately optimized LoRA learning rate.

---

# 21. Training smoke tests

## 21.1 Scripts

Pilot smoke launcher:

```text
run_training_smokes.sh
```

Corrected smoke launcher:

```text
run_training_smokes_v2.sh
```

## 21.2 Tested configurations

```text
Llama2 full
Llama2 LoRA
Qwen2 full
Qwen2 LoRA
```

## 21.3 Observed peak memory

Approximate peaks:

```text
Llama2 full:  65.9-67.6 GiB
Llama2 LoRA:  17.1 GiB
Qwen2 full:   81.6 GiB
Qwen2 LoRA:   27.8 GiB
```

## 21.4 Acceptance criteria

A smoke test must verify:

- four distributed processes initialize;
- NCCL communication works;
- bf16 is active;
- FlashAttention2 is active;
- forward and backward passes complete;
- optimizer updates complete;
- loss is finite;
- checkpoint saving works;
- full checkpoints contain full weights;
- LoRA checkpoints contain adapter files;
- LoRA trainable parameters match the intended targets;
- corrected LR behavior is monotonic after warmup.

## 21.5 Run

```bash
cd /data/saral/wdir/smart || exit 1

./run_training_smokes_v2.sh
```

---

# 22. Production fine-tuning

## 22.1 Pilot launchers

These belong to the original pre-correction workflow:

```text
run_one_production_training.sh
run_remaining_production_trainings.sh
```

They are retained for provenance.

Their outputs must not be mixed with corrected final results.

## 22.2 Corrected single-run launcher

```text
run_one_production_training_v2.sh
```

It uses:

```text
instruction_tuner_local.py
accelerate_4gpu_bf16.yaml
scheduler-fixed training
r64/a32/projection-seven LoRA
```

Example full run:

```bash
cd /data/saral/wdir/smart || exit 1

./run_one_production_training_v2.sh \
  llama2_7b \
  /data/saral/wdir/smart/llama2_7b \
  25000 \
  full
```

Example LoRA run:

```bash
./run_one_production_training_v2.sh \
  qwen2_7b \
  /data/saral/wdir/smart/qwen2_7b \
  50000 \
  lora
```

## 22.3 All-experiment launcher

```text
run_all_finetuning_v2.sh
```

The launcher runs all eight experiments sequentially because each child process uses all four GPUs.

Experiments:

```text
Llama2 25K full
Llama2 50K full
Qwen2 25K full
Qwen2 50K full

Llama2 25K LoRA
Llama2 50K LoRA
Qwen2 25K LoRA
Qwen2 50K LoRA
```

## 22.4 Launch with tmux

```bash
cd /data/saral/wdir/smart || exit 1

tmux new -s smart_finetuning_v2
```

Inside tmux:

```bash
./run_all_finetuning_v2.sh
```

Detach:

```text
Ctrl-b
d
```

Reattach:

```bash
tmux attach -t smart_finetuning_v2
```

## 22.5 Monitoring

```bash
watch -n 5 \
  'nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu,power.draw --format=csv'
```

```bash
tail -f \
  /mnt/warm_storage/saral/smart/logs/production_training_v2/batch_*/master.log
```

---

# 23. Checkpoint naming

## 23.1 Corrected full checkpoints

```text
llama2_7b_smart_25000_full_scheduler_fixed_seed23

llama2_7b_smart_50000_full_scheduler_fixed_seed23

qwen2_7b_smart_25000_full_scheduler_fixed_seed23

qwen2_7b_smart_50000_full_scheduler_fixed_seed23
```

## 23.2 Corrected LoRA checkpoints

```text
llama2_7b_smart_25000_lora_scheduler_fixed_r64_a32_proj7_seed23

llama2_7b_smart_50000_lora_scheduler_fixed_r64_a32_proj7_seed23

qwen2_7b_smart_25000_lora_scheduler_fixed_r64_a32_proj7_seed23

qwen2_7b_smart_50000_lora_scheduler_fixed_r64_a32_proj7_seed23
```

## 23.3 Output root

```text
/mnt/warm_storage/saral/smart/models
```

## 23.4 Expected full-checkpoint files

```text
config.json
generation_config.json
model*.safetensors
model.safetensors.index.json
tokenizer_config.json
special_tokens_map.json
all_results.json
```

## 23.5 Expected LoRA files

```text
adapter_config.json
adapter_model.safetensors
all_results.json
```

---

# 24. Training-result verification

Every completed run should contain finite metrics and the expected optimizer-step count.

Expected steps:

```text
SMART-25K: 391
SMART-50K: 782
```

Expected `all_results.json` fields:

```text
completed_steps
max_train_steps
eval_loss
perplexity
```

A completed result must satisfy:

```text
completed_steps == expected steps
max_train_steps == expected steps
eval_loss is finite
perplexity is finite
```

If the local trainer does not emit `max_train_steps`, either:

1. add it when writing `all_results.json`; or
2. change the verifier to rely on `completed_steps`.

Do not treat the existence of a directory alone as proof of completion.

## 24.1 LoRA config verification

```bash
python3 - <<'PY'
import json
from pathlib import Path

root = Path(
    "/mnt/warm_storage/saral/smart/models"
)

expected_targets = {
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
}

for path in sorted(
    root.glob(
        "*scheduler_fixed_r64_a32_proj7*/"
        "adapter_config.json"
    )
):
    config = json.loads(
        path.read_text(encoding="utf-8")
    )

    assert int(config["r"]) == 64
    assert int(config["lora_alpha"]) == 32
    assert float(config["lora_dropout"]) == 0.05
    assert set(config["target_modules"]) == expected_targets

    print("OK:", path.parent.name)
PY
```

---

# 25. Leaderboard evaluation

Evaluation repository:

```text
https://github.com/EleutherAI/lm-evaluation-harness.git
```

## 25.1 Evaluation tasks

```text
leaderboard_mmlu_pro
leaderboard_bbh
leaderboard_musr
leaderboard_math_hard
leaderboard_ifeval
leaderboard_gpqa
```

The leaderboard task definitions contain their intended few-shot settings.

The commands do not override `--num_fewshot`.

## 25.2 Formatting decision

Do not use:

```text
--apply_chat_template
--fewshot_as_multiturn
```

The models were trained using plain:

```text
prompt + " " + response
```

Applying a chat template during evaluation would introduce a train-evaluation formatting mismatch.

## 25.3 Full checkpoint arguments

For a full fine-tuned checkpoint:

```text
pretrained=<saved model>
tokenizer=<saved model>
dtype=bfloat16
attn_implementation=flash_attention_2
```

Example:

```bash
MODEL=/mnt/warm_storage/saral/smart/models/qwen2_7b_smart_25000_full_scheduler_fixed_seed23

OUT=/mnt/warm_storage/saral/smart/evaluations/lm_eval_leaderboard_corrected/$(basename "$MODEL")

mkdir -p "$OUT"

lm_eval \
  --model hf \
  --model_args "pretrained=$MODEL,tokenizer=$MODEL,dtype=bfloat16,attn_implementation=flash_attention_2" \
  --tasks leaderboard_mmlu_pro \
  --device cuda:0 \
  --batch_size 8 \
  --seed 23 \
  --output_path "$OUT/mmlu_pro.json" \
  > "$OUT/mmlu_pro.log" 2>&1
```

## 25.4 LoRA arguments

For LoRA:

```text
pretrained=<original base model>
peft=<saved adapter>
tokenizer=<original base model>
dtype=bfloat16
attn_implementation=flash_attention_2
```

Example:

```bash
BASE=/data/saral/wdir/smart/qwen2_7b

ADAPTER=/mnt/warm_storage/saral/smart/models/qwen2_7b_smart_25000_lora_scheduler_fixed_r64_a32_proj7_seed23

OUT=/mnt/warm_storage/saral/smart/evaluations/lm_eval_leaderboard_corrected/$(basename "$ADAPTER")

mkdir -p "$OUT"

lm_eval \
  --model hf \
  --model_args "pretrained=$BASE,peft=$ADAPTER,tokenizer=$BASE,dtype=bfloat16,attn_implementation=flash_attention_2" \
  --tasks leaderboard_math_hard \
  --device cuda:0 \
  --batch_size 8 \
  --seed 23 \
  --output_path "$OUT/math_hard.json" \
  > "$OUT/math_hard.log" 2>&1
```

Depending on the installed harness version, the executable may be:

```text
lm_eval
```

or:

```text
lm-eval run
```

---

# 26. Four-GPU evaluation scheduler

Two evaluation launchers were prepared.

## 26.1 Dynamic checkpoint-level launcher

```text
run_all_leaderboard_evals.sh
```

This launcher treats each checkpoint as a job and evaluates all six task groups inside that job.

It dynamically starts the next checkpoint when a GPU becomes free.

## 26.2 Fixed-wave launcher

```text
run_all_lm_eval_wait4.sh
```

This is the explicit `wait`-barrier implementation.

Total evaluations:

\[
8\text{ checkpoints}
\times
6\text{ tasks}
=
48\text{ jobs}.
\]

Execution pattern:

```text
launch four jobs on GPUs 0,1,2,3
wait for all four
validate all four result files
launch the next four
wait
...
```

## 26.3 Launch

```bash
cd /data/saral/wdir/smart || exit 1

tmux new -s smart_lm_eval
```

Inside tmux:

```bash
./run_all_lm_eval_wait4.sh
```

## 26.4 Output root

```text
/mnt/warm_storage/saral/smart/evaluations/lm_eval_leaderboard_corrected
```

## 26.5 Per-checkpoint layout

```text
<checkpoint-name>/
├── mmlu_pro.json
├── mmlu_pro.log
├── mmlu_pro.command.sh
├── bbh.json
├── bbh.log
├── bbh.command.sh
├── musr.json
├── musr.log
├── musr.command.sh
├── math_hard.json
├── math_hard.log
├── math_hard.command.sh
├── ifeval.json
├── ifeval.log
├── ifeval.command.sh
├── gpqa.json
├── gpqa.log
└── gpqa.command.sh
```

## 26.6 Scheduler behavior

The script:

- validates all checkpoint paths before starting;
- verifies four visible GPUs;
- assigns one process per GPU;
- redirects each job to a dedicated log;
- saves one JSON result per task;
- saves model-response caches;
- skips already valid outputs;
- preserves invalid previous JSON files;
- waits for all four jobs in each wave;
- stops before the next wave if any job fails;
- writes a scheduler status TSV;
- writes a final results manifest.

---

# 27. Script inventory

## 27.1 Author-release files

| File | Purpose |
|---|---|
| `data_generation_scripts/get_SMART_mixture.py` | Author mixture-construction entry point. |
| `data_generation_scripts/get_SMART_mixture.sh` | Shell wrapper for the author mixture pipeline. |
| `data_generation_scripts/utils/gc_ordering.py` | Graph Cut ordering implementation. |
| `data_generation_scripts/utils/fl_ordering.py` | Facility Location ordering implementation. |
| `data_generation_scripts/utils/logdet_ordering.py` | Log Determinant ordering implementation. |
| `data_generation_scripts/utils/random_ordering.py` | Random-ordering baseline. |
| `instruction_tuner.py` | Released instruction-tuning implementation. |
| `instruction_tuner.sh` | Released trainer shell wrapper. |
| `config.yaml` | Released/default experiment configuration. |

## 27.2 Local data scripts

| File | Purpose |
|---|---|
| `local_dataset.py` | Local dataset-loading utilities. |
| `data_generation_scripts/run_stage2_selection.py` | Production exact/candidate-restricted Facility Location. |
| `data_generation_scripts/build_reduced_validation_datasets.py` | Creates fixed 10K validation copies. |

The following stages were also implemented during the replication:

```text
source-task audit
GTE-large embedding generation
task-embedding generation
Stage 1 Graph Cut
allocation reconciliation
dataset materialization
tokenized-supervision audit
eval-safe validation construction
```

Some were executed as inline Python heredocs rather than retained with stable filenames.

Before release, move the exact executed code into committed scripts.

Recommended packaging names are:

```text
data_generation_scripts/audit_local_tasks.py
data_generation_scripts/embed_prompts_gte_large.py
data_generation_scripts/run_stage1_graph_cut.py
data_generation_scripts/materialize_smart_datasets.py
data_generation_scripts/audit_tokenized_supervision.py
data_generation_scripts/build_eval_safe_datasets.py
```

These are recommended final names, not a claim that these exact names were used during the first execution.

## 27.3 Training scripts

| File | Status | Purpose |
|---|---|---|
| `instruction_tuner_local.py` | Current | Local trainer with scheduler correction. |
| `instruction_tuner_local.before_scheduler_fix.py` | Provenance | Pre-correction trainer backup. |
| `accelerate_4gpu_bf16.yaml` | Current | Four-GPU bf16 Accelerate configuration. |
| `run_training_smokes.sh` | Pilot | Original smoke-test launcher. |
| `run_training_smokes_v2.sh` | Current | Corrected full/LoRA smoke tests. |
| `run_one_production_training.sh` | Pilot | Original production launcher. |
| `run_one_production_training_v2.sh` | Current | Corrected single-run launcher. |
| `run_remaining_production_trainings.sh` | Pilot | Original remaining-run batch launcher. |
| `run_all_finetuning_v2.sh` | Current | Runs all eight corrected experiments. |

## 27.4 Evaluation scripts

| File | Purpose |
|---|---|
| `run_all_leaderboard_evals.sh` | Dynamic one-checkpoint-per-GPU evaluator. |
| `run_all_lm_eval_wait4.sh` | Four-job wave evaluator using `wait`. |

---

# 28. Reproducibility invariants

A correct clean rerun should satisfy all of the following.

## 28.1 Data invariants

```text
Task count:                 309
Valid source train rows:    6,266,471
Original validation rows:     183,870
Eval-safe validation rows:     183,751
SMART-25K train rows:           25,000
SMART-50K train rows:           50,000
```

## 28.2 Selection invariants

```text
Prompt embedding dimension:  1024
Stage 1 objective:            Graph Cut
Graph Cut lambda:             0.4
Stage 2 objective:            Facility Location
Stage 2 exact threshold:      5,000
Large-task candidates:        2,048
Seed:                         23
SMART-25K nested in 50K:      yes
```

## 28.3 Training invariants

```text
Epochs:                       1
Global batch:                 64
SMART-25K optimizer steps:    391
SMART-50K optimizer steps:    782
Learning rate:                2e-5
Weight decay:                 0.1
Warmup ratio:                 0.01
Maximum sequence length:      4096
Precision:                    bf16
```

## 28.4 LoRA invariants

```text
r:                            64
alpha:                        32
dropout:                      0.05
bias:                         none

targets:
q_proj
k_proj
v_proj
o_proj
gate_proj
up_proj
down_proj
```

## 28.5 Evaluation invariants

```text
Task groups:                  6
Chat template:                disabled
Processes per GPU:            1
Results per task:             one JSON
Logs per task:                one log
```

---

# 29. Known deviations

## 29.1 Ground-set difference

The paper used the complete FLAN 2022 collection.

The local reconstruction contains 309 tasks from five locally available collections.

This can change:

- task similarities;
- task ranking;
- Graph Cut gains;
- allocation weights;
- selected examples;
- downstream benchmark scores.

The work reproduces the SMART procedure, not the exact full data universe.

## 29.2 Candidate-restricted large-task selection

Tasks above 5,000 rows do not use exact all-to-all selectable Facility Location.

The represented set remains complete, but selectable facilities are restricted to 2,048 candidates.

This is the main algorithmic approximation.

## 29.3 Qwen2

Qwen2 was not one of the three primary models listed in the original SMART experiment section.

Its results are a model-family extension.

## 29.4 LoRA

The primary author-comparable setup is full fine-tuning.

LoRA is a controlled parameter-efficient extension.

## 29.5 Reduced train-time validation

The 10K validation subset gives a less precise estimate of full-corpus validation loss.

It does not change final model weights.

## 29.6 Pilot checkpoints

Checkpoints without `scheduler_fixed` used the incorrect oscillating distributed LR schedule.

They must not be included in final result tables.

---

# 30. Frequently asked questions

## Did the pipeline subset QQP before embedding?

No.

Every valid QQP training prompt was embedded.

## Did Facility Location represent only 2,048 QQP examples?

No.

All 363,846 valid QQP examples remained in the represented set.

Only 2,048 examples were eligible to be selected as facilities.

## Was QQP itself over 400 GB?

No.

The raw local QQP file was approximately 109 MiB.

The approximately 493 GiB estimate referred to its dense float32 pairwise similarity matrix.

## Why not build the 493 GiB matrix?

It would make the direct dense implementation impractical and would require far more memory than the available process budget.

## Is the candidate pool a subset?

It is a subset of selectable facilities.

It is not a subset of the examples represented by the Facility Location objective.

## Can candidate restriction change the final examples?

Yes.

The candidate-restricted result need not be identical to exact dense selection.

## Does candidate restriction guarantee lower benchmark scores?

No.

It creates approximation risk, but benchmark impact is empirical.

The validation showed approximately 99.7-99.9% objective retention on the tested tasks.

That supports the approximation but does not prove equal downstream accuracy.

## Is selecting 25K or 50K itself a compromise?

No.

The purpose of SMART is to construct a small representative coreset.

The additional approximation is restricting selectable facilities for very large tasks.

## Was the training set reduced when validation was reduced?

No.

Only the post-training validation split was changed.

## Does 10K validation affect model weights?

No.

Validation occurs after all optimizer updates and is not used for early stopping or model selection.

## Why is no chat template used in lm-eval?

Training used plain prompt-response concatenation.

Adding a chat template during evaluation would create a formatting mismatch.

## Why are there V2 scripts?

V2 scripts contain the scheduler correction and corrected LoRA configuration.

## Which checkpoints should be evaluated?

Only checkpoints whose names contain:

```text
scheduler_fixed
```

---

# 31. Clean rerun checklist

1. Verify the five source-data directories.
2. Audit all 309 task files.
3. Verify the 6,266,471 valid training-row count.
4. Verify the 183,870 original validation count.
5. Generate GTE-large embeddings for every valid training prompt.
6. Save source-index maps alongside embeddings.
7. Compute mean task embeddings.
8. Run exact Graph Cut with `lambda=0.4`.
9. Apply Taylor-softmax to Stage 1 gains.
10. Reconcile exact 25K and 50K integer allocations.
11. Verify allocation sums.
12. Validate candidate sizes against exact reference tasks.
13. Freeze candidate size 2,048 and threshold 5,000.
14. Run Stage 2 over all 309 tasks.
15. Verify cumulative gains against manual Facility Location coverage.
16. Verify selected source-index uniqueness.
17. Materialize SMART-25K.
18. Materialize SMART-50K.
19. Verify trainer columns are `prompt` and `response`.
20. Run Llama2 tokenization-supervision audit.
21. Run Qwen2 tokenization-supervision audit.
22. Remove the union of zero-supervision validation rows.
23. Verify the common validation count is 183,751.
24. Optionally create fixed 10K train-time validation copies.
25. Run corrected Llama2 full smoke.
26. Run corrected Llama2 LoRA smoke.
27. Run corrected Qwen2 full smoke.
28. Run corrected Qwen2 LoRA smoke.
29. Verify LR monotonicity after warmup.
30. Run all eight corrected fine-tuning experiments.
31. Verify optimizer-step counts and saved files.
32. Run all 48 leaderboard jobs.
33. Aggregate only corrected checkpoint results.
34. Archive logs, manifests, hashes, and software versions.

---

# 32. Reporting recommendations

Final results should be separated into three categories.

## 32.1 Author-comparable method track

```text
Llama2-7B SMART-25K full fine-tuning
Llama2-7B SMART-50K full fine-tuning
```

## 32.2 Model-family extension

```text
Qwen2-7B SMART-25K full fine-tuning
Qwen2-7B SMART-50K full fine-tuning
```

## 32.3 Parameter-efficient extension

```text
Llama2-7B SMART-25K LoRA
Llama2-7B SMART-50K LoRA
Qwen2-7B SMART-25K LoRA
Qwen2-7B SMART-50K LoRA
```

Every result table should disclose:

```text
Local 309-task ground set
Graph Cut lambda 0.4
Facility Location Stage 2
Candidate-restricted selection for tasks above 5,000 rows
Candidate pool 2,048
Seed 23
```

This prevents readers from interpreting the results as an exact reproduction of the complete 1,840-task experiment.

---

# 33. Provenance to retain

Archive the following files with the final experiment:

```text
SMART source commit
task-file inventory
source-data audit
task_allocations.csv
embedding metadata
embedding hashes
source_indices.npy
stage2_selection_catalog.json
stage2_selection_summary.json
selected_source_indices.npy
candidate_full_local_indices.npy
selection.csv
dataset fingerprints
tokenizer-audit report
eval-safe filtering report
reduced_validation_summary.json
pip freeze
nvidia-smi report
Accelerate config
trainer source hash
training command files
training logs
all_results.json
adapter_config.json
lm-eval commit
lm-eval command files
lm-eval result JSON files
lm-eval logs
scheduler status TSV
results_manifest.tsv
```

The selected source-index arrays are especially important because they permit reconstruction of the exact 25K and 50K mixtures without rerunning submodular optimization.

---

# 34. References

## SMART

Relevant sections:

```text
Section 3.1: two-stage submodular selection
Section 3.2: prompt and task embeddings
Section 3.3: submodular-function selection
Section 4.1: fine-tuning data
Section 4.2: fine-tuning procedure
Section 4.8: objective-function comparisons
```

Released SMART source snapshot used during reconstruction:

```text
e6fe8080b2c01980f80a0ef62c71219fa8e55b93
```

## LM Evaluation Harness

```text
https://github.com/EleutherAI/lm-evaluation-harness.git
```

---

# Summary

The local replication preserves the central SMART pipeline:

```text
raw prompt-response task files
        |
        v
source audit and stable source indices
        |
        v
GTE-large prompt embeddings
        |
        v
mean task embeddings
        |
        v
Graph Cut task weighting
        |
        v
Taylor-softmax task allocations
        |
        v
Facility Location instance selection
        |
        v
SMART-25K and SMART-50K DatasetDicts
        |
        v
response-only supervised fine-tuning
        |
        v
leaderboard evaluation
```

The important large-task qualification is:

```text
QQP raw text was not 493 GiB
the dense square kernel was approximately 493 GiB

QQP was not reduced to 2,048 represented rows
all 363,846 valid rows remained represented

2,048 rows were eligible to become selected facilities
the resulting cross-kernel was approximately 2.78 GiB

the approach is a validated approximation
it is not exact dense large-task Facility Location
```

This distinction allows the replication to be described accurately:

- the SMART methodology is preserved;
- exact author behavior is used where computationally practical;
- the large-task approximation is disclosed and validated;
- infrastructure adaptations are separated from algorithm changes;
- Qwen2 and LoRA are reported as extensions;
- pilot scheduler-bug checkpoints are excluded from final evaluation.
