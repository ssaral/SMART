One of the diagnostic confirms the failure was a **true mathematical tie**, not an approximation error:

```text
candidate 293 gain = 0.084697898913
candidate 434 gain = 0.084697898913
gap                = 0
```

Submodlib selected candidate 434; our deterministic heap selected candidate 293. Both are valid greedy maximizers at rank 86.

## Important distinction

Our planned blockwise implementation is **not an approximation of Facility Location**:

* complete task pool is used;
* cosine similarities are computed from the original embeddings;
* the same Facility Location objective is used;
* exact marginal gains are computed;
* LazyGreedy is used;
* no sampling, ANN graph, clustering, or candidate truncation is introduced.

The only possible difference is which example is selected when multiple candidates have identical or numerically indistinguishable gains.

The authors’ mixture script takes prefixes of saved instance orderings, so different tie choices can change the final examples even when the Facility Location objective is unchanged. 

## What we can and cannot guarantee

We can guarantee:

* no objective approximation;
* no candidate-pool reduction;
* deterministic output;
* exact gain computation under a frozen numeric policy;
* selected candidates are true maximizers or objective-equivalent ties;
* final Facility Location coverage remains equivalent within a strict tolerance.

We cannot guarantee that two objective-equivalent example sets produce absolutely identical model-training results. They contain different examples, so a small downstream variation is possible. We must measure that rather than assume it away.

# Frozen Stage 2 policy

## Engine selection

For pools where the dense matrix is safely feasible:

```text
Authors’ dense Submodlib Facility Location
```

For pools where dense storage is infeasible:

```text
Full-pool exact blockwise Facility Location
```

No sparse or approximate method will be used in the primary baseline.

## Deterministic tie rule

For blockwise ordering:

1. Compare marginal gains using chunked CPU float64 arbitration when the leading candidates are close.
2. Treat candidates as tied only when their common-reference gains differ by no more than:

[
10^{-10}\max(1,|g_{\max}|)
]

3. Among tied candidates, choose the smallest original author-format dataset index.
4. Record the entire tie set and its gains.

This gives a stable result across GPUs, heap ordering, and chunk sizes.

The tie rule may differ from Submodlib’s undocumented internal queue behavior, but it does not change the greedy objective value.

# Verification contract

Before generating all 309 orderings, the implementation must pass four levels of verification.

## 1. Per-step correctness

For every selected candidate:

* independently recompute its marginal gain using chunked CPU float64;
* verify the saved gain;
* verify that the updated coverage objective increases by exactly that amount;
* verify there are no duplicate selected indices.

## 2. Tie-aware dense parity

On dense-feasible audit pools, classify every divergence as:

| Classification  | Meaning                                       |
| --------------- | --------------------------------------------- |
| `index_match`   | Same candidate selected                       |
| `exact_tie`     | Different candidate, identical float64 gain   |
| `numerical_tie` | Difference is within the frozen tie tolerance |
| `true_mismatch` | Blockwise candidate has materially lower gain |

Acceptance requires:

```text
true_mismatch count = 0
```

The `sglue::copa` rank-86 result is an `exact_tie`.

## 3. Prefix-objective parity

At useful prefixes such as:

```text
1, 10, 25, 50, 100, required_budget
```

compare:

* Facility Location objective;
* mean maximum cosine coverage;
* minimum coverage;
* selected-set overlap;
* number of exact or numerical ties.

The primary acceptance criterion is objective agreement, not selected-index overlap.

For dense audit pools:

```text
relative objective difference <= 1e-7
absolute mean-coverage difference <= 1e-7
```

## 4. Reproducibility

Repeat blockwise selection using:

* CUDA device 0 and CUDA device 1;
* at least two embedding chunk sizes;
* two independent program executions.

After float64 tie arbitration, all runs must produce the same source-index sequence and gains.

# Audit suite

We should not validate only `sglue::copa`. Use at least:

```text
Tiny:
  sglue::copa
  t0::wiki_qa_Direct_Answer_to_Question

Small:
  cot::stream_qed_ii
  flan2021::ai2_arc_ARC_Challenge_1.0.0
  sglue::rte

Medium:
  t0::wiki_qa_Decide_good_answer
  flan2021::glue_mrpc_2.0.0
  t0::social_i_qa_Check_if_a_random_answer_is_valid_or_not
```

The dense oracle can use either the required prefix or the complete (n-1) ordering where practical.

# Downstream sensitivity check

Even exact ties can substitute one example for another. To determine whether that affects model results, generate one alternate tie-policy dataset:

```text
Primary tie policy:   smallest dataset index
Sensitivity policy:   largest dataset index
```

Only exact/numerical ties differ; all task budgets and non-tied selections remain identical.

A single lower-cost sensitivity experiment is sufficient initially:

```text
Llama2-7B LoRA
25K mixture
same training seed and settings
primary versus alternate tie policy
```

Compare:

* validation loss;
* task-level metrics;
* external benchmark scores;
* difference relative to ordinary training-seed variation.

This will tell us whether tie choices materially affect the empirical conclusion. We should not claim zero impact before running that check.

## Baseline description

The final primary baseline can be described as:

> SMART with the authors’ task-level Graph Cut and allocation procedure, dense Submodlib Facility Location where feasible, and a full-pool matrix-free implementation of the identical Facility Location objective for pools where dense similarity storage is infeasible. Objective-equivalent ties are resolved with a deterministic, high-precision rule.

The next implementation step is to replace exact-index parity with this tie-aware verification harness and run it across the audit suite before processing any large pool.
