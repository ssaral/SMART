Refer to this doc only if you have already installed lm-evaluation library.
Use the current `lm-eval run` CLI. The six leaderboard task configurations already contain their intended shot counts, so do **not** add `--num_fewshot`: MMLU-Pro is 5-shot, BBH 3-shot, MATH-Hard 4-shot, and MuSR, IFEval, and GPQA are 0-shot. ([GitHub][1])

`--output_path` may point directly to a JSON file, while `--use_cache` provides resumable model-response caching. ([GitHub][2])

## 1. Select the saved model

### Full fine-tuned checkpoint

Change only the `MODEL` path:

```bash
set -o pipefail

MODEL=/mnt/warm_storage/saral/smart/models/llama2_7b_smart_25000_full_scheduler_fixed_seed23

RUN_NAME=$(basename "$MODEL")

OUT=/mnt/warm_storage/saral/smart/evaluations/lm_eval_leaderboard/$RUN_NAME
CACHE=/mnt/warm_storage/saral/smart/cache/lm_eval/$RUN_NAME

MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,attn_implementation=flash_attention_2"

mkdir -p \
  "$OUT"/{mmlu_pro,bbh,musr,math_hard,ifeval,gpqa} \
  "$CACHE"
```

### LoRA checkpoint

For a LoRA run, use the original base model in `pretrained=` and the saved adapter in `peft=`. The Hugging Face backend supports both local model paths and PEFT adapter paths. ([GitHub][3])

Example for Llama2:

```bash
set -o pipefail

BASE=/data/saral/wdir/smart/llama2_7b

ADAPTER=/mnt/warm_storage/saral/smart/models/llama2_7b_smart_25000_lora_scheduler_fixed_r64_a32_proj7_seed23

RUN_NAME=$(basename "$ADAPTER")

OUT=/mnt/warm_storage/saral/smart/evaluations/lm_eval_leaderboard/$RUN_NAME
CACHE=/mnt/warm_storage/saral/smart/cache/lm_eval/$RUN_NAME

MODEL_ARGS="pretrained=$BASE,peft=$ADAPTER,tokenizer=$BASE,dtype=bfloat16,attn_implementation=flash_attention_2"

mkdir -p \
  "$OUT"/{mmlu_pro,bbh,musr,math_hard,ifeval,gpqa} \
  "$CACHE"
```

For a Qwen2 LoRA adapter, change both `BASE` and `ADAPTER`:

```bash
BASE=/data/saral/wdir/smart/qwen2_7b
```

The same six commands below then work unchanged.

## 2. Validate the six task configurations

Run this once before the actual evaluation:

```bash
lm-eval validate \
  --tasks leaderboard_mmlu_pro,leaderboard_bbh,leaderboard_musr,leaderboard_math_hard,leaderboard_ifeval,leaderboard_gpqa
```

The current CLI provides `validate` specifically to check task existence, configuration, datasets, metrics, and templates. ([GitHub][2])

## 3. MMLU-Pro

```bash
CUDA_VISIBLE_DEVICES=0 \
lm-eval run \
  --model hf \
  --model_args "$MODEL_ARGS" \
  --tasks leaderboard_mmlu_pro \
  --device cuda:0 \
  --batch_size auto:4 \
  --max_batch_size 32 \
  --seed 23 \
  --cache_requests true \
  --use_cache "$CACHE/mmlu_pro_" \
  --output_path "$OUT/mmlu_pro/results.json" \
  --show_config \
  2>&1 | tee "$OUT/mmlu_pro/run.log"
```

Result:

```text
$OUT/mmlu_pro/results.json
```

## 4. BBH

```bash
CUDA_VISIBLE_DEVICES=0 \
lm-eval run \
  --model hf \
  --model_args "$MODEL_ARGS" \
  --tasks leaderboard_bbh \
  --device cuda:0 \
  --batch_size auto:4 \
  --max_batch_size 32 \
  --seed 23 \
  --cache_requests true \
  --use_cache "$CACHE/bbh_" \
  --output_path "$OUT/bbh/results.json" \
  --show_config \
  2>&1 | tee "$OUT/bbh/run.log"
```

Result:

```text
$OUT/bbh/results.json
```

## 5. MuSR

```bash
CUDA_VISIBLE_DEVICES=0 \
lm-eval run \
  --model hf \
  --model_args "$MODEL_ARGS" \
  --tasks leaderboard_musr \
  --device cuda:0 \
  --batch_size auto:4 \
  --max_batch_size 32 \
  --seed 23 \
  --cache_requests true \
  --use_cache "$CACHE/musr_" \
  --output_path "$OUT/musr/results.json" \
  --show_config \
  2>&1 | tee "$OUT/musr/run.log"
```

Result:

```text
$OUT/musr/results.json
```

## 6. MATH-Hard

The leaderboard MATH-Hard group requires the harness’s math dependencies because its answer-equivalence logic uses SymPy. ([GitHub][1])

```bash
CUDA_VISIBLE_DEVICES=0 \
lm-eval run \
  --model hf \
  --model_args "$MODEL_ARGS" \
  --tasks leaderboard_math_hard \
  --device cuda:0 \
  --batch_size auto:4 \
  --max_batch_size 16 \
  --seed 23 \
  --cache_requests true \
  --use_cache "$CACHE/math_hard_" \
  --output_path "$OUT/math_hard/results.json" \
  --show_config \
  2>&1 | tee "$OUT/math_hard/run.log"
```

Result:

```text
$OUT/math_hard/results.json
```

I used a lower maximum batch size for MATH because it is generative and can produce much longer sequences.

## 7. IFEval

```bash
CUDA_VISIBLE_DEVICES=0 \
lm-eval run \
  --model hf \
  --model_args "$MODEL_ARGS" \
  --tasks leaderboard_ifeval \
  --device cuda:0 \
  --batch_size auto:4 \
  --max_batch_size 16 \
  --seed 23 \
  --cache_requests true \
  --use_cache "$CACHE/ifeval_" \
  --output_path "$OUT/ifeval/results.json" \
  --show_config \
  2>&1 | tee "$OUT/ifeval/run.log"
```

Result:

```text
$OUT/ifeval/results.json
```

## 8. GPQA

```bash
CUDA_VISIBLE_DEVICES=0 \
lm-eval run \
  --model hf \
  --model_args "$MODEL_ARGS" \
  --tasks leaderboard_gpqa \
  --device cuda:0 \
  --batch_size auto:4 \
  --max_batch_size 32 \
  --seed 23 \
  --cache_requests true \
  --use_cache "$CACHE/gpqa_" \
  --output_path "$OUT/gpqa/results.json" \
  --show_config \
  2>&1 | tee "$OUT/gpqa/run.log"
```

Result:

```text
$OUT/gpqa/results.json
```

## Result layout

After all six commands:

```text
/mnt/warm_storage/saral/smart/evaluations/lm_eval_leaderboard/
└── <checkpoint-name>/
    ├── mmlu_pro/
    │   ├── results.json
    │   └── run.log
    ├── bbh/
    │   ├── results.json
    │   └── run.log
    ├── musr/
    │   ├── results.json
    │   └── run.log
    ├── math_hard/
    │   ├── results.json
    │   └── run.log
    ├── ifeval/
    │   ├── results.json
    │   └── run.log
    └── gpqa/
        ├── results.json
        └── run.log
```

## Important formatting decision

Do not add:

```bash
--apply_chat_template
--fewshot_as_multiturn
```

Your SMART models were trained using plain:

```text
prompt + " " + response
```

rather than a chat-message template. Omitting chat-template flags keeps evaluation formatting consistent with training.

## Optional per-example outputs

The commands above save aggregate results only. To additionally save prompts and individual model outputs, append:

```bash
--log_samples
```

The harness supports sample logging when an output path is provided, but it can consume substantial additional storage. ([GitHub][2])

[1]: https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/leaderboard/README.md?utm_source=chatgpt.com "lm-evaluation-harness/lm_eval/tasks/leaderboard/README.md at main · EleutherAI/lm-evaluation-harness · GitHub"
[2]: https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/interface.md?utm_source=chatgpt.com "lm-evaluation-harness/docs/interface.md at main · EleutherAI/lm-evaluation-harness · GitHub"
[3]: https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/models/huggingface.py?utm_source=chatgpt.com "lm-evaluation-harness/lm_eval/models/huggingface.py at main · EleutherAI/lm-evaluation-harness · GitHub"
