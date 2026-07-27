This script evaluates all **8 corrected checkpoints × 6 leaderboard tasks = 48 evaluations**. It launches exactly four evaluations—one per GPU—calls `wait`, verifies all four outputs, and only then launches the next wave.

The harness supports local `pretrained=` paths, PEFT adapters through `peft=PATH`, explicit CUDA devices, integer batch sizes, and JSON-file output paths. Current `lm-eval` also accepts the legacy single-command syntax used below. ([GitHub][1])

```bash
cd /data/saral/wdir/smart || exit 1

cat > run_all_lm_eval_wait4.sh <<'BASH'
#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT=/data/saral/wdir/smart
ROOT=/mnt/warm_storage/saral/smart

MODEL_ROOT="$ROOT/models"

EVAL_ROOT="$ROOT/evaluations/lm_eval_leaderboard_corrected"
CACHE_ROOT="$ROOT/cache/lm_eval_leaderboard_corrected"

BATCH_SIZE=${BATCH_SIZE:-8}
SEED=${SEED:-23}
NUM_GPUS=4

DTYPE=${DTYPE:-bfloat16}
ATTENTION=${ATTENTION:-flash_attention_2}

LLAMA_BASE="$PROJECT/llama2_7b"
QWEN_BASE="$PROJECT/qwen2_7b"

RUN_TIMESTAMP=$(date -u +%Y%m%dT%H%M%SZ)

SCHEDULER_DIR="$EVAL_ROOT/scheduler"
MASTER_LOG="$SCHEDULER_DIR/scheduler_${RUN_TIMESTAMP}.log"
STATUS_FILE="$SCHEDULER_DIR/status_${RUN_TIMESTAMP}.tsv"

mkdir -p \
    "$EVAL_ROOT" \
    "$CACHE_ROOT" \
    "$SCHEDULER_DIR"

exec > >(tee -a "$MASTER_LOG") 2>&1


# ------------------------------------------------------------
# Find the lm-eval executable
# ------------------------------------------------------------

if command -v lm_eval >/dev/null 2>&1; then
    LM_EVAL=(lm_eval)
elif command -v lm-eval >/dev/null 2>&1; then
    # The current executable supports the legacy single-command form.
    LM_EVAL=(lm-eval)
else
    echo "Neither lm_eval nor lm-eval is available in PATH." >&2
    exit 1
fi

echo "lm-eval executable: ${LM_EVAL[*]}"


# ------------------------------------------------------------
# Corrected checkpoint definitions
#
# Format:
#   name|kind|checkpoint_or_adapter|base_model
#
# For full checkpoints, base_model is unused.
# ------------------------------------------------------------

CHECKPOINTS=(
    "llama2_7b_smart_25000_full_scheduler_fixed_seed23|full|$MODEL_ROOT/llama2_7b_smart_25000_full_scheduler_fixed_seed23|none"
    "llama2_7b_smart_50000_full_scheduler_fixed_seed23|full|$MODEL_ROOT/llama2_7b_smart_50000_full_scheduler_fixed_seed23|none"

    "qwen2_7b_smart_25000_full_scheduler_fixed_seed23|full|$MODEL_ROOT/qwen2_7b_smart_25000_full_scheduler_fixed_seed23|none"
    "qwen2_7b_smart_50000_full_scheduler_fixed_seed23|full|$MODEL_ROOT/qwen2_7b_smart_50000_full_scheduler_fixed_seed23|none"

    "llama2_7b_smart_25000_lora_scheduler_fixed_r64_a32_proj7_seed23|lora|$MODEL_ROOT/llama2_7b_smart_25000_lora_scheduler_fixed_r64_a32_proj7_seed23|$LLAMA_BASE"
    "llama2_7b_smart_50000_lora_scheduler_fixed_r64_a32_proj7_seed23|lora|$MODEL_ROOT/llama2_7b_smart_50000_lora_scheduler_fixed_r64_a32_proj7_seed23|$LLAMA_BASE"

    "qwen2_7b_smart_25000_lora_scheduler_fixed_r64_a32_proj7_seed23|lora|$MODEL_ROOT/qwen2_7b_smart_25000_lora_scheduler_fixed_r64_a32_proj7_seed23|$QWEN_BASE"
    "qwen2_7b_smart_50000_lora_scheduler_fixed_r64_a32_proj7_seed23|lora|$MODEL_ROOT/qwen2_7b_smart_50000_lora_scheduler_fixed_r64_a32_proj7_seed23|$QWEN_BASE"
)


# ------------------------------------------------------------
# Leaderboard tasks
#
# Format:
#   harness_task_name|output_slug
# ------------------------------------------------------------

TASKS=(
    "leaderboard_mmlu_pro|mmlu_pro"
    "leaderboard_bbh|bbh"
    "leaderboard_musr|musr"
    "leaderboard_math_hard|math_hard"
    "leaderboard_ifeval|ifeval"
    "leaderboard_gpqa|gpqa"
)


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

has_full_weights() {
    local directory=$1

    compgen -G "$directory/model*.safetensors" >/dev/null ||
    compgen -G "$directory/pytorch_model*.bin" >/dev/null
}


has_adapter_weights() {
    local directory=$1

    [[ -f "$directory/adapter_model.safetensors" ||
       -f "$directory/adapter_model.bin" ]]
}


validate_result_json() {
    local result_path=$1

    python3 - "$result_path" <<'PY'
import json
import math
import sys
from pathlib import Path

path = Path(sys.argv[1])

if not path.is_file() or path.stat().st_size == 0:
    raise SystemExit(1)

try:
    payload = json.loads(
        path.read_text(encoding="utf-8")
    )
except Exception:
    raise SystemExit(1)

if not isinstance(payload, dict):
    raise SystemExit(1)

results = payload.get("results")
groups = payload.get("groups")

if not isinstance(results, dict) and not isinstance(groups, dict):
    raise SystemExit(1)

metric_containers = []

if isinstance(results, dict):
    metric_containers.extend(results.values())

if isinstance(groups, dict):
    metric_containers.extend(groups.values())

numeric_metric_found = False

for container in metric_containers:
    if not isinstance(container, dict):
        continue

    for key, value in container.items():
        if key.endswith("_stderr"):
            continue

        if isinstance(value, (int, float)):
            if math.isfinite(float(value)):
                numeric_metric_found = True
                break

    if numeric_metric_found:
        break

if not numeric_metric_found:
    raise SystemExit(1)

raise SystemExit(0)
PY
}


# ------------------------------------------------------------
# Preflight all checkpoints before starting any evaluations
# ------------------------------------------------------------

echo
echo "=== Checkpoint preflight ==="

for specification in "${CHECKPOINTS[@]}"; do
    IFS='|' read -r \
        checkpoint_name \
        checkpoint_kind \
        checkpoint_path \
        base_path \
        <<< "$specification"

    if [[ ! -d "$checkpoint_path" ]]; then
        echo "Missing checkpoint: $checkpoint_path" >&2
        exit 1
    fi

    case "$checkpoint_kind" in
        full)
            if [[ ! -f "$checkpoint_path/config.json" ]]; then
                echo "Missing full-model config: $checkpoint_path/config.json" >&2
                exit 1
            fi

            if ! has_full_weights "$checkpoint_path"; then
                echo "Missing full-model weights: $checkpoint_path" >&2
                exit 1
            fi
            ;;

        lora)
            if [[ ! -d "$base_path" ]]; then
                echo "Missing LoRA base model: $base_path" >&2
                exit 1
            fi

            if [[ ! -f "$checkpoint_path/adapter_config.json" ]]; then
                echo "Missing adapter config: $checkpoint_path" >&2
                exit 1
            fi

            if ! has_adapter_weights "$checkpoint_path"; then
                echo "Missing adapter weights: $checkpoint_path" >&2
                exit 1
            fi
            ;;

        *)
            echo "Unknown checkpoint kind: $checkpoint_kind" >&2
            exit 1
            ;;
    esac

    echo "OK: $checkpoint_name"
done


VISIBLE_GPU_COUNT=$(
    nvidia-smi \
        --query-gpu=index \
        --format=csv,noheader |
    wc -l
)

if (( VISIBLE_GPU_COUNT < NUM_GPUS )); then
    echo "Expected at least $NUM_GPUS GPUs." >&2
    echo "Visible GPUs: $VISIBLE_GPU_COUNT" >&2
    exit 1
fi

echo
echo "=== GPU inventory ==="

nvidia-smi \
    --query-gpu=index,name,memory.total,memory.free \
    --format=csv


# ------------------------------------------------------------
# Scheduler state
# ------------------------------------------------------------

printf \
    'timestamp_utc\twave\tpid\tgpu\tcheckpoint\ttask\tstatus\texit_code\tresult_path\n' \
    > "$STATUS_FILE"

declare -a ACTIVE_PIDS=()
declare -a ACTIVE_NAMES=()
declare -a ACTIVE_TASKS=()
declare -a ACTIVE_GPUS=()
declare -a ACTIVE_RESULTS=()
declare -a ACTIVE_LOGS=()

WAVE_NUMBER=0
TOTAL_LAUNCHED=0
TOTAL_SKIPPED=0
TOTAL_COMPLETED=0


terminate_active_jobs() {
    local pid

    echo
    echo "Terminating active evaluation jobs..."

    for pid in "${ACTIVE_PIDS[@]:-}"; do
        if kill -0 "$pid" >/dev/null 2>&1; then
            kill "$pid" >/dev/null 2>&1 || true
        fi
    done

    for pid in "${ACTIVE_PIDS[@]:-}"; do
        wait "$pid" >/dev/null 2>&1 || true
    done
}

trap terminate_active_jobs INT TERM


# ------------------------------------------------------------
# Launch one evaluation
# ------------------------------------------------------------

launch_evaluation() {
    local checkpoint_name=$1
    local checkpoint_kind=$2
    local checkpoint_path=$3
    local base_path=$4
    local task_name=$5
    local task_slug=$6

    # GPU assignment is 0,1,2,3 within every wave.
    local gpu=${#ACTIVE_PIDS[@]}

    local output_dir="$EVAL_ROOT/$checkpoint_name"
    local result_path="$output_dir/${task_slug}.json"
    local log_path="$output_dir/${task_slug}.log"
    local command_path="$output_dir/${task_slug}.command.sh"

    local cache_dir="$CACHE_ROOT/$checkpoint_name"
    local cache_prefix="$cache_dir/${task_slug}_"

    mkdir -p \
        "$output_dir" \
        "$cache_dir"

    if validate_result_json "$result_path" >/dev/null 2>&1; then
        echo "SKIP completed: $checkpoint_name / $task_name"

        printf \
            '%s\t-\t-\t-\t%s\t%s\talready_complete\t0\t%s\n' \
            "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
            "$checkpoint_name" \
            "$task_name" \
            "$result_path" \
            >> "$STATUS_FILE"

        TOTAL_SKIPPED=$((TOTAL_SKIPPED + 1))
        return 0
    fi

    if [[ -e "$result_path" ]]; then
        mv \
            "$result_path" \
            "${result_path}.invalid.${RUN_TIMESTAMP}"
    fi

    local model_args

    if [[ "$checkpoint_kind" == "full" ]]; then
        model_args="pretrained=$checkpoint_path,tokenizer=$checkpoint_path,dtype=$DTYPE,attn_implementation=$ATTENTION"
    else
        model_args="pretrained=$base_path,peft=$checkpoint_path,tokenizer=$base_path,dtype=$DTYPE,attn_implementation=$ATTENTION"
    fi

    local command=(
        "${LM_EVAL[@]}"
        --model hf
        --model_args "$model_args"
        --tasks "$task_name"
        --device "cuda:$gpu"
        --batch_size "$BATCH_SIZE"
        --seed "$SEED"
        --use_cache "$cache_prefix"
        --show_config
        --output_path "$result_path"
    )

    {
        echo '#!/usr/bin/env bash'
        printf '%q ' "${command[@]}"
        printf '\n'
    } > "$command_path"

    chmod +x "$command_path"

    echo
    echo "------------------------------------------------------------"
    echo "LAUNCH"
    echo "Wave:       $((WAVE_NUMBER + 1))"
    echo "GPU:        $gpu"
    echo "Checkpoint: $checkpoint_name"
    echo "Kind:       $checkpoint_kind"
    echo "Task:       $task_name"
    echo "Result:     $result_path"
    echo "Log:        $log_path"
    echo "------------------------------------------------------------"

    "${command[@]}" \
        > "$log_path" \
        2>&1 &

    local pid=$!

    ACTIVE_PIDS+=("$pid")
    ACTIVE_NAMES+=("$checkpoint_name")
    ACTIVE_TASKS+=("$task_name")
    ACTIVE_GPUS+=("$gpu")
    ACTIVE_RESULTS+=("$result_path")
    ACTIVE_LOGS+=("$log_path")

    TOTAL_LAUNCHED=$((TOTAL_LAUNCHED + 1))

    printf \
        '%s\t%s\t%s\t%s\t%s\t%s\tlaunched\t0\t%s\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        "$((WAVE_NUMBER + 1))" \
        "$pid" \
        "$gpu" \
        "$checkpoint_name" \
        "$task_name" \
        "$result_path" \
        >> "$STATUS_FILE"
}


# ------------------------------------------------------------
# Wait for every process in the current wave
# ------------------------------------------------------------

wait_for_wave() {
    local count=${#ACTIVE_PIDS[@]}

    if (( count == 0 )); then
        return 0
    fi

    WAVE_NUMBER=$((WAVE_NUMBER + 1))

    echo
    echo "============================================================"
    echo "WAITING FOR WAVE $WAVE_NUMBER"
    echo "Active jobs: $count"
    echo "============================================================"

    local wave_failed=0
    local index
    local pid
    local rc
    local checkpoint_name
    local task_name
    local gpu
    local result_path
    local log_path

    # Wait for all jobs in this wave, even if one fails.
    for index in "${!ACTIVE_PIDS[@]}"; do
        pid=${ACTIVE_PIDS[$index]}
        checkpoint_name=${ACTIVE_NAMES[$index]}
        task_name=${ACTIVE_TASKS[$index]}
        gpu=${ACTIVE_GPUS[$index]}
        result_path=${ACTIVE_RESULTS[$index]}
        log_path=${ACTIVE_LOGS[$index]}

        if wait "$pid"; then
            rc=0
        else
            rc=$?
        fi

        if (( rc == 0 )); then
            if validate_result_json "$result_path"; then
                status=complete
                TOTAL_COMPLETED=$((TOTAL_COMPLETED + 1))
            else
                status=invalid_result
                rc=90
                wave_failed=1
            fi
        else
            status=failed
            wave_failed=1
        fi

        printf \
            '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
            "$WAVE_NUMBER" \
            "$pid" \
            "$gpu" \
            "$checkpoint_name" \
            "$task_name" \
            "$status" \
            "$rc" \
            "$result_path" \
            >> "$STATUS_FILE"

        echo
        echo "FINISHED"
        echo "Wave:       $WAVE_NUMBER"
        echo "PID:        $pid"
        echo "GPU:        $gpu"
        echo "Checkpoint: $checkpoint_name"
        echo "Task:       $task_name"
        echo "Status:     $status"
        echo "Exit code:  $rc"

        if (( rc != 0 )); then
            echo "Log:        $log_path"
        fi
    done

    ACTIVE_PIDS=()
    ACTIVE_NAMES=()
    ACTIVE_TASKS=()
    ACTIVE_GPUS=()
    ACTIVE_RESULTS=()
    ACTIVE_LOGS=()

    echo
    echo "Wave $WAVE_NUMBER finished."

    if (( wave_failed != 0 )); then
        echo "At least one evaluation in wave $WAVE_NUMBER failed." >&2
        echo "The next wave will not be launched." >&2
        return 1
    fi
}


# ------------------------------------------------------------
# Generate all 48 jobs
# ------------------------------------------------------------

echo
echo "============================================================"
echo "SMART LM-EVAL BATCH"
echo "Checkpoints: ${#CHECKPOINTS[@]}"
echo "Tasks:       ${#TASKS[@]}"
echo "Total jobs:  $((${#CHECKPOINTS[@]} * ${#TASKS[@]}))"
echo "Wave size:   $NUM_GPUS"
echo "Batch size:  $BATCH_SIZE"
echo "Output root: $EVAL_ROOT"
echo "Master log:  $MASTER_LOG"
echo "Status file: $STATUS_FILE"
echo "============================================================"

for checkpoint_specification in "${CHECKPOINTS[@]}"; do
    IFS='|' read -r \
        checkpoint_name \
        checkpoint_kind \
        checkpoint_path \
        base_path \
        <<< "$checkpoint_specification"

    for task_specification in "${TASKS[@]}"; do
        IFS='|' read -r \
            task_name \
            task_slug \
            <<< "$task_specification"

        launch_evaluation \
            "$checkpoint_name" \
            "$checkpoint_kind" \
            "$checkpoint_path" \
            "$base_path" \
            "$task_name" \
            "$task_slug"

        # Hard barrier: complete all four before launching more.
        if (( ${#ACTIVE_PIDS[@]} == NUM_GPUS )); then
            wait_for_wave
        fi
    done
done

# Handle a final partial wave when completed jobs were skipped.
if (( ${#ACTIVE_PIDS[@]} > 0 )); then
    wait_for_wave
fi


# ------------------------------------------------------------
# Final verification and compact summary
# ------------------------------------------------------------

SUMMARY_TSV="$EVAL_ROOT/results_manifest.tsv"

printf \
    'checkpoint\tkind\ttask\tresult_path\tlog_path\tstatus\n' \
    > "$SUMMARY_TSV"

MISSING_RESULTS=0

for checkpoint_specification in "${CHECKPOINTS[@]}"; do
    IFS='|' read -r \
        checkpoint_name \
        checkpoint_kind \
        checkpoint_path \
        base_path \
        <<< "$checkpoint_specification"

    for task_specification in "${TASKS[@]}"; do
        IFS='|' read -r \
            task_name \
            task_slug \
            <<< "$task_specification"

        output_dir="$EVAL_ROOT/$checkpoint_name"
        result_path="$output_dir/${task_slug}.json"
        log_path="$output_dir/${task_slug}.log"

        if validate_result_json "$result_path" >/dev/null 2>&1; then
            status=complete
        else
            status=missing_or_invalid
            MISSING_RESULTS=$((MISSING_RESULTS + 1))
        fi

        printf \
            '%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$checkpoint_name" \
            "$checkpoint_kind" \
            "$task_name" \
            "$result_path" \
            "$log_path" \
            "$status" \
            >> "$SUMMARY_TSV"
    done
done

echo
echo "============================================================"
echo "LM-EVAL BATCH FINISHED"
echo "Waves completed: $WAVE_NUMBER"
echo "Jobs launched:   $TOTAL_LAUNCHED"
echo "Jobs completed:  $TOTAL_COMPLETED"
echo "Jobs skipped:    $TOTAL_SKIPPED"
echo "Invalid/missing: $MISSING_RESULTS"
echo "Manifest:        $SUMMARY_TSV"
echo "Master log:      $MASTER_LOG"
echo "============================================================"

if (( MISSING_RESULTS != 0 )); then
    exit 1
fi

echo "All 48 leaderboard evaluations are complete."
BASH

chmod +x run_all_lm_eval_wait4.sh

bash -n run_all_lm_eval_wait4.sh
```

Run it inside `tmux`:

```bash
cd /data/saral/wdir/smart || exit 1

tmux new -s smart_lm_eval
```

Then:

```bash
./run_all_lm_eval_wait4.sh
```

The execution pattern will be:

```text
Wave 1: jobs  1–4  on GPUs 0–3
wait
Wave 2: jobs  5–8  on GPUs 0–3
wait
...
Wave 12: jobs 45–48 on GPUs 0–3
wait
```

Results will be stored under:

```text
/mnt/warm_storage/saral/smart/evaluations/lm_eval_leaderboard_corrected/
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

Rerunning the script skips any result JSON that already parses successfully and contains numeric evaluation metrics.

[1]: https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/interface.md?utm_source=chatgpt.com "lm-evaluation-harness/docs/interface.md at main · EleutherAI/lm-evaluation-harness · GitHub"
