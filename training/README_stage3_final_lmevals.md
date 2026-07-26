The scheduler below treats each checkpoint as one job, and each job runs all six official leaderboard groups:

```text
leaderboard_mmlu_pro
leaderboard_bbh
leaderboard_musr
leaderboard_math_hard
leaderboard_ifeval
leaderboard_gpqa
```

This produces eight jobs rather than 48 checkpoint-task jobs, avoiding six separate 7B model loads per checkpoint. Four jobs run concurrently—one per GPU—and the next queued checkpoint starts as soon as any GPU becomes free.

The leaderboard task configurations already encode the intended shot counts: MMLU-Pro 5-shot, BBH 3-shot, MATH-Hard 4-shot, and GPQA, MuSR, and IFEval 0-shot. Therefore, the script deliberately does not override `--num_fewshot`. ([GitHub][1])

The Hugging Face backend supports local `pretrained=` paths and PEFT adapters through `peft=PATH`; the current CLI uses `lm-eval run`. Response and request caches are also supported for interrupted-run resumption. ([GitHub][2])

Because these models were trained using plain `prompt + response` formatting—not a chat-message template—the default is **no `--apply_chat_template`**. The script provides an opt-in switch, but the plain protocol is the consistent choice for these checkpoints.

## End-to-end launcher

```bash
cd /data/saral/wdir/smart || exit 1

cat > run_all_leaderboard_evals.sh <<'BASH'
#!/usr/bin/env bash
set -Eeuo pipefail
shopt -s nullglob

if (( BASH_VERSINFO[0] < 5 )); then
    echo "Bash 5 or newer is required for wait -n -p." >&2
    exit 2
fi

# ------------------------------------------------------------
# Paths and configuration
# ------------------------------------------------------------

PROJECT=${PROJECT:-/data/saral/wdir/smart}
ROOT=${ROOT:-/mnt/warm_storage/saral/smart}

MODEL_ROOT=${MODEL_ROOT:-$ROOT/models}
EVAL_ROOT=${EVAL_ROOT:-$ROOT/evaluations/lm_eval_leaderboard}

TOOLS_ROOT=${TOOLS_ROOT:-$ROOT/tools}
LMEVAL_REPO=${LMEVAL_REPO:-$TOOLS_ROOT/lm-evaluation-harness}
LMEVAL_VENV=${LMEVAL_VENV:-$ROOT/venvs/lm_eval}

LMEVAL_URL=${LMEVAL_URL:-https://github.com/EleutherAI/lm-evaluation-harness.git}
LMEVAL_REF=${LMEVAL_REF:-main}
LMEVAL_UPDATE=${LMEVAL_UPDATE:-0}
FORCE_INSTALL=${FORCE_INSTALL:-0}
ALLOW_DIRTY_LMEVAL=${ALLOW_DIRTY_LMEVAL:-0}

NUM_GPUS=${NUM_GPUS:-4}
SEED=${SEED:-23}

# Re-estimate the maximum usable batch size four times through
# each task. Limit the discovered batch size to avoid extreme
# allocations on short examples.
BATCH_SIZE=${BATCH_SIZE:-auto:4}
MAX_BATCH_SIZE=${MAX_BATCH_SIZE:-32}

# Per-sample logs can consume substantial disk space.
LOG_SAMPLES=${LOG_SAMPLES:-0}

# These SMART checkpoints were trained without chat templates.
APPLY_CHAT_TEMPLATE=${APPLY_CHAT_TEMPLATE:-0}

RUN_PREFLIGHT_SMOKE=${RUN_PREFLIGHT_SMOKE:-1}
ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION:-flash_attention_2}
CPU_THREADS_PER_JOB=${CPU_THREADS_PER_JOB:-12}

LLAMA_BASE=${LLAMA_BASE:-$PROJECT/llama2_7b}
QWEN_BASE=${QWEN_BASE:-$PROJECT/qwen2_7b}

HF_CACHE_ROOT=${HF_CACHE_ROOT:-$ROOT/cache/lm_eval_hf}
REQUEST_CACHE_ROOT=${REQUEST_CACHE_ROOT:-$ROOT/cache/lm_eval_requests}

TASKS=(
    leaderboard_mmlu_pro
    leaderboard_bbh
    leaderboard_musr
    leaderboard_math_hard
    leaderboard_ifeval
    leaderboard_gpqa
)

TASK_CSV=$(IFS=,; echo "${TASKS[*]}")

mkdir -p \
    "$TOOLS_ROOT" \
    "$EVAL_ROOT" \
    "$HF_CACHE_ROOT" \
    "$REQUEST_CACHE_ROOT"

export HF_HOME="$HF_CACHE_ROOT"
export HF_HUB_CACHE="$HF_CACHE_ROOT/hub"
export HF_DATASETS_CACHE="$HF_CACHE_ROOT/datasets"
export LM_HARNESS_CACHE_PATH="$REQUEST_CACHE_ROOT"

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LMEVAL_LOG_LEVEL=INFO

# ------------------------------------------------------------
# Install and freeze lm-evaluation-harness
# ------------------------------------------------------------

checkout_ref() {
    git -C "$LMEVAL_REPO" fetch --tags origin "$LMEVAL_REF"
    git -C "$LMEVAL_REPO" checkout --detach FETCH_HEAD
}

if [[ ! -d "$LMEVAL_REPO/.git" ]]; then
    echo "Cloning lm-evaluation-harness..."

    git clone \
        --filter=blob:none \
        "$LMEVAL_URL" \
        "$LMEVAL_REPO"

    checkout_ref

elif [[ "$LMEVAL_UPDATE" == "1" ]]; then
    echo "Updating lm-evaluation-harness to $LMEVAL_REF..."
    checkout_ref
else
    echo "Using existing lm-evaluation-harness checkout without updating."
fi

if [[ "$ALLOW_DIRTY_LMEVAL" != "1" ]] &&
   [[ -n "$(git -C "$LMEVAL_REPO" status --porcelain)" ]]
then
    echo "lm-evaluation-harness checkout is dirty:" >&2
    git -C "$LMEVAL_REPO" status --short >&2

    echo \
      "Set ALLOW_DIRTY_LMEVAL=1 only if this is intentional." \
      >&2

    exit 1
fi

LMEVAL_SHA=$(
    git -C "$LMEVAL_REPO" rev-parse HEAD
)

echo "lm-evaluation-harness commit: $LMEVAL_SHA"

if [[ ! -x "$LMEVAL_VENV/bin/python" ]]; then
    echo "Creating persistent lm-eval virtual environment..."

    python3 -m venv \
        --system-site-packages \
        "$LMEVAL_VENV"
fi

# shellcheck disable=SC1091
source "$LMEVAL_VENV/bin/activate"

INSTALL_MARKER="$LMEVAL_VENV/.lm_eval_commit"
INSTALLED_SHA=""

if [[ -f "$INSTALL_MARKER" ]]; then
    INSTALLED_SHA=$(cat "$INSTALL_MARKER")
fi

if [[ "$FORCE_INSTALL" == "1" ||
      "$INSTALLED_SHA" != "$LMEVAL_SHA" ]]
then
    echo "Installing lm-eval and leaderboard extras..."

    python -m pip install \
        --upgrade \
        pip \
        setuptools \
        wheel

    python -m pip install \
        -e "${LMEVAL_REPO}[hf,math,ifeval,sentencepiece]"

    python -m pip install \
        --upgrade \
        peft

    printf '%s\n' "$LMEVAL_SHA" \
        > "$INSTALL_MARKER"
else
    echo \
      "lm-eval environment already matches commit $LMEVAL_SHA."
fi

LM_EVAL="$LMEVAL_VENV/bin/lm-eval"

if [[ ! -x "$LM_EVAL" ]]; then
    echo "lm-eval executable is missing: $LM_EVAL" >&2
    exit 1
fi

if [[ "$ATTN_IMPLEMENTATION" == "flash_attention_2" ]]; then
    if ! python -c 'import flash_attn' >/dev/null 2>&1; then
        echo "flash_attn is unavailable; falling back to SDPA."
        ATTN_IMPLEMENTATION=sdpa
    fi
fi

# ------------------------------------------------------------
# Hardware preflight
# ------------------------------------------------------------

echo

python - <<'PY'
import torch

print("torch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("Visible GPUs:", torch.cuda.device_count())

for index in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(index)

    print(
        f"GPU {index}: "
        f"{props.name}, "
        f"{props.total_memory / 2**30:.2f} GiB"
    )
PY

VISIBLE_GPUS=$(
    nvidia-smi \
      --query-gpu=index \
      --format=csv,noheader |
    wc -l
)

if (( VISIBLE_GPUS < NUM_GPUS )); then
    echo \
      "Need $NUM_GPUS GPUs, but only $VISIBLE_GPUS are visible." \
      >&2

    exit 1
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
    echo \
      "HF_TOKEN is not set. Public datasets may still work; " \
      "authenticated or gated access will not."
fi

# ------------------------------------------------------------
# Checkpoint definitions and validation
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

choose_tokenizer() {
    local checkpoint=$1
    local base=$2

    if [[ -f "$checkpoint/tokenizer_config.json" ||
          -f "$checkpoint/tokenizer.json" ]]
    then
        printf '%s\n' "$checkpoint"
    else
        printf '%s\n' "$base"
    fi
}

declare -a JOBS=()

declare -A JOB_MODEL=()
declare -A JOB_BUDGET=()
declare -A JOB_MODE=()
declare -A JOB_KIND=()
declare -A JOB_BASE=()
declare -A JOB_CKPT=()
declare -A JOB_TOKENIZER=()

register_job() {
    local name=$1
    local model=$2
    local budget=$3
    local mode=$4
    local kind=$5
    local base=$6
    local checkpoint=$7
    local tokenizer

    tokenizer=$(
        choose_tokenizer \
          "$checkpoint" \
          "$base"
    )

    JOBS+=("$name")

    JOB_MODEL["$name"]=$model
    JOB_BUDGET["$name"]=$budget
    JOB_MODE["$name"]=$mode
    JOB_KIND["$name"]=$kind
    JOB_BASE["$name"]=$base
    JOB_CKPT["$name"]=$checkpoint
    JOB_TOKENIZER["$name"]=$tokenizer
}

register_job \
  llama2_7b_smart_25000_full_seed23 \
  llama2_7b \
  25000 \
  full \
  full \
  "$LLAMA_BASE" \
  "$MODEL_ROOT/llama2_7b_smart_25000_full_seed23"

register_job \
  llama2_7b_smart_50000_full_seed23 \
  llama2_7b \
  50000 \
  full \
  full \
  "$LLAMA_BASE" \
  "$MODEL_ROOT/llama2_7b_smart_50000_full_seed23"

register_job \
  qwen2_7b_smart_25000_full_seed23 \
  qwen2_7b \
  25000 \
  full \
  full \
  "$QWEN_BASE" \
  "$MODEL_ROOT/qwen2_7b_smart_25000_full_seed23"

register_job \
  qwen2_7b_smart_50000_full_seed23 \
  qwen2_7b \
  50000 \
  full \
  full \
  "$QWEN_BASE" \
  "$MODEL_ROOT/qwen2_7b_smart_50000_full_seed23"

register_job \
  llama2_7b_smart_25000_lora_seed23 \
  llama2_7b \
  25000 \
  lora \
  lora \
  "$LLAMA_BASE" \
  "$MODEL_ROOT/llama2_7b_smart_25000_lora_seed23"

register_job \
  llama2_7b_smart_50000_lora_seed23 \
  llama2_7b \
  50000 \
  lora \
  lora \
  "$LLAMA_BASE" \
  "$MODEL_ROOT/llama2_7b_smart_50000_lora_seed23"

register_job \
  qwen2_7b_smart_25000_lora_seed23 \
  qwen2_7b \
  25000 \
  lora \
  lora \
  "$QWEN_BASE" \
  "$MODEL_ROOT/qwen2_7b_smart_25000_lora_seed23"

register_job \
  qwen2_7b_smart_50000_lora_seed23 \
  qwen2_7b \
  50000 \
  lora \
  lora \
  "$QWEN_BASE" \
  "$MODEL_ROOT/qwen2_7b_smart_50000_lora_seed23"

for job in "${JOBS[@]}"; do
    base=${JOB_BASE[$job]}
    checkpoint=${JOB_CKPT[$job]}
    kind=${JOB_KIND[$job]}

    if [[ ! -d "$base" ]]; then
        echo "Missing base model: $base" >&2
        exit 1
    fi

    if [[ ! -d "$checkpoint" ]]; then
        echo "Missing checkpoint: $checkpoint" >&2
        exit 1
    fi

    if [[ "$kind" == "full" ]]; then
        if [[ ! -f "$checkpoint/config.json" ]]; then
            echo \
              "Missing full-model config: " \
              "$checkpoint/config.json" \
              >&2

            exit 1
        fi

        if ! has_full_weights "$checkpoint"; then
            echo "Missing full-model weights: $checkpoint" >&2
            exit 1
        fi
    else
        if [[ ! -f "$checkpoint/adapter_config.json" ]]; then
            echo \
              "Missing adapter config: " \
              "$checkpoint/adapter_config.json" \
              >&2

            exit 1
        fi

        if ! has_adapter_weights "$checkpoint"; then
            echo "Missing adapter weights: $checkpoint" >&2
            exit 1
        fi
    fi
done

if [[ "$APPLY_CHAT_TEMPLATE" == "1" ]]; then
    echo "Checking tokenizer chat templates..."

    for job in "${JOBS[@]}"; do
        tokenizer=${JOB_TOKENIZER[$job]}

        python - "$tokenizer" <<'PY'
import sys
from transformers import AutoTokenizer

path = sys.argv[1]

tokenizer = AutoTokenizer.from_pretrained(
    path,
    local_files_only=True,
    trust_remote_code=True,
)

if not getattr(tokenizer, "chat_template", None):
    raise RuntimeError(
        f"Tokenizer has no chat template: {path}"
    )

print("Chat template OK:", path)
PY
    done
fi

model_args_for() {
    local job=$1
    local kind=${JOB_KIND[$job]}
    local base=${JOB_BASE[$job]}
    local checkpoint=${JOB_CKPT[$job]}
    local tokenizer=${JOB_TOKENIZER[$job]}

    if [[ "$kind" == "full" ]]; then
        printf \
          'pretrained=%s,tokenizer=%s,dtype=bfloat16,attn_implementation=%s,low_cpu_mem_usage=True' \
          "$checkpoint" \
          "$tokenizer" \
          "$ATTN_IMPLEMENTATION"
    else
        printf \
          'pretrained=%s,peft=%s,tokenizer=%s,dtype=bfloat16,attn_implementation=%s,low_cpu_mem_usage=True' \
          "$base" \
          "$checkpoint" \
          "$tokenizer" \
          "$ATTN_IMPLEMENTATION"
    fi
}

# ------------------------------------------------------------
# Result validation
# ------------------------------------------------------------

validate_result_json() {
    local result_json=$1

    python - \
      "$result_json" \
      "${TASKS[@]}" <<'PY'
import json
import math
import sys
from pathlib import Path

path = Path(sys.argv[1])
required = sys.argv[2:]

if not path.is_file():
    raise SystemExit(1)

try:
    payload = json.loads(
        path.read_text(encoding="utf-8")
    )
except Exception:
    raise SystemExit(1)

results = payload.get("results") or {}
groups = payload.get("groups") or {}
group_subtasks = payload.get("group_subtasks") or {}
configs = payload.get("configs") or {}

present = (
    set(results)
    | set(groups)
    | set(group_subtasks)
    | set(configs)
)

missing = [
    name
    for name in required
    if name not in present
]

if missing:
    print(
        "Missing task/group outputs:",
        missing,
        file=sys.stderr,
    )
    raise SystemExit(1)

preferred = {
    "leaderboard_mmlu_pro": "acc",
    "leaderboard_bbh": "acc_norm",
    "leaderboard_musr": "acc_norm",
    "leaderboard_math_hard": "exact_match",
    "leaderboard_ifeval": "prompt_level_strict_acc",
    "leaderboard_gpqa": "acc_norm",
}

for name, metric in preferred.items():
    entry = (
        groups.get(name)
        or results.get(name)
    )

    if not isinstance(entry, dict):
        print(
            f"No metric dictionary for {name}",
            file=sys.stderr,
        )
        raise SystemExit(1)

    candidate_keys = [
        metric,
        f"{metric},none",
    ]

    candidate_keys.extend(
        key
        for key in entry
        if key.split(",", 1)[0] == metric
    )

    value = None

    for key in candidate_keys:
        if key in entry:
            value = entry[key]
            break

    try:
        value = float(value)
    except Exception:
        print(
            f"No numeric {metric} for {name}: {entry}",
            file=sys.stderr,
        )
        raise SystemExit(1)

    if not math.isfinite(value):
        print(
            f"Non-finite {metric} for {name}: {value}",
            file=sys.stderr,
        )
        raise SystemExit(1)
PY
}

# ------------------------------------------------------------
# Record full provenance
# ------------------------------------------------------------

PROVENANCE_DIR="$EVAL_ROOT/provenance"

mkdir -p "$PROVENANCE_DIR"

printf '%s\n' "$LMEVAL_SHA" \
    > "$PROVENANCE_DIR/lm_eval_commit.txt"

python -m pip freeze \
    > "$PROVENANCE_DIR/pip_freeze.txt"

git -C "$LMEVAL_REPO" status --short \
    > "$PROVENANCE_DIR/lm_eval_git_status.txt"

find "$LMEVAL_REPO/lm_eval/tasks/leaderboard" \
    -type f \
    -print0 |
sort -z |
xargs -0 sha256sum \
    > "$PROVENANCE_DIR/leaderboard_task_files_sha256.txt"

nvidia-smi -q \
    > "$PROVENANCE_DIR/nvidia_smi_q.txt"

JOB_MANIFEST="$EVAL_ROOT/job_manifest.tsv"

printf \
  'job_name\tmodel\tbudget\tmode\tkind\tbase_model\tcheckpoint\ttokenizer\tresult_json\n' \
  > "$JOB_MANIFEST"

for job in "${JOBS[@]}"; do
    printf \
      '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$job" \
      "${JOB_MODEL[$job]}" \
      "${JOB_BUDGET[$job]}" \
      "${JOB_MODE[$job]}" \
      "${JOB_KIND[$job]}" \
      "${JOB_BASE[$job]}" \
      "${JOB_CKPT[$job]}" \
      "${JOB_TOKENIZER[$job]}" \
      "$EVAL_ROOT/runs/$job/results.json" \
      >> "$JOB_MANIFEST"
done

cat > "$PROVENANCE_DIR/evaluation_config.txt" <<EOF
lm_eval_commit=$LMEVAL_SHA
tasks=$TASK_CSV
seed=$SEED
batch_size=$BATCH_SIZE
max_batch_size=$MAX_BATCH_SIZE
attention_implementation=$ATTN_IMPLEMENTATION
apply_chat_template=$APPLY_CHAT_TEMPLATE
log_samples=$LOG_SAMPLES
num_parallel_jobs=$NUM_GPUS
EOF

# Validate task existence, syntax, dataset paths, metrics and templates.
"$LM_EVAL" validate \
    --tasks "$TASK_CSV" \
    2>&1 |
tee "$PROVENANCE_DIR/task_validation.log"

COMMON_EXTRA_ARGS=()

if [[ "$APPLY_CHAT_TEMPLATE" == "1" ]]; then
    COMMON_EXTRA_ARGS+=(
        --apply_chat_template
    )
fi

if [[ "$LOG_SAMPLES" == "1" ]]; then
    COMMON_EXTRA_ARGS+=(
        --log_samples
    )
fi

# ------------------------------------------------------------
# Small end-to-end model/task smoke test
# ------------------------------------------------------------

if [[ "$RUN_PREFLIGHT_SMOKE" == "1" ]]; then
    SMOKE_DIR="$EVAL_ROOT/preflight"
    SMOKE_RESULT="$SMOKE_DIR/results.json"

    mkdir -p "$SMOKE_DIR"

    if validate_result_json \
        "$SMOKE_RESULT" \
        >/dev/null 2>&1
    then
        echo "Preflight smoke result already valid; skipping."
    else
        echo \
          "Running one-example-per-subtask preflight smoke " \
          "on GPU 0..."

        SMOKE_JOB=${JOBS[0]}
        SMOKE_MODEL_ARGS=$(
            model_args_for "$SMOKE_JOB"
        )

        rm -f "$SMOKE_RESULT"

        CUDA_VISIBLE_DEVICES=0 \
        OMP_NUM_THREADS="$CPU_THREADS_PER_JOB" \
        MKL_NUM_THREADS="$CPU_THREADS_PER_JOB" \
        OPENBLAS_NUM_THREADS="$CPU_THREADS_PER_JOB" \
        "$LM_EVAL" run \
            --model hf \
            --model_args "$SMOKE_MODEL_ARGS" \
            --tasks "$TASK_CSV" \
            --device cuda:0 \
            --batch_size 1 \
            --limit 1 \
            --seed "$SEED" \
            --trust_remote_code \
            --cache_requests true \
            --show_config \
            --output_path "$SMOKE_RESULT" \
            "${COMMON_EXTRA_ARGS[@]}" \
            2>&1 |
        tee "$SMOKE_DIR/smoke.log"

        validate_result_json "$SMOKE_RESULT"

        echo "Preflight smoke passed."
    fi
fi

# ------------------------------------------------------------
# Per-job execution
# ------------------------------------------------------------

write_status_json() {
    local path=$1
    local job=$2
    local gpu=$3
    local status=$4
    local exit_code=$5
    local started=$6
    local finished=$7
    local result_json=$8

    python - \
      "$path" \
      "$job" \
      "$gpu" \
      "$status" \
      "$exit_code" \
      "$started" \
      "$finished" \
      "$result_json" \
      "$LMEVAL_SHA" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])

payload = {
    "job_name": sys.argv[2],
    "physical_gpu_slot": int(sys.argv[3]),
    "status": sys.argv[4],
    "exit_code": int(sys.argv[5]),
    "started_utc": sys.argv[6],
    "finished_utc": sys.argv[7],
    "result_json": sys.argv[8],
    "lm_eval_commit": sys.argv[9],
}

temporary = path.with_suffix(
    path.suffix + ".tmp"
)

temporary.write_text(
    json.dumps(payload, indent=2) + "\n",
    encoding="utf-8",
)

temporary.replace(path)
PY
}

run_job() {
    local job=$1
    local gpu=$2

    local run_dir="$EVAL_ROOT/runs/$job"
    local result_json="$run_dir/results.json"

    local started
    local finished
    local attempt
    local log_path
    local time_path
    local command_path
    local status_path
    local model_args
    local rc

    mkdir -p \
        "$run_dir/response_cache" \
        "$run_dir/tmp"

    if validate_result_json \
        "$result_json" \
        >/dev/null 2>&1
    then
        echo "[$job] result already complete; skipping."
        return 0
    fi

    if [[ -f "$result_json" ]]; then
        mv \
          "$result_json" \
          "$run_dir/results.invalid.$(
              date -u +%Y%m%dT%H%M%SZ
          ).json"
    fi

    attempt=$(date -u +%Y%m%dT%H%M%SZ)

    log_path="$run_dir/eval.$attempt.log"
    time_path="$run_dir/time.$attempt.txt"
    command_path="$run_dir/command.$attempt.sh"
    status_path="$run_dir/status.json"

    started=$(date -u +%Y-%m-%dT%H:%M:%SZ)

    model_args=$(
        model_args_for "$job"
    )

    local cmd=(
        "$LM_EVAL" run
        --model hf
        --model_args "$model_args"
        --tasks "$TASK_CSV"
        --device cuda:0
        --batch_size "$BATCH_SIZE"
        --max_batch_size "$MAX_BATCH_SIZE"
        --seed "$SEED"
        --trust_remote_code
        --cache_requests true
        --use_cache "$run_dir/response_cache/cache"
        --show_config
        --output_path "$result_json"
        "${COMMON_EXTRA_ARGS[@]}"
    )

    {
        echo '#!/usr/bin/env bash'

        printf \
          'CUDA_VISIBLE_DEVICES=%q ' \
          "$gpu"

        printf '%q ' "${cmd[@]}"
        printf '\n'
    } > "$command_path"

    chmod +x "$command_path"

    write_status_json \
      "$status_path" \
      "$job" \
      "$gpu" \
      running \
      0 \
      "$started" \
      "" \
      "$result_json"

    echo
    echo "============================================================"
    echo "START $job"
    echo "GPU slot:   $gpu"
    echo "Kind:       ${JOB_KIND[$job]}"
    echo "Checkpoint: ${JOB_CKPT[$job]}"
    echo "Started:    $started"
    echo "============================================================"

    set +e

    (
        export CUDA_VISIBLE_DEVICES="$gpu"
        export CUDA_DEVICE_ORDER=PCI_BUS_ID

        # 12 threads × four workers leaves substantial CPU headroom
        # on the 180-thread host.
        export OMP_NUM_THREADS="$CPU_THREADS_PER_JOB"
        export MKL_NUM_THREADS="$CPU_THREADS_PER_JOB"
        export OPENBLAS_NUM_THREADS="$CPU_THREADS_PER_JOB"
        export NUMEXPR_NUM_THREADS="$CPU_THREADS_PER_JOB"

        export TMPDIR="$run_dir/tmp"

        /usr/bin/time \
          -v \
          -o "$time_path" \
          "${cmd[@]}"
    ) 2>&1 |
    tee "$log_path"

    rc=${PIPESTATUS[0]}

    set -e

    if (( rc == 0 )); then
        if ! validate_result_json \
            "$result_json" \
            2>>"$log_path"
        then
            echo \
              "[$job] result validation failed." |
            tee -a "$log_path"

            rc=3
        fi
    fi

    finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)

    if (( rc == 0 )); then
        write_status_json \
          "$status_path" \
          "$job" \
          "$gpu" \
          complete \
          0 \
          "$started" \
          "$finished" \
          "$result_json"

        echo \
          "COMPLETE $job on GPU slot $gpu at $finished"
    else
        write_status_json \
          "$status_path" \
          "$job" \
          "$gpu" \
          failed \
          "$rc" \
          "$started" \
          "$finished" \
          "$result_json"

        echo \
          "FAILED $job on GPU slot $gpu with exit code " \
          "$rc at $finished" \
          >&2
    fi

    return "$rc"
}

# ------------------------------------------------------------
# Dynamic four-GPU scheduler
# ------------------------------------------------------------

SCHEDULER_DIR="$EVAL_ROOT/scheduler"

mkdir -p "$SCHEDULER_DIR"

SCHEDULER_LOG="$SCHEDULER_DIR/scheduler.$(
    date -u +%Y%m%dT%H%M%SZ
).log"

STATUS_TSV="$SCHEDULER_DIR/status.tsv"

printf \
  'timestamp_utc\tjob_name\tgpu_slot\tstatus\texit_code\n' \
  > "$STATUS_TSV"

exec > >(tee -a "$SCHEDULER_LOG") 2>&1

echo
echo \
  "Queueing ${#JOBS[@]} checkpoint-level jobs; " \
  "each job runs all six leaderboard groups."

echo "Maximum concurrent jobs: $NUM_GPUS"
echo "Scheduler log: $SCHEDULER_LOG"

declare -a PENDING=()

for job in "${JOBS[@]}"; do
    result_json="$EVAL_ROOT/runs/$job/results.json"

    if validate_result_json \
        "$result_json" \
        >/dev/null 2>&1
    then
        echo "SKIP complete: $job"

        printf \
          '%s\t%s\t%s\t%s\t%s\n' \
          "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
          "$job" \
          - \
          already_complete \
          0 \
          >> "$STATUS_TSV"
    else
        PENDING+=("$job")
    fi
done

declare -a FREE_GPUS=()

for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
    FREE_GPUS+=("$gpu")
done

declare -A PID_TO_JOB=()
declare -A PID_TO_GPU=()

declare -a FAILED_JOBS=()

NEXT_INDEX=0

start_one() {
    local job=$1
    local gpu=$2

    run_job "$job" "$gpu" &

    local pid=$!

    PID_TO_JOB["$pid"]=$job
    PID_TO_GPU["$pid"]=$gpu

    printf \
      '%s\t%s\t%s\t%s\t%s\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
      "$job" \
      "$gpu" \
      launched \
      0 \
      >> "$STATUS_TSV"

    echo \
      "LAUNCHED $job as PID $pid on GPU slot $gpu"
}

while (( NEXT_INDEX < ${#PENDING[@]} )) ||
      (( ${#PID_TO_JOB[@]} > 0 ))
do
    while (( NEXT_INDEX < ${#PENDING[@]} )) &&
          (( ${#FREE_GPUS[@]} > 0 ))
    do
        job=${PENDING[$NEXT_INDEX]}
        gpu=${FREE_GPUS[0]}

        FREE_GPUS=(
            "${FREE_GPUS[@]:1}"
        )

        NEXT_INDEX=$((NEXT_INDEX + 1))

        start_one \
          "$job" \
          "$gpu"
    done

    if (( ${#PID_TO_JOB[@]} == 0 )); then
        continue
    fi

    FINISHED_PID=""

    set +e

    wait \
      -n \
      -p FINISHED_PID \
      "${!PID_TO_JOB[@]}"

    rc=$?

    set -e

    if [[ -z "$FINISHED_PID" ||
          -z "${PID_TO_JOB[$FINISHED_PID]+x}" ]]
    then
        echo \
          "Unable to identify completed PID; " \
          "scheduler cannot continue safely." \
          >&2

        exit 1
    fi

    job=${PID_TO_JOB[$FINISHED_PID]}
    gpu=${PID_TO_GPU[$FINISHED_PID]}

    unset 'PID_TO_JOB[$FINISHED_PID]'
    unset 'PID_TO_GPU[$FINISHED_PID]'

    FREE_GPUS+=("$gpu")

    if (( rc == 0 )); then
        state=complete
    else
        state=failed
        FAILED_JOBS+=("$job")
    fi

    printf \
      '%s\t%s\t%s\t%s\t%s\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
      "$job" \
      "$gpu" \
      "$state" \
      "$rc" \
      >> "$STATUS_TSV"

    echo \
      "REAPED PID $FINISHED_PID: " \
      "$job, GPU slot $gpu, status=$state, rc=$rc"
done

# ------------------------------------------------------------
# Aggregate the eight result files
# ------------------------------------------------------------

SUMMARY_CSV="$EVAL_ROOT/leaderboard_summary.csv"
SUMMARY_JSON="$EVAL_ROOT/leaderboard_summary.json"

set +e

python - \
  "$JOB_MANIFEST" \
  "$SUMMARY_CSV" \
  "$SUMMARY_JSON" <<'PY'
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path


manifest_path = Path(sys.argv[1])
csv_path = Path(sys.argv[2])
json_path = Path(sys.argv[3])


with manifest_path.open(
    encoding="utf-8",
    newline="",
) as handle:
    manifest = list(
        csv.DictReader(
            handle,
            delimiter="\t",
        )
    )


def entry_for(
    payload: dict,
    name: str,
) -> dict:
    groups = payload.get("groups") or {}
    results = payload.get("results") or {}

    return (
        groups.get(name)
        or results.get(name)
        or {}
    )


def metric(
    entry: dict,
    name: str,
) -> float:
    for key in (
        f"{name},none",
        name,
    ):
        if key in entry:
            return float(entry[key])

    for key, value in entry.items():
        if key.split(",", 1)[0] == name:
            return float(value)

    raise KeyError(name)


def optional_metric(
    entry: dict,
    name: str,
) -> float | None:
    try:
        return metric(entry, name)
    except Exception:
        return None


rows: list[dict] = []
failed: list[str] = []

for specification in manifest:
    result_path = Path(
        specification["result_json"]
    )

    if not result_path.is_file():
        failed.append(
            specification["job_name"]
        )
        continue

    try:
        payload = json.loads(
            result_path.read_text(
                encoding="utf-8"
            )
        )

        mmlu = entry_for(
            payload,
            "leaderboard_mmlu_pro",
        )
        bbh = entry_for(
            payload,
            "leaderboard_bbh",
        )
        musr = entry_for(
            payload,
            "leaderboard_musr",
        )
        math_hard = entry_for(
            payload,
            "leaderboard_math_hard",
        )
        ifeval = entry_for(
            payload,
            "leaderboard_ifeval",
        )
        gpqa = entry_for(
            payload,
            "leaderboard_gpqa",
        )

        row = {
            "job_name": specification[
                "job_name"
            ],
            "model": specification["model"],
            "budget": int(
                specification["budget"]
            ),
            "mode": specification["mode"],
            "mmlu_pro_acc": metric(
                mmlu,
                "acc",
            ),
            "bbh_acc_norm": metric(
                bbh,
                "acc_norm",
            ),
            "musr_acc_norm": metric(
                musr,
                "acc_norm",
            ),
            "math_hard_exact_match": metric(
                math_hard,
                "exact_match",
            ),
            "ifeval_prompt_strict": metric(
                ifeval,
                "prompt_level_strict_acc",
            ),
            "ifeval_inst_strict": optional_metric(
                ifeval,
                "inst_level_strict_acc",
            ),
            "ifeval_prompt_loose": optional_metric(
                ifeval,
                "prompt_level_loose_acc",
            ),
            "ifeval_inst_loose": optional_metric(
                ifeval,
                "inst_level_loose_acc",
            ),
            "gpqa_acc_norm": metric(
                gpqa,
                "acc_norm",
            ),
            "result_json": str(result_path),
            "status": "complete",
        }

        for key, value in row.items():
            if (
                isinstance(value, float)
                and not math.isfinite(value)
            ):
                raise ValueError(
                    f"non-finite {key}"
                )

        rows.append(row)

    except Exception as exc:
        print(
            "Failed to summarize "
            f"{specification['job_name']}: "
            f"{exc}",
            file=sys.stderr,
        )

        failed.append(
            specification["job_name"]
        )


rows.sort(
    key=lambda row: (
        row["model"],
        row["mode"],
        row["budget"],
    )
)


if rows:
    with csv_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
        )

        writer.writeheader()
        writer.writerows(rows)


json_path.write_text(
    json.dumps(
        {
            "status": (
                "complete"
                if (
                    not failed
                    and len(rows)
                    == len(manifest)
                )
                else "incomplete"
            ),
            "completed_count": len(rows),
            "expected_count": len(manifest),
            "failed_jobs": failed,
            "experiments": rows,
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)


print()
print(
    f"{'model':12s} "
    f"{'budget':>7s} "
    f"{'mode':>6s} "
    f"{'MMLU-Pro':>10s} "
    f"{'BBH':>10s} "
    f"{'MuSR':>10s} "
    f"{'MATH':>10s} "
    f"{'IFEval':>10s} "
    f"{'GPQA':>10s}"
)

for row in rows:
    print(
        f"{row['model']:12s} "
        f"{row['budget']:7,d} "
        f"{row['mode']:>6s} "
        f"{row['mmlu_pro_acc']:10.4f} "
        f"{row['bbh_acc_norm']:10.4f} "
        f"{row['musr_acc_norm']:10.4f} "
        f"{row['math_hard_exact_match']:10.4f} "
        f"{row['ifeval_prompt_strict']:10.4f} "
        f"{row['gpqa_acc_norm']:10.4f}"
    )

print()
print("CSV:", csv_path)
print("JSON:", json_path)

if failed or len(rows) != len(manifest):
    raise SystemExit(1)
PY

summary_rc=$?

set -e

if (( ${#FAILED_JOBS[@]} > 0 )); then
    echo
    echo "Failed scheduler jobs:"

    printf \
      '  %s\n' \
      "${FAILED_JOBS[@]}"
fi

if (( ${#FAILED_JOBS[@]} > 0 ||
      summary_rc != 0 ))
then
    echo \
      "Evaluation batch finished with failures. " \
      "Rerun the same script to resume cached jobs." \
      >&2

    exit 1
fi

echo
echo \
  "All eight checkpoints completed all six " \
  "leaderboard evaluations."

echo "Summary CSV:   $SUMMARY_CSV"
echo "Summary JSON:  $SUMMARY_JSON"
echo "Scheduler log: $SCHEDULER_LOG"
BASH

chmod +x run_all_leaderboard_evals.sh

bash -n run_all_leaderboard_evals.sh
```

## Launch

The first run needs internet access to clone the harness and obtain benchmark datasets. Set `HF_TOKEN` beforehand when authenticated dataset access is required.

```bash
cd /data/saral/wdir/smart || exit 1

tmux new -s smart_leaderboard
```

Inside the session:

```bash
./run_all_leaderboard_evals.sh
```

Detach with `Ctrl-b`, then `d`. Reattach with:

```bash
tmux attach -t smart_leaderboard
```

## Monitoring

Scheduler activity:

```bash
tail -f \
  /mnt/warm_storage/saral/smart/evaluations/lm_eval_leaderboard/scheduler/scheduler.*.log
```

GPU use:

```bash
watch -n 5 \
  'nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu,power.draw --format=csv'
```

Current job states:

```bash
column -t -s $'\t' \
  /mnt/warm_storage/saral/smart/evaluations/lm_eval_leaderboard/scheduler/status.tsv
```

## Final outputs

```text
/mnt/warm_storage/saral/smart/evaluations/lm_eval_leaderboard/
├── runs/
│   ├── llama2_7b_smart_25000_full_seed23/
│   ├── llama2_7b_smart_50000_full_seed23/
│   ├── llama2_7b_smart_25000_lora_seed23/
│   ├── llama2_7b_smart_50000_lora_seed23/
│   ├── qwen2_7b_smart_25000_full_seed23/
│   ├── qwen2_7b_smart_50000_full_seed23/
│   ├── qwen2_7b_smart_25000_lora_seed23/
│   └── qwen2_7b_smart_50000_lora_seed23/
├── provenance/
├── scheduler/
├── job_manifest.tsv
├── leaderboard_summary.csv
└── leaderboard_summary.json
```

Rerunning the same script skips validated completed outputs. Failed or interrupted evaluations reuse both the cached task requests and model responses where available.

[1]: https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/leaderboard/README.md?utm_source=chatgpt.com "lm-evaluation-harness/lm_eval/tasks/leaderboard/README.md at main · EleutherAI/lm-evaluation-harness · GitHub"
[2]: https://github.com/EleutherAI/lm-evaluation-harness/blob/main/README.md?utm_source=chatgpt.com "lm-evaluation-harness/README.md at main · EleutherAI/lm-evaluation-harness · GitHub"
