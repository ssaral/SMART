cd /data/saral/wdir/smart || exit 1

chmod +x data_generation_scripts/run_stage2_selection.py

python3 -m py_compile data_generation_scripts/run_stage2_selection.py
  
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
  --task-id 'cot::stream_qed_ii' \
  --task-id 'flan2021::anli_r3_0.1.0' \
  --show-progress \
  2>&1 | tee \
  "$STAGE2/stage2_smoke.log"
