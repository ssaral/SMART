cd /data/saral/wdir/smart || exit 1

ROOT=/mnt/warm_storage/saral/smart
STAGE2="$ROOT/artifacts/stage2_selection"

python3 \
  data_generation_scripts/run_stage2_selection.py \
  --allocations "$ROOT/artifacts/stage1_allocations/task_allocations.csv" \
  --embedding-root "$ROOT/artifacts/prompt_embeddings/gte-large" \
  --output-root "$STAGE2" \
  --exact-threshold 5000 \
  --candidate-size 2048 \
  --seed 23 \
  2>&1 | tee \
  "$STAGE2/stage2_production.log"
