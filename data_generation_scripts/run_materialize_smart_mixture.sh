cd /data/saral/wdir/smart || exit 1

chmod +x data_generation_scripts/materialize_smart_mixtures.py

python3 -m py_compile data_generation_scripts/materialize_smart_mixtures.py

ROOT=/mnt/warm_storage/saral/smart
DATASETS="$ROOT/datasets"

mkdir -p "$DATASETS"

python3 \
  data_generation_scripts/materialize_smart_mixtures.py \
  --manifest "$ROOT/prepared_data/clean_task_manifest.csv" \
  --allocations "$ROOT/artifacts/stage1_allocations/task_allocations.csv" \
  --stage2-root "$ROOT/artifacts/stage2_selection" \
  --output-root "$DATASETS" \
  --seed 23 \
  --expected-validation-count 183870 \
  2>&1 | tee \
  "$DATASETS/materialization.log"
