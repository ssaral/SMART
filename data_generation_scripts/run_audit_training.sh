cd /data/saral/wdir/smart || exit 1

ROOT=/mnt/warm_storage/saral/smart
AUDIT="$ROOT/artifacts/tokenization_audit"

mkdir -p "$AUDIT"

export TOKENIZERS_PARALLELISM=true

python3 \
  data_generation_scripts/audit_training_tokenization.py \
  --dataset-25000 "$ROOT/datasets/trainer/smart_25000" \
  --dataset-50000 "$ROOT/datasets/trainer/smart_50000" \
  --model "llama2_7b=/data/saral/wdir/smart/llama2_7b" \
  --model "qwen2_7b=/data/saral/wdir/smart/qwen2_7b" \
  --output-root "$AUDIT" \
  --max-seq-length 4096 \
  --batch-size 16 \
  --top-outliers 100 \
  --trust-remote-code \
  --local-files-only \
  2>&1 | tee \
  "$AUDIT/tokenization_audit.log"
