source ./smart_env.sh

export CUDA_VISIBLE_DEVICES=0,1,2,3
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

GTE_PATH=$(cat /mnt/warm_storage/saral/smart/artifacts/prompt_embeddings/gte-large/encoder_snapshot_path.txt)

python3 data_generation_scripts/embed_all_tasks.py \
  --manifest /mnt/warm_storage/saral/smart/prepared_data/clean_task_manifest.csv \
  --inventory-manifest /mnt/warm_storage/saral/smart/dataset_inventory/task_manifest.csv \
  --encoder "$GTE_PATH" \
  --output-root /mnt/warm_storage/saral/smart/artifacts/prompt_embeddings/gte-large \
  --devices auto \
  --batch-size 128 \
  --seed 23 \
  2>&1 | tee \
  /mnt/warm_storage/saral/smart/artifacts/prompt_embeddings/gte-large/full_embedding_run.log
