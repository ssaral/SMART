watch -n 10 '
echo "Completed task metadata files:"
find /mnt/warm_storage/saral/smart/artifacts/prompt_embeddings/gte-large/tasks \
  -name metadata.json | wc -l

echo
echo "Storage used:"
du -sh /mnt/warm_storage/saral/smart/artifacts/prompt_embeddings/gte-large

echo
echo "Recent worker messages:"
tail -n 3 /mnt/warm_storage/saral/smart/artifacts/prompt_embeddings/gte-large/worker_cuda_*.log 2>/dev/null
'
