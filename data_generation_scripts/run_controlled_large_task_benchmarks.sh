for N in 2000 5000 10000; do
  echo
  echo "===== ANLI R3 benchmark n=$N ====="

  /usr/bin/time -v \
    -o "$BENCH/time_anli_r3_n${N}.txt" \
    python3 \
    /data/saral/wdir/smart/data_generation_scripts/benchmark_exact_facility_location.py \
    --allocations "$ROOT/artifacts/stage1_allocations/task_allocations.csv" \
    --embedding-root "$ROOT/artifacts/prompt_embeddings/gte-large" \
    --task-id 'flan2021::anli_r3_0.1.0' \
    --sample-size "$N" \
    --budget-column final_allocation_50000 \
    --output-root "$BENCH" \
    --show-progress \
    2>&1 | tee "$BENCH/anli_r3_n${N}.log"
done
