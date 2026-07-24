THREADS=$(nproc)

# Use at most 32 CPU threads for comparable measurements:

if [ "$THREADS" -gt 32 ]; then
  THREADS=32
fi

export OMP_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS="$THREADS"

echo "Benchmark threads: $THREADS"

ROOT=/mnt/warm_storage/saral/smart
BENCH="$ROOT/artifacts/stage2_exact_benchmarks"

mkdir -p "$BENCH"

/usr/bin/time -v \
  -o "$BENCH/time_stream_qed_ii_full.txt" \
  python3 \
  /data/saral/wdir/smart/data_generation_scripts/benchmark_exact_facility_location.py \
  --allocations "$ROOT/artifacts/stage1_allocations/task_allocations.csv" \
  --embedding-root "$ROOT/artifacts/prompt_embeddings/gte-large" \
  --task-id 'cot::stream_qed_ii' \
  --sample-size 0 \
  --budget-column final_allocation_50000 \
  --output-root "$BENCH" \
  --show-progress \
  2>&1 | tee "$BENCH/stream_qed_ii_full.log"
