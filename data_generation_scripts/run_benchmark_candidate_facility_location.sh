chmod +x data_generation_scripts/benchmark_candidate_facility_location.py
python3 -m py_compile  data_generation_scripts/benchmark_candidate_facility_location.py

THREADS=$(nproc)

if [ "$THREADS" -gt 32 ]; then
  THREADS=32
fi

export OMP_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS="$THREADS"

echo "Threads: $THREADS"

ROOT=/mnt/warm_storage/saral/smart
EXACT="$ROOT/artifacts/stage2_exact_benchmarks"
CAND="$ROOT/artifacts/stage2_candidate_validation"

mkdir -p "$CAND"

/usr/bin/time -v \
  -o "$CAND/time_anli_r3_rep2000_cand2000.txt" \
  python3 \
  /data/saral/wdir/smart/data_generation_scripts/benchmark_candidate_facility_location.py \
  --allocations "$ROOT/artifacts/stage1_allocations/task_allocations.csv" \
  --embedding-root "$ROOT/artifacts/prompt_embeddings/gte-large" \
  --exact-benchmark-root "$EXACT" \
  --task-id 'flan2021::anli_r3_0.1.0' \
  --sample-size 2000 \
  --candidate-size 2000 \
  --seed 23 \
  --output-root "$CAND" \
  --show-progress \
  2>&1 | tee \
  "$CAND/anli_r3_rep2000_cand2000.log"

# Complete-task quality curve
for C in 512 1024 2048 2961; do
  echo
  echo "===== stream_qed_ii candidate size $C ====="

  /usr/bin/time -v \
    -o "$CAND/time_stream_qed_ii_cand${C}.txt" \
    python3 \
    /data/saral/wdir/smart/data_generation_scripts/benchmark_candidate_facility_location.py \
    --allocations "$ROOT/artifacts/stage1_allocations/task_allocations.csv" \
    --embedding-root "$ROOT/artifacts/prompt_embeddings/gte-large" \
    --exact-benchmark-root "$EXACT" \
    --task-id 'cot::stream_qed_ii' \
    --sample-size 2961 \
    --candidate-size "$C" \
    --seed 23 \
    --output-root "$CAND" \
    --show-progress \
    2>&1 | tee \
    "$CAND/stream_qed_ii_cand${C}.log"
done
