Yes, keep the project root fixed as:

```text
/data/saral/wdir/smart
```

The failure is unrelated to the GPU setup or the four-worker logic. The encoder argument was empty.

Your log shows:

```text
Encoder: /data/saral/wdir/smart
```

The launch script does this:

```python
Path(args.encoder).resolve()
```

When `--encoder` receives an empty string, `Path("")` resolves to the current working directory. Since you launched from `/data/saral/wdir/smart`, the code mistakenly tried to load the entire SMART project as a Sentence Transformers model.

No embeddings were generated, so it is safe to correct the path and rerun.

# Step 7 correction — Resolve the GTE snapshot

## 1. Check the stored path file

Inside the container:

```bash
cd /data/saral/wdir/smart || exit 1
source ./smart_env.sh

GTE_PATH_FILE=/mnt/warm_storage/saral/smart/artifacts/prompt_embeddings/gte-large/encoder_snapshot_path.txt

echo "Path file:"
ls -l "$GTE_PATH_FILE" 2>/dev/null || true

echo
echo "Contents:"
cat -A "$GTE_PATH_FILE" 2>/dev/null || true
```

It is probably missing or empty.

## 2. Locate the downloaded model

Search persistent cache storage:

```bash
find /mnt/warm_storage/saral/smart/cache \
  -path '*/models--thenlper--gte-large/snapshots/*/modules.json' \
  -print
```

A valid result should resemble:

```text
/mnt/warm_storage/saral/smart/cache/huggingface/hub/models--thenlper--gte-large/snapshots/<revision>/modules.json
```

Set `GTE_PATH` from that result:

```bash
GTE_MODULES_FILE=$(
  find /mnt/warm_storage/saral/smart/cache \
    -path '*/models--thenlper--gte-large/snapshots/*/modules.json' \
    -print \
    -quit
)

if [[ -z "$GTE_MODULES_FILE" ]]; then
    echo "ERROR: GTE-large snapshot was not found in persistent cache."
    exit 1
fi

GTE_PATH=$(dirname "$GTE_MODULES_FILE")

echo "Resolved GTE path:"
echo "$GTE_PATH"
```

Validate its contents:

```bash
ls -lah "$GTE_PATH"

for required in modules.json config.json; do
    if [[ -e "$GTE_PATH/$required" ]]; then
        echo "OK      $required"
    else
        echo "MISSING $required"
    fi
done
```

## 3. Verify that Sentence Transformers can load it

```bash
python3 - "$GTE_PATH" <<'PY'
import sys
from sentence_transformers import SentenceTransformer

path = sys.argv[1]

model = SentenceTransformer(
    path,
    device="cpu",
    local_files_only=True,
)

print("Loaded:", path)
print("Embedding dimension:", model.get_sentence_embedding_dimension())
print("Maximum sequence length:", model.max_seq_length)
PY
```

Expected:

```text
Embedding dimension: 1024
Maximum sequence length: 512
```

## 4. Save the corrected snapshot path

```bash
mkdir -p \
  /mnt/warm_storage/saral/smart/artifacts/prompt_embeddings/gte-large

printf '%s\n' "$GTE_PATH" > \
  /mnt/warm_storage/saral/smart/artifacts/prompt_embeddings/gte-large/encoder_snapshot_path.txt

cat \
  /mnt/warm_storage/saral/smart/artifacts/prompt_embeddings/gte-large/encoder_snapshot_path.txt
```

# When no snapshot is found

Download it explicitly into persistent storage:

```bash
cd /data/saral/wdir/smart || exit 1
source ./smart_env.sh

python3 - <<'PY'
import os
from pathlib import Path

from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer

output_root = Path(
    "/mnt/warm_storage/saral/smart/"
    "artifacts/prompt_embeddings/gte-large"
)
output_root.mkdir(parents=True, exist_ok=True)

snapshot_path = snapshot_download(
    repo_id="thenlper/gte-large",
    cache_dir=os.environ["HF_HUB_CACHE"],
)

model = SentenceTransformer(
    snapshot_path,
    device="cpu",
    local_files_only=True,
)

path_file = output_root / "encoder_snapshot_path.txt"
path_file.write_text(snapshot_path + "\n", encoding="utf-8")

print("Snapshot:", snapshot_path)
print("Dimension:", model.get_sentence_embedding_dimension())
print("Maximum sequence length:", model.max_seq_length)
print("Saved path:", path_file)
PY
```

# Add a defensive check to the embedding script

The current script should reject an empty encoder path instead of interpreting it as the working directory.

In `data_generation_scripts/embed_all_tasks.py`, replace:

```python
encoder_path = str(
    Path(args.encoder).resolve()
)

if not Path(encoder_path).is_dir():
    raise FileNotFoundError(
        f"Local encoder snapshot not found: "
        f"{encoder_path}"
    )
```

with:

```python
encoder_argument = args.encoder.strip()

if not encoder_argument:
    raise ValueError(
        "--encoder is empty. Supply the local GTE-large "
        "Sentence Transformers snapshot directory."
    )

encoder_directory = Path(
    encoder_argument
).expanduser().resolve()

if not encoder_directory.is_dir():
    raise FileNotFoundError(
        f"Local encoder snapshot not found: "
        f"{encoder_directory}"
    )

modules_path = encoder_directory / "modules.json"
config_path = encoder_directory / "config.json"

if not modules_path.is_file():
    raise FileNotFoundError(
        f"The encoder directory is not a Sentence Transformers "
        f"snapshot because modules.json is missing: "
        f"{encoder_directory}"
    )

if not config_path.is_file():
    raise FileNotFoundError(
        f"Encoder config.json is missing: "
        f"{encoder_directory}"
    )

encoder_path = str(encoder_directory)
```

This prevents the same silent conversion from an empty string to the project root.

# Relaunch safely

Use explicit shell validation before starting four workers:

```bash
cd /data/saral/wdir/smart || exit 1
source ./smart_env.sh

export CUDA_VISIBLE_DEVICES=0,1,2,3
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

GTE_PATH_FILE=/mnt/warm_storage/saral/smart/artifacts/prompt_embeddings/gte-large/encoder_snapshot_path.txt

if [[ ! -s "$GTE_PATH_FILE" ]]; then
    echo "ERROR: Missing or empty encoder path file: $GTE_PATH_FILE"
    exit 1
fi

GTE_PATH=$(cat "$GTE_PATH_FILE")

if [[ -z "$GTE_PATH" ]]; then
    echo "ERROR: GTE_PATH is empty."
    exit 1
fi

if [[ ! -f "$GTE_PATH/modules.json" ]]; then
    echo "ERROR: Invalid Sentence Transformers directory: $GTE_PATH"
    echo "modules.json was not found."
    exit 1
fi

echo "Using encoder: $GTE_PATH"
```

Optionally remove only the failed-worker logs:

```bash
rm -f \
  /mnt/warm_storage/saral/smart/artifacts/prompt_embeddings/gte-large/worker_cuda_*.error.log
```

Then rerun:

```bash
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
```

The corrected startup should show an encoder path similar to:

```text
Encoder: /mnt/warm_storage/saral/smart/cache/huggingface/hub/models--thenlper--gte-large/snapshots/<revision>
```

and each worker should report:

```text
Encoder ready: dimension=1024, max_sequence_length=512
```

followed by task-level `START` and `DONE` messages.
