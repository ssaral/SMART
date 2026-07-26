The SMART data-generation pipeline when complete and verified -- we can proceed. Before tokenization or training, we need to inspect the local Llama tokenizer and the authors’ released training code so that we do not guess the prompt serialization, BOS/EOS handling, or loss masking.

## Step 17 — Audit the Llama training interface

This step does **not** load model weights or start training.

### 17.1 Locate the released training files

```bash
cd /data/saral/wdir/smart_v2 || exit 1
source ./smart_v2_env.sh

find /data/saral/wdir/smart_v2 \
  -maxdepth 5 \
  -type f \
  \( -name '*.py' -o -name '*.sh' -o -name '*.yaml' -o -name '*.json' \) \
  | sort \
  | grep -Ei \
    'train|finetun|sft|supervised|launcher|deepspeed|accelerate|dataset|collator' \
  | tee \
    /mnt/warm_storage/saral/smart_v2/logs/training_code_inventory.txt
```

Also search for the fields and formatting operations used by the trainer:

```bash
grep -RInE \
  'inputs|targets|messages|chat_template|apply_chat_template|response_template|completion|labels|IGNORE_INDEX|mask|DataCollator|SFTTrainer|Trainer' \
  /data/saral/wdir/smart_v2 \
  --include='*.py' \
  --include='*.sh' \
  2>/dev/null \
  | tee \
    /mnt/warm_storage/saral/smart_v2/logs/training_format_search.txt
```

### 17.2 Create the local Llama tokenizer/config audit

```bash
mkdir -p src/training

cat > src/training/audit_local_model.py <<'PY'
"""Inspect a local Transformers model without loading its weights."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import transformers
from transformers import AutoConfig, AutoTokenizer


FILES_TO_HASH = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    if value is None:
        return None

    if isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, dict):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            json_safe(item)
            for item in value
        ]

    return str(value)


def main() -> int:
    args = parse_args()

    model_path = args.model_path.resolve()
    output_path = args.output.resolve()

    if not (model_path / "config.json").is_file():
        raise FileNotFoundError(
            f"Missing config.json under {model_path}"
        )

    config = AutoConfig.from_pretrained(
        str(model_path),
        local_files_only=True,
        trust_remote_code=False,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        local_files_only=True,
        trust_remote_code=False,
        use_fast=True,
    )

    chat_template = getattr(
        tokenizer,
        "chat_template",
        None,
    )

    file_hashes = {}

    for filename in FILES_TO_HASH:
        path = model_path / filename

        if path.is_file():
            file_hashes[filename] = {
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }

    report = {
        "status": "complete",
        "model_path": str(model_path),
        "transformers_version": transformers.__version__,
        "config": {
            "class": type(config).__name__,
            "model_type": getattr(
                config,
                "model_type",
                None,
            ),
            "architectures": getattr(
                config,
                "architectures",
                None,
            ),
            "vocab_size": getattr(
                config,
                "vocab_size",
                None,
            ),
            "hidden_size": getattr(
                config,
                "hidden_size",
                None,
            ),
            "intermediate_size": getattr(
                config,
                "intermediate_size",
                None,
            ),
            "num_hidden_layers": getattr(
                config,
                "num_hidden_layers",
                None,
            ),
            "num_attention_heads": getattr(
                config,
                "num_attention_heads",
                None,
            ),
            "num_key_value_heads": getattr(
                config,
                "num_key_value_heads",
                None,
            ),
            "max_position_embeddings": getattr(
                config,
                "max_position_embeddings",
                None,
            ),
            "torch_dtype": json_safe(
                getattr(
                    config,
                    "torch_dtype",
                    None,
                )
            ),
            "rope_theta": getattr(
                config,
                "rope_theta",
                None,
            ),
            "rope_scaling": json_safe(
                getattr(
                    config,
                    "rope_scaling",
                    None,
                )
            ),
            "bos_token_id": getattr(
                config,
                "bos_token_id",
                None,
            ),
            "eos_token_id": getattr(
                config,
                "eos_token_id",
                None,
            ),
            "pad_token_id": getattr(
                config,
                "pad_token_id",
                None,
            ),
        },
        "tokenizer": {
            "class": type(tokenizer).__name__,
            "is_fast": tokenizer.is_fast,
            "length": len(tokenizer),
            "vocab_size": tokenizer.vocab_size,
            "model_max_length": tokenizer.model_max_length,
            "padding_side": tokenizer.padding_side,
            "truncation_side": tokenizer.truncation_side,
            "bos_token": tokenizer.bos_token,
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token": tokenizer.eos_token,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token": tokenizer.pad_token,
            "pad_token_id": tokenizer.pad_token_id,
            "unk_token": tokenizer.unk_token,
            "unk_token_id": tokenizer.unk_token_id,
            "add_bos_token": getattr(
                tokenizer,
                "add_bos_token",
                None,
            ),
            "add_eos_token": getattr(
                tokenizer,
                "add_eos_token",
                None,
            ),
            "special_tokens_map": json_safe(
                tokenizer.special_tokens_map
            ),
            "additional_special_tokens": json_safe(
                tokenizer.additional_special_tokens
            ),
            "has_chat_template": bool(
                chat_template
            ),
            "chat_template": chat_template,
        },
        "files": file_hashes,
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            report,
            handle,
            indent=2,
            ensure_ascii=False,
        )
        handle.write("\n")

    print("=== Local model audit ===")
    print(f"Path:                    {model_path}")
    print(f"Config class:            {type(config).__name__}")
    print(f"Model type:              {config.model_type}")
    print(f"Architectures:           {config.architectures}")
    print(
        "Max position embeddings:",
        getattr(
            config,
            "max_position_embeddings",
            None,
        ),
    )
    print(f"Tokenizer class:         {type(tokenizer).__name__}")
    print(f"Tokenizer length:        {len(tokenizer):,}")
    print(f"Tokenizer max length:    {tokenizer.model_max_length}")
    print(f"BOS:                     {tokenizer.bos_token!r} / {tokenizer.bos_token_id}")
    print(f"EOS:                     {tokenizer.eos_token!r} / {tokenizer.eos_token_id}")
    print(f"PAD:                     {tokenizer.pad_token!r} / {tokenizer.pad_token_id}")
    print(
        "Adds BOS automatically: ",
        getattr(
            tokenizer,
            "add_bos_token",
            None,
        ),
    )
    print(
        "Adds EOS automatically: ",
        getattr(
            tokenizer,
            "add_eos_token",
            None,
        ),
    )
    print(f"Has chat template:       {bool(chat_template)}")
    print(f"Report:                  {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
PY

python3 -m py_compile \
  src/training/audit_local_model.py
```

Run it only for Llama at this stage:

```bash
python3 -m src.training.audit_local_model \
  --model-path /data/saral/wdir/smart_v2/llama2_7b \
  --output /mnt/warm_storage/saral/smart_v2/manifests/llama2_7b_model_audit.json \
  2>&1 | tee \
  /mnt/warm_storage/saral/smart_v2/logs/audit_llama2_7b.log
```

### 17.3 Print the fields needed for the next decision

```bash
python3 - <<'PY'
import json
from pathlib import Path

path = Path(
    "/mnt/warm_storage/saral/smart_v2/"
    "manifests/llama2_7b_model_audit.json"
)

with path.open(encoding="utf-8") as handle:
    report = json.load(handle)

config = report["config"]
tokenizer = report["tokenizer"]

print("Model type:", config["model_type"])
print("Architectures:", config["architectures"])
print(
    "Max positions:",
    config["max_position_embeddings"],
)
print("Config dtype:", config["torch_dtype"])

print("\nTokenizer")
print("Class:", tokenizer["class"])
print("Length:", tokenizer["length"])
print(
    "Model max length:",
    tokenizer["model_max_length"],
)
print(
    "BOS:",
    repr(tokenizer["bos_token"]),
    tokenizer["bos_token_id"],
)
print(
    "EOS:",
    repr(tokenizer["eos_token"]),
    tokenizer["eos_token_id"],
)
print(
    "PAD:",
    repr(tokenizer["pad_token"]),
    tokenizer["pad_token_id"],
)
print(
    "add_bos_token:",
    tokenizer["add_bos_token"],
)
print(
    "add_eos_token:",
    tokenizer["add_eos_token"],
)
print(
    "has_chat_template:",
    tokenizer["has_chat_template"],
)

if tokenizer["has_chat_template"]:
    print("\nChat template:")
    print(tokenizer["chat_template"])
PY
```

The next step will use these outputs and the released trainer code to freeze exactly:

```text
serialized example = ?
BOS handling       = ?
EOS handling       = ?
pad token policy   = ?
loss on prompt     = included or masked
truncation policy  = prompt side / target side
maximum length     = 4096
packing            = enabled or disabled
```

We should not create the full-fine-tuning launcher until those six decisions are supported by the local tokenizer and authors’ training implementation.
