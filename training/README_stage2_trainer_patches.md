I found a **real learning-rate scheduler error** in the production setup we used, plus a LoRA configuration mismatch.

## 1. LoRA configuration actually used

The completed LoRA runs used:

```text
r              = 64
alpha          = 16
dropout        = 0.05
target_modules = q_proj,k_proj,v_proj,o_proj,
                 gate_proj,up_proj,down_proj,lm_head
bias           = none
```

They did **not** use `alpha=32`.

Both the original SMART repository and the launcher I provided defaulted or explicitly passed:

```bash
--peft_lora_alpha 16
```

Your reported trainable parameter counts confirm that `lm_head` was included:

```text
Llama2: 162,217,984
Qwen2:  171,442,176
```

With the same seven transformer projection targets but without `lm_head`, the expected counts would be:

```text
Llama2: 159,907,840
Qwen2:  161,480,704
```

Check the saved adapters directly:

```bash
python3 - <<'PY'
import json
from pathlib import Path

root = Path("/mnt/warm_storage/saral/smart/models")

for path in sorted(root.glob("*lora*/adapter_config.json")):
    config = json.loads(path.read_text(encoding="utf-8"))

    print()
    print(path.parent.name)
    print("  r:             ", config.get("r"))
    print("  lora_alpha:    ", config.get("lora_alpha"))
    print("  lora_dropout:  ", config.get("lora_dropout"))
    print(
        "  target_modules:",
        sorted(config.get("target_modules") or []),
    )
    print("  bias:          ", config.get("bias"))
PY
```

## 2. Scheduler error affecting every completed checkpoint

The trainer creates a cosine scheduler using:

```python
num_training_steps=args.max_train_steps
```

Our production launcher explicitly supplied:

```text
25K: max_train_steps=391
50K: max_train_steps=782
```

However, Accelerate's scheduler wrapper advances the underlying scheduler **once per process** when `split_batches=False`. With four GPUs, each optimizer update advanced the cosine scheduler four times. ([GitHub][1])

Therefore, the effective schedules were approximately:

```text
25K:
configured horizon       = 391 scheduler ticks
scheduler ticks/update   = 4
horizon reached near     = optimizer step 98

50K:
configured horizon       = 782 scheduler ticks
scheduler ticks/update   = 4
horizon reached near     = optimizer step 196
```

The Transformers cosine function is not clamped after its configured horizon. Past the endpoint, its cosine expression can increase again, creating unintended oscillations rather than one cosine decay across the epoch. ([GitHub][2])

The paper calls for one cosine decay over one training epoch, with 1% warmup. 

This means:

* all four full-fine-tuning checkpoints have the wrong LR trajectory;
* all four LoRA checkpoints have the wrong LR trajectory;
* the LoRA checkpoints additionally have `alpha=16`, not `32`.

The one-step smoke tests could not reveal this because a one-step scheduler has no meaningful full curve.

### Confirm it from the logs

After warmup, a correct half-cosine schedule must decrease monotonically. This audit reports LR increases after the first logged production step:

```bash
python3 - <<'PY'
from pathlib import Path
import re

root = Path(
    "/mnt/warm_storage/saral/smart/"
    "logs/production_training"
)

pattern = re.compile(
    r"optimizer_step=(\d+).*?"
    r"lr=([0-9.eE+-]+)"
)

for log_path in sorted(root.glob("*/training.log")):
    points = []

    for line in log_path.read_text(
        encoding="utf-8",
        errors="replace",
    ).splitlines():
        match = pattern.search(line)

        if match:
            points.append(
                (
                    int(match.group(1)),
                    float(match.group(2)),
                )
            )

    if not points:
        continue

    increases = []

    for previous, current in zip(
        points,
        points[1:],
    ):
        if current[1] > previous[1] * (1.0 + 1e-8):
            increases.append(
                (
                    previous[0],
                    previous[1],
                    current[0],
                    current[1],
                )
            )

    print()
    print(log_path.parent.name)
    print("  logged points:       ", len(points))
    print("  first point:         ", points[0])
    print("  final point:         ", points[-1])
    print("  post-warmup increases:", len(increases))

    if increases:
        print("  first increase:      ", increases[0])
PY
```

## 3. Correct LoRA targets

There are two defensible definitions.

### Recommended LoRA extension

Use LoRA on all transformer attention and MLP projection layers:

```text
q_proj
k_proj
v_proj
o_proj
gate_proj
up_proj
down_proj
```

Official PEFT/TRL guidance identifies these seven modules as the all-transformer-linear configuration. `lm_head` is an additional output projection, not part of the transformer blocks. ([GitHub][3])

I recommend freezing:

```text
LORA_R       = 64
LORA_ALPHA   = 32
LORA_DROPOUT = 0.05

TARGETS =
q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
```

This is also more comparable across Llama2 and Qwen2. Qwen's much larger vocabulary makes an `lm_head` adapter substantially larger.

### Repository-default PEFT targets

The released SMART code includes:

```text
q_proj,k_proj,v_proj,o_proj,
gate_proj,up_proj,down_proj,lm_head
```

That is valid, but SMART's paper did not report LoRA experiments, so this is not an author-comparable experimental setting. The author-comparable track is full fine-tuning.

## 4. Patch the scheduler correctly

The clean solution is to stop Accelerate from automatically stepping the scheduler with its process multiplier, then step it exactly once per synchronized optimizer update.

Back up and patch the local trainer:

```bash
cd /data/saral/wdir/smart || exit 1

cp \
  instruction_tuner_local.py \
  instruction_tuner_local.before_scheduler_fix.py

python3 - <<'PY'
from pathlib import Path
import re

path = Path("instruction_tuner_local.py")
text = path.read_text(encoding="utf-8")

# Require explicit optimizer-step counts. Our production launcher
# supplies 391 for 25K and 782 for 50K.
needle = "    args=parse_args()\n"

replacement = """    args=parse_args()

    if args.max_train_steps is None:
        raise ValueError(
            "The corrected local trainer requires an explicit "
            "--max_train_steps value."
        )
"""

if replacement not in text:
    if needle not in text:
        raise RuntimeError(
            "Could not find args=parse_args()."
        )

    text = text.replace(
        needle,
        replacement,
        1,
    )

old_accelerator = (
    "    accelerator=Accelerator("
    "gradient_accumulation_steps="
    "args.gradient_accumulation_steps, "
    "**accelerator_log_kwargs)\n"
)

new_accelerator = """    accelerator=Accelerator(
        gradient_accumulation_steps=(
            args.gradient_accumulation_steps
        ),
        # The scheduler is stepped manually once per synchronized
        # optimizer update. This prevents Accelerate from advancing
        # it once per distributed process.
        step_scheduler_with_optimizer=False,
        **accelerator_log_kwargs,
    )
"""

if "step_scheduler_with_optimizer=False" not in text:
    if old_accelerator not in text:
        raise RuntimeError(
            "Could not find Accelerator construction."
        )

    text = text.replace(
        old_accelerator,
        new_accelerator,
        1,
    )

old_optimizer_block = """                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
"""

new_optimizer_block = """                    optimizer.step()
                    optimizer.zero_grad()
"""

if old_optimizer_block in text:
    text = text.replace(
        old_optimizer_block,
        new_optimizer_block,
        1,
    )
elif new_optimizer_block not in text:
    raise RuntimeError(
        "Could not patch scheduler step location."
    )

old_sync_block = """                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    completed_steps += 1
"""

new_sync_block = """                if accelerator.sync_gradients:
                    # AcceleratedScheduler was constructed with
                    # step_scheduler_with_optimizer=False, so this
                    # advances the underlying scheduler exactly once.
                    if not accelerator.optimizer_step_was_skipped:
                        lr_scheduler.step()

                    progress_bar.update(1)
                    completed_steps += 1
"""

if new_sync_block not in text:
    if old_sync_block not in text:
        raise RuntimeError(
            "Could not find synchronized-step block."
        )

    text = text.replace(
        old_sync_block,
        new_sync_block,
        1,
    )

# Add a final invariant: one scheduler tick per completed
# optimizer update.
tracking_pattern = re.compile(
    r"""(
    if args\.with_tracking:
        accelerator\.end_training\(\)
\s*
)(    if args\.output_dir is not None:)""",
    re.MULTILINE,
)

scheduler_check = """\\1    scheduler_core = getattr(
        lr_scheduler,
        "scheduler",
        lr_scheduler,
    )
    scheduler_last_epoch = int(
        scheduler_core.last_epoch
    )

    if scheduler_last_epoch != completed_steps:
        raise RuntimeError(
            "Scheduler/optimizer step mismatch: "
            f"scheduler={scheduler_last_epoch}, "
            f"optimizer={completed_steps}."
        )

\\2"""

if "Scheduler/optimizer step mismatch" not in text:
    text, count = tracking_pattern.subn(
        scheduler_check,
        text,
        count=1,
    )

    if count != 1:
        raise RuntimeError(
            "Could not insert final scheduler invariant."
        )

path.write_text(
    text,
    encoding="utf-8",
)

print("Corrected scheduler stepping.")
PY

python3 -m py_compile \
  instruction_tuner_local.py

grep -nE \
  "step_scheduler_with_optimizer|lr_scheduler.step|Scheduler/optimizer" \
  instruction_tuner_local.py
```

The corrected sequence is:

```text
forward/backward on each microbatch
optimizer update every 16 microbatches
scheduler update once per optimizer update
```

## 5. Patch LoRA settings

Create corrected copies rather than modifying provenance for the old runs:

```bash
cd /data/saral/wdir/smart || exit 1

cp \
  run_one_production_training.sh \
  run_one_production_training_v2.sh

cp \
  run_training_smokes.sh \
  run_training_smokes_v2.sh

python3 - <<'PY'
from pathlib import Path

paths = [
    Path("run_one_production_training_v2.sh"),
    Path("run_training_smokes_v2.sh"),
]

for path in paths:
    text = path.read_text(encoding="utf-8")

    text = text.replace(
        "--peft_lora_alpha 16",
        "--peft_lora_alpha 32",
    )

    text = text.replace(
        "q_proj,k_proj,v_proj,o_proj,"
        "gate_proj,up_proj,down_proj,lm_head",
        "q_proj,k_proj,v_proj,o_proj,"
        "gate_proj,up_proj,down_proj",
    )

    path.write_text(
        text,
        encoding="utf-8",
    )

    print("Patched:", path)
PY

bash -n run_one_production_training_v2.sh
bash -n run_training_smokes_v2.sh

grep -nE \
  "peft_lora_(r|alpha|dropout)|peft_target_modules" \
  run_one_production_training_v2.sh \
  run_training_smokes_v2.sh
```

Also change the output run naming in the V2 production launcher to prevent collision:

```bash
python3 - <<'PY'
from pathlib import Path

path = Path(
    "run_one_production_training_v2.sh"
)

text = path.read_text(encoding="utf-8")

old = '''RUN_NAME="${MODEL_LABEL}_smart_${BUDGET}_${MODE}_seed${SEED}"
'''

new = '''if [[ "$MODE" == "full" ]]; then
    RUN_TAG="scheduler_fixed"
else
    RUN_TAG="scheduler_fixed_r64_a32_proj7"
fi

RUN_NAME="${MODEL_LABEL}_smart_${BUDGET}_${MODE}_${RUN_TAG}_seed${SEED}"
'''

if old not in text:
    raise RuntimeError(
        "Could not find production RUN_NAME."
    )

text = text.replace(old, new, 1)

path.write_text(
    text,
    encoding="utf-8",
)

print("V2 run names patched.")
PY

bash -n run_one_production_training_v2.sh
```

## 6. Required action

The scientifically clean path is:

1. Keep the existing checkpoints as pilot artifacts.
2. Do not use them for final leaderboard reporting.
3. Apply the scheduler correction.
4. Freeze LoRA as `r=64`, `alpha=32`, dropout `0.05`, projection-seven targets.
5. Rerun the four smoke tests.
6. Retrain all eight experiments from the original base checkpoints.

After the corrected LoRA smoke tests, expected trainable counts are:

```text
Llama2 LoRA: 159,907,840
Qwen2 LoRA:  161,480,704
```

The full-fine-tuning counts remain:

```text
Llama2 full: 6,738,415,616
Qwen2 full:  7,615,616,512
```

No SMART data generation, Stage 1, Stage 2, materialization, or tokenization work needs to be repeated. Only model training and subsequent leaderboard evaluation need rerunning.

[1]: https://raw.githubusercontent.com/huggingface/accelerate/main/src/accelerate/scheduler.py "raw.githubusercontent.com"
[2]: https://github.com/huggingface/transformers/blob/main/src/transformers/optimization.py?utm_source=chatgpt.com "transformers/src/transformers/optimization.py at main · huggingface/transformers · GitHub"
[3]: https://github.com/huggingface/peft/blob/main/docs/source/developer_guides/lora.md?utm_source=chatgpt.com "peft/docs/source/developer_guides/lora.md at main · huggingface/peft · GitHub"
