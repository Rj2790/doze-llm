"""Model-free helpers shared by the trainable backends (MLX, HF/PEFT).

Loss masking convention: a training sequence is prompt_ids + completion_ids.
Only completion tokens are under the loss. With mlx_lm's default_loss the
mask is `prompt_len <= step <= total_len` over 1-based target positions, so
`lengths_row(prompt_len, total_len)` gives exactly `total_len - prompt_len`
supervised tokens. HF uses label = -100 on the prompt part instead
(`labels_for`). Both count the same training tokens.
"""

from __future__ import annotations

from collections.abc import Sequence

# PREREG §3: LoRA r=16 on all attention and MLP projections.
LORA_RANK = 16
LORA_TARGET_KEYS = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                    "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
LORA_TARGET_MODULES = tuple(k.split(".")[-1] for k in LORA_TARGET_KEYS)   # PEFT style
DEFAULT_LR = 1e-4          # tunable
LORA_ALPHA = 32            # PEFT: scaling = alpha / r = 2.0
DEFAULT_LORA_SCALE = LORA_ALPHA / LORA_RANK   # same scaling on MLX (mlx_lm default is 20; numerics still differ)


def lengths_row(prompt_len: int, total_len: int) -> list[int]:
    if not 0 < prompt_len < total_len:
        raise ValueError(f"need 0 < prompt_len < total_len, got {prompt_len}, {total_len}")
    return [prompt_len, total_len]


def supervised_tokens(prompt_len: int, total_len: int) -> int:
    lengths_row(prompt_len, total_len)
    return total_len - prompt_len


def labels_for(prompt_ids: Sequence[int], completion_ids: Sequence[int], ignore_index: int = -100) -> list[int]:
    if not prompt_ids or not completion_ids:
        raise ValueError("prompt and completion must be non-empty")
    return [ignore_index] * len(prompt_ids) + list(completion_ids)


def cycle_examples(examples: Sequence, steps: int, seed: int) -> list:
    """Order of examples over `steps` single-example steps: a seeded shuffle
    of the set, repeated as needed. Deterministic per (examples, seed)."""
    import random
    if not examples or steps <= 0:
        return []
    order = list(range(len(examples)))
    rng = random.Random(seed)
    out: list = []
    while len(out) < steps:
        rng.shuffle(order)
        out.extend(examples[i] for i in order)
    return out[:steps]
