"""MLX backend for Apple silicon: generation + LoRA training (development
only; recorded numbers come from the HF/PEFT backend on Modal, and the two
are never mixed in one comparison).

Install:  pip install mlx-lm
Models:   mlx-community/Qwen3-4B-bf16   (~8 GB, primary precision)
          mlx-community/Qwen3-4B-4bit   (~2.5 GB, fast iteration)

Qwen3 thinking is disabled via the chat template so all reasoning is in
visible, countable tokens (PREREG §3). LoRA: r=16 on all attention and MLP
projections of every layer (PREREG §3); batch size 1 per step; only
completion tokens are under the loss.
"""

from __future__ import annotations

import time
from typing import Sequence

from backends import training_utils as tu
from backends.base import GenResult, TrainStats


class MLXBackend:
    def __init__(self, model_id: str = "mlx-community/Qwen3-4B-bf16", system: str | None = None,
                 lora: bool = False, lr: float = tu.DEFAULT_LR, lora_scale: float = tu.DEFAULT_LORA_SCALE,
                 seed: int = 0):
        from mlx_lm import load  # imported lazily so the rest of the repo runs without mlx
        self.name = f"mlx:{model_id}"
        self.model, self.tokenizer = load(model_id)
        self.system = system
        self.lr = lr
        self.lora_scale = lora_scale
        self.optimizer = None
        self.has_lora = False
        self.seed = seed
        self.set_seed(seed)            # A1: LoRA init and every sampler draw follow the run seed
        if lora:
            self.attach_lora()

    def set_seed(self, seed: int) -> None:
        import mlx.core as mx
        self.seed = seed
        mx.random.seed(seed)

    # ------------------------------------------------------------------ generation
    def _chat(self, prompt: str) -> str:
        messages = []
        if self.system:
            messages.append({"role": "system", "content": self.system})
        messages.append({"role": "user", "content": prompt})
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )

    def generate(self, prompts: Sequence[str], max_tokens: int = 128,
                 temperature: float = 0.0) -> list[GenResult]:
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler

        sampler = make_sampler(temp=temperature)
        self.model.eval()
        out = []
        for p in prompts:
            chat = self._chat(p)
            text = generate(self.model, self.tokenizer, prompt=chat,
                            max_tokens=max_tokens, sampler=sampler, verbose=False)
            out.append(GenResult(text=text, prompt_tokens=self.count_tokens(chat),
                                 completion_tokens=self.count_tokens(text)))
        return out

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    # ------------------------------------------------------------------ training
    def attach_lora(self) -> None:
        import mlx.optimizers as optim
        from mlx_lm.tuner.utils import linear_to_lora_layers

        self.model.freeze()
        linear_to_lora_layers(self.model, num_layers=len(self.model.layers),
                              config={"rank": tu.LORA_RANK, "scale": self.lora_scale, "dropout": 0.0,
                                      "keys": set(tu.LORA_TARGET_KEYS)})
        self.optimizer = optim.Adam(learning_rate=self.lr)
        self.has_lora = True
        self.model.eval()

    def _lora_modules(self):
        from mlx_lm.tuner.lora import LoRALinear
        return [(k, m) for k, m in self.model.named_modules() if isinstance(m, LoRALinear)]

    def _encode_example(self, prompt: str, completion: str):
        import mlx.core as mx
        chat = self._chat(prompt)
        p_ids = self.tokenizer.encode(chat)
        eos = self.tokenizer.eos_token_id
        c_ids = self.tokenizer.encode(completion.strip()) + ([eos] if eos is not None else [])
        ids = p_ids + c_ids
        lengths = tu.lengths_row(len(p_ids), len(ids))
        return mx.array([ids]), mx.array([lengths]), tu.supervised_tokens(len(p_ids), len(ids))

    def train(self, examples: Sequence[tuple[str, str]], steps: int, seed: int) -> TrainStats:
        import mlx.core as mx
        import mlx.nn as nn
        from mlx_lm.tuner.trainer import default_loss

        if not self.has_lora:
            raise RuntimeError("attach_lora() first (construct with lora=True)")
        order = tu.cycle_examples(list(examples), steps, seed)
        if not order:
            return TrainStats(steps=0, training_tokens=0)
        loss_and_grad = nn.value_and_grad(self.model, default_loss)
        self.model.train()
        t0 = time.time()
        tokens, losses = 0, []
        for prompt, completion in order:
            batch, lengths, ntok = self._encode_example(prompt, completion)
            (loss, _), grads = loss_and_grad(self.model, batch, lengths)
            self.optimizer.update(self.model, grads)
            mx.eval(self.model.parameters(), self.optimizer.state, loss)
            tokens += ntok
            losses.append(loss.item())
        self.model.eval()
        mx.clear_cache()
        return TrainStats(steps=len(order), training_tokens=tokens, seconds=time.time() - t0,
                          loss=sum(losses) / len(losses))

    def decay_adapter(self, factor: float) -> None:
        """W_lora *= (1 - factor): shrink lora_b so the delta a@b scales once."""
        import mlx.core as mx
        if not 0 <= factor < 1:
            raise ValueError("factor must be in [0, 1)")
        for _, m in self._lora_modules():
            m.lora_b = m.lora_b * (1.0 - factor)
        mx.eval(self.model.parameters())

    def reset_adapter(self) -> None:
        import math
        import mlx.core as mx
        import mlx.optimizers as optim
        mx.random.seed(self.seed)
        for _, m in self._lora_modules():
            in_dims, r = m.lora_a.shape
            s = 1 / math.sqrt(in_dims)
            m.lora_a = mx.random.uniform(low=-s, high=s, shape=(in_dims, r))
            m.lora_b = mx.zeros(m.lora_b.shape)
        self.optimizer = optim.Adam(learning_rate=self.lr)
        mx.eval(self.model.parameters())

    def save_adapter(self, path: str) -> None:
        import mlx.core as mx
        from mlx.utils import tree_flatten
        mx.save_safetensors(path, dict(tree_flatten(self.model.trainable_parameters())))

    def lora_norm(self) -> float:
        """Diagnostic: L2 norm of all lora_b (0 at init; grows with training)."""
        import mlx.core as mx
        return float(sum(mx.sum(m.lora_b.astype(mx.float32) ** 2) for _, m in self._lora_modules()).item() ** 0.5)
