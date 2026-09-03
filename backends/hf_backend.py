"""Transformers + PEFT backend (CUDA). The backend for every recorded number
(CONTEXT.md §4: MLX and PEFT LoRA numerics differ; never mix them).

Same contract as backends/mlx_backend.py: chat template with thinking
disabled, greedy or sampled generation, LoRA r=16 / alpha=32 on all
attention and MLP projections, batch size 1 per step, loss on completion
tokens only, `decay_adapter` shrinks lora_B, `reset_adapter` re-inits.

NOT YET EXECUTED anywhere (no torch locally). First run must be the
Baseline arm for a few episodes on Modal (CONTEXT.md §8 step 6).
"""

from __future__ import annotations

import time
from typing import Sequence

from backends import training_utils as tu
from backends.base import GenResult, TrainStats


class HFBackend:
    def __init__(self, model_id: str = "Qwen/Qwen3-4B", system: str | None = None, lora: bool = False,
                 lr: float = tu.DEFAULT_LR, device: str = "cuda", dtype: str = "bfloat16", seed: int = 0):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = f"hf:{model_id}"
        self.seed = seed
        self.set_seed(seed)            # A1: PEFT init and sampling follow the run seed (seeded, not bit-identical on CUDA)
        self.system = system
        self.lr = lr
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"          # batched generation
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=getattr(torch, dtype)).to(device)
        if self.model.dtype != getattr(torch, dtype):    # B3: older transformers ignore dtype= and load fp32
            raise RuntimeError(f"model loaded as {self.model.dtype}, expected {dtype}")
        self.model.eval()
        self.batch_size = 16
        self.optimizer = None
        self.has_lora = False
        if lora:
            self.attach_lora()

    def set_seed(self, seed: int) -> None:
        import random
        import torch
        self.seed = seed
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # ------------------------------------------------------------------ generation
    def _chat(self, prompt: str) -> str:
        messages = []
        if self.system:
            messages.append({"role": "system", "content": self.system})
        messages.append({"role": "user", "content": prompt})
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                  enable_thinking=False)

    def generate(self, prompts: Sequence[str], max_tokens: int = 128,
                 temperature: float = 0.0) -> list[GenResult]:
        """Left-padded batches of `batch_size` (B1). Sampling is pure
        temperature: top_k=0, top_p=1.0 (C5), matching the MLX backend."""
        import torch
        self.model.eval()
        kw = dict(max_new_tokens=max_tokens, pad_token_id=self.tokenizer.pad_token_id)
        if temperature > 0:
            kw.update(do_sample=True, temperature=temperature, top_p=1.0, top_k=0)
        else:
            kw.update(do_sample=False, temperature=None, top_p=None, top_k=None)
        out = []
        for i in range(0, len(prompts), self.batch_size):
            chats = [self._chat(p) for p in prompts[i:i + self.batch_size]]
            enc = self.tokenizer(chats, return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                gen = self.model.generate(**enc, **kw)
            plen = enc["input_ids"].shape[1]
            for row, mask, chat in zip(gen, enc["attention_mask"], chats):
                new = row[plen:]
                eos = (new == self.tokenizer.eos_token_id).nonzero()
                new = new[: int(eos[0].item()) + 1] if len(eos) else new
                new = new[new != self.tokenizer.pad_token_id]
                text = self.tokenizer.decode(new, skip_special_tokens=True)
                out.append(GenResult(text=text, prompt_tokens=int(mask.sum().item()),
                                     completion_tokens=int(new.numel())))
        return out

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    # ------------------------------------------------------------------ training
    def attach_lora(self) -> None:
        import torch
        from peft import LoraConfig, get_peft_model

        cfg = LoraConfig(r=tu.LORA_RANK, lora_alpha=tu.LORA_ALPHA, lora_dropout=0.0, bias="none",
                         target_modules=list(tu.LORA_TARGET_MODULES), task_type="CAUSAL_LM")
        self.model = get_peft_model(self.model, cfg)
        self.optimizer = torch.optim.Adam([p for p in self.model.parameters() if p.requires_grad], lr=self.lr)
        self.has_lora = True
        self.model.eval()

    def _encode_example(self, prompt: str, completion: str):
        import torch
        p_ids = self.tokenizer.encode(self._chat(prompt))
        c_ids = self.tokenizer.encode(completion.strip(), add_special_tokens=False) + [self.tokenizer.eos_token_id]
        labels = tu.labels_for(p_ids, c_ids)
        ids = torch.tensor([p_ids + c_ids], device=self.device)
        lab = torch.tensor([labels], device=self.device)
        return ids, lab, tu.supervised_tokens(len(p_ids), len(p_ids) + len(c_ids))

    def train(self, examples: Sequence[tuple[str, str]], steps: int, seed: int) -> TrainStats:
        if not self.has_lora:
            raise RuntimeError("attach_lora() first (construct with lora=True)")
        order = tu.cycle_examples(list(examples), steps, seed)
        if not order:
            return TrainStats(steps=0, training_tokens=0)
        self.model.train()
        t0 = time.time()
        tokens, losses = 0, []
        for prompt, completion in order:
            ids, labels, ntok = self._encode_example(prompt, completion)
            loss = self.model(input_ids=ids, labels=labels).loss
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            tokens += ntok
            losses.append(float(loss.item()))
        self.model.eval()
        return TrainStats(steps=len(order), training_tokens=tokens, seconds=time.time() - t0,
                          loss=sum(losses) / len(losses), losses=losses)

    def decay_adapter(self, factor: float) -> None:
        import torch
        if not 0 <= factor < 1:
            raise ValueError("factor must be in [0, 1)")
        with torch.no_grad():
            for name, p in self.model.named_parameters():
                if "lora_B" in name:
                    p.mul_(1.0 - factor)

    def reset_adapter(self) -> None:
        import math
        import torch
        torch.manual_seed(self.seed)
        with torch.no_grad():
            for name, p in self.model.named_parameters():
                if "lora_A" in name:
                    torch.nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                elif "lora_B" in name:
                    p.zero_()
        self.optimizer = torch.optim.Adam([p for p in self.model.parameters() if p.requires_grad], lr=self.lr)

    def save_adapter(self, path: str) -> None:
        self.model.save_pretrained(path)

    def save_state(self, sdir) -> None:
        """Adapter params + optimizer + RNG states, for resume after preemption."""
        import random
        import torch
        from pathlib import Path
        st = {"trainable": {n: p.detach().cpu() for n, p in self.model.named_parameters() if p.requires_grad},
              "optimizer": self.optimizer.state_dict() if self.optimizer is not None else None,
              "torch_rng": torch.get_rng_state(), "py_random": random.getstate(), "seed": self.seed}
        if torch.cuda.is_available():
            st["cuda_rng"] = torch.cuda.get_rng_state_all()
        tmp = Path(sdir, "backend.pt.tmp")
        torch.save(st, tmp)
        tmp.replace(Path(sdir, "backend.pt"))

    def load_state(self, sdir) -> None:
        import random
        import torch
        from pathlib import Path
        st = torch.load(Path(sdir, "backend.pt"), map_location="cpu", weights_only=False)
        with torch.no_grad():
            params = dict(self.model.named_parameters())
            for n, v in st["trainable"].items():
                params[n].copy_(v.to(params[n].device))
        if st["optimizer"] is not None and self.optimizer is not None:
            self.optimizer.load_state_dict(st["optimizer"])
        torch.set_rng_state(st["torch_rng"])
        if "cuda_rng" in st and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(st["cuda_rng"])
        random.setstate(st["py_random"])

    def describe(self) -> dict:
        trainable = [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]
        return {"backend": self.name, "seed": self.seed, "lora": self.has_lora,
                "model_dtype": str(self.model.dtype),
                "adapter_dtypes": sorted({str(p.dtype) for _, p in trainable}),
                "trainable_params": int(sum(p.numel() for _, p in trainable)),
                "n_layers": int(getattr(self.model.config, "num_hidden_layers", 0))}

    def lora_norm(self) -> float:
        import torch
        with torch.no_grad():
            return float(torch.sqrt(sum((p.float() ** 2).sum() for n, p in self.model.named_parameters()
                                        if "lora_B" in n)).item())
