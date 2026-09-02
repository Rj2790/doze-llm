"""Model-free backend for dry runs and harness tests.

`policy` is 'literal' (follows the rule exactly, guesses on probes) or
`shortcut` (uses r_N == r_{N-5}). `noise` flips a fraction of answers so
accuracy is not exactly 0 or 1, which exercises the reporting code.
"""

from __future__ import annotations

import random
from typing import Sequence

from backends.base import GenResult, TrainStats
from tasks import number_reduction as nr


class ScriptedBackend:
    def __init__(self, policy: str = "literal", noise: float = 0.0, seed: int = 0):
        assert policy in ("literal", "shortcut")
        self.name = f"scripted-{policy}"
        self.policy = policy
        self.noise = noise
        self.rng = random.Random(seed)

    @staticmethod
    def _digits(prompt: str) -> list[str]:
        return nr.parse_digits_line(prompt)

    def _answer(self, toks: list[str]) -> str:
        L = len(toks)
        visible = "".join(t for t in toks if t != nr.MASK)
        if self.policy == "shortcut":
            return nr.solve(visible[: nr.prefix_len(L)])[-1]
        if nr.MASK in toks:
            return self.rng.choice(nr.DIGITS)
        return nr.solve(visible)[-1]

    def set_seed(self, seed: int) -> None:
        self.rng = random.Random(seed)

    def generate(self, prompts: Sequence[str], max_tokens: int = 128,
                 temperature: float = 0.0) -> list[GenResult]:
        out = []
        for p in prompts:
            toks = self._digits(p)
            ans = self._answer(toks)
            if self.rng.random() < self.noise:
                ans = self.rng.choice([d for d in nr.DIGITS if d != ans])
            if "STEPS:" in p and nr.MASK not in toks and self.policy == "literal":
                inst = nr.Instance.from_digits("".join(toks))
                mode = "work" if "previous,digit->result" in p else "full"
                text = nr.gold_response(inst, mode)
                if ans != inst.answer:   # noise flips the answer line only
                    text = text.rsplit("ANSWER:", 1)[0] + f"ANSWER: {ans}"
            else:
                text = f"ANSWER: {ans}"
            out.append(GenResult(text=text, prompt_tokens=self.count_tokens(p),
                                 completion_tokens=self.count_tokens(text)))
        return out

    def count_tokens(self, text: str) -> int:
        return len(text.split())


class FakeTrainableBackend(ScriptedBackend):
    """Scripted backend with a fake LoRA: records what it was trained on,
    counts steps/tokens, and (optionally) 'learns' the shortcut once it has
    taken `learn_after_steps` gradient steps. Dream prompts get a valid
    variation of the seed puzzle (last five digits perturbed so the prefix is
    unchanged); a fraction `dream_error_rate` of dreams carry a wrong step."""

    DREAM_MARKER = "Invent a NEW puzzle"

    def __init__(self, learn_after_steps: int | None = None, dream_error_rate: float = 0.0, seed: int = 0,
                 noise: float = 0.0):
        super().__init__(policy="literal", noise=noise, seed=seed)
        self.name = "scripted-fake-trainable"
        self.learn_after_steps = learn_after_steps
        self.dream_error_rate = dream_error_rate
        self.trained: list[tuple[str, str]] = []
        self.gradient_steps = 0
        self.decays = 0

    def _dream(self, prompt: str) -> str:
        seed = list(nr.parse_digits_line(prompt))
        L = len(seed)
        i = self.rng.randrange(nr.prefix_len(L), L)
        seed[i] = self.rng.choice([d for d in nr.DIGITS if d != seed[i]])
        inst = nr.Instance.from_digits("".join(seed))
        body = nr.gold_response(inst, "work")
        if self.rng.random() < self.dream_error_rate:
            lines = body.splitlines()
            j = self.rng.randrange(0, L - 1)
            pair, res = lines[j].split("->")
            lines[j] = f"{pair}->{[d for d in nr.DIGITS if d != res][0]}"
            body = "\n".join(lines)
        return f"{nr.digits_line(inst.digits)}\n{body}"

    def generate(self, prompts: Sequence[str], max_tokens: int = 128,
                 temperature: float = 0.0) -> list[GenResult]:
        out = []
        for p in prompts:
            if self.DREAM_MARKER in p:
                t = self._dream(p)
                out.append(GenResult(text=t, prompt_tokens=self.count_tokens(p),
                                     completion_tokens=self.count_tokens(t)))
            else:
                out.extend(super().generate([p], max_tokens=max_tokens, temperature=temperature))
        return out

    def train(self, examples: Sequence[tuple[str, str]], steps: int, seed: int) -> TrainStats:
        if not examples or steps <= 0:
            return TrainStats(steps=0, training_tokens=0)
        tokens = 0
        for k in range(steps):
            prompt, completion = examples[k % len(examples)]
            self.trained.append((prompt, completion))
            tokens += self.count_tokens(completion)
        self.gradient_steps += steps
        if self.learn_after_steps is not None and self.gradient_steps >= self.learn_after_steps:
            self.policy = "shortcut"
        return TrainStats(steps=steps, training_tokens=tokens)

    def decay_adapter(self, factor: float) -> None:
        self.decays += 1

    def reset_adapter(self) -> None:
        self.policy = "literal"
        self.gradient_steps = 0

    def save_adapter(self, path: str) -> None:
        pass
