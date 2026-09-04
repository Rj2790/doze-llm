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

    def _state(self) -> dict:
        return {"policy": self.policy, "noise": self.noise, "rng": self.rng.getstate()}

    def _load(self, d: dict) -> None:
        self.policy, self.noise = d["policy"], d["noise"]
        st = d["rng"]
        self.rng.setstate((st[0], tuple(st[1]), st[2]))

    def save_state(self, sdir) -> None:
        import json
        from pathlib import Path
        Path(sdir, "backend.json").write_text(json.dumps(self._state()))

    def load_state(self, sdir) -> None:
        import json
        from pathlib import Path
        self._load(json.loads(Path(sdir, "backend.json").read_text()))

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

    DREAM_MARKER = "Write a NEW string"      # the digits-proposal prompt (sleep/dreamer.py PROPOSE_MARKER)

    def __init__(self, learn_after_steps: int | None = None, dream_error_rate: float = 0.0, seed: int = 0,
                 noise: float = 0.0):
        super().__init__(policy="literal", noise=noise, seed=seed)
        self.name = "scripted-fake-trainable"
        self.learn_after_steps = learn_after_steps
        self.dream_error_rate = dream_error_rate
        self.trained: list[tuple[str, str]] = []
        self.gradient_steps = 0
        self.decays = 0
        self.proposed: set[str] = set()      # digit strings this fake proposed as dreams

    def _dream(self, prompt: str) -> str:
        """Digits-only proposal: perturb one of the last five digits (prefix kept)."""
        seed = list(nr.parse_digits_line(prompt))
        L = len(seed)
        i = self.rng.randrange(nr.prefix_len(L), L)
        seed[i] = self.rng.choice([d for d in nr.DIGITS if d != seed[i]])
        digits = "".join(seed)
        self.proposed.add(digits)
        return nr.digits_line(digits)

    def _solve_dream(self, inst: "nr.Instance") -> str:
        """Solving a proposed string: gold, or (dream_error_rate) 3 wrong STEPS (fails C3)."""
        if self.rng.random() < self.dream_error_rate:
            steps = list(inst.responses)
            for j in self.rng.sample(range(inst.length - 1), 3):
                steps[j] = [d for d in nr.DIGITS if d != steps[j]][0]
            return f"{nr._work_lines(inst.digits)}\nSTEPS: {' '.join(steps)}\nANSWER: {inst.answer}"
        return nr.gold_response(inst, "work")

    def generate(self, prompts: Sequence[str], max_tokens: int = 128,
                 temperature: float = 0.0) -> list[GenResult]:
        out = []
        for p in prompts:
            if self.DREAM_MARKER in p:
                t = self._dream(p)
                out.append(GenResult(text=t, prompt_tokens=self.count_tokens(p), completion_tokens=self.count_tokens(t)))
                continue
            toks = self._digits(p)
            if nr.MASK not in toks and "".join(toks) in self.proposed and self.dream_error_rate > 0:
                t = self._solve_dream(nr.Instance.from_digits("".join(toks)))
                out.append(GenResult(text=t, prompt_tokens=self.count_tokens(p), completion_tokens=self.count_tokens(t)))
                continue
            out.extend(super().generate([p], max_tokens=max_tokens, temperature=temperature))
        return out

    def _state(self) -> dict:
        d = super()._state()
        d.update({"learn_after_steps": self.learn_after_steps, "dream_error_rate": self.dream_error_rate,
                  "trained": self.trained, "gradient_steps": self.gradient_steps, "decays": self.decays,
                  "proposed": sorted(self.proposed)})
        return d

    def _load(self, d: dict) -> None:
        super()._load(d)
        self.learn_after_steps, self.dream_error_rate = d["learn_after_steps"], d["dream_error_rate"]
        self.trained = [tuple(x) for x in d["trained"]]
        self.gradient_steps, self.decays = d["gradient_steps"], d["decays"]
        self.proposed = set(d.get("proposed", []))

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
