"""Online arm (PREREG §5, amended A2): LoRA; exactly one gradient step after
every episode, on a *kept* trajectory (keep rule = sleep/filters.py):

    the current trajectory if it is kept,
    else a random trajectory from today's kept pool (since the last checkpoint),
    else the most recently kept trajectory ever,
    else skip the step and log it.

So gradient steps == K per K episodes whenever anything has ever been kept.
`filtered=False` gives the optional arm "online_unfiltered": one step on
every trajectory as written, right or wrong.
"""

from __future__ import annotations

import random
import time
from collections import Counter
from collections.abc import Sequence

from arms.harness import Episode, RunConfig, batched_attempt_many
from backends.base import GenResult, TrainableBackend
from eval import compute_ledger as cl
from sleep import filters
from sleep.filters import Example
from tasks import number_reduction as nr


class OnlineArm:
    def __init__(self, filtered: bool = True):
        self.filtered = filtered
        self.name = "online" if filtered else "online_unfiltered"
        self.today: list[Example] = []
        self.most_recent: Example | None = None
        self.skipped = 0
        self.substitutions: Counter = Counter()
        self.log: list[dict] = []

    def attempt(self, prompt: str, backend: TrainableBackend, max_tokens: int,
                phase: str = "day") -> tuple[GenResult, int]:
        return backend.generate([prompt], max_tokens=max_tokens)[0], 1

    def _choose(self, ep: Episode, cfg: RunConfig) -> tuple[Example | None, str]:
        ex = filters.to_example(ep, cfg.mode, cfg.numbered)
        if not self.filtered:
            return ex, "current"
        if (ep.episode - 1) % cfg.k == 0:      # first episode of a new day
            self.today = []
        if filters.keep(ep):
            self.today.append(ex)
            self.most_recent = ex
            return ex, "current"
        if self.today:
            rng = random.Random(cfg.seed * 100_000 + ep.episode)
            return rng.choice(self.today), "today"
        if self.most_recent is not None:
            return self.most_recent, "recent"
        return None, "skip"

    def attempt_many(self, prompts: Sequence[str], backend, max_tokens: int,
                     phase: str = "eval") -> list[tuple[GenResult, int]]:
        return batched_attempt_many(prompts, backend, max_tokens)

    def after_episode(self, ep: Episode, backend: TrainableBackend, ledger: cl.Ledger, cfg: RunConfig) -> None:
        ex, how = self._choose(ep, cfg)
        if ex is None:
            self.skipped += 1
            self.log.append({"episode": ep.episode, "skipped": True})
            return
        if how != "current":
            self.substitutions[how] += 1
        t0 = time.time()
        st = backend.train([(ex.prompt, ex.completion)], steps=1, seed=cfg.seed * 100_000 + ep.episode)
        ledger.add(gradient_steps=st.steps, training_tokens=st.training_tokens, gpu_seconds=time.time() - t0)

    def night(self, episodes: Sequence[Episode], backend: TrainableBackend, ledger: cl.Ledger,
              cfg: RunConfig, split: nr.Split) -> dict:
        return {}
