"""Baseline arm (PREREG §5): frozen weights, one attempt per prompt. The floor."""

from __future__ import annotations

from collections.abc import Sequence

from arms.harness import Episode, RunConfig, batched_attempt_many
from backends.base import Backend, GenResult
from eval import compute_ledger as cl
from tasks import number_reduction as nr


class BaselineArm:
    name = "baseline"

    def attempt(self, prompt: str, backend: Backend, max_tokens: int,
                phase: str = "day") -> tuple[GenResult, int]:
        return backend.generate([prompt], max_tokens=max_tokens)[0], 1

    def attempt_many(self, prompts: Sequence[str], backend, max_tokens: int,
                     phase: str = "eval") -> list[tuple[GenResult, int]]:
        return batched_attempt_many(prompts, backend, max_tokens)

    def after_episode(self, ep: Episode, backend: Backend, ledger: cl.Ledger, cfg: RunConfig) -> None:
        return None

    def night(self, episodes: Sequence[Episode], backend: Backend, ledger: cl.Ledger,
              cfg: RunConfig, split: nr.Split) -> dict:
        return {}
