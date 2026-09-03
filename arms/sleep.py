"""Sleep and Sleep-NoDream arms (PREREG §5): LoRA; zero updates by day; a
night (sleep/consolidate.py) after every K episodes."""

from __future__ import annotations

from collections.abc import Sequence

from arms.harness import Episode, RunConfig, batched_attempt_many
from backends.base import GenResult, TrainableBackend
from eval import compute_ledger as cl
from sleep.consolidate import SleepConfig, run_night
from sleep.replay_buffer import ReplayBuffer
from tasks import number_reduction as nr

__all__ = ["SleepArm", "SleepConfig"]


class SleepArm:
    def __init__(self, scfg: SleepConfig | None = None, dream: bool = True):
        self.scfg = scfg or SleepConfig()
        self.dream = dream
        self.name = "sleep" if dream else "sleep_nodream"
        self.buffer: ReplayBuffer | None = None
        self.nights = 0

    def save_state(self, sdir) -> None:
        import json
        from pathlib import Path
        if self.buffer is not None:
            self.buffer.save(Path(sdir, "buffer.json"))
        Path(sdir, "sleep.json").write_text(json.dumps({"nights": self.nights, "has_buffer": self.buffer is not None}))

    def load_state(self, sdir, episodes, cfg, split) -> None:
        import json
        from pathlib import Path
        d = json.loads(Path(sdir, "sleep.json").read_text())
        self.nights = d["nights"]
        self.buffer = ReplayBuffer.load(Path(sdir, "buffer.json"), split.train_prefixes) if d["has_buffer"] else None

    def attempt(self, prompt: str, backend: TrainableBackend, max_tokens: int,
                phase: str = "day") -> tuple[GenResult, int]:
        return backend.generate([prompt], max_tokens=max_tokens)[0], 1

    def attempt_many(self, prompts: Sequence[str], backend, max_tokens: int,
                     phase: str = "eval") -> list[tuple[GenResult, int]]:
        return batched_attempt_many(prompts, backend, max_tokens)

    def after_episode(self, ep: Episode, backend: TrainableBackend, ledger: cl.Ledger, cfg: RunConfig) -> None:
        return None   # no updates by day

    def night(self, episodes: Sequence[Episode], backend: TrainableBackend, ledger: cl.Ledger,
              cfg: RunConfig, split: nr.Split) -> dict:
        if self.buffer is None:
            self.buffer = ReplayBuffer(split.train_prefixes)
        self.nights += 1
        return run_night(episodes, backend, ledger, cfg, split, self.scfg, self.buffer,
                         night_index=self.nights, dream_enabled=self.dream)
