"""Replay buffer of kept trajectories from all previous nights (PREREG §5
steps 4 and 7). Guarded: an example whose prefix is held out can never enter."""

from __future__ import annotations

import json
import random
from collections.abc import Iterable, Sequence
from dataclasses import asdict
from pathlib import Path

from sleep.filters import Example


class LeakError(RuntimeError):
    pass


class ReplayBuffer:
    def __init__(self, train_prefixes: Iterable[str]):
        self.train_prefixes = set(train_prefixes)
        if not self.train_prefixes:
            raise ValueError("train_prefixes must be non-empty")
        self.examples: list[Example] = []

    def __len__(self) -> int:
        return len(self.examples)

    def check(self, examples: Sequence[Example]) -> None:
        bad = [e.digits for e in examples if e.prefix not in self.train_prefixes]
        if bad:
            raise LeakError(f"{len(bad)} example(s) with non-training prefix, e.g. {bad[0]}")

    def add(self, examples: Sequence[Example], night: int | None = None) -> None:
        self.check(examples)
        for e in examples:
            e.night = night
            self.examples.append(e)

    def sample(self, n: int, seed: int) -> list[Example]:
        """Uniform sample without replacement over everything ever added."""
        n = min(n, len(self.examples))
        return random.Random(seed).sample(self.examples, n) if n > 0 else []

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps([asdict(e) for e in self.examples]))

    @classmethod
    def load(cls, path: str | Path, train_prefixes: Iterable[str]) -> "ReplayBuffer":
        buf = cls(train_prefixes)
        exs = [Example(**d) for d in json.loads(Path(path).read_text())]
        buf.check(exs)
        buf.examples = exs
        return buf
