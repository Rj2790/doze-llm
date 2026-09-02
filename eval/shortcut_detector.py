"""Detect whether a solver has discovered the hidden shortcut.

Two signals, both model-agnostic. The caller supplies a `generate` callable
(prompt -> completion text); this module never touches a model directly, so
it can be unit-tested with scripted solvers.

1. Probe accuracy (primary, PREREG §6.1)
   Digits after the shortcut prefix are masked. The literal rule reaches only
   r_{N-5}. A solver answers r_N correctly above chance (1/3) only by using
   r_N == r_{N-5}. Probe items come from held-out prefixes, so a lookup
   table learned on training prefixes does not help.

   Items are sampled stratified by target (n/3 each) so constant-answer and
   answer-frequency heuristics score exactly chance, and `above_chance` is a
   one-sided exact binomial test vs 1/3 at ALPHA (not a fixed margin, which
   tripped on n=40 in calibration; see CONTEXT.md §4).

2. Token collapse (secondary)
   Median completion length on held-out full-format items, over correct
   answers only. Reported, not used for the criterion.
"""

from __future__ import annotations

import math
import random
import statistics
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from tasks import number_reduction as nr

Generate = Callable[[str], str]
CHANCE = 1.0 / len(nr.DIGITS)


ALPHA = 0.01  # one-sided binomial test vs CHANCE for the above_chance flag


@dataclass
class ProbeResult:
    n: int
    correct: int
    accuracy: float
    above_chance: bool
    p_value: float = float("nan")

    def meets_criterion(self, threshold: float) -> bool:
        return self.accuracy >= threshold


def binom_sf(k: int, n: int, p: float) -> float:
    """P(X >= k) for X ~ Binomial(n, p). Exact, no scipy."""
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))


def stratified_probe_items(pool: Sequence[nr.Instance], n: int, seed: int) -> list[nr.Instance]:
    """Draw n probe items from `pool` with exactly n/3 per target digit, so a
    constant-answer or answer-frequency heuristic scores exactly CHANCE.
    Deterministic in `seed`; order is shuffled so targets are not grouped."""
    if n % len(nr.DIGITS):
        raise ValueError(f"n_probe must be divisible by {len(nr.DIGITS)}, got {n}")
    per = n // len(nr.DIGITS)
    rng = random.Random(seed)
    by_target: dict[str, list[nr.Instance]] = {d: [] for d in nr.DIGITS}
    for x in pool:
        by_target[nr.probe_target(x)].append(x)
    short = {d: len(v) for d, v in by_target.items() if len(v) < per}
    if short:
        raise ValueError(f"pool too small for {per} per target: {short}")
    out: list[nr.Instance] = []
    for d in nr.DIGITS:
        out.extend(rng.sample(by_target[d], per))
    rng.shuffle(out)
    return out


def score_probe(items: Sequence[nr.Instance], texts: Sequence[str], alpha: float = ALPHA) -> ProbeResult:
    assert len(items) == len(texts)
    correct = sum(nr.parse_answer(t) == nr.probe_target(x) for x, t in zip(items, texts))
    acc = correct / len(items) if items else 0.0
    p = binom_sf(correct, len(items), CHANCE) if items else 1.0
    return ProbeResult(n=len(items), correct=correct, accuracy=acc, above_chance=p < alpha, p_value=p)


def run_probe(generate: Generate, items: Sequence[nr.Instance], alpha: float = ALPHA,
              numbered: bool = False) -> ProbeResult:
    return score_probe(items, [generate(nr.format_probe(x, numbered=numbered)) for x in items], alpha)


def target_distribution(items: Sequence[nr.Instance]) -> dict[str, int]:
    return dict(sorted(Counter(nr.probe_target(x) for x in items).items()))


@dataclass
class LengthResult:
    median_tokens: float
    n_correct: int


def token_collapse(generate: Generate, items: Sequence[nr.Instance],
                   count_tokens: Callable[[str], int] = lambda s: len(s.split()),
                   mode: str = "full", numbered: bool = False) -> LengthResult:
    lengths = []
    for inst in items:
        out = generate(nr.format_prompt(inst, mode=mode, numbered=numbered))
        if nr.score(inst, out)["correct"]:
            lengths.append(count_tokens(out))
    return LengthResult(median_tokens=statistics.median(lengths) if lengths else float("nan"),
                        n_correct=len(lengths))


@dataclass
class CriterionTracker:
    """PREREG §6.1: probe accuracy >= threshold on two consecutive checkpoints."""
    threshold: float = 0.70
    consecutive_required: int = 2
    _streak: int = 0
    first_met_at: int | None = None

    def update(self, episode: int, probe: ProbeResult) -> bool:
        self._streak = self._streak + 1 if probe.meets_criterion(self.threshold) else 0
        if self._streak >= self.consecutive_required and self.first_met_at is None:
            self.first_met_at = episode
        return self.first_met_at is not None
