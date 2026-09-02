"""Per-arm compute ledger and the PREREG §5 matching check.

Every arm records tokens generated, gradient steps, training tokens and
GPU-seconds. `check_matched` implements the matching rule and raises
`BudgetMismatch` so no analysis ever compares arms outside tolerance:

    Awake.tokens_generated            == Sleep.tokens_generated      +-tol
    Online.gradient_steps             == Sleep.gradient_steps        +-tol
    Online.training_tokens            == Sleep.training_tokens       +-tol
    Sleep-NoDream.gradient_steps      == Sleep.gradient_steps        +-tol  (H3 is at matched budget)
    Sleep-NoDream.training_tokens     == Sleep.training_tokens       +-tol

Sleep is always the reference. Baseline has no matching constraint (floor).
Ledgers from different backends (mlx vs hf) are never compared.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

ARM_NAMES = ("baseline", "awake", "online", "sleep", "sleep_nodream", "online_unfiltered")
REFERENCE = "sleep"

# (arm, field) pairs that must match the reference arm.
MATCH_RULES: tuple[tuple[str, str], ...] = (
    ("awake", "tokens_generated"),
    ("online", "gradient_steps"),
    ("online", "training_tokens"),
    ("sleep_nodream", "gradient_steps"),
    ("sleep_nodream", "training_tokens"),
    ("online_unfiltered", "gradient_steps"),
    ("online_unfiltered", "training_tokens"),
)


class BudgetMismatch(RuntimeError):
    pass


def within(a: float, b: float, tol: float) -> bool:
    """|a-b| <= tol * max(|a|,|b|); equal zeros are within."""
    if a == b:
        return True
    return abs(a - b) <= tol * max(abs(a), abs(b))


@dataclass
class Ledger:
    arm: str
    backend: str = ""
    tokens_generated: int = 0
    gradient_steps: int = 0
    training_tokens: int = 0
    gpu_seconds: float = 0.0
    checkpoints: list[dict] = field(default_factory=list)

    def add(self, tokens_generated: int = 0, gradient_steps: int = 0,
            training_tokens: int = 0, gpu_seconds: float = 0.0) -> None:
        for name, v in (("tokens_generated", tokens_generated), ("gradient_steps", gradient_steps),
                        ("training_tokens", training_tokens), ("gpu_seconds", gpu_seconds)):
            if v < 0:
                raise ValueError(f"{name} must be >= 0, got {v}")
        self.tokens_generated += tokens_generated
        self.gradient_steps += gradient_steps
        self.training_tokens += training_tokens
        self.gpu_seconds += gpu_seconds

    def totals(self) -> dict:
        return {"tokens_generated": self.tokens_generated, "gradient_steps": self.gradient_steps,
                "training_tokens": self.training_tokens, "gpu_seconds": round(self.gpu_seconds, 3)}

    def checkpoint(self, episode: int) -> dict:
        """Snapshot cumulative totals at an evaluation checkpoint."""
        snap = {"episode": episode, **self.totals()}
        self.checkpoints.append(snap)
        return snap

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "Ledger":
        return cls(**json.loads(Path(path).read_text()))


def check_matched(arms: dict[str, Ledger], tol: float = 0.05) -> None:
    """Raise BudgetMismatch if any present arm violates MATCH_RULES against
    the Sleep reference, or if ledgers come from different backends."""
    if REFERENCE not in arms:
        raise BudgetMismatch(f"reference arm '{REFERENCE}' missing; have {sorted(arms)}")
    backends = {L.backend for L in arms.values()}
    if len(backends) > 1:
        raise BudgetMismatch(f"ledgers from different backends must never be compared: {sorted(backends)}")
    ref = arms[REFERENCE]
    problems = []
    for arm, fld in MATCH_RULES:
        if arm not in arms:
            continue
        a, b = getattr(arms[arm], fld), getattr(ref, fld)
        if not within(a, b, tol):
            drift = (a - b) / b if b else float("inf")
            problems.append(f"{arm}.{fld}={a} vs {REFERENCE}.{fld}={b} (drift {drift:+.1%}, tol {tol:.0%})")
    if problems:
        raise BudgetMismatch("compute budgets not matched:\n  " + "\n  ".join(problems))
