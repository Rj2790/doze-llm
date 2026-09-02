"""Keep rule (PREREG §5 night step 2, amended C3; shared by Sleep, Online and
the dreamer): a trajectory is kept iff its final answer is correct AND at
least 9 of 11 intermediate responses are correct. Wrong-answer near-misses
are not kept."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from arms.harness import Episode
from tasks import number_reduction as nr

NEAR_MISS_MAX_WRONG = 2   # 11 - 9 at L=12


@dataclass
class Example:
    digits: str
    prompt: str
    completion: str
    source: str            # "day" | "dream"
    night: int | None = None

    @property
    def prefix(self) -> str:
        return self.digits[: nr.prefix_len(len(self.digits))]


def keep_scores(correct: bool, steps_correct: int, length: int) -> bool:
    """The C3 keep rule on raw scores; shared by Sleep, Online and the dreamer."""
    return bool(correct) and steps_correct >= (length - 1) - NEAR_MISS_MAX_WRONG


def keep(ep: Episode) -> bool:
    return keep_scores(ep.correct, ep.steps_correct, len(ep.digits))


def filter_day(episodes: Sequence[Episode]) -> list[Episode]:
    return [e for e in episodes if keep(e)]


def to_example(ep: Episode, mode: str, numbered: bool) -> Example:
    inst = nr.Instance.from_digits(ep.digits)
    return Example(digits=ep.digits, prompt=nr.format_prompt(inst, mode, numbered=numbered),
                   completion=ep.text.strip(), source="day")
