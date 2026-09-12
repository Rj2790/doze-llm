"""Transfer puzzle, variant 2 (records/transfer_prereg.md): four symbols and
two position-dependent rules.

Symbols 1, 4, 7, 9. Combine left to right as in the original puzzle, but the
rule alternates by step:
  odd steps  (1st, 3rd, ...): same -> that symbol; different -> the SMALLER of
                              the two symbols not involved
  even steps (2nd, 4th, ...): same -> that symbol; different -> the LARGER of
                              the two symbols not involved
No hidden structure is imposed on the strings: this is a transfer study, not
an insight study. Output format is identical to the original puzzle so that
format-following transfers and parsing is shared in spirit (own parsers here
because the symbol set differs).
"""
from __future__ import annotations

import random
import re
from collections.abc import Sequence
from dataclasses import dataclass

SYMBOLS = ("1", "4", "7", "9")
DEFAULT_LENGTH = 12
NAME = "fold_variant2"

# the original puzzle's rule, for the negative-transfer measure (pairs within {1,4,9})
_OLD = {frozenset("14"): "9", frozenset("19"): "4", frozenset("49"): "1"}


def combine(a: str, b: str, step: int) -> str:
    """step is 1-based."""
    if a == b:
        return a
    others = sorted(s for s in SYMBOLS if s not in (a, b))
    return others[0] if step % 2 == 1 else others[-1]


def solve(digits: str) -> list[str]:
    r = combine(digits[0], digits[1], 1)
    out = [r]
    for i, d in enumerate(digits[2:], start=2):
        r = combine(r, d, i)
        out.append(r)
    return out


def old_rule_value(a: str, b: str) -> str | None:
    """What the ORIGINAL puzzle's rule would output for this pair, if defined."""
    if a == b:
        return a
    return _OLD.get(frozenset((a, b)))


@dataclass(frozen=True)
class Instance:
    digits: str
    responses: tuple[str, ...]
    answer: str

    @property
    def length(self) -> int:
        return len(self.digits)

    @classmethod
    def from_digits(cls, digits: str) -> "Instance":
        r = solve(digits)
        return cls(digits=digits, responses=tuple(r), answer=r[-1])


def make_items(n: int, seed: int, length: int = DEFAULT_LENGTH) -> list[Instance]:
    """n distinct random strings, deterministic in seed; answers as balanced
    as n allows (round-robin over symbols)."""
    rng = random.Random(f"{NAME}-{seed}")
    by_answer: dict[str, list[Instance]] = {s: [] for s in SYMBOLS}
    seen: set[str] = set()
    target = -(-n // len(SYMBOLS))
    while sum(len(v) for v in by_answer.values()) < n:
        d = "".join(rng.choice(SYMBOLS) for _ in range(length))
        if d in seen:
            continue
        seen.add(d)
        inst = Instance.from_digits(d)
        if len(by_answer[inst.answer]) < target:
            by_answer[inst.answer].append(inst)
    out: list[Instance] = []
    i = 0
    while len(out) < n:
        s = SYMBOLS[i % len(SYMBOLS)]
        if by_answer[s]:
            out.append(by_answer[s].pop(0))
        i += 1
    return out


# ---- prompting (mirrors the original work format) --------------------------------

SYSTEM_PROMPT = ("You solve digit-combination puzzles. Output only the requested lines, exactly in the "
                 "requested format, with no explanation, no restatement of the problem, and no markdown.")

EXAMPLE_DIGITS = "149741"


def _table(step_kind: str) -> str:
    rows = []
    for a in SYMBOLS:
        for b in SYMBOLS:
            if a < b:
                rows.append(f"{a},{b}->{combine(a, b, 1 if step_kind == 'odd' else 2)}")
    return "  ".join(rows)


def rule_text(length: int = DEFAULT_LENGTH) -> str:
    return (f"You are given a string of {length} digits, each 1, 4, 7 or 9.\n"
            "Combine digits left to right. Start with the first two digits (that is step 1). Then combine the "
            "previous result with the next digit (step 2), and so on, until all digits are used. The answer is the final result.\n"
            "The rule depends on the step number:\n"
            "  - if the two digits are the same, the result is that digit (any step)\n"
            f"  - on ODD steps (1, 3, 5, ...) if they differ, the result is the SMALLER of the two digits not involved: {_table('odd')}\n"
            f"  - on EVEN steps (2, 4, 6, ...) if they differ, the result is the LARGER of the two digits not involved: {_table('even')}")


def _work_lines(digits: str) -> str:
    r = solve(digits)
    lines = [f"{digits[0]},{digits[1]}->{r[0]}"]
    prev = r[0]
    for d, res in zip(digits[2:], r[1:]):
        lines.append(f"{prev},{d}->{res}")
        prev = res
    return "\n".join(lines)


def example_block() -> str:
    r = solve(EXAMPLE_DIGITS)
    return (f"Example with {len(EXAMPLE_DIGITS)} digits:\nDigits: {' '.join(EXAMPLE_DIGITS)}\n{_work_lines(EXAMPLE_DIGITS)}\n"
            f"STEPS: {' '.join(r)}\nANSWER: {r[-1]}")


def work_format(length: int = DEFAULT_LENGTH) -> str:
    return (f"First write each of the {length - 1} comparisons on its own line as 'previous,digit->result'. "
            "Then write all results on a line starting with 'STEPS:', then the final result on a line starting with 'ANSWER:'.")


def format_prompt(inst: Instance) -> str:
    return f"{rule_text(inst.length)}\n\n{example_block()}\n\n{work_format(inst.length)}\n\nDigits: {' '.join(inst.digits)}"


# ---- parsing / scoring ----------------------------------------------------------------

_ANSWER_RE = re.compile(r"ANSWER\W{0,6}([1479])", re.IGNORECASE)
_STEPS_RE = re.compile(r"STEPS\W{0,6}([1479][\s,>\-1479]*)", re.IGNORECASE)
_LINE_RE = re.compile(r"^\s*([1479])\s*,\s*([1479])\s*->\s*([1479])\s*$", re.MULTILINE)


def parse_answer(text: str) -> str | None:
    hits = _ANSWER_RE.findall(text)
    return hits[-1] if hits else None


def parse_steps(text: str) -> list[str] | None:
    m = _STEPS_RE.search(text)
    if not m:
        return None
    toks = re.findall(r"[1479]", m.group(1))
    return toks or None


def parse_work_lines(text: str) -> list[tuple[str, str, str]]:
    return [(a, b, c) for a, b, c in _LINE_RE.findall(text)]


def score(inst: Instance, text: str) -> dict:
    """Correctness plus error anatomy up to the first divergence:
    first_wrong_step (1-based; None if all written steps are right), whether
    that first error was a rule error (right operands, wrong result) or a
    tracking error (wrong operands), whether the wrong result equals what the
    ORIGINAL puzzle's rule gives (negative transfer), and the number of
    consecutive correct leading steps."""
    ans = parse_answer(text)
    steps = parse_steps(text)
    steps_correct = sum(a == b for a, b in zip(steps, inst.responses)) if steps else 0
    lines = parse_work_lines(text)
    first_wrong = None; kind = None; old_hit = False; leading = 0
    for k, (a, b, c) in enumerate(lines[: inst.length - 1], start=1):
        exp_a = inst.digits[0] if k == 1 else inst.responses[k - 2]
        exp_b = inst.digits[k]
        if (a, b) != (exp_a, exp_b):
            first_wrong, kind = k, "tracking"; break
        if c != inst.responses[k - 1]:
            first_wrong, kind = k, "rule"; old_hit = old_rule_value(a, b) == c; break
        leading += 1
    if first_wrong is None and len(lines) < inst.length - 1:
        first_wrong, kind = len(lines) + 1, "missing"
    return {"answer": ans, "correct": ans == inst.answer, "steps_correct": steps_correct,
            "steps_total": len(inst.responses), "steps_parsed": steps is not None, "n_lines": len(lines),
            "leading_correct": leading, "first_wrong_step": first_wrong, "first_error_kind": kind,
            "first_error_old_rule": old_hit}
