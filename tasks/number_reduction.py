"""Number Reduction Task (Wagner et al., 2004), adapted for LLMs.

An instance is an L-digit string over the alphabet {1, 4, 9}. The pairwise
rule is:

    same, same       -> that digit           (e.g. 1,1 -> 1)
    different pair   -> the third digit      (e.g. 1,4 -> 9)

Applied left to right:

    r1 = f(d1, d2)
    r_i = f(r_{i-1}, d_{i+1})   for i = 2..L-1

The task answer is r_{L-1}.

Hidden structure: instances are built so that the last three responses
mirror the preceding three:

    r_N == r_{N-5},   r_{N-1} == r_{N-4},   r_{N-2} == r_{N-3}    (N = L-1)

so the answer equals r_{N-5}, which needs only the first L-5 digits. A
solver that has noticed this can stop early.

Counting: digits d1..d_{L-3} are free, the last three are forced, so there
are 3^(L-3) structured instances. Wagner's L=8 gives only 243 and every
probe prefix has 9 instances sharing one answer — trivially memorisable.
Default L=12 gives 19,683 instances and 2,187 distinct prefixes.

Splits are **by prefix**: no probe/held-out prefix ever appears in training,
so beating chance on the probe requires the rule, not a lookup table.

Everything here is deterministic given a seed and has no model dependency.
"""

from __future__ import annotations

import itertools
import random
import re
from collections.abc import Sequence
from dataclasses import dataclass, field

DIGITS = ("1", "4", "9")
DEFAULT_LENGTH = 12
MASK = "_"


def combine(a: str, b: str) -> str:
    """Pairwise rule: same -> same, different -> the third digit."""
    if a == b:
        return a
    (third,) = set(DIGITS) - {a, b}
    return third


def solve(digits: str) -> list[str]:
    """Return the L-1 intermediate responses for a full string."""
    assert len(digits) >= 3 and set(digits) <= set(DIGITS), digits
    responses = [combine(digits[0], digits[1])]
    for d in digits[2:]:
        responses.append(combine(responses[-1], d))
    return responses


def prefix_len(length: int) -> int:
    """Digits needed to compute the shortcut response r_{N-5}."""
    return length - 5


def has_mirror_structure(digits: str) -> bool:
    r = solve(digits)
    n = len(r) - 1
    return r[n] == r[n - 5] and r[n - 1] == r[n - 4] and r[n - 2] == r[n - 3]


@dataclass(frozen=True)
class Instance:
    digits: str
    responses: tuple[str, ...]
    answer: str
    structured: bool

    @property
    def length(self) -> int:
        return len(self.digits)

    @property
    def prefix(self) -> str:
        return self.digits[: prefix_len(self.length)]

    @property
    def shortcut_response(self) -> str:
        """r_{N-5}: equals the answer iff the instance is structured."""
        return self.responses[len(self.responses) - 6]

    @classmethod
    def from_digits(cls, digits: str) -> "Instance":
        r = solve(digits)
        return cls(digits=digits, responses=tuple(r), answer=r[-1],
                   structured=has_mirror_structure(digits))


def structured_instances(length: int = DEFAULT_LENGTH) -> list[Instance]:
    """Construct all structured instances directly: free digits d1..d_{L-3},
    then force the last three so the mirror holds. 3^(L-3) instances."""
    out = []
    for free in itertools.product(DIGITS, repeat=length - 3):
        r = solve("".join(free))          # r1..r_{L-4}
        n = length - 1                    # index of final response (1-based)
        digits = list(free)
        prev = r[-1]
        # targets for r_{N-2}, r_{N-1}, r_N are r_{N-3}, r_{N-4}, r_{N-5}
        for target in (r[n - 4], r[n - 5], r[n - 6]):
            # choose d so that combine(prev, d) == target
            d = target if target == prev else (set(DIGITS) - {prev, target}).pop()
            digits.append(d)
            prev = target
        inst = Instance.from_digits("".join(digits))
        assert inst.structured
        out.append(inst)
    return out


def all_instances(length: int = DEFAULT_LENGTH) -> list[Instance]:
    """Full space. 3^L strings; fine for L<=12 (531k), avoid above that."""
    return [Instance.from_digits("".join(t)) for t in itertools.product(DIGITS, repeat=length)]


# PREREG §4.1 (amended 2026-09-02): sizes are multiples of 3 so every part is
# exactly uniform over answers. Probe items are disjoint from the eval items.
N_TRAIN = 801
N_HELDOUT = 201
N_PROBE = 60


@dataclass
class Split:
    train: list[Instance]
    heldout: list[Instance]          # unseen prefixes; held-out task accuracy at checkpoints
    probe: list[Instance]            # unseen prefixes; shortcut probe; disjoint from heldout
    length: int
    train_prefixes: set[str] = field(default_factory=set)
    heldout_prefixes: set[str] = field(default_factory=set)


def _stratified(pool: list[Instance], n: int, rng: random.Random, key=lambda x: x.answer) -> list[Instance]:
    """Exactly n/3 per answer digit, drawn without replacement, then shuffled."""
    if n % len(DIGITS):
        raise ValueError(f"size must be a multiple of {len(DIGITS)}, got {n}")
    per = n // len(DIGITS)
    by = {d: [x for x in pool if key(x) == d] for d in DIGITS}
    short = {d: len(v) for d, v in by.items() if len(v) < per}
    if short:
        raise ValueError(f"pool too small for {per} per answer: {short}")
    out = [x for d in DIGITS for x in rng.sample(by[d], per)]
    rng.shuffle(out)
    return out


def make_split(seed: int, length: int = DEFAULT_LENGTH,
               n_train: int = N_TRAIN, n_heldout: int = N_HELDOUT, n_probe: int = N_PROBE,
               heldout_prefix_frac: float = 0.2) -> Split:
    """Prefix-disjoint train / held-out split of structured instances, each
    part exactly uniform over answers; probe items disjoint from held-out
    eval items. Deterministic in `seed`."""
    rng = random.Random(seed)
    pool = structured_instances(length)
    prefixes = sorted({x.prefix for x in pool})
    rng.shuffle(prefixes)
    k = int(len(prefixes) * heldout_prefix_frac)
    ho_pref, tr_pref = set(prefixes[:k]), set(prefixes[k:])
    train_pool = [x for x in pool if x.prefix in tr_pref]
    ho_pool = [x for x in pool if x.prefix in ho_pref]
    train = _stratified(train_pool, n_train, rng)
    heldout = _stratified(ho_pool, n_heldout, rng)
    used = {x.digits for x in heldout}
    probe = _stratified([x for x in ho_pool if x.digits not in used], n_probe, rng)
    return Split(train=train, heldout=heldout, probe=probe, length=length,
                 train_prefixes=tr_pref, heldout_prefixes=ho_pref)


_UNSTRUCTURED_CACHE: dict = {}


def _unstructured_pool(length: int, heldout_prefixes: frozenset) -> list[Instance]:
    key = (length, heldout_prefixes)
    if key not in _UNSTRUCTURED_CACHE:
        out = []
        for pref in sorted(heldout_prefixes):
            for tail in itertools.product(DIGITS, repeat=length - len(pref)):
                inst = Instance.from_digits(pref + "".join(tail))
                if not inst.structured:
                    out.append(inst)
        _UNSTRUCTURED_CACHE[key] = out
    return _UNSTRUCTURED_CACHE[key]


def unstructured_heldout(split: Split, n: int, seed: int) -> list[Instance]:
    """n UNSTRUCTURED strings (no mirror) whose prefix is held out, exactly
    uniform over answers, disjoint from the structured held-out/probe items.
    Used by the implicit-shortcut metrics (mirror_bias, short_gap)."""
    rng = random.Random(f"unstructured-{seed}")
    used = {x.digits for x in split.heldout + split.probe}
    pool = [x for x in _unstructured_pool(split.length, frozenset(split.heldout_prefixes)) if x.digits not in used]
    return _stratified(pool, n, rng)


# --------------------------------------------------------------------------
# Prompting
# --------------------------------------------------------------------------

def rule_text(length: int) -> str:
    return (
        f"You are given a string of {length} digits, each 1, 4 or 9.\n"
        "Combine digits left to right using this rule:\n"
        "  - if the two digits are the same, the result is that digit\n"
        "  - if they are different, the result is the third digit (1,4->9  1,9->4  4,9->1)\n"
        "Start with the first two digits. Then combine the previous result with the next digit, "
        "and so on, until all digits are used. The answer is the final result."
    )


def full_format(length: int) -> str:
    return (f"Show the {length - 1} intermediate results as a space-separated list on a line "
            "starting with 'STEPS:', then the final result on a line starting with 'ANSWER:'.")


SYSTEM_PROMPT = (
    "You solve digit-combination puzzles. Output only the requested lines, exactly in the "
    "requested format, with no explanation, no restatement of the problem, and no markdown."
)

# Worked example on a 6-digit string. Six digits is too short for the mirror
# structure to exist (needs L >= 8), so this cannot leak the shortcut.
EXAMPLE_DIGITS = "149914"
_EX_STEPS = solve(EXAMPLE_DIGITS)


def digits_line(tokens: Sequence[str], numbered: bool = False) -> str:
    """'Digits: 1 4 9 ...' or, numbered, 'Digits: 1:1 2:4 3:9 ...'.
    Tokens may include MASK. Numbering is a tunable prompt detail
    (CONTEXT.md §4): it helps the model keep its place in runs of repeats."""
    if numbered:
        return "Digits: " + " ".join(f"{i}:{t}" for i, t in enumerate(tokens, 1))
    return "Digits: " + " ".join(tokens)


_DIGITS_LINE_RE = re.compile(r"^Digits:\s*(.*)$", re.MULTILINE)


def parse_digits_line(prompt: str) -> list[str]:
    """Recover the item's digit tokens (digits or MASK) from a prompt built by
    format_prompt/format_probe, numbered or not. Last 'Digits:' line wins
    (the worked example comes first)."""
    hits = _DIGITS_LINE_RE.findall(prompt)
    if not hits:
        raise ValueError("no Digits: line in prompt")
    toks = hits[-1].split()
    return [t.split(":", 1)[1] if ":" in t else t for t in toks]


def _work_lines(digits: str) -> str:
    """One line per comparison: 'prev,digit->result'."""
    r = solve(digits)
    lines = [f"{digits[0]},{digits[1]}->{r[0]}"]
    for i, d in enumerate(digits[2:], start=1):
        lines.append(f"{r[i-1]},{d}->{r[i]}")
    return "\n".join(lines)


def example_block(mode: str, numbered: bool = False) -> str:
    head = f"Example with {len(EXAMPLE_DIGITS)} digits:\n{digits_line(EXAMPLE_DIGITS, numbered)}\n"
    tail = f"STEPS: {' '.join(_EX_STEPS)}\nANSWER: {_EX_STEPS[-1]}"
    if mode == "full":
        return head + tail
    if mode == "work":
        return head + _work_lines(EXAMPLE_DIGITS) + "\n" + tail
    if mode == "short":
        return head + f"ANSWER: {_EX_STEPS[-1]}"
    raise ValueError(mode)


FULL_EXAMPLE = example_block("full")
SHORT_EXAMPLE = example_block("short")
WORK_EXAMPLE = example_block("work")

NUMBERED_NOTE = "Each digit is written as position:digit."


def work_format(length: int) -> str:
    return (f"First write each of the {length - 1} comparisons on its own line as "
            "'previous,digit->result'. Then write all results on a line starting with 'STEPS:', "
            "then the final result on a line starting with 'ANSWER:'.")


SHORT_FORMAT = "Reply with only the final result on a line starting with 'ANSWER:'."

PROBE_FORMAT = ("Some digits are hidden and shown as '_'. Give your best guess for the final "
                "result on a line starting with 'ANSWER:'.")

_FORMATS = {"full": full_format, "work": work_format, "short": lambda length: SHORT_FORMAT}


def _rule(length: int, numbered: bool) -> str:
    return rule_text(length) + (f"\n{NUMBERED_NOTE}" if numbered else "")


def format_prompt(inst: Instance, mode: str = "full", numbered: bool = False) -> str:
    if mode not in _FORMATS:
        raise ValueError(mode)
    return (f"{_rule(inst.length, numbered)}\n\n{example_block(mode, numbered)}\n\n"
            f"{_FORMATS[mode](inst.length)}\n\n{digits_line(inst.digits, numbered)}")


def format_probe(inst: Instance, numbered: bool = False) -> str:
    """Mask everything after the shortcut prefix. The literal rule can then
    reach only r_{N-5}; the answer is guessable only via the shortcut.
    Mask positions and target are frozen (PREREG §4.1); wording is not."""
    k = prefix_len(inst.length)
    shown = list(inst.digits[:k]) + [MASK] * (inst.length - k)
    return (f"{_rule(inst.length, numbered)}\n\n{example_block('short', numbered)}\n\n{PROBE_FORMAT}"
            f"\n\n{digits_line(shown, numbered)}")


def probe_target(inst: Instance) -> str:
    return inst.shortcut_response


# --------------------------------------------------------------------------
# Parsing model output
# --------------------------------------------------------------------------

_ANSWER_RE = re.compile(r"ANSWER\W{0,6}([149])\b", re.IGNORECASE)
_STEPS_RE = re.compile(r"STEPS\W{0,6}((?:[149][\s,→\->]*){2,})", re.IGNORECASE)


def parse_answer(text: str) -> str | None:
    """Last ANSWER tag wins. Tolerates markdown (**ANSWER:** 9), colons of
    either kind, and trailing punctuation."""
    hits = _ANSWER_RE.findall(text)
    return hits[-1] if hits else None


def parse_steps(text: str) -> list[str] | None:
    """First STEPS tag. Digits may be separated by spaces, commas or arrows."""
    m = _STEPS_RE.search(text)
    if not m:
        return None
    toks = re.findall(r"[149]", m.group(1))
    return toks or None


def score(inst: Instance, text: str) -> dict:
    ans = parse_answer(text)
    steps = parse_steps(text)
    steps_correct = sum(a == b for a, b in zip(steps, inst.responses)) if steps else 0
    return {"answer": ans, "correct": ans == inst.answer,
            "steps_correct": steps_correct, "steps_total": len(inst.responses),
            "steps_parsed": steps is not None}


def gold_response(inst: Instance, mode: str = "full") -> str:
    tail = f"STEPS: {' '.join(inst.responses)}\nANSWER: {inst.answer}"
    if mode == "work":
        return f"{_work_lines(inst.digits)}\n{tail}"
    return tail
