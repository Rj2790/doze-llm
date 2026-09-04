"""Night step 3 (PREREG §5): dreaming = verified self-generation.

For each kept trajectory the current model is asked for `n_variations` new
puzzles: change one to three of the LAST FIVE digits of the source string,
keep the first seven, and solve the new string. A dream enters training
only if it passes the shared C3 keep rule against the ground-truth solver
(final answer correct AND >= 9/11 STEPS entries correct) and keeps the
source prefix (so it can never carry a held-out prefix: the source is a
training instance). We do NOT filter dreams for the mirror structure; the
fraction of accepted dreams that happen to be structured is reported.

Digit-7 boundary note: digits 1-7 determine r6, the shortcut target.
Varying only digits 8-12 leaves r6 fixed and changes r11 unless the mirror
happens to hold, so most accepted dreams are unstructured strings solved
literally. This is tunable prompt wording (CONTEXT.md §4).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from backends.base import Backend
from sleep import filters
from sleep.filters import Example
from tasks import number_reduction as nr

DREAM_TEMPERATURE = 0.8   # tunable; the digits proposal needs diversity; the solve is greedy like the day

PROPOSE_MARKER = "Write a NEW string"
MAX_CHANGED = 3           # a proposal may change 1..3 of the last five digits
PROPOSE_INSTRUCTION = (
    "{marker}: keep the first seven digits exactly as they are and change between one and three of "
    "the last five digits (positions 8 to 12); every digit must be 1, 4 or 9. Output exactly one line, "
    "'Digits: ' followed by the {length} digits separated by spaces, and nothing else."
)
DREAM_INSTRUCTION = PROPOSE_INSTRUCTION   # legacy name


def propose_prompt(seed: Example) -> str:
    """Step 1 of the decoupled dreamer: propose new digits only (no solving)."""
    length = len(seed.digits)
    return (f"Here is a string of {length} digits, each 1, 4 or 9.\n{nr.digits_line(seed.digits)}\n\n"
            f"{PROPOSE_INSTRUCTION.format(marker=PROPOSE_MARKER, length=length)}")


def dream_prompt(seed: Example, mode: str, numbered: bool) -> str:
    """Legacy one-step prompt (pilot). Kept for reference; not used by dream()."""
    length = len(seed.digits)
    solved = f"{nr.digits_line(seed.digits, numbered)}\n{seed.completion.strip()}"
    return (f"{nr.rule_text(length)}\n\n{solved}\n\nAbove is a solved puzzle. Invent a NEW puzzle from it: "
            f"{PROPOSE_INSTRUCTION.format(marker='write a new string', length=length)} Then solve it: "
            f"the {length - 1} comparison lines as 'previous,digit->result', then 'STEPS:', then 'ANSWER:'. "
            "Output only these lines (last five digits only).")


def _verify_scores(inst: nr.Instance, body: str, split: nr.Split, mode: str) -> str:
    s = nr.score(inst, body)
    if s["answer"] is None or not s["steps_parsed"] or (mode == "work" and body.count("->") == 0):
        return "unparsable"
    if inst.prefix in split.heldout_prefixes:      # unreachable when the source is a training instance
        return "heldout_prefix"
    if not filters.keep_scores(s["correct"], s["steps_correct"], inst.length):
        return "wrong"
    return "ok"


def parse_proposal(text: str, split: nr.Split, source: Example) -> tuple[nr.Instance | None, str]:
    """Validate a proposed digit string: parsable, right length/alphabet,
    same prefix as the source, not a verbatim copy, 1-3 digits changed."""
    try:
        toks = nr.parse_digits_line(text)
    except ValueError:
        return None, "unparsable"
    if len(toks) != split.length or not set(toks) <= set(nr.DIGITS):
        return None, "unparsable"
    digits = "".join(toks)
    inst = nr.Instance.from_digits(digits)
    if inst.prefix != source.prefix:
        return None, "prefix_changed"
    if digits == source.digits:
        return None, "duplicate"
    n_changed = sum(a != b for a, b in zip(digits, source.digits))
    if not 1 <= n_changed <= MAX_CHANGED:
        return None, "out_of_spec"          # instruction says change 1-3 of the last five
    if inst.prefix in split.heldout_prefixes:
        return None, "heldout_prefix"
    return inst, "ok"


def verify(text: str, split: nr.Split, mode: str, source_prefix: str) -> tuple[Example | None, str]:
    """Verify a combined 'Digits: ...' + solution text (used by tests/guard
    checks). Reasons: unparsable, prefix_changed, heldout_prefix, wrong."""
    try:
        toks = nr.parse_digits_line(text)
    except ValueError:
        return None, "unparsable"
    if len(toks) != split.length or not set(toks) <= set(nr.DIGITS):
        return None, "unparsable"
    digits = "".join(toks)
    inst = nr.Instance.from_digits(digits)
    body = text[text.rfind("Digits:"):].split("\n", 1)
    body = body[1].strip() if len(body) > 1 else ""
    if nr.parse_answer(body) is None or not nr.parse_steps(body) or (mode == "work" and body.count("->") == 0):
        return None, "unparsable"
    if inst.prefix != source_prefix:
        return None, "prefix_changed"
    reason = _verify_scores(inst, body, split, mode)
    if reason != "ok":
        return None, reason
    return Example(digits=digits, prompt="", completion=body, source="dream"), "ok"


def dream(backend: Backend, kept: Sequence[Example], n_variations: int, split: nr.Split,
          mode: str, numbered: bool, max_tokens: int,
          temperature: float = DREAM_TEMPERATURE) -> tuple[list[Example], dict]:
    """Decoupled dreaming (post-pilot): (1) propose new digits at `temperature`
    for every kept trajectory x n_variations; reject unparsable / prefix
    changed / duplicate; (2) solve each accepted string with the ordinary day
    prompt, greedy; keep iff the C3 rule holds against the solver."""
    seeds = [e for e in kept for _ in range(n_variations)]
    rejected: Counter = Counter()
    tokens = 0
    out: list[Example] = []
    accepted: list[dict] = []
    structured = 0
    if not seeds:
        return out, {"generated": 0, "tokens": 0, "kept": 0, "rejected": {}, "structured": 0,
                     "structured_frac": None, "duplicates": 0, "accepted": []}
    proposals = backend.generate([propose_prompt(e) for e in seeds], max_tokens=48, temperature=temperature)
    todo: list[tuple[Example, nr.Instance]] = []
    for seed, g in zip(seeds, proposals):
        tokens += g.completion_tokens
        inst, reason = parse_proposal(g.text, split, seed)
        if inst is None:
            rejected[reason] += 1
        else:
            todo.append((seed, inst))
    if todo:
        solves = backend.generate([nr.format_prompt(inst, mode, numbered=numbered) for _, inst in todo],
                                  max_tokens=max_tokens)
        for (seed, inst), g in zip(todo, solves):
            tokens += g.completion_tokens
            body = g.text.strip()
            reason = _verify_scores(inst, body, split, mode)
            if reason != "ok":
                rejected[reason] += 1
                continue
            ex = Example(digits=inst.digits, prompt=nr.format_prompt(inst, mode, numbered=numbered),
                         completion=body, source="dream")
            structured += inst.structured
            accepted.append({"digits": inst.digits, "source": seed.digits, "structured": bool(inst.structured),
                             "duplicate": False, "tokens": g.completion_tokens})
            out.append(ex)
    return out, {"generated": len(seeds), "tokens": tokens, "kept": len(out), "rejected": dict(rejected),
                 "structured": structured, "structured_frac": (structured / len(out)) if out else None,
                 "duplicates": rejected.get("duplicate", 0), "accepted": accepted}


def interleave(a: Sequence, b: Sequence) -> list:
    """1:1 interleave, remainder appended."""
    out: list = []
    for x, y in zip(a, b):
        out += [x, y]
    n = min(len(a), len(b))
    out += list(a[n:]) + list(b[n:])
    return out
