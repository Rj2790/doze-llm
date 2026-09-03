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

DREAM_TEMPERATURE = 0.8   # tunable; dreams need diversity

DREAM_INSTRUCTION = (
    "Above is a solved puzzle. Invent a NEW puzzle from it: keep the first seven digits exactly as "
    "they are and change between one and three of the last five digits (positions 8 to 12), each "
    "still 1, 4 or 9. Then solve the new string with the same rule. Output exactly: a line "
    "'Digits: ' with the {length} new digits separated by spaces, then the {steps} comparison "
    "lines as 'previous,digit->result', then 'STEPS:' with all {steps} results, then 'ANSWER:' "
    "with the final result. Output only these lines."
)


def dream_prompt(seed: Example, mode: str, numbered: bool) -> str:
    length = len(seed.digits)
    solved = f"{nr.digits_line(seed.digits, numbered)}\n{seed.completion.strip()}"
    return (f"{nr.rule_text(length)}\n\n{solved}\n\n"
            f"{DREAM_INSTRUCTION.format(length=length, steps=length - 1)}")


def verify(text: str, split: nr.Split, mode: str, source_prefix: str) -> tuple[Example | None, str]:
    """Return (example, 'ok') or (None, reason). Reasons: unparsable,
    prefix_changed, heldout_prefix, wrong. 'wrong' = fails the C3 keep rule."""
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
    s = nr.score(inst, body)
    if s["answer"] is None or not s["steps_parsed"] or (mode == "work" and body.count("->") == 0):
        return None, "unparsable"
    if inst.prefix != source_prefix:
        return None, "prefix_changed"
    if inst.prefix in split.heldout_prefixes:      # unreachable when the source is a training instance
        return None, "heldout_prefix"
    if not filters.keep_scores(s["correct"], s["steps_correct"], inst.length):
        return None, "wrong"
    return Example(digits=digits, prompt="", completion=body, source="dream"), "ok"


def dream(backend: Backend, kept: Sequence[Example], n_variations: int, split: nr.Split,
          mode: str, numbered: bool, max_tokens: int,
          temperature: float = DREAM_TEMPERATURE) -> tuple[list[Example], dict]:
    seeds = [e for e in kept for _ in range(n_variations)]
    prompts = [dream_prompt(e, mode, numbered) for e in seeds]
    out: list[Example] = []
    rejected: Counter = Counter()
    tokens = 0
    structured = 0
    duplicates = 0
    accepted: list[dict] = []
    if prompts:
        gens = backend.generate(prompts, max_tokens=max_tokens, temperature=temperature)
        for seed, g in zip(seeds, gens):
            tokens += g.completion_tokens
            ex, reason = verify(g.text, split, mode, source_prefix=seed.prefix)
            if ex is None:
                rejected[reason] += 1
            else:
                inst = nr.Instance.from_digits(ex.digits)
                ex.prompt = nr.format_prompt(inst, mode, numbered=numbered)
                dup = ex.digits == seed.digits          # a verbatim copy of the source (logged, still accepted)
                structured += inst.structured
                duplicates += dup
                accepted.append({"digits": ex.digits, "source": seed.digits, "structured": bool(inst.structured),
                                 "duplicate": bool(dup), "tokens": backend.count_tokens(ex.completion)})
                out.append(ex)
    return out, {"generated": len(prompts), "tokens": tokens, "kept": len(out), "rejected": dict(rejected),
                 "structured": structured, "structured_frac": (structured / len(out)) if out else None,
                 "duplicates": duplicates, "accepted": accepted}


def interleave(a: Sequence, b: Sequence) -> list:
    """1:1 interleave, remainder appended."""
    out: list = []
    for x, y in zip(a, b):
        out += [x, y]
    n = min(len(a), len(b))
    out += list(a[n:]) + list(b[n:])
    return out
