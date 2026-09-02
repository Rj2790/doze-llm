"""Night step 3 (PREREG §5): dreaming = verified self-generation.

For each kept trajectory the current model is asked for `n_variations` new
puzzles (a changed digit string plus its full solution). A dream enters
training only if it verifies against the ground-truth solver: every work
line, every STEPS entry and the ANSWER must be exactly right. Dreams whose
prefix is held out are rejected (eval leak guard). Nothing else is filtered:
in particular we do NOT check for the mirror structure, because that would
inject the experimenter's knowledge of the hidden rule into the treatment.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from backends.base import Backend
from sleep.filters import Example
from tasks import number_reduction as nr

DREAM_TEMPERATURE = 0.8   # tunable; dreams need diversity

DREAM_INSTRUCTION = (
    "Above is a solved puzzle. Invent a NEW puzzle of {length} digits, each 1, 4 or 9, by "
    "changing several digits of the one above, then solve it with the same rule. Output exactly: "
    "a line 'Digits: ' with the {length} new digits separated by spaces, then the {steps} "
    "comparison lines as 'previous,digit->result', then 'STEPS:' with all {steps} results, then "
    "'ANSWER:' with the final result. Output only these lines."
)


def dream_prompt(seed: Example, mode: str, numbered: bool) -> str:
    length = len(seed.digits)
    solved = f"{nr.digits_line(seed.digits, numbered)}\n{seed.completion.strip()}"
    return (f"{nr.rule_text(length)}\n\n{solved}\n\n"
            f"{DREAM_INSTRUCTION.format(length=length, steps=length - 1)}")


def _verify_work_lines(inst: nr.Instance, body: str) -> bool:
    lines = [l.strip() for l in body.splitlines() if "->" in l]
    if len(lines) != inst.length - 1:
        return False
    prev = inst.digits[0]
    for i, line in enumerate(lines):
        try:
            pair, res = line.split("->")
            a, b = [t.strip() for t in pair.split(",")]
        except ValueError:
            return False
        a, b, res = a.split(":")[-1], b.split(":")[-1], res.strip()
        if a != prev or b != inst.digits[i + 1] or res != inst.responses[i]:
            return False
        prev = res
    return True


def verify(text: str, split: nr.Split, mode: str) -> tuple[Example | None, str]:
    """Return (example, 'ok') or (None, reason). Reasons: unparsable,
    heldout_prefix, wrong."""
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
    if inst.prefix in split.heldout_prefixes:
        return None, "heldout_prefix"
    ok = s["correct"] and s["steps_correct"] == inst.length - 1 and nr.parse_steps(body) == list(inst.responses)
    if mode == "work":
        ok = ok and _verify_work_lines(inst, body)
    if not ok:
        return None, "wrong"
    return Example(digits=digits, prompt="", completion=body, source="dream"), "ok"


def dream(backend: Backend, kept: Sequence[Example], n_variations: int, split: nr.Split,
          mode: str, numbered: bool, max_tokens: int,
          temperature: float = DREAM_TEMPERATURE) -> tuple[list[Example], dict]:
    prompts = [dream_prompt(e, mode, numbered) for e in kept for _ in range(n_variations)]
    out: list[Example] = []
    rejected: Counter = Counter()
    tokens = 0
    if prompts:
        gens = backend.generate(prompts, max_tokens=max_tokens, temperature=temperature)
        for g in gens:
            tokens += g.completion_tokens
            ex, reason = verify(g.text, split, mode)
            if ex is None:
                rejected[reason] += 1
            else:
                ex.prompt = nr.format_prompt(nr.Instance.from_digits(ex.digits), mode, numbered=numbered)
                out.append(ex)
    return out, {"generated": len(prompts), "tokens": tokens, "kept": len(out), "rejected": dict(rejected)}


def interleave(a: Sequence, b: Sequence) -> list:
    """1:1 interleave, remainder appended."""
    out: list = []
    for x, y in zip(a, b):
        out += [x, y]
    n = min(len(a), len(b))
    out += list(a[n:]) + list(b[n:])
    return out
