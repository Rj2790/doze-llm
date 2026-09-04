"""Secondary insight metrics (added after the pilot; DEVIATIONS.md 2026-09-04).
The primary metric (probe accuracy) is unchanged. All model-free.

mirror_bias   on UNSTRUCTURED held-out strings, among the model's erroneous
              steps 9-11, the fraction whose value equals the mirror value
              (r9<->r8, r10<->r7, r11<->r6). Chance: 1/2 per erroneous step
              when the mirror value differs from the truth (two wrong digits,
              one of them the mirror), 0 when it coincides.
short_gap     short-mode (answer-only) accuracy on structured minus on
              unstructured held-out items.
late_vs_early conditional error rates from work-mode completions: steps 9-11
              given steps 1-8 correct, vs steps 3-8 given steps 1-2 correct.
volatility    population std of held-out accuracy across checkpoints.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence

from tasks import number_reduction as nr

LATE_STEPS = (9, 10, 11)
EARLY_STEPS = (3, 4, 5, 6, 7, 8)


def mirror_value(responses: Sequence[str], k: int) -> str:
    """Mirror of step k (1-based): step 17-k, i.e. r9<->r8, r10<->r7, r11<->r6."""
    return responses[(17 - k) - 1]


def mirror_bias(items: Sequence[nr.Instance], texts: Sequence[str]) -> dict:
    n_err, n_mirror, chance_sum, n_parsed = 0, 0, 0.0, 0
    for x, t in zip(items, texts):
        steps = nr.parse_steps(t)
        if not steps or len(steps) < 11:
            continue
        n_parsed += 1
        for k in LATE_STEPS:
            truth, got, mv = x.responses[k - 1], steps[k - 1], mirror_value(x.responses, k)
            if got == truth:
                continue
            n_err += 1
            n_mirror += (got == mv)
            chance_sum += 0.5 if mv != truth else 0.0
    return {"mirror_bias": (n_mirror / n_err) if n_err else None, "chance": (chance_sum / n_err) if n_err else None,
            "n_error_steps": n_err, "n_items_parsed": n_parsed}


def short_gap(structured: Sequence[nr.Instance], s_texts: Sequence[str],
              unstructured: Sequence[nr.Instance], u_texts: Sequence[str]) -> dict:
    sa = sum(nr.parse_answer(t) == x.answer for x, t in zip(structured, s_texts)) / max(len(structured), 1)
    ua = sum(nr.parse_answer(t) == x.answer for x, t in zip(unstructured, u_texts)) / max(len(unstructured), 1)
    return {"structured_acc": sa, "unstructured_acc": ua, "gap": sa - ua}


def late_vs_early(items: Sequence[nr.Instance], texts: Sequence[str]) -> dict:
    late_err = late_tot = early_err = early_tot = 0
    n_late = n_early = 0
    for x, t in zip(items, texts):
        steps = nr.parse_steps(t)
        if not steps or len(steps) < 11:
            continue
        ok = [steps[i] == x.responses[i] for i in range(11)]
        if all(ok[:8]):
            n_late += 1
            late_tot += 3
            late_err += sum(not ok[k - 1] for k in LATE_STEPS)
        if all(ok[:2]):
            n_early += 1
            early_tot += 6
            early_err += sum(not ok[k - 1] for k in EARLY_STEPS)
    return {"late_error_rate": (late_err / late_tot) if late_tot else None, "n_late": n_late,
            "early_error_rate": (early_err / early_tot) if early_tot else None, "n_early": n_early}


def volatility(values: Sequence[float]) -> float | None:
    vals = [v for v in values if v is not None]
    return statistics.pstdev(vals) if len(vals) >= 2 else None
