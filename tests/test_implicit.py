"""Secondary insight metrics added after the pilot (DEVIATIONS.md 2026-09-04);
the primary metric (probe) is unchanged. Model-free."""
import statistics

import pytest

from eval import implicit
from tasks import number_reduction as nr

SPLIT = nr.make_split(seed=0)


def test_unstructured_heldout_items():
    items = nr.unstructured_heldout(SPLIT, n=120, seed=0)
    assert len(items) == 120 and len({x.digits for x in items}) == 120
    assert all(not x.structured and x.prefix in SPLIT.heldout_prefixes for x in items)
    from collections import Counter
    assert Counter(x.answer for x in items) == {"1": 40, "4": 40, "9": 40}
    assert [x.digits for x in nr.unstructured_heldout(SPLIT, 120, 0)] == [x.digits for x in items]
    assert not {x.digits for x in items} & {x.digits for x in SPLIT.heldout + SPLIT.probe}


def test_mirror_value_definition():
    x = SPLIT.heldout[0]                      # structured: r9=r8, r10=r7, r11=r6
    r = x.responses
    assert implicit.mirror_value(r, 9) == r[7] and implicit.mirror_value(r, 10) == r[6] and implicit.mirror_value(r, 11) == r[5]
    assert all(r[k - 1] == implicit.mirror_value(r, k) for k in (9, 10, 11))


def _literal(items):
    return [nr.gold_response(x, "work") for x in items]


def _mirror_biased(items):
    """Solver that computes the chain but overwrites steps 9-11 with the mirror
    values (as if it had internalised r_k = r_{17-k})."""
    out = []
    for x in items:
        steps = list(x.responses)
        for k in (9, 10, 11):
            steps[k - 1] = implicit.mirror_value(x.responses, k)
        out.append(f"STEPS: {' '.join(steps)}\nANSWER: {steps[-1]}")
    return out


def _random_wrong(items, seed=0):
    import random
    rng = random.Random(seed)
    out = []
    for x in items:
        steps = list(x.responses)
        for k in (9, 10, 11):
            steps[k - 1] = rng.choice([d for d in nr.DIGITS if d != steps[k - 1]])
        out.append(f"STEPS: {' '.join(steps)}\nANSWER: {steps[-1]}")
    return out


def test_mirror_bias_on_unstructured_items():
    items = nr.unstructured_heldout(SPLIT, n=120, seed=0)
    lit = implicit.mirror_bias(items, _literal(items))
    assert lit["n_error_steps"] == 0 and lit["mirror_bias"] is None
    mb = implicit.mirror_bias(items, _mirror_biased(items))
    assert mb["n_error_steps"] > 0 and mb["mirror_bias"] == 1.0
    rw = implicit.mirror_bias(items, _random_wrong(items))
    assert 0.3 < rw["mirror_bias"] < 0.7 and abs(rw["mirror_bias"] - rw["chance"]) < 0.15
    # chance: for an erroneous step, P(error == mirror) = 1/2 when mirror != truth, 0 when mirror == truth
    assert 0.2 < rw["chance"] <= 0.5


def test_short_gap():
    structured = SPLIT.heldout[:120]
    unstructured = nr.unstructured_heldout(SPLIT, n=120, seed=0)
    shortcut = lambda items: [f"ANSWER: {x.shortcut_response}" for x in items]     # answers r6 always
    g = implicit.short_gap(structured, shortcut(structured), unstructured, shortcut(unstructured))
    assert g["structured_acc"] == 1.0 and g["unstructured_acc"] < 0.6 and g["gap"] > 0.4
    literal = lambda items: [f"ANSWER: {x.answer}" for x in items]
    g2 = implicit.short_gap(structured, literal(structured), unstructured, literal(unstructured))
    assert g2["gap"] == 0.0


def test_late_vs_early_conditional_error_rates():
    items = SPLIT.heldout[:60]
    lit = implicit.late_vs_early(items, _literal(items))
    assert lit["late_error_rate"] == 0.0 and lit["early_error_rate"] == 0.0 and lit["n_late"] == 60 and lit["n_early"] == 60
    # errors only at steps 9-11: late rate 1.0 (all 3 late steps wrong for every item), early rate 0
    rw = implicit.late_vs_early(items, _random_wrong(items))
    assert rw["late_error_rate"] == 1.0 and rw["early_error_rate"] == 0.0
    # unparsable completions are excluded from the denominators
    part = implicit.late_vs_early(items, ["nonsense"] * 60)
    assert part["n_late"] == 0 and part["late_error_rate"] is None


def test_volatility():
    assert implicit.volatility([0.5, 0.5, 0.5]) == 0.0
    assert implicit.volatility([0.373, 0.527, 0.522, 0.692]) == pytest.approx(statistics.pstdev([0.373, 0.527, 0.522, 0.692]))
    assert implicit.volatility([0.4]) is None


def test_short_gap_reports_unparsable_fraction():
    structured = SPLIT.heldout[:6]
    unstructured = nr.unstructured_heldout(SPLIT, n=6, seed=0)
    good = [f"ANSWER: {x.answer}" for x in structured]
    cut = ["1,4->9\n9,1->4\n4,9->1"] * 6                      # work lines, no ANSWER within the short budget
    g = implicit.short_gap(structured, good, unstructured, cut)
    assert g["structured_unparsable"] == 0.0 and g["unstructured_unparsable"] == 1.0
    assert g["unstructured_acc"] == 0.0 and g["gap"] == 1.0
