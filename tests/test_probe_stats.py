"""Probe sampling and statistics (CONTEXT.md §4: the fixed-margin flag was an
artifact; replace with stratified sampling + one-sided binomial test)."""
import math
from collections import Counter

import pytest

from eval import shortcut_detector as sd
from tasks import number_reduction as nr


def test_binomial_sf_exact_values():
    # P(X >= k | n, p) against hand-computed values
    assert sd.binom_sf(0, 10, 1 / 3) == 1.0
    assert math.isclose(sd.binom_sf(10, 10, 1 / 3), (1 / 3) ** 10)
    assert math.isclose(sd.binom_sf(1, 2, 0.5), 0.75)
    # the run-3 artifact: 20/40 vs 1/3 is p ~ 0.019 -> not significant at 0.01
    p = sd.binom_sf(20, 40, 1 / 3)
    assert 0.015 < p < 0.025
    # 60 items at 0.70 (the PREREG criterion) is overwhelmingly significant
    assert sd.binom_sf(42, 60, 1 / 3) < 1e-6


def test_stratified_probe_is_balanced_and_deterministic():
    pool = nr.make_split(seed=0).heldout
    items = sd.stratified_probe_items(pool, n=120, seed=0)
    assert len(items) == 120
    assert Counter(nr.probe_target(x) for x in items) == {"1": 40, "4": 40, "9": 40}
    assert len({x.digits for x in items}) == 120
    assert all(x in pool for x in items)
    again = sd.stratified_probe_items(pool, n=120, seed=0)
    assert [x.digits for x in again] == [x.digits for x in items]
    other = sd.stratified_probe_items(pool, n=120, seed=1)
    assert [x.digits for x in other] != [x.digits for x in items]


def test_stratified_probe_rejects_bad_n():
    pool = nr.make_split(seed=0).heldout
    with pytest.raises(ValueError):
        sd.stratified_probe_items(pool, n=40, seed=0)   # not divisible by 3
    with pytest.raises(ValueError):
        sd.stratified_probe_items(pool, n=600, seed=0)  # pool too small


def test_constant_and_last_digit_heuristics_not_flagged():
    """The two heuristics the untrained model actually used (CONTEXT.md §4)
    must not trip the above-chance flag on a stratified probe."""
    pool = nr.make_split(seed=0).heldout
    items = sd.stratified_probe_items(pool, n=120, seed=0)
    for d in nr.DIGITS:
        r = sd.run_probe(lambda p, d=d: f"ANSWER: {d}", items)
        assert math.isclose(r.accuracy, 1 / 3) and not r.above_chance

    def last_visible(prompt: str) -> str:
        toks = nr.parse_digits_line(prompt)
        return f"ANSWER: {[t for t in toks if t != nr.MASK][-1]}"

    r = sd.run_probe(last_visible, items)
    assert not r.above_chance, (r.accuracy, r.p_value)


def test_run_probe_reports_p_value():
    pool = nr.make_split(seed=0).heldout
    items = sd.stratified_probe_items(pool, n=60, seed=0)
    r = sd.run_probe(lambda p: f"ANSWER: {nr.solve(''.join(t for t in nr.parse_digits_line(p) if t != nr.MASK))[-1]}", items)
    assert r.accuracy == 1.0 and r.above_chance and r.p_value < 1e-20
    assert r.n == 60 and r.correct == 60
