"""Exploratory deep-analysis helpers (records/analysis_deep.md). Pure functions
on run dicts; model-free."""
import math

import pytest

from eval import deep_analysis as da


def test_pooled_binomial_two_sided_and_ci():
    r = da.pooled_binomial(correct=20, n=60, p0=1 / 3)
    assert r["rate"] == pytest.approx(1 / 3) and r["p_two_sided"] > 0.8      # discrete: 0.90 at n=60
    r = da.pooled_binomial(correct=1300, n=3900, p0=1 / 3)      # exactly chance, large n
    assert r["p_two_sided"] > 0.9 and r["ci95"][0] < 1 / 3 < r["ci95"][1]
    r = da.pooled_binomial(correct=1500, n=3900, p0=1 / 3)      # +5 points over chance, n=3900 -> tiny p
    assert r["p_two_sided"] < 1e-6 and r["ci95"][0] > 1 / 3
    assert da.pooled_binomial(0, 0, 0.5)["p_two_sided"] is None


def test_block_accuracy_and_error_position():
    eps = [{"episode": i, "correct": i % 2 == 0, "steps_correct": 11 if i % 2 == 0 else i % 11, "tokens": 90 + i}
           for i in range(100)]
    blocks = da.block_accuracy(eps, block=50)
    assert [b["episode_end"] for b in blocks] == [50, 100] and all(b["accuracy"] == 0.5 for b in blocks)
    assert blocks[0]["tokens_mean"] == pytest.approx(sum(90 + i for i in range(50)) / 50)
    pos = da.error_position_hist(eps, n_steps=11)
    assert sum(pos.values()) == 50 and set(pos) <= set(range(0, 11))


def test_series_helpers():
    s = [0.4, 0.9, 0.7, 0.95, 0.8]
    assert da.max_drawdown(s) == pytest.approx(0.2)
    d = da.step_changes(s)
    assert d == pytest.approx([0.5, -0.2, 0.25, -0.15])
    assert da.volatility(s) == pytest.approx(math.sqrt(sum((x - sum(d) / 4) ** 2 for x in d) / 4))
    assert da.max_drawdown([0.5]) == 0.0 and da.volatility([0.5]) == 0.0


def test_buffer_composition():
    buffer = [{"digits": "111111111111", "answer": "1"}, {"digits": "111111111111", "answer": "1"},
              {"digits": "444444444444", "answer": "4"}, {"digits": "999999999999", "answer": "9"}]
    train = {"111111111111", "444444444444", "999999999999", "149149149149"}
    r = da.buffer_composition(buffer, train)
    assert r["n"] == 4 and r["n_distinct_digits"] == 3 and r["train_coverage"] == pytest.approx(3 / 4)
    assert r["answer_dist"] == {"1": 0.5, "4": 0.25, "9": 0.25} and r["outside_train"] == 0


def test_detectable_effect_n5():
    # With n=5 and the exact sign-flip test the smallest attainable two-sided p is 2/32
    assert da.min_attainable_p(5) == pytest.approx(0.0625)
    assert da.min_attainable_p(8) == pytest.approx(2 / 256)
