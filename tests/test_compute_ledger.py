"""PREREG §5 compute matching: Awake tokens = Sleep tokens ±5%; Online (and
Sleep-NoDream) gradient steps and training tokens = Sleep ±5%. The analysis
refuses to compare arms outside tolerance."""
import json

import pytest

from eval import compute_ledger as cl


def _ledger(arm, gen=0, steps=0, train_tok=0, gpu=0.0):
    L = cl.Ledger(arm=arm)
    L.add(tokens_generated=gen, gradient_steps=steps, training_tokens=train_tok, gpu_seconds=gpu)
    return L


def test_ledger_accumulates_and_snapshots():
    L = cl.Ledger(arm="sleep")
    L.add(tokens_generated=100, gpu_seconds=1.0)
    L.add(tokens_generated=50, gradient_steps=3, training_tokens=600, gpu_seconds=2.5)
    assert L.tokens_generated == 150 and L.gradient_steps == 3
    assert L.training_tokens == 600 and L.gpu_seconds == 3.5
    L.checkpoint(episode=50)
    L.add(tokens_generated=10)
    L.checkpoint(episode=100)
    assert [c["episode"] for c in L.checkpoints] == [50, 100]
    assert L.checkpoints[0]["tokens_generated"] == 150
    assert L.checkpoints[1]["tokens_generated"] == 160


def test_ledger_rejects_negative():
    L = cl.Ledger(arm="x")
    with pytest.raises(ValueError):
        L.add(tokens_generated=-1)


def test_ledger_json_roundtrip(tmp_path):
    L = _ledger("online", gen=10, steps=5, train_tok=500, gpu=3.0)
    L.checkpoint(episode=50)
    p = tmp_path / "online.json"
    L.save(p)
    M = cl.Ledger.load(p)
    assert M == L
    assert json.loads(p.read_text())["arm"] == "online"


def test_within_tolerance_helper():
    assert cl.within(100, 105, tol=0.05)
    assert cl.within(105, 100, tol=0.05)
    assert not cl.within(100, 106, tol=0.05)
    assert cl.within(0, 0, tol=0.05)
    assert not cl.within(0, 1, tol=0.05)


def test_check_matched_passes_when_all_within_5pct():
    arms = {
        "baseline": _ledger("baseline", gen=1000),
        "awake": _ledger("awake", gen=5100),                       # 5000 * 1.02
        "online": _ledger("online", gen=1000, steps=296, train_tok=59000),
        "sleep": _ledger("sleep", gen=5000, steps=300, train_tok=60000),
        "sleep_nodream": _ledger("sleep_nodream", gen=1000, steps=300, train_tok=60000),
    }
    cl.check_matched(arms, tol=0.05)  # does not raise


def test_check_matched_raises_on_token_drift():
    arms = {
        "awake": _ledger("awake", gen=5400),                       # +8%
        "online": _ledger("online", steps=300, train_tok=60000),
        "sleep": _ledger("sleep", gen=5000, steps=300, train_tok=60000),
        "sleep_nodream": _ledger("sleep_nodream", steps=300, train_tok=60000),
    }
    with pytest.raises(cl.BudgetMismatch, match="awake.*tokens_generated"):
        cl.check_matched(arms, tol=0.05)


def test_check_matched_raises_on_gradient_drift():
    arms = {
        "awake": _ledger("awake", gen=5000),
        "online": _ledger("online", steps=320, train_tok=60000),   # +6.7% steps
        "sleep": _ledger("sleep", gen=5000, steps=300, train_tok=60000),
        "sleep_nodream": _ledger("sleep_nodream", steps=300, train_tok=60000),
    }
    with pytest.raises(cl.BudgetMismatch, match="online.*gradient_steps"):
        cl.check_matched(arms, tol=0.05)
    arms["online"] = _ledger("online", steps=300, train_tok=64000)  # +6.7% tokens
    with pytest.raises(cl.BudgetMismatch, match="online.*training_tokens"):
        cl.check_matched(arms, tol=0.05)
    arms["online"] = _ledger("online", steps=300, train_tok=60000)
    arms["sleep_nodream"] = _ledger("sleep_nodream", steps=280, train_tok=60000)  # -6.7%
    with pytest.raises(cl.BudgetMismatch, match="sleep_nodream.*gradient_steps"):
        cl.check_matched(arms, tol=0.05)


def test_check_matched_requires_sleep_reference():
    with pytest.raises(cl.BudgetMismatch, match="sleep"):
        cl.check_matched({"awake": _ledger("awake", gen=10)}, tol=0.05)


def test_check_matched_never_mixes_backends():
    sleep = _ledger("sleep", gen=5000, steps=300, train_tok=60000); sleep.backend = "hf"
    awake = _ledger("awake", gen=5000); awake.backend = "mlx"
    with pytest.raises(cl.BudgetMismatch, match="backend"):
        cl.check_matched({"sleep": sleep, "awake": awake}, tol=0.05)


def test_online_unfiltered_is_matched_like_online():
    assert ("online_unfiltered", "gradient_steps") in cl.MATCH_RULES
    arms = {"sleep": _ledger("sleep", gen=5000, steps=300, train_tok=60000),
            "online_unfiltered": _ledger("online_unfiltered", steps=330, train_tok=60000)}
    with pytest.raises(cl.BudgetMismatch, match="online_unfiltered.*gradient_steps"):
        cl.check_matched(arms, tol=0.05)
