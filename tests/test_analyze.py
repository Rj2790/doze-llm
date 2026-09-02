"""A3: analysis loads ledgers via Ledger.load, persists the check_matched
verdict per seed into the run JSONs, and refuses to produce anything on
unmatched or mixed-backend arms."""
import json

import pytest

from eval import analyze
from eval import compute_ledger as cl


def _ledger(arm, backend="hf:Qwen/Qwen3-4B", gen=5000, steps=300, tok=60000):
    L = cl.Ledger(arm=arm, backend=backend)
    L.add(tokens_generated=gen, gradient_steps=steps, training_tokens=tok)
    L.checkpoint(50)
    return L


def _write_seed(root, seed, ledgers, criterion=None):
    for arm, L in ledgers.items():
        L.save(root / "ledgers" / f"{arm}_seed{seed}.json")
        run = {"arm": arm, "backend": L.backend, "config": {"seed": seed, "n_episodes": 100, "k": 50},
               "criterion_episode": criterion, "eval_tokens": 1, "ledger": {}, "matched": None,
               "checkpoints": [{"episode": 0, "probe_accuracy": 0.3, "heldout_accuracy": 0.4, "control_accuracy": 0.5},
                               {"episode": 50, "probe_accuracy": 0.8, "heldout_accuracy": 0.6, "control_accuracy": 0.45}],
               "nights": [], "episodes": []}
        (root / "runs").mkdir(exist_ok=True, parents=True)
        (root / "runs" / f"{arm}_seed{seed}.json").write_text(json.dumps(run))


def _good():
    return {"sleep": _ledger("sleep"), "awake": _ledger("awake", gen=5100, steps=0, tok=0),
            "online": _ledger("online", gen=1000, steps=296, tok=59000),
            "sleep_nodream": _ledger("sleep_nodream", gen=1000, steps=300, tok=60000),
            "baseline": _ledger("baseline", gen=1000, steps=0, tok=0)}


def test_check_seed_persists_verdict(tmp_path):
    _write_seed(tmp_path, 0, _good())
    v = analyze.check_seed(tmp_path, seed=0)
    assert v["ok"] and v["backend"] == "hf:Qwen/Qwen3-4B" and set(v["arms"]) == set(_good())
    for arm in _good():
        d = json.loads((tmp_path / "runs" / f"{arm}_seed0.json").read_text())
        assert d["matched"]["ok"] is True and d["matched"]["seed"] == 0


def test_check_seed_refuses_unmatched(tmp_path):
    bad = _good(); bad["online"] = _ledger("online", gen=1000, steps=340, tok=60000)
    _write_seed(tmp_path, 0, bad)
    with pytest.raises(analyze.AnalysisRefused, match="online.*gradient_steps"):
        analyze.check_seed(tmp_path, seed=0)
    d = json.loads((tmp_path / "runs" / "online_seed0.json").read_text())
    assert d["matched"]["ok"] is False and "gradient_steps" in d["matched"]["message"]


def test_check_seed_refuses_mixed_backends(tmp_path):
    mixed = _good(); mixed["awake"] = _ledger("awake", backend="mlx:foo", gen=5000, steps=0, tok=0)
    _write_seed(tmp_path, 0, mixed)
    with pytest.raises(analyze.AnalysisRefused, match="backend"):
        analyze.check_seed(tmp_path, seed=0)


def test_summarize_only_after_all_seeds_pass(tmp_path):
    _write_seed(tmp_path, 0, _good(), criterion=50)
    _write_seed(tmp_path, 1, _good(), criterion=None)
    report = analyze.summarize(tmp_path, seeds=[0, 1])
    assert "sleep" in report["arms"] and report["arms"]["sleep"]["criterion_episodes"] == [50, None]
    assert report["arms"]["sleep"]["forgetting"][0] == pytest.approx(0.45 - 0.5)
    assert report["seeds_checked"] == [0, 1]
    bad = _good(); bad["awake"] = _ledger("awake", gen=9000, steps=0, tok=0)
    _write_seed(tmp_path, 2, bad)
    with pytest.raises(analyze.AnalysisRefused):
        analyze.summarize(tmp_path, seeds=[0, 1, 2])
