"""Final-analysis machinery (PREREG §7): per-seed summaries, paired sign-flip
test, two-sample log-rank with censoring, hypothesis verdicts, records layout."""
import json
import math

import pytest

from eval import analyze, stats
from eval import compute_ledger as cl


# ---- statistics -------------------------------------------------------------

def test_sign_flip_exact_pvalue():
    # n=5 all positive: only the all-positive and all-negative flips are as extreme -> p = 2/32
    r = stats.sign_flip_test([0.1, 0.2, 0.15, 0.3, 0.05])
    assert r["n"] == 5 and r["mean"] == pytest.approx(0.16) and r["p"] == pytest.approx(2 / 32)
    r2 = stats.sign_flip_test([0.1, -0.1, 0.2, -0.2, 0.0])
    assert r2["mean"] == pytest.approx(0.0) and r2["p"] == 1.0
    assert stats.sign_flip_test([0.3])["p"] == 1.0                       # n=1: no test possible
    assert stats.sign_flip_test([])["n"] == 0 and stats.sign_flip_test([])["p"] is None


def test_logrank_two_sample():
    # identical groups -> statistic ~0, p ~1
    a = [(100, True), (200, True), (300, True)]
    b = [(100, True), (200, True), (300, True)]
    r = stats.logrank(a, b)
    assert r["events"] == 6 and abs(r["statistic"]) < 1e-9 and r["p"] == pytest.approx(1.0)
    # group a reaches criterion early, group b never (censored at 600) -> significant
    a = [(50, True), (100, True), (100, True), (150, True), (200, True)]
    b = [(600, False)] * 5
    r = stats.logrank(a, b)
    assert r["events"] == 5 and r["statistic"] > 3.84 and r["p"] < 0.05
    # all censored -> degenerate, reported as such
    r = stats.logrank([(600, False)] * 5, [(600, False)] * 5)
    assert r["events"] == 0 and r["p"] is None and r["degenerate"]


def test_chi2_sf_one_df():
    assert stats.chi2_sf_1df(3.841) == pytest.approx(0.05, abs=0.001)
    assert stats.chi2_sf_1df(0.0) == 1.0
    assert stats.chi2_sf_1df(10.83) == pytest.approx(0.001, abs=0.0002)


# ---- per-seed summaries and verdicts -----------------------------------------

def _run(arm, seed, probe, held, ctrl, crit=None, gen=5000, steps=300, tok=60000):
    cks = [{"episode": 50 * i, "probe_accuracy": p, "probe_p_value": 0.5, "probe_above_chance": False, "probe_n": 60,
            "probe_correct": round(p * 60), "heldout_accuracy": h, "heldout_n": 201, "median_tokens_correct": 98,
            "control_accuracy": c, "control_unparsable": 0.0, "eval_tokens": 1, "seconds": 1.0,
            "ledger_tokens_generated": gen, "ledger_gradient_steps": steps, "ledger_training_tokens": tok, "ledger_gpu_seconds": 1.0}
           for i, (p, h, c) in enumerate(zip(probe, held, ctrl))]
    return {"arm": arm, "backend": "hf:Qwen/Qwen3-4B", "config": {"seed": seed, "n_episodes": 600, "k": 50},
            "criterion_episode": crit, "eval_tokens": 1, "ledger": {"arm": arm, "backend": "hf:Qwen/Qwen3-4B",
            "tokens_generated": gen, "gradient_steps": steps, "training_tokens": tok, "gpu_seconds": 1.0, "checkpoints": []},
            "matched": None, "partial": False, "checkpoints": cks, "nights": [], "episodes": []}


def _write_records(root, seeds, crit_sleep=None):
    for s in seeds:
        d = root / f"seed{s}"; (d / "runs").mkdir(parents=True); (d / "ledgers").mkdir()
        arms = {
            "sleep": _run("sleep", s, [0.33] * 13, [0.4] + [0.9] * 12, [0.58] + [0.60] * 12, crit=crit_sleep, gen=5000, steps=300, tok=60000),
            "online": _run("online", s, [0.33] * 13, [0.4] + [0.85] * 12, [0.58] + [0.75] * 12, gen=1000, steps=296, tok=59000),
            "sleep_nodream": _run("sleep_nodream", s, [0.33] * 13, [0.4] + [0.9] * 12, [0.58] + [0.65] * 12, gen=1000, steps=300, tok=60000),
            "awake": _run("awake", s, [0.30] * 13, [0.45] * 13, [0.59] * 13, gen=5100, steps=0, tok=0),
            "baseline": _run("baseline", s, [0.37] * 13, [0.4] * 13, [0.58] * 13, gen=1000, steps=0, tok=0),
        }
        for a, r in arms.items():
            (d / "runs" / f"{a}_seed{s}.json").write_text(json.dumps(r))
            L = cl.Ledger(arm=a, backend="hf:Qwen/Qwen3-4B"); L.add(tokens_generated=r["ledger"]["tokens_generated"], gradient_steps=r["ledger"]["gradient_steps"], training_tokens=r["ledger"]["training_tokens"])
            L.save(d / "ledgers" / f"{a}_seed{s}.json")


def test_records_layout_and_seed_summaries(tmp_path):
    _write_records(tmp_path, [0, 1, 2])
    fa = analyze.final_analysis(tmp_path, seeds=[0, 1, 2], layout="per-seed")
    assert fa["seeds_checked"] == [0, 1, 2] and all(v["ok"] for v in fa["verdicts"])
    s = fa["per_seed"]["sleep"][0]
    assert s["final_heldout"] == 0.9 and s["peak_heldout"] == 0.9 and s["last3_heldout"] == pytest.approx(0.9)
    assert s["forgetting"] == pytest.approx(0.02) and s["probe_max"] == 0.33 and s["criterion_episode"] is None
    # paired comparisons exist with sign-flip p
    h4 = fa["paired"]["H4_forgetting_sleep_minus_online"]
    assert h4["n"] == 3 and h4["mean"] == pytest.approx(0.02 - 0.17) and 0 < h4["p"] <= 1.0


def test_hypothesis_verdicts_all_censored(tmp_path):
    _write_records(tmp_path, [0, 1, 2, 3, 4])
    fa = analyze.final_analysis(tmp_path, seeds=[0, 1, 2, 3, 4], layout="per-seed")
    H = fa["hypotheses"]
    assert H["H1"]["verdict"] == "not supported" and "censored" in H["H1"]["note"]
    assert H["H2"]["verdict"] == "not supported" and H["H2"]["logrank"]["degenerate"]
    assert H["H3"]["verdict"] == "not supported"
    # H4 predicted Sleep forgets less (delta larger) than Online; fixture has the opposite -> not supported, direction reversed
    assert H["H4"]["verdict"] == "not supported" and H["H4"]["direction"] == "reversed"
    md = analyze.final_report_markdown(fa)
    assert "H2" in md and "not supported" in md and "Exploratory" in md and "preliminary" not in md.lower()


def test_hypothesis_supported_when_sleep_hits_criterion(tmp_path):
    _write_records(tmp_path, [0, 1, 2, 3, 4], crit_sleep=150)
    fa = analyze.final_analysis(tmp_path, seeds=[0, 1, 2, 3, 4], layout="per-seed")
    assert fa["hypotheses"]["H2"]["logrank"]["events"] == 5 and fa["hypotheses"]["H2"]["logrank"]["p"] < 0.05
    assert fa["hypotheses"]["H2"]["verdict"] == "supported"


def test_preliminary_flag_when_seeds_incomplete(tmp_path):
    _write_records(tmp_path, [0, 1])
    fa = analyze.final_analysis(tmp_path, seeds=[0, 1], layout="per-seed", required_seeds=5)
    assert fa["preliminary"] is True
    assert "PRELIMINARY" in analyze.final_report_markdown(fa)


def test_exploratory_schedule_block(tmp_path):
    _write_records(tmp_path, [0, 1, 2])
    fa = analyze.final_analysis(tmp_path, seeds=[0, 1, 2], layout="per-seed")
    ex = fa["exploratory"]["schedule"]
    assert set(ex) >= {"online", "sleep", "sleep_nodream"}
    assert ex["online"]["gsm8k_delta_mean"] == pytest.approx(0.17) and ex["sleep"]["gsm8k_delta_mean"] == pytest.approx(0.02)
    assert "post_update_gsm8k_change_mean" in ex["sleep"]
