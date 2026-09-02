from arms import harness as hz
from backends.scripted import FakeTrainableBackend, ScriptedBackend
from eval import preflight as pf
from tasks import number_reduction as nr


def test_agreement_rows_and_summary():
    items = pf.agreement_items(seed=0, n=12)
    rows = pf.run_agreement(ScriptedBackend("literal"), items)
    s = pf.summarize_agreement(rows)
    assert s["n"] == 12 and s["accuracy"] == 1.0 and s["steps_exact"] == 1.0 and s["kept"] == 12
    noisy = pf.run_agreement(ScriptedBackend("literal", noise=0.5, seed=1), items)
    c = pf.compare_agreement(rows, noisy)
    assert c["n"] == 12 and c["both_correct"] + c["only_a_correct"] == 12 and c["only_b_correct"] == 0


def test_dream_stats_from_kept_rows():
    split = nr.make_split(seed=0)
    rows = pf.run_agreement(FakeTrainableBackend(), split.heldout[:6])
    # dream seeds must be training instances in the real night; here we only exercise the stats
    d = pf.dream_stats(FakeTrainableBackend(), [dict(r, digits=t.digits) for r, t in zip(rows, split.train[:6])], split, n_seed=4)
    assert d["seeds"] == 4 and d["generated"] == 8 and d["kept"] + sum(d["rejected"].values()) == 8
    assert d["dream_completion_tokens_mean"] is not None


def test_projected_ratio_is_one_when_lengths_match():
    r = pf.project_training_token_ratio(100.0, 100.0, 0.5, keep_rate=0.5)
    assert abs(r["ratio_sleep_over_online"] - 1.0) < 1e-9 and r["within_5pct"]
    r = pf.project_training_token_ratio(100.0, 130.0, 0.5, keep_rate=0.5)   # dreams 30% longer, half the pool
    assert r["ratio_sleep_over_online"] > 1.05 and not r["within_5pct"]
    r = pf.project_training_token_ratio(100.0, 130.0, 0.0, keep_rate=0.5)   # no dreams survive -> Online == Sleep
    assert r["within_5pct"]
