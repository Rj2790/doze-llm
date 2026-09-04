"""Episode harness + frozen-weight arms (PREREG §5, §6). Model-free."""
import pytest

from arms import harness as hz
from arms.awake import AwakeArm
from arms.baseline import BaselineArm
from backends.scripted import ScriptedBackend
from eval import compute_ledger as cl
from tasks import number_reduction as nr


def test_checkpoint_trigger_is_every_k_episodes_one_based():
    # Episodes are 1-based. A night/checkpoint happens after episode 50, 100, ... 600.
    assert not hz.is_checkpoint(0, 50)
    assert not hz.is_checkpoint(49, 50)
    assert hz.is_checkpoint(50, 50)
    assert not hz.is_checkpoint(51, 50)
    assert hz.is_checkpoint(600, 50)
    assert sum(hz.is_checkpoint(e, 50) for e in range(1, 601)) == 12


def test_episode_sequence_is_seeded_and_train_only():
    cfg = hz.RunConfig(seed=0, n_episodes=600)
    split = nr.make_split(seed=cfg.seed)
    a = hz.episode_sequence(split, cfg)
    b = hz.episode_sequence(split, cfg)
    assert len(a) == 600 and [x.digits for x in a] == [x.digits for x in b]
    assert all(x.prefix in split.train_prefixes for x in a)
    assert not any(x.prefix in split.heldout_prefixes for x in a)
    c = hz.episode_sequence(nr.make_split(seed=1), hz.RunConfig(seed=1, n_episodes=600))
    assert [x.digits for x in c] != [x.digits for x in a]


def _cfg(**kw):
    base = dict(seed=0, n_episodes=100, k=50, n_probe=30, n_heldout_eval=20, mode="work")
    base.update(kw)
    return hz.RunConfig(**base)


def test_baseline_literal_solver_never_meets_criterion():
    res = hz.run_arm(BaselineArm(), ScriptedBackend("literal", noise=0.0), _cfg())
    assert [c.episode for c in res.checkpoints] == [0, 50, 100]          # A5: episode-0 checkpoint
    assert all(c.heldout_accuracy == 1.0 for c in res.checkpoints)
    assert all(not c.probe.above_chance for c in res.checkpoints)
    assert res.criterion_episode is None                       # censored
    assert res.ledger.arm == "baseline" and res.ledger.gradient_steps == 0
    assert res.ledger.tokens_generated > 0
    assert [c["episode"] for c in res.ledger.checkpoints] == [0, 50, 100]
    assert res.ledger.checkpoints[0]["tokens_generated"] == 0
    assert len(res.episodes) == 100 and all(e.correct for e in res.episodes)


def test_baseline_shortcut_solver_meets_criterion_at_second_checkpoint():
    res = hz.run_arm(BaselineArm(), ScriptedBackend("shortcut"), _cfg())
    assert all(c.probe.accuracy == 1.0 for c in res.checkpoints)
    assert res.criterion_episode == 50                          # checkpoints 0 and 50 are consecutive


def test_eval_items_come_only_from_heldout_prefixes_and_probe_is_disjoint():
    cfg = _cfg()
    split = nr.make_split(seed=cfg.seed)
    probe, heldout = hz.eval_items(split, cfg)
    assert len(probe) == 30 and len(heldout) == 20
    assert all(x.prefix in split.heldout_prefixes for x in probe + heldout)
    assert not {x.digits for x in probe} & {x.digits for x in split.heldout}   # C7
    from collections import Counter
    assert Counter(nr.probe_target(x) for x in probe) == {"1": 10, "4": 10, "9": 10}
    full = hz.eval_items(split, hz.RunConfig(seed=0))
    assert len(full[0]) == 60 and len(full[1]) == 201
    with pytest.raises(ValueError):
        hz.eval_items(split, hz.RunConfig(seed=0, n_probe=90))


def test_same_seed_is_reproducible_scripted():
    """A1: same seed, fresh backend -> identical run (timings excluded)."""
    from backends.scripted import FakeTrainableBackend
    from arms.sleep import SleepArm

    def run():
        res = hz.run_arm(SleepArm(dream=True), FakeTrainableBackend(seed=0), _cfg())
        return [(e.digits, e.text, e.tokens) for e in res.episodes], \
               [(c.episode, c.probe.correct, c.heldout_accuracy) for c in res.checkpoints], \
               [{k: v for k, v in n.items() if k != "seconds"} for n in res.nights], \
               (res.ledger.tokens_generated, res.ledger.gradient_steps, res.ledger.training_tokens)
    assert run() == run()


def test_awake_spends_its_token_budget_cumulatively():
    budget = 40
    arm = AwakeArm(token_budget_per_episode=budget)
    res = hz.run_arm(arm, ScriptedBackend("literal"), _cfg(n_episodes=10, k=5))
    total = res.ledger.tokens_generated
    one_completion = max(e.tokens for e in res.episodes)
    assert 10 * budget <= total <= 10 * budget + one_completion
    assert sum(e.rounds for e in res.episodes) > 10                # some self-critique happened
    assert res.ledger.gradient_steps == 0 and res.ledger.arm == "awake"
    # eval attempts do not touch the day accounting
    assert arm.spent == total and arm.allowed == 10 * budget


def test_awake_budget_from_reference_ledger_matches_within_tolerance():
    sleep = cl.Ledger(arm="sleep", backend="scripted-literal"); sleep.add(tokens_generated=4000)
    budget = AwakeArm.budget_from_reference(sleep, n_episodes=100)
    assert budget == 40
    res = hz.run_arm(AwakeArm(token_budget_per_episode=budget), ScriptedBackend("literal"), _cfg())
    cl.check_matched({"sleep": sleep, "awake": res.ledger}, tol=0.05)


def test_eval_tokens_are_tracked_separately_from_matched_budget():
    res = hz.run_arm(BaselineArm(), ScriptedBackend("literal"), _cfg())
    assert res.eval_tokens > 0
    day_tokens = sum(e.tokens for e in res.episodes)
    assert res.ledger.tokens_generated == day_tokens


def test_wall_clock_is_recorded_per_episode_and_checkpoint():
    res = hz.run_arm(BaselineArm(), ScriptedBackend("literal"), _cfg())
    assert all(e.seconds >= 0 for e in res.episodes)
    assert res.checkpoints[0].day_seconds_mean is None and res.checkpoints[0].night_seconds is None
    assert all(c.day_seconds_mean is not None and c.night_seconds is not None for c in res.checkpoints[1:])
    assert "day_seconds_mean" in res.checkpoints[1].flat() and "night_seconds" in res.checkpoints[1].flat()


def test_run_result_json_roundtrip(tmp_path):
    res = hz.run_arm(BaselineArm(), ScriptedBackend("literal"), _cfg(n_episodes=50))
    p = tmp_path / "run.json"
    res.save(p)
    d = hz.load_run(p)
    assert d["arm"] == "baseline" and d["config"]["seed"] == 0 and d["backend"] == "scripted-literal"
    assert [c["episode"] for c in d["checkpoints"]] == [0, 50] and "probe_accuracy" in d["checkpoints"][0]
    assert d["matched"] is None                                  # A3: verdict filled in by eval/analyze.py


def test_awake_uses_probe_specific_critique():
    """C4: at probe time the critique prompt asks for the best guess from the
    visible digits; critique rounds still happen."""
    from arms import awake as aw
    x = nr.make_split(seed=0).probe[0]
    probe_prompt = nr.format_probe(x)
    day_prompt = nr.format_prompt(x, "work")
    assert aw.CRITIQUE_PROBE != aw.CRITIQUE
    assert aw.critique_prompt(probe_prompt, "ANSWER: 1").endswith("ANSWER: 1")
    assert aw.CRITIQUE_PROBE.strip().splitlines()[0] in aw.critique_prompt(probe_prompt, "ANSWER: 1")
    assert aw.CRITIQUE.strip().splitlines()[0] in aw.critique_prompt(day_prompt, "x")
    arm = AwakeArm(token_budget_per_episode=6)
    r, rounds = arm.attempt(probe_prompt, ScriptedBackend("literal"), max_tokens=64, phase="eval")
    assert rounds >= 2 and nr.parse_answer(r.text) is not None


# ---- B1: batched eval generation and per-checkpoint saving ----------------

class CountingBackend(ScriptedBackend):
    def __init__(self):
        super().__init__("literal")
        self.calls = []

    def generate(self, prompts, max_tokens=128, temperature=0.0):
        self.calls.append(len(prompts))
        return super().generate(prompts, max_tokens=max_tokens, temperature=temperature)


def test_checkpoint_eval_is_batched_for_single_shot_arms():
    be = CountingBackend()
    hz.run_arm(BaselineArm(), be, _cfg(n_episodes=50, k=50, n_probe=30, n_heldout_eval=20))
    # two checkpoints (0, 50): probe batch of 30 and held-out batch of 20 each; day episodes are single
    assert be.calls.count(30) == 2 and be.calls.count(20) == 2
    assert sum(c == 1 for c in be.calls) == 50


def test_awake_eval_is_round_batched_and_matches_sequential():
    """Awake's eval attempt_many batches each critique round across prompts;
    per-prompt results and rounds equal the sequential attempt() loop, and
    the number of generate calls is bounded by max_rounds, not by prompts."""
    items = nr.make_split(seed=0).heldout[:20]
    prompts = [nr.format_prompt(x, "work") for x in items] + [nr.format_probe(x) for x in items]
    # shortcut policy: deterministic on probes too (literal guesses probes from its RNG, so call order matters)
    seq = [AwakeArm(token_budget_per_episode=40).attempt(p, ScriptedBackend("shortcut"), 64, phase="eval") for p in prompts]
    be = CountingBackend(); be.policy = "shortcut"
    bat = AwakeArm(token_budget_per_episode=40).attempt_many(prompts, be, 64, phase="eval")
    assert [(r.text, r.completion_tokens, k) for r, k in bat] == [(r.text, r.completion_tokens, k) for r, k in seq]
    assert len(be.calls) <= AwakeArm(token_budget_per_episode=40).max_rounds and max(be.calls) == 40
    assert all(k >= 2 for _, k in bat)                                  # critique happened at probe time too (C4)


def test_partial_results_saved_after_every_checkpoint(tmp_path):
    class Boom(BaselineArm):
        def attempt(self, prompt, backend, max_tokens, phase="day"):
            if phase == "day" and "boom" in getattr(self, "flag", ""):
                raise RuntimeError("boom")
            return super().attempt(prompt, backend, max_tokens, phase)

    out = tmp_path / "partial.json"
    arm = Boom()

    def log(msg):
        if "ep=50" in msg:
            arm.flag = "boom"            # crash during the second day

    with pytest.raises(RuntimeError):
        hz.run_arm(arm, ScriptedBackend("literal"), _cfg(), save_path=out, log=log)
    d = hz.load_run(out)
    assert [c["episode"] for c in d["checkpoints"]] == [0, 50] and d["partial"] is True
    res = hz.run_arm(BaselineArm(), ScriptedBackend("literal"), _cfg(), save_path=out)
    d = hz.load_run(out)
    assert [c["episode"] for c in d["checkpoints"]] == [0, 50, 100] and d["partial"] is False
    assert len(res.checkpoints) == 3


# ---- post-pilot checkpoint fields (DEVIATIONS 2026-09-04) ----------------------

def test_checkpoint_records_implicit_metrics_rows_and_control_parse_rate(tmp_path):
    from eval import control_bench as cb
    cfg = _cfg(n_episodes=50, k=50, n_probe=12, n_heldout_eval=9, n_implicit=12)
    items = cb.load_sample(tmp_path / "x.jsonl", n=4, seed=0, ids=None) if False else None
    res = hz.run_arm(BaselineArm(), ScriptedBackend("literal"), cfg, save_path=tmp_path / "r.json")
    c = res.checkpoints[0]
    f = c.flat()
    for k in ("mirror_bias", "mirror_bias_chance", "mirror_bias_n", "short_structured_acc", "short_unstructured_acc",
              "short_gap", "late_error_rate", "early_error_rate", "control_unparsable"):
        assert k in f, k
    assert f["short_structured_acc"] == 1.0 and f["short_gap"] == 0.0        # literal solver
    assert f["control_unparsable"] is None                                   # no control items configured
    assert len(c.heldout_rows) == 9 and set(c.heldout_rows[0]) == {"digits", "steps", "answer"}
    d = hz.load_run(tmp_path / "r.json")
    assert len(d["checkpoints"][0]["heldout_rows"]) == 9
    assert (tmp_path / "r.json.state").is_dir()                               # final state saved too


def test_implicit_metrics_can_be_disabled():
    res = hz.run_arm(BaselineArm(), ScriptedBackend("literal"), _cfg(n_episodes=50, k=50, n_implicit=0))
    assert res.checkpoints[0].flat()["mirror_bias"] is None and res.checkpoints[0].flat()["short_gap"] is None
