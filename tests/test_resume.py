"""Preemption safety: resumable runs and idempotent orchestration (model-free).

A run interrupted after checkpoint c and resumed from its partial file must
reproduce the uninterrupted run exactly: episodes, checkpoints, ledger,
nights. Backend + arm state (adapter/optimizer/RNG/buffer/pools) are saved
at every checkpoint; episodes after the last checkpoint are redone."""
import json
import time

import pytest

from arms import harness as hz
from arms.awake import AwakeArm
from arms.baseline import BaselineArm
from arms.online import OnlineArm
from arms.sleep import SleepArm
from backends.scripted import FakeTrainableBackend, ScriptedBackend
from eval import orchestrate


def _cfg(**kw):
    base = dict(seed=0, n_episodes=100, k=25, n_probe=12, n_heldout_eval=6, mode="work")
    base.update(kw)
    return hz.RunConfig(**base)


def _fingerprint(res):
    return {
        "episodes": [(e.episode, e.digits, e.text, e.tokens, e.rounds) for e in res.episodes],
        "checkpoints": [(c.episode, c.probe.correct, round(c.heldout_accuracy, 6), c.median_tokens_correct,
                         c.ledger["tokens_generated"], c.ledger["gradient_steps"], c.ledger["training_tokens"])
                        for c in res.checkpoints],
        "ledger": (res.ledger.tokens_generated, res.ledger.gradient_steps, res.ledger.training_tokens),
        "nights": [{k: v for k, v in n.items() if k not in ("seconds",)} for n in res.nights],
        "criterion": res.criterion_episode,
    }


class Crash(RuntimeError):
    pass


def _crash_after(arm_cls_factory, crash_episode):
    """Wrap an arm so its day attempt raises at `crash_episode`."""
    arm = arm_cls_factory()
    seen = {"n": 0}
    orig = arm.attempt

    def attempt(prompt, backend, max_tokens, phase="day"):
        if phase == "day":
            seen["n"] += 1
            if seen["n"] == crash_episode:
                raise Crash(f"crash at day attempt {crash_episode}")
        return orig(prompt, backend, max_tokens, phase)

    arm.attempt = attempt
    return arm


ARMS = {
    "baseline": (lambda: BaselineArm(), lambda: ScriptedBackend("literal", noise=0.2)),
    "awake": (lambda: AwakeArm(token_budget_per_episode=40), lambda: ScriptedBackend("literal", noise=0.2)),
    "online": (lambda: OnlineArm(), lambda: FakeTrainableBackend(noise=0.3, learn_after_steps=60)),
    "sleep": (lambda: SleepArm(dream=True), lambda: FakeTrainableBackend(noise=0.3, dream_error_rate=0.3)),
}


@pytest.mark.parametrize("name", list(ARMS))
def test_resume_reproduces_uninterrupted_run(tmp_path, name):
    make_arm, make_backend = ARMS[name]
    cfg = _cfg()
    full = hz.run_arm(make_arm(), make_backend(), cfg)

    out = tmp_path / f"{name}.json"
    with pytest.raises(Crash):
        hz.run_arm(_crash_after(make_arm, 63), make_backend(), cfg, save_path=out)   # dies in block 3 (ep 51-75)
    d = hz.load_run(out)
    assert d["partial"] is True and [c["episode"] for c in d["checkpoints"]] == [0, 25, 50]
    assert (tmp_path / f"{name}.json.state").is_dir()

    resumed = hz.run_arm(make_arm(), make_backend(), cfg, save_path=out, resume=True)
    assert _fingerprint(resumed) == _fingerprint(full)
    d2 = hz.load_run(out)
    assert d2["partial"] is False and len(d2["episodes"]) == 100 and d2["resumed_from"] == 50


def test_resume_of_complete_run_returns_it_without_rerunning(tmp_path):
    out = tmp_path / "b.json"
    res = hz.run_arm(BaselineArm(), ScriptedBackend("literal"), _cfg(n_episodes=50), save_path=out)
    be = ScriptedBackend("literal")
    calls = []
    orig = be.generate
    be.generate = lambda *a, **k: (calls.append(1), orig(*a, **k))[1]
    again = hz.run_arm(BaselineArm(), be, _cfg(n_episodes=50), save_path=out, resume=True)
    assert calls == [] and _fingerprint(again) == _fingerprint(res)


def test_resume_ignored_when_no_file(tmp_path):
    out = tmp_path / "none.json"
    res = hz.run_arm(BaselineArm(), ScriptedBackend("literal"), _cfg(n_episodes=25), save_path=out, resume=True)
    assert len(res.episodes) == 25 and hz.load_run(out)["resumed_from"] is None


def test_fake_backend_state_roundtrip(tmp_path):
    be = FakeTrainableBackend(noise=0.3, dream_error_rate=0.2, seed=3)
    be.train([("p", "a b c")], steps=3, seed=1)
    be.generate(["Digits: 1 4 9 1 4 9 1 4 9 1 4 9\nSTEPS:"])
    be.save_state(tmp_path)
    a = be.generate(["Digits: 1 4 9 1 4 9 1 4 9 1 4 9\nSTEPS:"])[0].text
    be2 = FakeTrainableBackend(noise=0.3, dream_error_rate=0.2, seed=99)
    be2.load_state(tmp_path)
    b = be2.generate(["Digits: 1 4 9 1 4 9 1 4 9 1 4 9\nSTEPS:"])[0].text
    assert a == b and be2.gradient_steps == 3 and be2.policy == be.policy


# ---- orchestrator decisions (pure) --------------------------------------------

def test_orchestrator_plan():
    now = 1_000_000.0
    # nothing on disk, no known call -> spawn
    assert orchestrate.plan(None, None, now) == "spawn"
    # complete run on disk -> load it, never spawn
    assert orchestrate.plan({"partial": False, "mtime": now - 99999}, None, now) == "load"
    assert orchestrate.plan({"partial": False, "mtime": now - 99999}, "fc-123", now) == "load"
    # in-progress run with a known call id -> re-attach to the call
    assert orchestrate.plan({"partial": True, "mtime": now - 100}, "fc-123", now) == "attach"
    # in-progress run, fresh file, no call id -> wait for whoever is writing it
    assert orchestrate.plan({"partial": True, "mtime": now - 100}, None, now) == "wait"
    # in-progress but stale (> stale_after) -> spawn (with resume) to take over
    assert orchestrate.plan({"partial": True, "mtime": now - 10_000}, None, now, stale_after=3600) == "spawn"
    # known call id but no file yet (container still loading) -> attach
    assert orchestrate.plan(None, "fc-123", now) == "attach"


def test_run_tag_paths():
    p = orchestrate.paths("/results", "20260903T1200Z", "sleep", 0)
    assert p["run"] == "/results/20260903T1200Z/runs/sleep_seed0.json"
    assert p["ledger"] == "/results/20260903T1200Z/ledgers/sleep_seed0.json"
    assert p["calls"] == "/results/20260903T1200Z/grid_calls_seed0.json"
    assert p["summary"] == "/results/20260903T1200Z/grid_seed0.json"
    assert orchestrate.new_tag().endswith("Z") and len(orchestrate.new_tag()) == 14
