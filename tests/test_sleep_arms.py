"""Online / Sleep / Sleep-NoDream arms on a fake trainable backend (PREREG §5).
Checks the silent-failure watchlist: night fires after episode K and before
the checkpoint; S = K steps per night so gradient budgets match Online by
construction; dreams enter training only after verification; the buffer
never holds a held-out prefix; Awake budget derived from Sleep matches."""
from arms import harness as hz
from arms.awake import AwakeArm
from arms.online import OnlineArm
from arms.sleep import SleepArm, SleepConfig
from backends.scripted import FakeTrainableBackend
from eval import compute_ledger as cl
from tasks import number_reduction as nr


def _cfg(**kw):
    base = dict(seed=0, n_episodes=100, k=50, n_probe=30, n_heldout_eval=12, mode="work")
    base.update(kw)
    return hz.RunConfig(**base)


def test_fake_trainable_backend_accounting():
    be = FakeTrainableBackend()
    exs = [("p1", "a b c"), ("p2", "d e")]
    st = be.train(exs, steps=5, seed=0)
    assert st.steps == 5 and st.training_tokens == 3 + 2 + 3 + 2 + 3   # cycles in order
    assert be.gradient_steps == 5 and be.decays == 0
    be.decay_adapter(0.05)
    assert be.decays == 1


def test_online_filtered_one_step_every_episode():
    """A2: Online trains exactly one step per episode: on the current
    trajectory if kept, else a random one from today's kept pool, else the
    most recent kept ever, else skip and log. Steps == K per K episodes."""
    be = FakeTrainableBackend(noise=0.3)                  # ~30% wrong answers -> not kept
    arm = OnlineArm()
    res = hz.run_arm(arm, be, _cfg())
    assert arm.name == "online"
    n_wrong = sum(not e.correct for e in res.episodes)
    assert 10 < n_wrong < 60                              # the fixture really produces misses
    assert res.ledger.gradient_steps + arm.skipped == 100 and arm.skipped <= 3   # skips only before the first kept
    assert [c["gradient_steps"] for c in res.ledger.checkpoints] == [0, 50 - arm.skipped, 100 - arm.skipped]
    assert res.nights == [] and be.decays == 0
    # every trained completion is a kept trajectory of the right instance
    kept_texts = {e.text.strip() for e in res.episodes if e.correct}
    for prompt, completion in be.trained:
        assert completion in kept_texts
        digits = "".join(nr.parse_digits_line(prompt))
        assert nr.score(nr.Instance.from_digits(digits), completion)["correct"]
    assert arm.substitutions["today"] + arm.substitutions["recent"] == n_wrong - arm.skipped


def test_online_skips_and_logs_when_nothing_kept_yet():
    be = FakeTrainableBackend(noise=1.0)                  # everything wrong
    arm = OnlineArm()
    res = hz.run_arm(arm, be, _cfg(n_episodes=10, k=5))
    assert res.ledger.gradient_steps == 0 and arm.skipped == 10


def test_online_today_pool_resets_each_day():
    be = FakeTrainableBackend(noise=0.3)
    arm = OnlineArm()
    hz.run_arm(arm, be, _cfg(n_episodes=100, k=50))
    assert len(arm.today) <= 50 and arm.most_recent is not None


def test_online_unfiltered_trains_on_every_trajectory():
    be = FakeTrainableBackend(noise=0.3)
    arm = OnlineArm(filtered=False)
    res = hz.run_arm(arm, be, _cfg())
    assert arm.name == "online_unfiltered" and res.ledger.gradient_steps == 100
    assert [c for _, c in be.trained] == [e.text.strip() for e in res.episodes]


def test_sleep_night_after_k_episodes_trains_k_steps_and_buffers():
    be = FakeTrainableBackend()
    arm = SleepArm(SleepConfig(), dream=True)
    res = hz.run_arm(arm, be, _cfg())
    assert res.ledger.gradient_steps == 100                       # 2 nights x S=K=50
    assert [n["episode"] for n in res.nights] == [50, 100]
    n1, n2 = res.nights
    assert n1["kept"] == 50 and n1["dreams_generated"] == 100     # 2 per kept trajectory
    assert n1["dreams_kept"] > 0 and n1["replay"] == 0             # no earlier nights yet
    assert n2["replay"] == min(n2["new"], n1["kept"])              # C2: buffer holds real trajectories only
    assert n1["steps"] == 50 and n2["steps"] == 50
    assert be.decays == 2
    assert len(arm.buffer) == n1["kept"] + n2["kept"]
    split = nr.make_split(seed=0)
    assert all(e.prefix in split.train_prefixes for e in arm.buffer.examples)
    assert {e.source for e in arm.buffer.examples} == {"day"}      # C2
    # dream tokens are in the ledger (matched quantity), on top of day tokens
    assert res.ledger.tokens_generated == sum(e.tokens for e in res.episodes) + n1["dream_tokens"] + n2["dream_tokens"]


def test_sleep_nodream_has_no_dream_tokens():
    be = FakeTrainableBackend()
    arm = SleepArm(SleepConfig(), dream=False)
    res = hz.run_arm(arm, be, _cfg())
    assert all(n["dreams_generated"] == 0 and n["dream_tokens"] == 0 for n in res.nights)
    assert res.ledger.tokens_generated == sum(e.tokens for e in res.episodes)
    assert res.ledger.gradient_steps == 100 and be.decays == 2
    assert all(e.source == "day" for e in arm.buffer.examples)


def test_unverified_dreams_never_reach_training():
    be = FakeTrainableBackend(dream_error_rate=0.5)              # half the dreams have a wrong step
    arm = SleepArm(SleepConfig(), dream=True)
    res = hz.run_arm(arm, be, _cfg(n_episodes=50))
    n1 = res.nights[0]
    assert n1["dreams_rejected"].get("wrong", 0) > 0
    assert n1["dreams_kept"] + sum(n1["dreams_rejected"].values()) == n1["dreams_generated"]
    for prompt, completion in be.trained:
        digits = "".join(nr.parse_digits_line(prompt))
        s = nr.score(nr.Instance.from_digits(digits), completion)
        assert s["correct"] and s["steps_correct"] == 11


def test_all_arms_match_budgets_with_fake_backend():
    cfg = _cfg()
    sleep = hz.run_arm(SleepArm(SleepConfig(), dream=True), FakeTrainableBackend(), cfg)
    nodream = hz.run_arm(SleepArm(SleepConfig(), dream=False), FakeTrainableBackend(), cfg)
    online = hz.run_arm(OnlineArm(), FakeTrainableBackend(), cfg)
    budget = AwakeArm.budget_from_reference(sleep.ledger, cfg.n_episodes)
    awake = hz.run_arm(AwakeArm(token_budget_per_episode=budget), FakeTrainableBackend(), cfg)
    cl.check_matched({"sleep": sleep.ledger, "sleep_nodream": nodream.ledger,
                      "online": online.ledger, "awake": awake.ledger}, tol=0.05)


def test_criterion_detected_at_checkpoint_after_night():
    # the fake switches to the shortcut policy once it has taken 50 gradient steps
    res = hz.run_arm(SleepArm(SleepConfig(), dream=False), FakeTrainableBackend(learn_after_steps=50), _cfg())
    accs = [round(c.probe.accuracy, 2) for c in res.checkpoints]
    assert accs[0] < 0.7 and accs[1:] == [1.0, 1.0]               # ep 0 untrained; nights at 50, 100
    assert res.criterion_episode == 100
    res = hz.run_arm(OnlineArm(), FakeTrainableBackend(learn_after_steps=75), _cfg())
    accs = [round(c.probe.accuracy, 2) for c in res.checkpoints]
    assert accs[0] < 0.7 and accs[1] < 0.7 and accs[2] == 1.0 and res.criterion_episode is None


def test_sleep_config_defaults_follow_prereg():
    c = SleepConfig()
    assert c.n_variations == 2 and c.steps_per_night is None      # None -> K
    assert 0 < c.weight_decay < 1
