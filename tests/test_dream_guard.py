"""Pre-flight checks on the dream held-out-prefix guard (CONTEXT.md §5:
4 of 6 smoke dreams were rejected as held-out). Model-free."""
import random

from arms import harness as hz
from arms.sleep import SleepArm
from backends.scripted import FakeTrainableBackend
from sleep import consolidate, dreamer
from tasks import number_reduction as nr


def test_guard_rejection_rate_on_uniform_random_strings_is_20pct():
    """2,000 uniformly random 12-digit strings, gold solutions attached so only
    the prefix guard can reject them: rejection rate must be 0.20 +- 0.03
    (437 of 2,187 prefixes are held out)."""
    split = nr.make_split(seed=0)
    rng = random.Random(12345)
    n, rejected, other = 2000, 0, {}
    for _ in range(n):
        digits = "".join(rng.choice(nr.DIGITS) for _ in range(12))
        inst = nr.Instance.from_digits(digits)
        text = f"{nr.digits_line(digits)}\n{nr.gold_response(inst, 'work')}"
        ex, reason = dreamer.verify(text, split, mode="work", source_prefix=inst.prefix)   # isolate the held-out guard
        if reason == "heldout_prefix":
            rejected += 1
        elif reason != "ok":
            other[reason] = other.get(reason, 0) + 1
    assert other == {}, other
    rate = rejected / n
    assert 0.17 <= rate <= 0.23, rate
    assert abs(len(split.heldout_prefixes) / 3 ** 7 - 0.2) < 0.001


def test_dreamer_checks_against_the_harness_split_object(monkeypatch):
    """The split passed to the dreamer must be the very object the harness
    built for this seed: one make_split call per run, identity preserved."""
    built = []
    real_make_split = nr.make_split

    def spy_make_split(*a, **k):
        s = real_make_split(*a, **k)
        built.append(s)
        return s

    seen = []
    real_dream = dreamer.dream

    def spy_dream(backend, kept, n_variations, split, *a, **k):
        seen.append(split)
        return real_dream(backend, kept, n_variations, split, *a, **k)

    monkeypatch.setattr(hz.nr, "make_split", spy_make_split)
    monkeypatch.setattr(consolidate.dreamer, "dream", spy_dream)
    cfg = hz.RunConfig(seed=3, n_episodes=20, k=10, n_probe=6, n_heldout_eval=3)
    hz.run_arm(SleepArm(dream=True), FakeTrainableBackend(), cfg)
    assert len(built) == 1, "harness must build the split exactly once per run"
    assert len(seen) == 2 and all(s is built[0] for s in seen)
    assert built[0].heldout_prefixes and built[0].train_prefixes
    assert not built[0].heldout_prefixes & built[0].train_prefixes
