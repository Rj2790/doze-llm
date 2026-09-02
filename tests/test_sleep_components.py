"""Night-procedure components (PREREG §5). Model-free. Watchlist items:
held-out prefixes never enter the buffer or the dreams; unverified dreams
never enter training; near-miss threshold is >= 9/11 steps."""
import json

import pytest

from arms.harness import Episode
from backends.base import GenResult
from sleep import dreamer, replay_buffer as rb, filters
from tasks import number_reduction as nr

SPLIT = nr.make_split(seed=0)
TRAIN = SPLIT.train[0]
HELD = SPLIT.heldout[0]


def _ep(inst, correct, steps_correct, n=1):
    return Episode(episode=n, digits=inst.digits, text=nr.gold_response(inst, "work"), answer=inst.answer,
                   correct=correct, steps_correct=steps_correct, tokens=50)


def test_keep_rule_correct_answer_and_at_least_9_of_11_steps():
    """C3: keep iff the final answer is correct AND >= 9/11 steps are correct."""
    assert filters.keep(_ep(TRAIN, True, 11))
    assert filters.keep(_ep(TRAIN, True, 9))
    assert filters.keep(_ep(TRAIN, True, 10))
    assert not filters.keep(_ep(TRAIN, True, 8))           # right answer, sloppy chain
    assert not filters.keep(_ep(TRAIN, False, 11))         # wrong answer (impossible in practice, still out)
    assert not filters.keep(_ep(TRAIN, False, 10))         # wrong-answer near miss: removed by C3
    assert not filters.keep(_ep(TRAIN, False, 0))
    kept = filters.filter_day([_ep(TRAIN, True, 11), _ep(TRAIN, False, 10), _ep(TRAIN, True, 9), _ep(TRAIN, True, 5)])
    assert len(kept) == 2


def test_example_from_episode_uses_prompt_and_trajectory():
    ex = filters.to_example(_ep(TRAIN, True, 11), mode="work", numbered=False)
    assert ex.digits == TRAIN.digits and ex.prompt.endswith(" ".join(TRAIN.digits))
    assert ex.completion.strip().endswith(f"ANSWER: {TRAIN.answer}")
    assert ex.source == "day"


def test_replay_buffer_rejects_heldout_prefixes(tmp_path):
    buf = rb.ReplayBuffer(train_prefixes=SPLIT.train_prefixes)
    good = filters.to_example(_ep(TRAIN, True, 11), "work", False)
    bad = filters.to_example(_ep(HELD, True, 11), "work", False)
    buf.add([good])
    with pytest.raises(rb.LeakError):
        buf.add([bad])
    assert len(buf) == 1
    p = tmp_path / "buf.json"
    buf.save(p)
    buf2 = rb.ReplayBuffer.load(p, train_prefixes=SPLIT.train_prefixes)
    assert len(buf2) == 1 and buf2.examples[0].digits == TRAIN.digits


def test_replay_buffer_uniform_sample_over_all_nights():
    buf = rb.ReplayBuffer(train_prefixes=SPLIT.train_prefixes)
    for night, inst in enumerate(SPLIT.train[:30]):
        buf.add([filters.to_example(_ep(inst, True, 11), "work", False)], night=night)
    s = buf.sample(10, seed=0)
    assert len(s) == 10 and len({e.digits for e in s}) == 10
    assert buf.sample(10, seed=0) == s
    assert buf.sample(0, seed=0) == []
    assert len(buf.sample(100, seed=0)) == 30                   # capped at buffer size
    assert {e.night for e in buf.examples} == set(range(30))


class CannedBackend:
    """Returns canned texts in order; counts words as tokens."""
    name = "canned"

    def __init__(self, texts):
        self.texts = list(texts)
        self.prompts = []

    def generate(self, prompts, max_tokens=128, temperature=0.0):
        out = []
        for p in prompts:
            self.prompts.append(p)
            t = self.texts.pop(0) if self.texts else ""
            out.append(GenResult(text=t, prompt_tokens=len(p.split()), completion_tokens=len(t.split())))
        return out

    def count_tokens(self, text):
        return len(text.split())


def _dream_text(inst, break_step=None):
    body = nr.gold_response(inst, "work")
    if break_step is not None:
        lines = body.splitlines()
        a, b, _ = lines[break_step - 1].replace("->", ",").split(",")
        wrong = [d for d in nr.DIGITS if d != nr.combine(a, b)][0]
        lines[break_step - 1] = f"{a},{b}->{wrong}"
        body = "\n".join(lines)
    return f"Digits: {' '.join(inst.digits)}\n{body}"


def test_dreams_are_verified_against_solver_and_split():
    seed_ex = filters.to_example(_ep(TRAIN, True, 11), "work", False)
    t_ok = SPLIT.train[5]
    t_unstructured = nr.Instance.from_digits("111111111114")     # valid rule-wise, not mirror
    assert not t_unstructured.structured
    canned = [
        _dream_text(t_ok),                     # verifies
        _dream_text(SPLIT.train[6], break_step=4),  # wrong step -> rejected
        _dream_text(HELD),                     # held-out prefix -> rejected (leak guard)
        "Digits: 1 2 3\nnonsense",             # unparsable -> rejected
        _dream_text(t_unstructured),           # correct but unstructured -> kept (no structure filter)
        f"Digits: {' '.join(t_ok.digits)}\nANSWER: {t_ok.answer}",  # no work lines -> rejected
    ]
    be = CannedBackend(canned)
    kept, stats = dreamer.dream(be, [seed_ex] * 3, n_variations=2, split=SPLIT, mode="work",
                                numbered=False, max_tokens=400)
    assert stats["generated"] == 6 and stats["tokens"] == sum(len(t.split()) for t in canned)
    assert [e.digits for e in kept] == [t_ok.digits, t_unstructured.digits]
    assert all(e.source == "dream" for e in kept)
    assert stats["rejected"] == {"wrong": 1, "heldout_prefix": 1, "unparsable": 2}
    # the verified completion is the model's own text, and it scores perfectly
    for e in kept:
        s = nr.score(nr.Instance.from_digits(e.digits), e.completion)
        assert s["correct"] and s["steps_correct"] == 11


def test_dream_prompt_mentions_seed_and_format():
    seed_ex = filters.to_example(_ep(TRAIN, True, 11), "work", False)
    p = dreamer.dream_prompt(seed_ex, mode="work", numbered=False)
    assert TRAIN.digits[0] in p and "Digits:" in p and "STEPS:" in p and "12" in p


def test_interleave_one_to_one():
    a = [f"n{i}" for i in range(4)]
    b = [f"r{i}" for i in range(2)]
    assert dreamer.interleave(a, b) == ["n0", "r0", "n1", "r1", "n2", "n3"]
    assert dreamer.interleave([], b) == b


def test_night_checks_all_new_examples_for_leaks_before_training(monkeypatch):
    """C1: defence in depth — a held-out-prefix example must raise before any
    gradient step, even if the dreamer let it through."""
    from arms.harness import RunConfig
    from backends.scripted import FakeTrainableBackend
    from eval import compute_ledger as cl
    from sleep import consolidate

    leaked = filters.to_example(_ep(HELD, True, 11), "work", False)
    leaked.source = "dream"
    monkeypatch.setattr(consolidate.dreamer, "dream",
                        lambda *a, **k: ([leaked], {"generated": 1, "tokens": 5, "kept": 1, "rejected": {}}))
    be = FakeTrainableBackend()
    buf = rb.ReplayBuffer(SPLIT.train_prefixes)
    day = [_ep(SPLIT.train[i], True, 11, n=i + 1) for i in range(4)]
    with pytest.raises(rb.LeakError):
        consolidate.run_night(day, be, cl.Ledger("sleep"), RunConfig(seed=0, n_episodes=4, k=4), SPLIT,
                              consolidate.SleepConfig(), buf, night_index=1, dream_enabled=True)
    assert be.trained == [] and be.gradient_steps == 0 and len(buf) == 0
