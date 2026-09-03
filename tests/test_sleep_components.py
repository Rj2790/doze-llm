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


def _dream_text(inst, break_steps=()):
    steps = list(inst.responses)
    for j in break_steps:
        steps[j] = [d for d in nr.DIGITS if d != steps[j]][0]
    return f"Digits: {' '.join(inst.digits)}\n{nr._work_lines(inst.digits)}\nSTEPS: {' '.join(steps)}\nANSWER: {inst.answer}"


def test_dreams_are_verified_against_solver_and_split():
    src = TRAIN
    seed_ex = filters.to_example(_ep(src, True, 11), "work", False)
    v = lambda tail: nr.Instance.from_digits(src.digits[:7] + tail)
    t_ok, t_ok2 = v("14919"), v("99141")
    canned = [
        _dream_text(t_ok),                       # verifies
        _dream_text(v("41191"), break_steps=(3, 5, 7)),   # 3 wrong steps -> rejected (C3)
        _dream_text(HELD),                       # different prefix -> rejected before the held-out check
        "Digits: 1 2 3\nnonsense",               # unparsable -> rejected
        _dream_text(t_ok2, break_steps=(9,)),    # one wrong step, answer right -> kept (C3)
        f"Digits: {' '.join(t_ok.digits)}\nANSWER: {t_ok.answer}",  # no work lines -> rejected
    ]
    be = CannedBackend(canned)
    kept, stats = dreamer.dream(be, [seed_ex] * 3, n_variations=2, split=SPLIT, mode="work",
                                numbered=False, max_tokens=400)
    assert stats["generated"] == 6 and stats["tokens"] == sum(len(t.split()) for t in canned)
    assert [e.digits for e in kept] == [t_ok.digits, t_ok2.digits]
    assert all(e.source == "dream" and e.prefix == src.prefix for e in kept)
    assert stats["rejected"] == {"wrong": 1, "prefix_changed": 1, "unparsable": 2}
    assert stats["structured"] == sum(nr.Instance.from_digits(e.digits).structured for e in kept)
    for e in kept:
        s = nr.score(nr.Instance.from_digits(e.digits), e.completion)
        assert s["correct"] and s["steps_correct"] >= 9


def test_dream_prompt_mentions_seed_and_format():
    seed_ex = filters.to_example(_ep(TRAIN, True, 11), "work", False)
    p = dreamer.dream_prompt(seed_ex, mode="work", numbered=False)
    assert TRAIN.digits[0] in p and "Digits:" in p and "STEPS:" in p and "12" in p and "last five" in p


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


# ---- item 2: dreamer verification == shared C3 keep rule ------------------

def _dream_with_steps_wrong(inst, wrong_positions, wrong_answer=False):
    """Gold work response with the STEPS line altered at `wrong_positions`
    (0-based) and optionally a wrong ANSWER."""
    steps = list(inst.responses)
    for j in wrong_positions:
        steps[j] = [d for d in nr.DIGITS if d != steps[j]][0]
    ans = inst.answer if not wrong_answer else [d for d in nr.DIGITS if d != inst.answer][0]
    body = f"{nr._work_lines(inst.digits)}\nSTEPS: {' '.join(steps)}\nANSWER: {ans}"
    return f"{nr.digits_line(inst.digits)}\n{body}", steps, ans


def test_dreamer_and_sleep_filter_agree_on_fixtures():
    src = SPLIT.train[3]
    # dreams keep the source prefix: vary only the last five digits
    variants = []
    for tail in ("14919", "91141", "44911", "19914"):
        d = src.digits[:7] + tail
        variants.append(nr.Instance.from_digits(d))
    cases = [(variants[0], [], False), (variants[1], [4], False), (variants[2], [2, 6], False),
             (variants[3], [1, 5, 8], False), (variants[0], [], True), (variants[1], [3], True)]
    expected = [True, True, True, False, False, False]
    for (inst, wrong, bad_ans), want in zip(cases, expected):
        text, steps, ans = _dream_with_steps_wrong(inst, wrong, bad_ans)
        ex, reason = dreamer.verify(text, SPLIT, "work", source_prefix=src.prefix)
        ep = Episode(episode=1, digits=inst.digits, text=text, answer=ans, correct=(ans == inst.answer),
                     steps_correct=sum(a == b for a, b in zip(steps, inst.responses)), tokens=1)
        assert filters.keep(ep) is want, (wrong, bad_ans)
        assert (ex is not None) is want, (wrong, bad_ans, reason)
        if not want:
            assert reason == "wrong"


def test_dream_verification_uses_keep_scores_not_exact_chain():
    """A dream with a correct answer and 9/11 steps is accepted (C3), not
    rejected for failing an exact-chain check."""
    src = SPLIT.train[5]
    inst = nr.Instance.from_digits(src.digits[:7] + "11491")
    text, _, _ = _dream_with_steps_wrong(inst, [0, 1])
    ex, reason = dreamer.verify(text, SPLIT, "work", source_prefix=src.prefix)
    assert ex is not None and reason == "ok"
    assert filters.keep_scores(True, 9, 12) and not filters.keep_scores(True, 8, 12) and not filters.keep_scores(False, 11, 12)


# ---- item 3: dreams vary only the last five digits -------------------------

def test_dream_prefix_must_equal_source_prefix():
    src = SPLIT.train[8]
    same = nr.Instance.from_digits(src.digits[:7] + "49111")
    changed = nr.Instance.from_digits(("1" if src.digits[0] != "1" else "4") + src.digits[1:])
    ok, r_ok = dreamer.verify(f"{nr.digits_line(same.digits)}\n{nr.gold_response(same, 'work')}", SPLIT, "work", source_prefix=src.prefix)
    bad, r_bad = dreamer.verify(f"{nr.digits_line(changed.digits)}\n{nr.gold_response(changed, 'work')}", SPLIT, "work", source_prefix=src.prefix)
    assert ok is not None and r_ok == "ok" and ok.prefix == src.prefix
    assert bad is None and r_bad == "prefix_changed"


def test_dream_prompt_instructs_last_five_digits_only():
    seed_ex = filters.to_example(_ep(TRAIN, True, 11), "work", False)
    p = dreamer.dream_prompt(seed_ex, mode="work", numbered=False)
    assert "first seven digits" in p.lower() or "first 7 digits" in p.lower()
    assert "last five" in p.lower() or "positions 8" in p.lower()


def test_accepted_dreams_always_keep_source_prefix_and_report_structure():
    """End to end through dream(): every accepted dream has the source prefix
    (so held-out rejections are impossible by construction) and the stats
    report how many accepted dreams are mirror-structured."""
    from backends.scripted import FakeTrainableBackend
    seeds = [filters.to_example(_ep(SPLIT.train[i], True, 11, n=i + 1), "work", False) for i in range(10)]
    be = FakeTrainableBackend(dream_error_rate=0.3)
    kept, st = dreamer.dream(be, seeds, 2, SPLIT, "work", False, max_tokens=400)
    assert st["generated"] == 20 and st["rejected"].get("heldout_prefix", 0) == 0
    assert st["rejected"].get("prefix_changed", 0) == 0
    src_prefixes = {s.prefix for s in seeds}
    assert all(d.prefix in src_prefixes for d in kept)
    assert "structured" in st and 0 <= st["structured"] <= st["kept"]
    assert st["kept"] + sum(st["rejected"].values()) == 20


def test_dream_stats_record_digits_and_duplicates():
    """Accepted dreams are recorded (digits, structured, duplicate-of-source)
    so copies of the source can be counted; acceptance is unchanged."""
    src = SPLIT.train[11]
    seed_ex = filters.to_example(_ep(src, True, 11), "work", False)
    copy = f"{nr.digits_line(src.digits)}\n{nr.gold_response(src, 'work')}"
    var = nr.Instance.from_digits(src.digits[:7] + ("14919" if src.digits[7:] != "14919" else "91491"))
    variation = f"{nr.digits_line(var.digits)}\n{nr.gold_response(var, 'work')}"
    be = CannedBackend([copy, variation])
    kept, st = dreamer.dream(be, [seed_ex], n_variations=2, split=SPLIT, mode="work", numbered=False, max_tokens=400)
    assert st["kept"] == 2 and st["duplicates"] == 1
    assert [d["digits"] for d in st["accepted"]] == [src.digits, var.digits]
    assert st["accepted"][0]["duplicate"] is True and st["accepted"][1]["duplicate"] is False
    assert st["accepted"][0]["structured"] is True
    assert all(d["source"] == src.digits for d in st["accepted"])
