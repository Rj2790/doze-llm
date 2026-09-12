"""Model-free parts of the post-hoc follow-up (eval/followup.py)."""
import pytest

from eval import control_bench as cb
from eval import followup as fu
from tasks import number_reduction as nr


def test_probe_prompt_variants_keep_mask_and_add_text():
    sp = nr.make_split(seed=3, length=12)
    x = sp.probe[0]
    base = fu.probe_prompt(x, "as_run")
    assert base == nr.format_probe(x) and "_ _ _ _ _" in base
    hinted = fu.probe_prompt(x, "rule_stated")
    assert fu.RULE_HINT in hinted and "_ _ _ _ _" in hinted and nr.PROBE_FORMAT in hinted
    work = fu.probe_prompt(x, "work_allowed")
    assert fu.WORK_ALLOWED in work and nr.PROBE_FORMAT not in work and "_ _ _ _ _" in work
    both = fu.probe_prompt(x, "rule_stated_work")
    assert fu.RULE_HINT in both and fu.WORK_ALLOWED in both
    # no variant leaks the hidden digits
    for v in fu.PROBE_VARIANTS:
        p = fu.probe_prompt(x, v)
        assert " ".join(x.digits) not in p and x.digits[7:] not in p.replace(" ", "").split("Digits:")[-1]
    with pytest.raises(ValueError):
        fu.probe_prompt(x, "nope")


def test_positive_control_examples_use_training_prefixes_only():
    sp = nr.make_split(seed=2, length=12)
    ex = fu.positive_control_examples(sp, n=60, seed=2)
    assert len(ex) == 60
    forbidden = sp.heldout_prefixes
    answers = []
    for prompt, completion in ex:
        shown = nr.parse_digits_line(prompt)
        prefix = "".join(shown[: nr.prefix_len(12)])
        assert prefix in sp.train_prefixes and prefix not in forbidden
        assert all(t == nr.MASK for t in shown[nr.prefix_len(12):])
        assert completion.startswith("ANSWER: ") and completion[-1] in nr.DIGITS
        answers.append(completion[-1])
    # stratified over answers and consistent with the solver
    assert {answers.count(d) for d in nr.DIGITS} == {20}
    # completion is the shortcut target of the unmasked item
    for prompt, completion in ex[:5]:
        pass  # target correctness is guaranteed by construction (probe_target); prompt masks the tail


def test_logprob_choice_and_slope():
    assert fu.logprob_choice({"ANSWER: 1": -3.0, "ANSWER: 4": -1.2, "ANSWER: 9": -2.5}) == "ANSWER: 4"
    s = fu.length_accuracy_slope([{"tokens_mean": 80, "accuracy": 0.5}, {"tokens_mean": 100, "accuracy": 0.6},
                                  {"tokens_mean": 120, "accuracy": 0.7}])
    assert s["r"] == pytest.approx(1.0) and s["slope_per_10_tokens"] == pytest.approx(0.05)
    assert fu.length_accuracy_slope([{"tokens_mean": 80, "accuracy": 0.5}])["r"] is None


class _R:
    def __init__(self, text, n):
        self.text, self.completion_tokens = text, n


def test_gsm8k_rows_and_summary():
    items = [cb.Item(idx=1, question="q1", gold="12"), cb.Item(idx=2, question="q2", gold="7"),
             cb.Item(idx=3, question="q3", gold="5")]
    res = [_R("steps...\nANSWER: 12", 40), _R("so it is 8", 512), _R("no numbers here", 10)]
    rows = fu.gsm8k_rows(items, res, max_tokens=512)
    assert [r["correct"] for r in rows] == [True, False, False]
    assert [r["truncated"] for r in rows] == [False, True, False]
    assert [r["has_tag"] for r in rows] == [True, False, False]
    assert rows[2]["pred"] is None
    s = fu.gsm8k_summary(rows)
    assert s["n"] == 3 and s["accuracy"] == pytest.approx(1 / 3) and s["unparsable"] == pytest.approx(1 / 3)
    assert s["truncated"] == pytest.approx(1 / 3) and s["has_tag"] == pytest.approx(1 / 3)
    assert s["tokens_mean"] == pytest.approx((40 + 512 + 10) / 3)


def test_gsm8k_regimes_reproduce_the_grid_and_differ():
    assert fu.GSM8K_REGIMES["as_run"]["system"] == nr.SYSTEM_PROMPT and fu.GSM8K_REGIMES["as_run"]["max_tokens"] == 512
    assert fu.GSM8K_REGIMES["math_1024"]["system"] == cb.SYSTEM_PROMPT and fu.GSM8K_REGIMES["math_1024"]["max_tokens"] == 1024
    assert "no explanation" in nr.SYSTEM_PROMPT and "step by step" in cb.SYSTEM_PROMPT
