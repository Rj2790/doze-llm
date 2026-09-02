"""PREREG §4.3: fixed 300-item GSM8K test sample, exact match, never trained on."""
import json

import pytest

from eval import control_bench as cb

FIXTURE = [
    {"question": f"Q{i}: what is {i} + {i}?", "answer": f"Add.\n#### {2 * i}"} for i in range(20)
]
FIXTURE[3]["answer"] = "Large.\n#### 1,234"       # thousands separator in gold


def _write(tmp_path):
    p = tmp_path / "gsm8k_test.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in FIXTURE) + "\n")
    return p


def test_gold_answer_parsing():
    assert cb.gold_answer("Add.\n#### 12") == "12"
    assert cb.gold_answer("Large.\n#### 1,234") == "1234"
    assert cb.gold_answer("neg\n#### -5") == "-5"


def test_model_answer_parsing_and_exact_match():
    assert cb.parse_answer("... so the answer is 42.") == "42"
    assert cb.parse_answer("ANSWER: 1,234") == "1234"
    assert cb.parse_answer("#### 7") == "7"
    assert cb.parse_answer("The total is $18.00") == "18"
    assert cb.parse_answer("no digits here") is None
    assert cb.exact_match("42", "42") and cb.exact_match("42.0", "42")
    assert not cb.exact_match("41", "42") and not cb.exact_match(None, "42")


def test_fixed_sample_is_deterministic_and_disjoint(tmp_path):
    p = _write(tmp_path)
    a = cb.load_sample(p, n=12, seed=0)
    b = cb.load_sample(p, n=12, seed=0)
    assert [x.idx for x in a] == [x.idx for x in b]
    assert len({x.idx for x in a}) == 12
    assert all(x.gold == cb.gold_answer(FIXTURE[x.idx]["answer"]) for x in a)
    c = cb.load_sample(p, n=12, seed=1)
    assert [x.idx for x in c] != [x.idx for x in a]


def test_sample_ids_are_frozen_in_repo():
    """The 300 item indices live in eval/control_bench_ids.json so every arm,
    seed and backend scores the same items."""
    ids = cb.frozen_ids()
    assert len(ids) == 300 and len(set(ids)) == 300
    assert all(0 <= i < 1319 for i in ids)  # GSM8K test has 1319 rows
    assert ids == sorted(ids)


def test_evaluate_scores_exact_match(tmp_path):
    p = _write(tmp_path)
    items = cb.load_sample(p, n=10, seed=0)
    perfect = lambda prompt: f"The answer is {cb._fixture_answer(prompt)}."
    res = cb.evaluate(perfect, items)
    assert res.accuracy == 1.0 and res.n == 10
    wrong = lambda prompt: "The answer is 999999."
    assert cb.evaluate(wrong, items).accuracy == 0.0


def test_prompt_contains_question_and_format():
    it = cb.Item(idx=0, question="What is 1 + 1?", gold="2")
    p = cb.format_prompt(it)
    assert "What is 1 + 1?" in p and "ANSWER:" in p
