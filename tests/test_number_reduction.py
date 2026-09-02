import random
from collections import Counter

import pytest

from eval import shortcut_detector as sd
from tasks import number_reduction as nr


def test_rule_matches_wagner():
    assert nr.combine("1", "1") == "1"
    assert nr.combine("1", "4") == "9"
    assert nr.combine("4", "9") == "1"
    assert nr.combine("9", "1") == "4"


def test_worked_example_len8():
    r = nr.solve("11449494")
    assert r == ["1", "9", "1", "4", "4", "1", "9"]
    assert nr.has_mirror_structure("11449494")


def test_structured_count_matches_formula():
    for L in (8, 10, 12):
        assert len(nr.structured_instances(L)) == 3 ** (L - 3)
    # cross-check construction against brute force at L=8
    brute = {x.digits for x in nr.all_instances(8) if x.structured}
    built = {x.digits for x in nr.structured_instances(8)}
    assert brute == built


def test_split_is_prefix_disjoint():
    s = nr.make_split(seed=0)
    assert len(s.train) == nr.N_TRAIN == 801 and len(s.heldout) == nr.N_HELDOUT == 201
    assert len(s.probe) == nr.N_PROBE == 60
    assert not {x.prefix for x in s.train} & {x.prefix for x in s.heldout + s.probe}
    assert all(x.prefix in s.heldout_prefixes for x in s.heldout + s.probe)
    assert all(x.structured for x in s.train + s.heldout + s.probe)
    assert all(x.length == 12 for x in s.train)


def test_split_probe_disjoint_from_eval_items():
    s = nr.make_split(seed=0)
    assert not {x.digits for x in s.probe} & {x.digits for x in s.heldout}
    assert len({x.digits for x in s.train + s.heldout + s.probe}) == 801 + 201 + 60


def test_split_answer_distribution_exactly_uniform():
    """PREREG §4.1 (amended): train, held-out eval and probe are each exactly
    uniform over {1,4,9}; sizes are multiples of 3 (801 / 201 / 60)."""
    for seed in (0, 1, 7):
        s = nr.make_split(seed=seed)
        for part, n in ((s.train, 801), (s.heldout, 201), (s.probe, 60)):
            c = Counter(x.answer for x in part)
            assert c == {"1": n // 3, "4": n // 3, "9": n // 3}, (seed, c)
        assert Counter(nr.probe_target(x) for x in s.probe) == {"1": 20, "4": 20, "9": 20}
    with pytest.raises(ValueError):
        nr.make_split(seed=0, n_train=800)


def test_split_is_seeded():
    a, b = nr.make_split(seed=0), nr.make_split(seed=0)
    assert [x.digits for x in a.train] == [x.digits for x in b.train]
    assert [x.digits for x in a.probe] == [x.digits for x in b.probe]
    assert [x.digits for x in nr.make_split(seed=1).train] != [x.digits for x in a.train]


def test_shortcut_equals_answer_on_structured():
    for x in nr.make_split(seed=1).train:
        assert nr.probe_target(x) == x.answer


def test_probe_prefix_uninformative_without_structure():
    """Across *unstructured* strings sharing a prefix, the answer must vary
    (no leak); across structured ones it must be fixed (shortcut works)."""
    L = 8  # brute force is cheap here
    by_pref_s, by_pref_u = {}, {}
    for x in nr.all_instances(L):
        (by_pref_s if x.structured else by_pref_u).setdefault(x.prefix, set()).add(x.answer)
    assert all(len(v) == 1 for v in by_pref_s.values())
    assert all(len(v) == 3 for v in by_pref_u.values())


def test_answer_distribution_balanced():
    c = Counter(x.answer for x in nr.structured_instances(12))
    assert len(set(c.values())) == 1  # exactly uniform


def test_parsing():
    inst = nr.Instance.from_digits("11449494")
    s = nr.score(inst, "reasoning...\nSTEPS: 1 9 1 4 4 1 9\nANSWER: 9\n")
    assert s["correct"] and s["steps_correct"] == 7 and s["steps_parsed"]
    assert nr.parse_answer("ANSWER: 7") is None
    assert nr.parse_answer("nothing") is None


# ---- scripted solvers -----------------------------------------------------

def _digits(prompt):
    line = [l for l in prompt.splitlines() if l.startswith("Digits:")][-1]
    return line.split(":", 1)[1].split()


def literal_solver(prompt: str) -> str:
    toks = _digits(prompt)
    if nr.MASK in toks:
        return f"ANSWER: {random.choice(nr.DIGITS)}"
    r = nr.solve("".join(toks))
    return f"STEPS: {' '.join(r)}\nANSWER: {r[-1]}"


def shortcut_solver(prompt: str) -> str:
    toks = _digits(prompt)
    L = len(toks)
    vis = "".join(t for t in toks if t != nr.MASK)[: nr.prefix_len(L)]
    return f"ANSWER: {nr.solve(vis)[-1]}"


def test_probe_separates_solvers():
    random.seed(0)
    items = sd.stratified_probe_items(nr.make_split(seed=2).heldout, n=120, seed=0)
    lit = sd.run_probe(literal_solver, items)
    sc = sd.run_probe(shortcut_solver, items)
    assert 0.2 < lit.accuracy < 0.5 and not lit.above_chance and lit.p_value > sd.ALPHA
    assert sc.accuracy == 1.0 and sc.above_chance and sc.p_value < 1e-20


def test_criterion_tracker_requires_two_consecutive():
    t = sd.CriterionTracker(threshold=0.7)
    hi = sd.ProbeResult(60, 50, 50 / 60, True)
    lo = sd.ProbeResult(60, 20, 20 / 60, False)
    assert not t.update(50, hi)
    assert not t.update(100, lo)
    assert not t.update(150, hi)
    assert t.update(200, hi)
    assert t.first_met_at == 200


def test_token_collapse_signal():
    items = nr.make_split(seed=3).heldout[:50]
    lit = sd.token_collapse(literal_solver, items)
    sc = sd.token_collapse(shortcut_solver, items)
    assert lit.n_correct == 50 and sc.n_correct == 50
    assert sc.median_tokens < lit.median_tokens


def test_lenient_parsing():
    assert nr.parse_answer("**ANSWER:** 9") == "9"
    assert nr.parse_answer("Answer: 4.") == "4"
    assert nr.parse_answer("ANSWER: 1\nANSWER: 9") == "9"
    assert nr.parse_answer("The answer is 9") is None
    assert nr.parse_steps("**STEPS:** 9, 9, 1 → 1") == ["9", "9", "1", "1"]
    assert nr.parse_steps("STEPS: 9 4 4 9 4 1 1 1 1 1 1\nANSWER: 1") == list("94494111111")


def test_example_cannot_leak_structure():
    assert len(nr.EXAMPLE_DIGITS) < 8
    assert "STEPS:" in nr.format_prompt(nr.make_split(seed=0).heldout[0])
    assert nr.format_probe(nr.make_split(seed=0).heldout[0]).count("Digits:") == 2  # example + item


def test_work_mode_prompt_and_gold():
    x = nr.make_split(seed=0).heldout[0]
    p = nr.format_prompt(x, "work")
    assert "->" in p and "STEPS:" in p and p.rstrip().endswith(" ".join(x.digits))
    g = nr.gold_response(x, "work")
    assert g.count("->") == x.length - 1
    s = nr.score(x, g)
    assert s["correct"] and s["steps_correct"] == x.length - 1


# ---- numbered-digit prompt variant (tunable, CONTEXT.md §4) -------------

def test_numbered_prompt_and_parse():
    x = nr.make_split(seed=0).heldout[0]
    for mode in ("full", "work", "short"):
        p = nr.format_prompt(x, mode, numbered=True)
        assert p.rstrip().endswith(f"12:{x.digits[-1]}")
        assert " 1:" + x.digits[0] in p or "Digits: 1:" + x.digits[0] in p
        assert nr.parse_digits_line(p) == list(x.digits)
        assert nr.parse_digits_line(nr.format_prompt(x, mode)) == list(x.digits)
    # the worked example is numbered too, and still 6 digits
    assert f"6:{nr.EXAMPLE_DIGITS[-1]}" in nr.format_prompt(x, "work", numbered=True)


def test_numbered_probe_masks_same_positions():
    x = nr.make_split(seed=0).heldout[0]
    plain = nr.parse_digits_line(nr.format_probe(x))
    numbered = nr.parse_digits_line(nr.format_probe(x, numbered=True))
    assert plain == numbered == list(x.digits[:7]) + [nr.MASK] * 5
    assert "8:_" in nr.format_probe(x, numbered=True)


def test_scripted_backend_handles_numbered_prompts():
    from backends.scripted import ScriptedBackend
    x = nr.make_split(seed=0).heldout[0]
    lit = ScriptedBackend("literal")
    for numbered in (False, True):
        g = lit.generate([nr.format_prompt(x, "work", numbered=numbered)])[0]
        assert nr.score(x, g.text)["correct"]
    sc = ScriptedBackend("shortcut")
    g = sc.generate([nr.format_probe(x, numbered=True)])[0]
    assert nr.parse_answer(g.text) == nr.probe_target(x)
