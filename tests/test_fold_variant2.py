"""Transfer puzzle variant 2: four symbols, alternating rules."""
import pytest

from tasks import fold_variant2 as fv


def test_combine_alternates_by_step():
    assert fv.combine("1", "1", 1) == "1" and fv.combine("9", "9", 2) == "9"
    assert fv.combine("1", "4", 1) == "7"          # odd: smaller of {7,9}
    assert fv.combine("1", "4", 2) == "9"          # even: larger of {7,9}
    assert fv.combine("7", "9", 3) == "1" and fv.combine("7", "9", 4) == "4"
    assert fv.combine("4", "7", 1) == "1" and fv.combine("4", "7", 2) == "9"


def test_solve_worked_example_and_old_rule_values():
    # 1 4 9 7 4 1: step1 (1,4)->7 ; step2 (7,9)->4 ; step3 (4,7)->1 ; step4 (1,4)->9 ; step5 (9,1)->4
    assert fv.solve("149741") == ["7", "4", "1", "9", "4"]
    inst = fv.Instance.from_digits("149741")
    assert inst.answer == "4" and inst.length == 6
    assert fv.old_rule_value("1", "4") == "9" and fv.old_rule_value("7", "9") is None and fv.old_rule_value("4", "4") == "4"


def test_make_items_deterministic_distinct_balanced():
    a = fv.make_items(40, seed=3); b = fv.make_items(40, seed=3)
    assert [x.digits for x in a] == [x.digits for x in b]
    assert len({x.digits for x in a}) == 40 and all(len(x.digits) == 12 for x in a)
    counts = {s: sum(x.answer == s for x in a) for s in fv.SYMBOLS}
    assert set(counts.values()) == {10}
    assert [x.digits for x in fv.make_items(40, seed=4)] != [x.digits for x in a]


def test_prompt_contains_rule_tables_example_and_format():
    inst = fv.make_items(1, seed=0)[0]
    p = fv.format_prompt(inst)
    assert "ODD steps" in p and "EVEN steps" in p and "1,4->7" in p and "1,4->9" in p
    assert "Digits: 1 4 9 7 4 1" in p and "ANSWER: 4" in p and "11 comparisons" in p
    assert p.rstrip().endswith(" ".join(inst.digits))


def test_score_error_anatomy():
    inst = fv.Instance.from_digits("149741")          # responses 7 4 1 9 4
    good = fv._work_lines("149741") + "\nSTEPS: 7 4 1 9 4\nANSWER: 4"
    s = fv.score(inst, good)
    assert s["correct"] and s["steps_correct"] == 5 and s["first_wrong_step"] is None and s["leading_correct"] == 5
    # step 1 has the right operands but the OLD rule's result (1,4->9 instead of 7)
    bad = "1,4->9\n9,9->9\n9,7->1\n1,4->7\n7,1->4\nSTEPS: 9 9 1 7 4\nANSWER: 4"
    s = fv.score(inst, bad)
    assert s["first_wrong_step"] == 1 and s["first_error_kind"] == "rule" and s["first_error_old_rule"] is True
    assert s["leading_correct"] == 0 and s["correct"] and s["steps_correct"] == 2
    # steps 1-2 right, step 3 uses the wrong operands
    track = "1,4->7\n7,9->4\n7,7->7\n7,4->9\n9,1->4\nSTEPS: 7 4 7 9 4\nANSWER: 4"
    s = fv.score(inst, track)
    assert s["first_wrong_step"] == 3 and s["first_error_kind"] == "tracking" and s["leading_correct"] == 2
    # fewer lines than steps, all correct so far -> "missing"
    s = fv.score(inst, "1,4->7\n7,9->4\nANSWER: 4")
    assert s["first_wrong_step"] == 3 and s["first_error_kind"] == "missing"
    assert fv.score(inst, "no tags here")["correct"] is False and fv.score(inst, "no tags")["steps_parsed"] is False
