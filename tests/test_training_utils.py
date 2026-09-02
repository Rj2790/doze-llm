import pytest

from backends import training_utils as tu


def test_lora_targets_cover_all_attention_and_mlp_projections():
    assert set(tu.LORA_TARGET_MODULES) == {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    assert tu.LORA_RANK == 16


def test_mask_conventions_count_the_same_tokens():
    p, c = [1, 2, 3], [4, 5]
    assert tu.lengths_row(len(p), len(p) + len(c)) == [3, 5]
    assert tu.supervised_tokens(3, 5) == 2
    labels = tu.labels_for(p, c)
    assert labels == [-100, -100, -100, 4, 5]
    assert sum(l != -100 for l in labels) == tu.supervised_tokens(3, 5)
    with pytest.raises(ValueError):
        tu.lengths_row(5, 5)


def test_cycle_examples_is_seeded_and_covers_all_before_repeating():
    ex = ["a", "b", "c"]
    order = tu.cycle_examples(ex, steps=7, seed=0)
    assert len(order) == 7 and set(order[:3]) == set(ex) and set(order[3:6]) == set(ex)
    assert order == tu.cycle_examples(ex, steps=7, seed=0)
    assert tu.cycle_examples([], 5, 0) == [] and tu.cycle_examples(ex, 0, 0) == []
