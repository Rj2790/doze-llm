"""First-execution validation of a real backend (REVIEW.md list, CONTEXT.md
§8 item 9), the frozen-Baseline backend-agreement items (item 4) and the A4
dream-yield / token-length statistics (item 5). Backend-agnostic: runs on
MLX locally and on HF inside Modal. Model-free parts are unit-tested.
"""

from __future__ import annotations

import math
import statistics
import time
from collections.abc import Sequence

from arms.harness import Episode
from sleep import dreamer, filters
from tasks import number_reduction as nr

AGREEMENT_N = 40


def validate_backend(backend, quick_prompt: str | None = None) -> dict:
    """Items 1-4 of the list. `backend` must be LoRA-enabled (train/decay/reset)."""
    out: dict = {}
    d = backend.describe()
    out["1_describe"] = d
    out["1_pass"] = ("bfloat16" in d["model_dtype"]) and d["lora"] and d["trainable_params"] > 0

    # 2. chat template / stop / token accounting
    chat = backend._chat("Say OK.")
    inst = nr.make_split(seed=0).heldout[0]
    r = backend.generate([nr.format_prompt(inst, "work")], max_tokens=400)[0]
    ct = backend.count_tokens(r.text)
    out["2_template_has_empty_think"] = ("<think>" in chat and "</think>" in chat)
    out["2_stopped_before_budget"] = r.completion_tokens < 400
    out["2_completion_tokens"] = r.completion_tokens
    out["2_count_tokens_text"] = ct
    out["2_no_special_tokens_in_text"] = "<|im_end|>" not in r.text and "<think>" not in r.text
    out["2_pass"] = out["2_template_has_empty_think"] and out["2_stopped_before_budget"] and abs(r.completion_tokens - ct) <= 1 and out["2_no_special_tokens_in_text"]
    out["2_sample"] = r.text[:300]

    # 3. greedy determinism
    r2 = backend.generate([nr.format_prompt(inst, "work")], max_tokens=400)[0]
    out["3_pass"] = (r.text == r2.text)

    # 4. training mechanics on one repeated example
    prompt = nr.format_prompt(inst, "work")
    completion = nr.gold_response(inst, "work")
    n0 = backend.lora_norm()
    t0 = time.time()
    st = backend.train([(prompt, completion)], steps=6, seed=0)
    n1 = backend.lora_norm()
    backend.decay_adapter(0.05)
    n2 = backend.lora_norm()
    backend.reset_adapter()
    n3 = backend.lora_norm()
    comp_len = backend.count_tokens(completion.strip())
    losses = st.losses or []
    out["4_losses"] = [round(x, 4) for x in losses]
    out["4_loss_finite"] = all(math.isfinite(x) for x in losses)
    out["4_loss_decreasing"] = len(losses) >= 2 and losses[-1] < losses[0]
    out["4_training_tokens"] = st.training_tokens
    out["4_expected_tokens"] = 6 * (comp_len + 1)
    out["4_norm_before_after_decay_reset"] = [n0, n1, n2, n3]
    out["4_decay_ratio"] = (n2 / n1) if n1 else None
    out["4_seconds"] = round(time.time() - t0, 1)
    out["4_pass"] = (out["4_loss_finite"] and out["4_loss_decreasing"] and st.training_tokens == out["4_expected_tokens"]
                     and n0 == 0.0 and n1 > 0 and n2 < n1 and abs(n2 / n1 - 0.95) < 1e-3 and n3 == 0.0)
    return out


def agreement_items(seed: int = 0, n: int = AGREEMENT_N) -> list[nr.Instance]:
    return nr.make_split(seed=seed).heldout[:n]


def run_agreement(backend, items: Sequence[nr.Instance], max_tokens: int = 400) -> list[dict]:
    gens = backend.generate([nr.format_prompt(x, "work") for x in items], max_tokens=max_tokens)
    out = []
    for x, g in zip(items, gens):
        s = nr.score(x, g.text)
        out.append({"digits": x.digits, "text": g.text, "answer": s["answer"], "correct": s["correct"],
                    "steps_correct": s["steps_correct"], "steps_exact": s["steps_correct"] == s["steps_total"],
                    "tokens": g.completion_tokens,
                    "kept": s["correct"] and s["steps_correct"] >= s["steps_total"] - filters.NEAR_MISS_MAX_WRONG})
    return out


def summarize_agreement(rows: Sequence[dict]) -> dict:
    n = len(rows)
    kept_tok = [r["tokens"] for r in rows if r["kept"]]
    return {"n": n, "accuracy": sum(r["correct"] for r in rows) / n, "steps_exact": sum(r["steps_exact"] for r in rows) / n,
            "kept": sum(r["kept"] for r in rows), "kept_tokens_mean": statistics.mean(kept_tok) if kept_tok else None,
            "kept_tokens_median": statistics.median(kept_tok) if kept_tok else None,
            "all_tokens_mean": statistics.mean(r["tokens"] for r in rows)}


def compare_agreement(a: Sequence[dict], b: Sequence[dict]) -> dict:
    assert [r["digits"] for r in a] == [r["digits"] for r in b]
    same_answer = sum(x["answer"] == y["answer"] for x, y in zip(a, b))
    same_text = sum(x["text"].strip() == y["text"].strip() for x, y in zip(a, b))
    both_correct = sum(x["correct"] and y["correct"] for x, y in zip(a, b))
    only_a = sum(x["correct"] and not y["correct"] for x, y in zip(a, b))
    only_b = sum(y["correct"] and not x["correct"] for x, y in zip(a, b))
    return {"n": len(a), "same_answer": same_answer, "same_text": same_text, "both_correct": both_correct,
            "only_a_correct": only_a, "only_b_correct": only_b,
            "a": summarize_agreement(a), "b": summarize_agreement(b)}


def dream_stats(backend, rows: Sequence[dict], split: nr.Split, n_seed: int = 10, n_variations: int = 2,
                max_tokens: int = 400) -> dict:
    """A4 inputs: dream yield and dream completion lengths from real kept
    trajectories (the same prompt the night uses)."""
    kept_rows = [r for r in rows if r["kept"]][:n_seed]
    eps = [Episode(episode=i + 1, digits=r["digits"], text=r["text"], answer=r["answer"], correct=r["correct"],
                   steps_correct=r["steps_correct"], tokens=r["tokens"]) for i, r in enumerate(kept_rows)]
    kept = [filters.to_example(e, "work", False) for e in eps]
    t0 = time.time()
    dreams, st = dreamer.dream(backend, kept, n_variations, split, "work", False, max_tokens=max_tokens)
    dream_tok = [backend.count_tokens(d.completion) for d in dreams]
    return {"seeds": len(kept), "generated": st["generated"], "kept": st["kept"], "rejected": st["rejected"],
            "yield": (st["kept"] / st["generated"]) if st["generated"] else None,
            "dream_tokens_generated_total": st["tokens"],
            "dream_completion_tokens_mean": statistics.mean(dream_tok) if dream_tok else None,
            "dream_completion_tokens_median": statistics.median(dream_tok) if dream_tok else None,
            "seconds": round(time.time() - t0, 1)}


def project_training_token_ratio(kept_tokens_mean: float, dream_tokens_mean: float | None, dream_yield: float | None,
                                 keep_rate: float, k: int = 50, n_variations: int = 2) -> dict:
    """Online trains 1 step/episode on a kept trajectory (length L_k).
    Sleep trains K steps/night over interleave(new, replay) where new = kept
    day trajectories + verified dreams; replay = kept trajectories. Expected
    length per Sleep step = weighted mean of L_k and L_d."""
    kept_per_night = keep_rate * k
    dreams_per_night = kept_per_night * n_variations * (dream_yield or 0.0)
    new = kept_per_night + dreams_per_night
    replay = kept_per_night if kept_per_night else 0      # ~1:1 with new once the buffer is non-empty
    L_d = dream_tokens_mean if dream_tokens_mean is not None else kept_tokens_mean
    sleep_mean = ((kept_per_night + replay) * kept_tokens_mean + dreams_per_night * L_d) / max(new + replay, 1e-9)
    ratio = sleep_mean / kept_tokens_mean if kept_tokens_mean else None
    return {"L_kept": kept_tokens_mean, "L_dream": L_d, "kept_per_night": kept_per_night,
            "dreams_per_night": dreams_per_night, "sleep_mean_tokens_per_step": sleep_mean,
            "online_mean_tokens_per_step": kept_tokens_mean, "ratio_sleep_over_online": ratio,
            "within_5pct": (ratio is not None and abs(ratio - 1) <= 0.05)}
