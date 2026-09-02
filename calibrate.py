"""Calibration: is the task in the right difficulty regime for this model?

Runs the untrained model on the held-out set (full-format) and the probe,
and reports:

  full_accuracy     fraction of held-out items answered correctly
  steps_exact       fraction with all intermediate steps correct
  probe_accuracy    accuracy on masked items (chance = 0.333); items are
                    stratified by target, flag = one-sided binomial p < 0.01
  median_tokens     completion length on correct answers

Read it like this:
  full_accuracy > 0.90  -> too easy at this length; raise --length
  full_accuracy < 0.15  -> too hard; lower --length or check the prompt
  probe_accuracy already >> 0.33 -> the model has the shortcut from pretraining
                                     or the probe leaks; investigate before training

Usage:
  python calibrate.py --backend scripted                  # dry run, no model
  python calibrate.py --backend mlx --model mlx-community/Qwen3-4B-4bit --n 60 --n-probe 60
  python calibrate.py --backend mlx --model mlx-community/Qwen3-4B-bf16 --n 200 --n-probe 60
  python calibrate.py --backend mlx --model mlx-community/Qwen3-4B-bf16 --number-digits --n 40 --n-probe 120
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import Counter
from pathlib import Path

from eval import shortcut_detector as sd
from tasks import number_reduction as nr


def build_backend(args):
    if args.backend == "scripted":
        from backends.scripted import ScriptedBackend
        return ScriptedBackend(policy=args.policy, noise=args.noise, seed=args.seed)
    if args.backend == "mlx":
        from backends.mlx_backend import MLXBackend
        return MLXBackend(model_id=args.model, system=nr.SYSTEM_PROMPT, seed=args.seed)
    raise ValueError(args.backend)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["scripted", "mlx"], default="scripted")
    ap.add_argument("--model", default="mlx-community/Qwen3-4B-bf16")
    ap.add_argument("--policy", default="literal", help="scripted only")
    ap.add_argument("--noise", type=float, default=0.1, help="scripted only")
    ap.add_argument("--length", type=int, default=nr.DEFAULT_LENGTH)
    ap.add_argument("--mode", choices=["full", "work"], default="work",
                    help="full = steps only; work = one line per comparison first (scratchpad)")
    ap.add_argument("--number-digits", action="store_true",
                    help="write digits as position:digit (tunable prompt detail)")
    ap.add_argument("--n", type=int, default=100, help="held-out items for full task (<= 201)")
    ap.add_argument("--n-probe", type=int, default=60,
                    help="probe items, stratified by target; multiple of 3, <= 261")
    ap.add_argument("--max-tokens", type=int, default=400, help="full-format budget")
    ap.add_argument("--probe-max-tokens", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/calibration.json")
    ap.add_argument("--show", type=int, default=2, help="print this many raw completions")
    args = ap.parse_args()

    # PREREG sizes: 801 train / 201 held-out eval / 60 probe. Calibration may pool eval+probe items.
    split = nr.make_split(seed=args.seed, length=args.length)
    items = split.heldout[: args.n]
    probe_items = sd.stratified_probe_items(split.heldout + split.probe, n=args.n_probe, seed=args.seed)
    backend = build_backend(args)
    numbered = args.number_digits
    print(f"backend={backend.name} length={args.length} mode={args.mode} numbered={numbered} "
          f"n={len(items)} n_probe={len(probe_items)}")

    # ---- full task -------------------------------------------------------
    t0 = time.time()
    gens = backend.generate([nr.format_prompt(x, args.mode, numbered=numbered) for x in items],
                            max_tokens=args.max_tokens)
    scores = [nr.score(x, g.text) for x, g in zip(items, gens)]
    full_acc = sum(s["correct"] for s in scores) / len(scores)
    steps_exact = sum(s["steps_correct"] == s["steps_total"] for s in scores) / len(scores)
    parsed = sum(s["steps_parsed"] for s in scores) / len(scores)
    unparsable = sum(s["answer"] is None for s in scores) / len(scores)
    tok_correct = [g.completion_tokens for s, g in zip(scores, gens) if s["correct"]]
    t_full = time.time() - t0

    # ---- probe -----------------------------------------------------------
    t0 = time.time()
    gen_one = lambda p: backend.generate([p], max_tokens=args.probe_max_tokens)[0].text
    probe = sd.run_probe(gen_one, probe_items, numbered=numbered)
    t_probe = time.time() - t0

    # ---- where does the literal chain break? -----------------------------
    # first wrong step index (1-based) among parsed-but-imperfect responses
    first_err = Counter()
    for x, s, g in zip(items, scores, gens):
        steps = nr.parse_steps(g.text)
        if steps and steps != list(x.responses):
            for i, (a, b) in enumerate(zip(steps, x.responses), 1):
                if a != b:
                    first_err[i] += 1
                    break

    report = {
        "backend": backend.name, "length": args.length, "mode": args.mode, "numbered": numbered,
        "n": len(items), "n_probe": len(probe_items),
        "full_accuracy": round(full_acc, 3), "steps_exact": round(steps_exact, 3),
        "steps_parsed": round(parsed, 3), "answer_unparsable": round(unparsable, 3),
        "median_tokens_correct": statistics.median(tok_correct) if tok_correct else None,
        "probe_accuracy": round(probe.accuracy, 3), "probe_chance": round(sd.CHANCE, 3),
        "probe_p_value": round(probe.p_value, 4), "probe_above_chance": probe.above_chance,
        "probe_target_dist": sd.target_distribution(probe_items),
        "first_error_step_hist": dict(sorted(first_err.items())),
        "seconds_full": round(t_full, 1), "seconds_probe": round(t_probe, 1),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))

    print(json.dumps(report, indent=2))
    for x, g in list(zip(items, gens))[: args.show]:
        print("\n--- digits:", x.digits, "gold:", nr.gold_response(x).replace("\n", " | "))
        print(g.text.strip()[:600])

    chance_band = abs(full_acc - sd.CHANCE) < 0.10 and steps_exact < 0.05
    if unparsable > 0.5:
        verdict = ("FORMAT PROBLEM: most completions never reached ANSWER. Raise --max-tokens "
                   "and inspect the raw completions below before judging difficulty")
    elif chance_band:
        verdict = ("AT FLOOR: accuracy is chance and no chain is ever fully correct. The model is "
                   "not computing. Try --mode work, bf16 weights, or a shorter --length")
    elif full_acc > 0.90:
        verdict = "TOO EASY: raise --length"
    elif full_acc < 0.15:
        verdict = "TOO HARD: lower --length or inspect completions"
    else:
        verdict = "OK: in the learnable regime"
    if probe.above_chance:
        verdict += " | WARNING: probe already above chance before training"
    print("\nverdict:", verdict)


if __name__ == "__main__":
    main()
