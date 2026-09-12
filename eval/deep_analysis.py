"""Exploratory deep analysis of the completed grid (not preregistered).

Reads the per-seed run files under records/ and writes
records/analysis_deep.md, records/analysis_deep.json and records/figures/*.png.
Everything here is descriptive or exploratory; the preregistered tests live in
eval/analyze.py --final. Run:

    python -m eval.deep_analysis --results records --seeds 0,1,2,3,4
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path

from eval import stats
from tasks import number_reduction as nr

ARMS = ("baseline", "awake", "online", "sleep", "sleep_nodream")
TRAINED = ("online", "sleep", "sleep_nodream")
LABEL = {"baseline": "Baseline", "awake": "Awake", "online": "Online", "sleep": "Sleep", "sleep_nodream": "Sleep-NoDream"}
COLOR = {"baseline": "#8a8f98", "awake": "#c9a227", "online": "#1f77b4", "sleep": "#7b3fa0", "sleep_nodream": "#2a9d8f"}


# ---- pure helpers (tested) ----------------------------------------------------

def _binom_tail(k: int, n: int, p: float) -> float:
    """P[X >= k] for X ~ Binomial(n, p), computed in log space (safe for n in the thousands)."""
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    lp, lq = math.log(p), math.log1p(-p)
    terms = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * lp + (n - i) * lq for i in range(k, n + 1)]
    m = max(terms)
    return min(1.0, math.exp(m) * sum(math.exp(t - m) for t in terms))


def pooled_binomial(correct: int, n: int, p0: float) -> dict:
    """Two-sided exact-ish binomial test of rate vs p0 plus a Wilson 95% CI."""
    if n <= 0:
        return {"correct": correct, "n": n, "rate": None, "p_two_sided": None, "ci95": (None, None)}
    rate = correct / n
    upper = _binom_tail(correct, n, p0)                                    # P[X >= correct]
    lower = 1.0 - (_binom_tail(correct + 1, n, p0) if correct < n else 0.0)  # P[X <= correct]
    p = min(1.0, 2 * min(upper, lower))
    z = 1.959964
    den = 1 + z * z / n
    centre = (rate + z * z / (2 * n)) / den
    half = z * math.sqrt(rate * (1 - rate) / n + z * z / (4 * n * n)) / den
    return {"correct": correct, "n": n, "rate": rate, "p_two_sided": p, "ci95": (centre - half, centre + half)}


def block_accuracy(episodes: Sequence[dict], block: int = 50) -> list[dict]:
    out = []
    eps = sorted(episodes, key=lambda e: e["episode"])
    for start in range(0, len(eps), block):
        chunk = eps[start:start + block]
        if not chunk:
            continue
        out.append({"episode_end": start + len(chunk),
                    "accuracy": sum(1 for e in chunk if e["correct"]) / len(chunk),
                    "steps_correct_mean": statistics.fmean(e.get("steps_correct", 0) or 0 for e in chunk),
                    "tokens_mean": statistics.fmean(e.get("tokens", 0) or 0 for e in chunk)})
    return out


def error_position_hist(episodes: Sequence[dict], n_steps: int = 11) -> dict[int, int]:
    """Among wrong episodes, how many correct leading steps were produced (0..n_steps-1)."""
    h: Counter = Counter()
    for e in episodes:
        if not e["correct"]:
            k = e.get("steps_correct", 0) or 0
            h[min(max(int(k), 0), n_steps - 1)] += 1
    return dict(h)


def step_changes(series: Sequence[float]) -> list[float]:
    return [series[i + 1] - series[i] for i in range(len(series) - 1)]


def volatility(series: Sequence[float]) -> float:
    d = step_changes(series)
    return statistics.pstdev(d) if len(d) > 1 else 0.0


def max_drawdown(series: Sequence[float]) -> float:
    peak, worst = -1.0, 0.0
    for x in series:
        peak = max(peak, x)
        worst = max(worst, peak - x)
    return worst


def buffer_composition(buffer: Sequence[dict], train_digits: set[str]) -> dict:
    digits = [b["digits"] for b in buffer]
    distinct = set(digits)
    answers = Counter(b.get("answer") or nr.solve(b["digits"])[-1] for b in buffer)
    n = len(buffer)
    pl = nr.prefix_len(len(next(iter(train_digits)))) if train_digits else 0
    train_prefixes = {d[:pl] for d in train_digits}
    buf_prefixes = {d[:pl] for d in distinct}
    return {"n": n, "n_distinct_digits": len(distinct),
            "train_coverage": len(distinct & train_digits) / max(len(train_digits), 1),
            "prefix_coverage": len(buf_prefixes & train_prefixes) / max(len(train_prefixes), 1),
            "outside_train": len(distinct - train_digits),
            "answer_dist": {k: v / n for k, v in sorted(answers.items())} if n else {}}


def min_attainable_p(n: int) -> float:
    return 2 / (2 ** n)


def pearson(x: Sequence[float], y: Sequence[float]) -> float | None:
    if len(x) < 3 or len(x) != len(y):
        return None
    mx, my = statistics.fmean(x), statistics.fmean(y)
    sx = math.sqrt(sum((a - mx) ** 2 for a in x))
    sy = math.sqrt(sum((b - my) ** 2 for b in y))
    if sx == 0 or sy == 0:
        return None
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy)


# ---- loading -----------------------------------------------------------------

def load_grid(root: Path, seeds: Sequence[int]) -> dict[tuple[int, str], dict]:
    grid = {}
    for s in seeds:
        for a in ARMS:
            p = root / f"seed{s}" / "runs" / f"{a}_seed{s}.json"
            if p.exists():
                grid[(s, a)] = json.loads(p.read_text())
    return grid


def load_buffers(root: Path, seeds: Sequence[int]) -> dict[tuple[int, str], list[dict]]:
    out = {}
    for s in seeds:
        for a in ("sleep", "sleep_nodream"):
            p = root / f"seed{s}" / "state" / f"{a}_seed{s}.buffer.json"
            if p.exists():
                b = json.loads(p.read_text())
                out[(s, a)] = b if isinstance(b, list) else b.get("items", [])
    return out


def load_timing(root: Path, seeds: Sequence[int]) -> dict[tuple[int, str], dict]:
    out = {}
    for s in seeds:
        for a in ARMS:
            p = root / f"seed{s}" / f"{a}_seed{s}.timing.json"
            if p.exists():
                out[(s, a)] = json.loads(p.read_text())
    return out


# ---- analyses ----------------------------------------------------------------

def _mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.fmean(xs) if xs else None


def curves(grid, seeds) -> dict:
    """Per arm, per checkpoint index: mean/min/max across seeds for the three headline metrics."""
    out = {}
    for a in ARMS:
        runs = [grid[(s, a)] for s in seeds if (s, a) in grid]
        n_ck = min(len(r["checkpoints"]) for r in runs)
        rows = []
        for i in range(n_ck):
            row = {"episode": runs[0]["checkpoints"][i]["episode"]}
            for m in ("heldout_accuracy", "probe_accuracy", "control_accuracy"):
                v = [r["checkpoints"][i][m] for r in runs]
                row[m] = {"mean": statistics.fmean(v), "min": min(v), "max": max(v), "per_seed": v}
            rows.append(row)
        out[a] = rows
    return out


def probe_pooled(grid, seeds) -> dict:
    out = {}
    for a in ARMS:
        c_all = n_all = c_last = n_last = 0
        per_ck = defaultdict(lambda: [0, 0])
        for s in seeds:
            r = grid.get((s, a))
            if not r:
                continue
            n_probe = r["config"].get("n_probe", 60)
            def _cn(ck):                                   # pilot-format checkpoints carry only the accuracy
                n = ck.get("probe_n") or n_probe
                return (ck["probe_correct"] if ck.get("probe_correct") is not None else round(ck["probe_accuracy"] * n)), n
            # frozen arms re-run the identical evaluation at every checkpoint: count it once per seed
            cks = r["checkpoints"] if a in TRAINED else r["checkpoints"][-1:]
            for ck in cks:
                c, n = _cn(ck)
                c_all += c; n_all += n
                per_ck[ck["episode"]][0] += c; per_ck[ck["episode"]][1] += n
            c, n = _cn(r["checkpoints"][-1]); c_last += c; n_last += n
        out[a] = {"all_checkpoints": pooled_binomial(c_all, n_all, 1 / 3),
                  "final_checkpoint": pooled_binomial(c_last, n_last, 1 / 3),
                  "by_episode": {ep: pooled_binomial(c, n, 1 / 3) for ep, (c, n) in sorted(per_ck.items())},
                  "best_single_checkpoint": max((ck["probe_accuracy"], s, ck["episode"]) for s in seeds if (s, a) in grid
                                                for ck in grid[(s, a)]["checkpoints"])}
    return out


def day_curves(grid, seeds, block=50) -> dict:
    out = {}
    for a in ARMS:
        per_seed = [block_accuracy(grid[(s, a)]["episodes"], block) for s in seeds if (s, a) in grid and grid[(s, a)]["episodes"]]
        if not per_seed:
            continue
        n = min(len(p) for p in per_seed)
        out[a] = [{"episode_end": per_seed[0][i]["episode_end"],
                   "accuracy_mean": statistics.fmean(p[i]["accuracy"] for p in per_seed),
                   "accuracy_min": min(p[i]["accuracy"] for p in per_seed),
                   "accuracy_max": max(p[i]["accuracy"] for p in per_seed),
                   "tokens_mean": statistics.fmean(p[i]["tokens_mean"] for p in per_seed),
                   "steps_correct_mean": statistics.fmean(p[i]["steps_correct_mean"] for p in per_seed)} for i in range(n)]
    return out


def error_positions(grid, seeds) -> dict:
    """Where do wrong day-attempts fail: first 100 episodes vs last 100, per arm, pooled over seeds."""
    out = {}
    for a in ARMS:
        early, late = Counter(), Counter()
        for s in seeds:
            r = grid.get((s, a))
            if not r or not r["episodes"]:
                continue
            eps = sorted(r["episodes"], key=lambda e: e["episode"])
            early.update(error_position_hist(eps[:100])); late.update(error_position_hist(eps[-100:]))
        out[a] = {"first100": dict(sorted(early.items())), "last100": dict(sorted(late.items())),
                  "n_wrong_first100": sum(early.values()), "n_wrong_last100": sum(late.values())}
    return out


def heldout_items(grid, seeds) -> dict:
    """Per-item held-out analysis from the saved model rows (digits, parsed steps, parsed answer)."""
    out = {"per_run_final": {}, "mirror_among_wrong": {}, "hard_items": {}, "first_wrong_step": {}}
    for a in ARMS:
        fw_early, fw_late = Counter(), Counter()
        mir_w = mir_n = 0
        for s in seeds:
            r = grid.get((s, a))
            if not r:
                continue
            cks = [c for c in r["checkpoints"] if c.get("heldout_rows")]
            if not cks:
                continue
            for ck in (cks[0], cks[-1]):
                for row in ck["heldout_rows"]:
                    truth = nr.solve(row["digits"])
                    steps = row.get("steps") or ""
                    k = 0
                    while k < len(truth) and k < len(steps) and steps[k] == truth[k]:
                        k += 1
                    if row.get("answer") != truth[-1]:
                        (fw_early if ck is cks[0] else fw_late)[min(k, 10)] += 1
            last = cks[-1]
            correct = sum(1 for row in last["heldout_rows"] if row.get("answer") == nr.solve(row["digits"])[-1])
            out["per_run_final"][f"{a}_seed{s}"] = {"episode": last["episode"], "correct": correct, "n": len(last["heldout_rows"]),
                                                      "reported": last["heldout_accuracy"]}
        out["first_wrong_step"][a] = {"first_eval": dict(sorted(fw_early.items())), "final_eval": dict(sorted(fw_late.items()))}
    # items no trained arm solved at the final checkpoint (pooled over seeds)
    unsolved = Counter(); total = Counter()
    for s in seeds:
        for a in TRAINED:
            r = grid.get((s, a))
            if not r:
                continue
            cks = [c for c in r["checkpoints"] if c.get("heldout_rows")]
            if not cks:
                continue
            for row in cks[-1]["heldout_rows"]:
                total[row["digits"]] += 1
                if row.get("answer") != nr.solve(row["digits"])[-1]:
                    unsolved[row["digits"]] += 1
    hard = [(d, unsolved[d], total[d]) for d in total if unsolved[d] == total[d] and total[d] >= 3]
    out["hard_items"] = {"n_items_seen": len(total), "never_solved_by_any_trained_arm_at_600": len(hard),
                         "examples": sorted(hard)[:10]}
    return out


def control_dynamics(grid, seeds) -> dict:
    out = {}
    for a in TRAINED:
        ch, worst, drops, per_seed = [], [], 0, {}
        for s in seeds:
            r = grid.get((s, a))
            if not r:
                continue
            c = [ck["control_accuracy"] for ck in r["checkpoints"]]
            d = step_changes(c)
            ch += d; worst.append(min(d)); drops += sum(1 for x in d if x <= -0.05)
            per_seed[s] = {"delta": c[-1] - c[0], "peak": max(c), "trough": min(c), "max_drawdown": max_drawdown(c), "volatility": volatility(c)}
        out[a] = {"per_checkpoint_change_mean": _mean(ch), "per_checkpoint_change_sd": statistics.pstdev(ch) if len(ch) > 1 else 0.0,
                  "worst_change_per_seed": worst, "n_drops_ge_0.05": drops, "n_changes": len(ch), "per_seed": per_seed}
    # Sleep: does the GSM8K change at a checkpoint track that night's training?
    xs = defaultdict(list)
    for s in seeds:
        r = grid.get((s, "sleep"))
        if not r or not r["nights"]:
            continue
        cks = {ck["episode"]: ck["control_accuracy"] for ck in r["checkpoints"]}
        eps = sorted(cks)
        for n in r["nights"]:
            e = n["episode"]
            if e not in cks:
                continue
            prev = eps[eps.index(e) - 1] if eps.index(e) > 0 else None
            if prev is None:
                continue
            dream_n = len(n.get("dreams_accepted", [])) if isinstance(n.get("dreams_accepted"), list) else (n.get("dreams_accepted") or 0)
            xs["d_control"].append(cks[e] - cks[prev]); xs["loss"].append(n.get("loss")); xs["buffer"].append(n.get("buffer_size"))
            xs["train_set"].append(n.get("train_set")); xs["dream_frac"].append(dream_n / max(n.get("train_set") or 1, 1))
            xs["new"].append(n.get("new"))
    out["sleep_night_correlations"] = {k: pearson(xs[k], xs["d_control"]) for k in ("loss", "buffer", "train_set", "dream_frac", "new")}
    out["sleep_night_n"] = len(xs["d_control"])
    return out


def nights_summary(grid, seeds) -> dict:
    out = {}
    for a in ("sleep", "sleep_nodream"):
        by_night = defaultdict(lambda: defaultdict(list))
        for s in seeds:
            r = grid.get((s, a))
            if not r:
                continue
            for n in r["nights"]:
                k = n["night"]
                gen = n.get("dreams_generated") or 0
                acc = n.get("dreams_accepted"); acc = len(acc) if isinstance(acc, list) else (acc or 0)
                by_night[k]["kept"].append(n.get("kept")); by_night[k]["new"].append(n.get("new"))
                by_night[k]["buffer"].append(n.get("buffer_size")); by_night[k]["train_set"].append(n.get("train_set"))
                by_night[k]["loss"].append(n.get("loss")); by_night[k]["seconds"].append(n.get("seconds"))
                if a == "sleep":
                    by_night[k]["dreams_generated"].append(gen); by_night[k]["dreams_accepted"].append(acc)
                    by_night[k]["yield"].append(acc / gen if gen else None)
                    for rj in ("duplicate", "out_of_spec", "prefix_changed", "unparsable", "wrong", "heldout"):
                        by_night[k][f"rej_{rj}"].append(n.get(f"dreams_rejected_{rj}", 0) or 0)
                    by_night[k]["structured_frac"].append(n.get("dreams_structured_frac"))
        out[a] = [{"night": k, **{m: _mean(v) for m, v in d.items()}} for k, d in sorted(by_night.items())]
    return out


def stability(grid, seeds) -> dict:
    out = {}
    for a in ARMS:
        rows = []
        for s in seeds:
            r = grid.get((s, a))
            if not r:
                continue
            h = [ck["heldout_accuracy"] for ck in r["checkpoints"]]
            rows.append({"seed": s, "peak": max(h), "final": h[-1], "max_drawdown": max_drawdown(h), "volatility": volatility(h),
                         "first_ckpt_ge_0.8": next((ck["episode"] for ck in r["checkpoints"] if ck["heldout_accuracy"] >= 0.8), None)})
        out[a] = {"per_seed": rows, "mean_drawdown": _mean(x["max_drawdown"] for x in rows), "mean_volatility": _mean(x["volatility"] for x in rows),
                  "mean_final_minus_peak": _mean(x["final"] - x["peak"] for x in rows)}
    return out


def secondary(grid, seeds) -> dict:
    out = {}
    for a in ARMS:
        mb_w = mb_c = mb_n = 0
        early, late, gap, med_first, med_last, unp = [], [], [], [], [], []
        for s in seeds:
            r = grid.get((s, a))
            if not r:
                continue
            for ck in r["checkpoints"]:
                if ck.get("mirror_bias") is not None and ck.get("mirror_bias_n"):
                    n = ck["mirror_bias_n"]; mb_w += ck["mirror_bias"] * n; mb_c += ck["mirror_bias_chance"] * n; mb_n += n
                if ck.get("control_unparsable") is not None:
                    unp.append(ck["control_unparsable"])
            last = r["checkpoints"][-1]
            if last.get("late_error_rate") is not None:
                late.append(last["late_error_rate"]); early.append(last["early_error_rate"])
            if last.get("short_gap") is not None:
                gap.append(last["short_gap"])
            if r["checkpoints"][0].get("median_tokens_correct") is not None:
                med_first.append(r["checkpoints"][0]["median_tokens_correct"]); med_last.append(last["median_tokens_correct"])
        out[a] = {"mirror_bias_pooled": (mb_w / mb_n) if mb_n else None, "mirror_chance_pooled": (mb_c / mb_n) if mb_n else None,
                  "mirror_n": mb_n, "late_error_final_mean": _mean(late), "early_error_final_mean": _mean(early),
                  "short_gap_final_mean": _mean(gap), "median_tokens_correct_first": _mean(med_first),
                  "median_tokens_correct_final": _mean(med_last), "control_unparsable_max": max(unp) if unp else None}
    return out


def buffers_summary(buffers, seeds) -> dict:
    out = {}
    for (s, a), b in sorted(buffers.items()):
        split = nr.make_split(seed=s, length=12)
        train = {x.digits for x in split.train}
        held = {x.digits for x in split.heldout} | {x.digits for x in split.probe}
        comp = buffer_composition(b, train)
        comp["touches_heldout_or_probe"] = len({x["digits"] for x in b} & held)
        comp["train_answer_dist"] = {k: v / len(split.train) for k, v in sorted(Counter(x.answer for x in split.train).items())}
        out[f"{a}_seed{s}"] = comp
    return out


def compute_summary(grid, timing, seeds) -> dict:
    out = {}
    for a in ARMS:
        gen, steps, tok, hours, gpus = [], [], [], [], Counter()
        for s in seeds:
            r = grid.get((s, a))
            if not r:
                continue
            L = r["ledger"]; gen.append(L["tokens_generated"]); steps.append(L["gradient_steps"]); tok.append(L["training_tokens"])
            t = timing.get((s, a))
            if t:
                hours.append(t["total_seconds"] / 3600); gpus[t.get("gpu", "?")] += 1
        out[a] = {"tokens_generated_mean": _mean(gen), "gradient_steps_mean": _mean(steps), "training_tokens_mean": _mean(tok),
                  "hours_mean": _mean(hours), "hours_n": len(hours), "gpus": dict(gpus)}
    return out


def awake_rounds(grid, seeds) -> dict:
    rounds = Counter(); acc_by_rounds = defaultdict(lambda: [0, 0])
    for s in seeds:
        r = grid.get((s, "awake"))
        if not r:
            continue
        for e in r["episodes"]:
            k = e.get("rounds", 1) or 1
            rounds[k] += 1; acc_by_rounds[k][1] += 1; acc_by_rounds[k][0] += int(bool(e["correct"]))
    base = Counter()
    for s in seeds:
        r = grid.get((s, "baseline"))
        if r:
            base["correct"] += sum(1 for e in r["episodes"] if e["correct"]); base["n"] += len(r["episodes"])
    return {"rounds_dist": dict(sorted(rounds.items())),
            "accuracy_by_rounds": {k: v[0] / v[1] for k, v in sorted(acc_by_rounds.items())},
            "awake_day_accuracy": sum(v[0] for v in acc_by_rounds.values()) / max(sum(v[1] for v in acc_by_rounds.values()), 1),
            "baseline_day_accuracy": base["correct"] / max(base["n"], 1)}


def power_note(grid, seeds) -> dict:
    diffs = []
    for s in seeds:
        a, b = grid.get((s, "sleep")), grid.get((s, "online"))
        if a and b:
            da_ = a["checkpoints"][-1]["control_accuracy"] - a["checkpoints"][0]["control_accuracy"]
            db_ = b["checkpoints"][-1]["control_accuracy"] - b["checkpoints"][0]["control_accuracy"]
            diffs.append(da_ - db_)
    sd = statistics.stdev(diffs) if len(diffs) > 1 else None
    m = statistics.fmean(diffs) if diffs else None
    # rough n for 80% power, two-sided 0.05, paired t: n ≈ ((1.96+0.84)·sd/|mean|)^2
    n_needed = ((1.96 + 0.84) * sd / abs(m)) ** 2 if (sd and m) else None
    return {"h4_diffs": diffs, "mean": m, "sd": sd, "min_attainable_p_n5": min_attainable_p(5),
            "approx_n_for_80pct_power_at_observed_effect": math.ceil(n_needed) if n_needed else None}


# ---- figures -----------------------------------------------------------------

def make_figures(res: dict, figdir: Path) -> list[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:                                   # pragma: no cover
        return []
    figdir.mkdir(parents=True, exist_ok=True)
    files = []

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, m, title in zip(axes, ("heldout_accuracy", "probe_accuracy", "control_accuracy"),
                            ("Held-out accuracy (201 unseen puzzles)", "Insight probe (60 masked items)", "GSM8K control (300 items)")):
        for a in ARMS:
            rows = res["curves"][a]
            x = [r["episode"] for r in rows]
            ax.plot(x, [r[m]["mean"] for r in rows], color=COLOR[a], label=LABEL[a], lw=2)
            ax.fill_between(x, [r[m]["min"] for r in rows], [r[m]["max"] for r in rows], color=COLOR[a], alpha=0.12)
        if m == "probe_accuracy":
            ax.axhline(1 / 3, ls=":", color="k", lw=1); ax.axhline(0.7, ls="--", color="k", lw=1)
            ax.text(5, 0.71, "criterion 0.70", fontsize=8); ax.text(5, 0.34, "chance", fontsize=8)
        ax.set_title(title, fontsize=11); ax.set_xlabel("episode"); ax.set_ylim(0, 1.02); ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, loc="lower right")
    fig.suptitle("Mean over 5 seeds; band = min–max across seeds", fontsize=10)
    fig.tight_layout(); f = figdir / "fig1_curves.png"; fig.savefig(f, dpi=130); plt.close(fig); files.append(f.name)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for a, rows in res["day_curves"].items():
        x = [r["episode_end"] for r in rows]
        axes[0].plot(x, [r["accuracy_mean"] for r in rows], color=COLOR[a], label=LABEL[a], lw=2)
        axes[0].fill_between(x, [r["accuracy_min"] for r in rows], [r["accuracy_max"] for r in rows], color=COLOR[a], alpha=0.12)
        axes[1].plot(x, [r["tokens_mean"] for r in rows], color=COLOR[a], label=LABEL[a], lw=2)
    axes[0].set_title("Day-time accuracy per 50-episode block", fontsize=11); axes[0].set_ylim(0, 1.02)
    axes[1].set_title("Generated tokens per day attempt (mean)", fontsize=11)
    for ax in axes:
        ax.set_xlabel("episode"); ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout(); f = figdir / "fig2_day_attempts.png"; fig.savefig(f, dpi=130); plt.close(fig); files.append(f.name)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    ns = res["nights"]["sleep"]; nd = res["nights"]["sleep_nodream"]
    x = [r["night"] for r in ns]
    axes[0].plot(x, [r["kept"] for r in ns], color=COLOR["sleep"], label="Sleep: real keeps / night", lw=2)
    axes[0].plot(x, [r["kept"] for r in nd], color=COLOR["sleep_nodream"], label="NoDream: real keeps / night", lw=2)
    axes[0].plot(x, [r["dreams_accepted"] for r in ns], color=COLOR["sleep"], ls="--", label="Sleep: dreams accepted / night", lw=2)
    axes[0].set_title("Night inputs (mean over seeds)", fontsize=11); axes[0].legend(fontsize=8)
    axes[1].plot(x, [r["loss"] for r in ns], color=COLOR["sleep"], label="Sleep", lw=2)
    axes[1].plot(x, [r["loss"] for r in nd], color=COLOR["sleep_nodream"], label="NoDream", lw=2)
    axes[1].set_title("Mean training loss per night", fontsize=11); axes[1].set_yscale("log"); axes[1].legend(fontsize=8)
    cats = ("duplicate", "wrong", "out_of_spec", "unparsable", "prefix_changed")
    bottom = [0.0] * len(x)
    for c in cats:
        vals = [r.get(f"rej_{c}") or 0 for r in ns]
        axes[2].bar(x, vals, bottom=bottom, label=c); bottom = [b + v for b, v in zip(bottom, vals)]
    axes[2].bar(x, [r["dreams_accepted"] for r in ns], bottom=bottom, label="accepted", color="k", alpha=0.6)
    axes[2].set_title("Dream proposals per night by outcome", fontsize=11); axes[2].legend(fontsize=7)
    for ax in axes:
        ax.set_xlabel("night"); ax.grid(alpha=0.3)
    fig.tight_layout(); f = figdir / "fig3_nights.png"; fig.savefig(f, dpi=130); plt.close(fig); files.append(f.name)

    fig, ax = plt.subplots(figsize=(8, 4))
    for i, a in enumerate(TRAINED):
        per = res["control_dynamics"][a]["per_seed"]
        ys = [per[str(s)]["delta"] if str(s) in per else per[s]["delta"] for s in sorted(per, key=int)]
        ax.scatter([i + (k - 2) * 0.08 for k in range(len(ys))], ys, color=COLOR[a], s=60, zorder=3)
        ax.hlines(statistics.fmean(ys), i - 0.25, i + 0.25, color=COLOR[a], lw=3)
    ax.axhline(0, color="k", lw=1); ax.set_xticks(range(3)); ax.set_xticklabels([LABEL[a] for a in TRAINED])
    ax.set_ylabel("GSM8K change, episode 0 → 600"); ax.set_title("Control-benchmark change per seed (dots) and mean (bar)", fontsize=11)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); f = figdir / "fig4_gsm8k_delta.png"; fig.savefig(f, dpi=130); plt.close(fig); files.append(f.name)
    return files


# ---- report ------------------------------------------------------------------

def _f(x, nd=3):
    return "—" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def report_markdown(res: dict, seeds: Sequence[int], figs: Sequence[str]) -> str:
    L = []
    L.append("# doze-llm — deep exploratory analysis of the completed 4B grid\n")
    L.append(f"Seeds {list(seeds)}; all five arms per seed. Everything below is descriptive or exploratory and was **not** "
             "preregistered; the preregistered verdicts are in `analysis_final.md`. Figures in `figures/`.\n")
    if figs:
        L.append("".join(f"![{f}](figures/{f})\n\n" for f in figs))

    L.append("## 1. Insight probe, pooled\n")
    L.append("Pooling every probe item over seeds and checkpoints gives far more power than any single checkpoint "
             "(≈3,900 item-evaluations per trained arm; 300 per frozen arm, whose 13 checkpoints repeat one identical evaluation). "
             "The same 60 items are re-asked at every checkpoint, so the pooled CI for trained arms is optimistic; the "
             "final-checkpoint column (300 independent items per arm) is the conservative reading. A shortcut used even "
             "5% of the time would show in either.\n")
    L.append("| arm | all checkpoints: correct / n | rate | 95% CI | p vs 1/3 | final checkpoint rate (n) | best single checkpoint |")
    L.append("|---|---|---|---|---|---|---|")
    for a in ARMS:
        p = res["probe_pooled"][a]; A = p["all_checkpoints"]; F = p["final_checkpoint"]; b = p["best_single_checkpoint"]
        L.append(f"| {LABEL[a]} | {A['correct']} / {A['n']} | {_f(A['rate'])} | {_f(A['ci95'][0])}–{_f(A['ci95'][1])} | {_f(A['p_two_sided'], 3)} | "
                 f"{_f(F['rate'])} ({F['n']}) | {b[0]:.2f} (seed {b[1]}, ep {b[2]}) |")
    L.append("")

    L.append("## 2. Learning during the day\n")
    L.append("Accuracy of the model's own day-time attempts per 50-episode block (these are the attempts that feed the "
             "filters), and how long a typical attempt is.\n")
    L.append("| arm | block 1 acc | block 6 acc | block 12 acc | tokens/attempt block 1 → 12 | correct steps/attempt block 1 → 12 |")
    L.append("|---|---|---|---|---|---|")
    for a, rows in res["day_curves"].items():
        r1, r6, r12 = rows[0], rows[min(5, len(rows) - 1)], rows[-1]
        L.append(f"| {LABEL[a]} | {r1['accuracy_mean']:.3f} | {r6['accuracy_mean']:.3f} | {r12['accuracy_mean']:.3f} | "
                 f"{r1['tokens_mean']:.0f} → {r12['tokens_mean']:.0f} | {r1['steps_correct_mean']:.1f} → {r12['steps_correct_mean']:.1f} |")
    L.append("")
    L.append("Where wrong attempts fail (number of correct leading steps before the first error; pooled over 5 seeds, so 500 attempts per column):\n")
    L.append("| arm | wrong in first 100 episodes | mode of first-error step | wrong in last 100 | mode of first-error step |")
    L.append("|---|---|---|---|---|")
    for a in ARMS:
        e = res["error_positions"][a]
        m1 = max(e["first100"], key=e["first100"].get) if e["first100"] else None
        m2 = max(e["last100"], key=e["last100"].get) if e["last100"] else None
        L.append(f"| {LABEL[a]} | {e['n_wrong_first100']} | {m1} | {e['n_wrong_last100']} | {m2} |")
    L.append("")

    L.append("## 3. Held-out items: what the model actually wrote\n")
    hi = res["heldout_items"]
    L.append("Recomputed from the saved per-item rows (model's parsed steps and answer) at the final checkpoint; "
             "`reported` is the harness's own number, as a consistency check.\n")
    L.append("| run | recomputed correct / n | reported |")
    L.append("|---|---|---|")
    for k, v in sorted(hi["per_run_final"].items()):
        L.append(f"| {k} | {v['correct']} / {v['n']} = {v['correct'] / v['n']:.3f} | {v['reported']:.3f} |")
    L.append("")
    L.append("First wrong step on held-out items (count of wrong items by number of correct leading steps), first vs final evaluation:\n")
    L.append("| arm | first eval | final eval |")
    L.append("|---|---|---|")
    for a in ARMS:
        w = hi["first_wrong_step"][a]
        L.append(f"| {LABEL[a]} | {w['first_eval']} | {w['final_eval']} |")
    L.append("")
    h = hi["hard_items"]
    L.append(f"Held-out items never solved by any trained arm at episode 600 (pooled over seeds, items seen ≥3 times): "
             f"**{h['never_solved_by_any_trained_arm_at_600']}** of {h['n_items_seen']}. Examples: {h['examples'][:5]}\n")

    L.append("## 4. Control benchmark dynamics\n")
    cd = res["control_dynamics"]
    L.append("| arm | mean change per checkpoint | SD | worst single change per seed | checkpoints with a drop ≥ 0.05 (of n) |")
    L.append("|---|---|---|---|---|")
    for a in TRAINED:
        c = cd[a]
        L.append(f"| {LABEL[a]} | {c['per_checkpoint_change_mean']:+.4f} | {c['per_checkpoint_change_sd']:.3f} | "
                 f"{[round(x, 3) for x in c['worst_change_per_seed']]} | {c['n_drops_ge_0.05']} (of {c['n_changes']}) |")
    L.append("")
    L.append("| arm | seed | GSM8K Δ | peak | trough | max drawdown | volatility |")
    L.append("|---|---|---|---|---|---|---|")
    for a in TRAINED:
        for s, v in sorted(cd[a]["per_seed"].items(), key=lambda kv: int(kv[0])):
            L.append(f"| {LABEL[a]} | {s} | {v['delta']:+.3f} | {v['peak']:.3f} | {v['trough']:.3f} | {v['max_drawdown']:.3f} | {v['volatility']:.3f} |")
    L.append("")
    corr = cd["sleep_night_correlations"]
    L.append(f"Sleep: Pearson correlation between the GSM8K change across a night and that night's training "
             f"(n = {cd['sleep_night_n']} nights): loss {_f(corr['loss'])}, buffer size {_f(corr['buffer'])}, "
             f"train-set size {_f(corr['train_set'])}, dream fraction of the train set {_f(corr['dream_frac'])}, "
             f"new keeps {_f(corr['new'])}. Values near 0 mean the forgetting is not explained by that quantity.\n")

    L.append("## 5. Nights\n")
    L.append("Mean over seeds per night (Sleep). `kept` = day trajectories passing the C3 filter (real keeps added to the buffer); "
             "`new` = items new to this night's train set = real keeps + accepted dreams; the train set is `new` plus a 1:1 replay draw.\n")
    L.append("| night | kept | new | buffer | dreams gen. | accepted | yield | dup | wrong | out-of-spec | unparsable | structured frac | loss | seconds |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in res["nights"]["sleep"]:
        L.append(f"| {r['night']} | {_f(r['kept'], 1)} | {_f(r['new'], 1)} | {_f(r['buffer'], 0)} | {_f(r['dreams_generated'], 1)} | {_f(r['dreams_accepted'], 1)} | "
                 f"{_f(r['yield'], 2)} | {_f(r['rej_duplicate'], 1)} | {_f(r['rej_wrong'], 1)} | {_f(r['rej_out_of_spec'], 1)} | {_f(r['rej_unparsable'], 1)} | "
                 f"{_f(r['structured_frac'], 3)} | {_f(r['loss'], 4)} | {_f(r['seconds'], 0)} |")
    L.append("")
    L.append("Sleep-NoDream nights (mean over seeds):\n")
    L.append("| night | kept | new | buffer | train set | loss | seconds |")
    L.append("|---|---|---|---|---|---|---|")
    for r in res["nights"]["sleep_nodream"]:
        L.append(f"| {r['night']} | {_f(r['kept'], 1)} | {_f(r['new'], 1)} | {_f(r['buffer'], 0)} | {_f(r['train_set'], 0)} | {_f(r['loss'], 4)} | {_f(r['seconds'], 0)} |")
    L.append("")

    L.append("## 6. Stability of the task metric\n")
    L.append("| arm | mean max drawdown (held-out) | mean volatility | mean final − peak | first checkpoint ≥ 0.80 per seed |")
    L.append("|---|---|---|---|---|")
    for a in ARMS:
        st = res["stability"][a]
        L.append(f"| {LABEL[a]} | {_f(st['mean_drawdown'])} | {_f(st['mean_volatility'])} | {_f(st['mean_final_minus_peak'])} | "
                 f"{[x['first_ckpt_ge_0.8'] for x in st['per_seed']]} |")
    L.append("")

    L.append("## 7. Secondary metrics (pooled)\n")
    L.append("| arm | mirror bias (pooled, n) | its chance level | late-step error rate (final) | early-step error rate (final) | short-answer gap (final) | median tokens per correct answer first → final | GSM8K unparsable max |")
    L.append("|---|---|---|---|---|---|---|---|")
    for a in ARMS:
        s_ = res["secondary"][a]
        L.append(f"| {LABEL[a]} | {_f(s_['mirror_bias_pooled'])} ({s_['mirror_n']}) | {_f(s_['mirror_chance_pooled'])} | {_f(s_['late_error_final_mean'])} | "
                 f"{_f(s_['early_error_final_mean'])} | {_f(s_['short_gap_final_mean'])} | {_f(s_['median_tokens_correct_first'], 0)} → {_f(s_['median_tokens_correct_final'], 0)} | {_f(s_['control_unparsable_max'])} |")
    L.append("")

    L.append("## 8. Replay buffers at the end of training\n")
    L.append("From the saved buffer state (seeds with a recovered `.state`). Coverage = fraction of the 801 training prefixes present.\n")
    L.append("| buffer | items | distinct puzzles | train coverage | train-prefix coverage | outside train | touches held-out/probe | answer dist (train) |")
    L.append("|---|---|---|---|---|---|---|---|")
    for k, v in res["buffers"].items():
        ad = ", ".join(f"{a}:{p:.2f}" for a, p in v["answer_dist"].items())
        td = ", ".join(f"{a}:{p:.2f}" for a, p in v["train_answer_dist"].items())
        L.append(f"| {k} | {v['n']} | {v['n_distinct_digits']} | {v['train_coverage']:.2f} | {v['prefix_coverage']:.2f} | {v['outside_train']} | {v['touches_heldout_or_probe']} | {ad} ({td}) |")
    L.append("")

    L.append("## 9. Awake: did critique rounds do anything?\n")
    aw = res["awake"]
    L.append(f"Rounds used per day attempt: {aw['rounds_dist']}. Accuracy by rounds used: "
             f"{ {k: round(v, 3) for k, v in aw['accuracy_by_rounds'].items()} }. Awake day accuracy {aw['awake_day_accuracy']:.3f} vs "
             f"Baseline day accuracy {aw['baseline_day_accuracy']:.3f}.\n")

    L.append("## 10. Compute and time\n")
    L.append("| arm | generated tokens (mean) | gradient steps | training tokens | wall hours (mean, n) | GPUs |")
    L.append("|---|---|---|---|---|---|")
    for a in ARMS:
        c = res["compute"][a]
        L.append(f"| {LABEL[a]} | {_f(c['tokens_generated_mean'], 0)} | {_f(c['gradient_steps_mean'], 0)} | {_f(c['training_tokens_mean'], 0)} | "
                 f"{_f(c['hours_mean'], 1)} ({c['hours_n']}) | {c['gpus']} |")
    L.append("")

    L.append("## 11. Power\n")
    pw = res["power"]
    L.append(f"H4 paired differences (Sleep − Online GSM8K Δ): {[round(x, 3) for x in pw['h4_diffs']]}, mean {pw['mean']:+.3f}, SD {_f(pw['sd'])}. "
             f"With n = 5 the exact sign-flip test cannot go below p = {pw['min_attainable_p_n5']}. At the observed effect and spread, "
             f"a paired test would need roughly n ≈ {pw['approx_n_for_80pct_power_at_observed_effect']} seeds for 80% power.\n")
    return "\n".join(L)


def run(root: str | Path, seeds: Sequence[int]) -> dict:
    root = Path(root)
    grid = load_grid(root, seeds)
    buffers = load_buffers(root, seeds)
    timing = load_timing(root, seeds)
    res = {"seeds": list(seeds), "curves": curves(grid, seeds), "probe_pooled": probe_pooled(grid, seeds),
           "day_curves": day_curves(grid, seeds), "error_positions": error_positions(grid, seeds),
           "heldout_items": heldout_items(grid, seeds), "control_dynamics": control_dynamics(grid, seeds),
           "nights": nights_summary(grid, seeds), "stability": stability(grid, seeds), "secondary": secondary(grid, seeds),
           "buffers": buffers_summary(buffers, seeds), "compute": compute_summary(grid, timing, seeds),
           "awake": awake_rounds(grid, seeds), "power": power_note(grid, seeds)}
    figs = make_figures(res, root / "figures")
    (root / "analysis_deep.json").write_text(json.dumps(res, indent=1, default=str))
    (root / "analysis_deep.md").write_text(report_markdown(res, seeds, figs))
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="records")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    a = ap.parse_args()
    seeds = [int(x) for x in a.seeds.split(",")]
    res = run(a.results, seeds)
    print(f"wrote {a.results}/analysis_deep.md (+ .json, figures/) for seeds {seeds}")
    for arm in ARMS:
        p = res["probe_pooled"][arm]["all_checkpoints"]
        print(f"  probe pooled {arm:14s} {p['correct']}/{p['n']} = {p['rate']:.3f}  p={p['p_two_sided']:.3g}")


if __name__ == "__main__":
    main()
