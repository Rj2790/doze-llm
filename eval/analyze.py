"""Analysis gate and summary (PREREG §7, amendment A3).

Nothing is plotted or tabulated for a seed until `check_seed` has loaded the
per-arm ledgers with Ledger.load, run check_matched (+-5%, one backend) and
written the verdict into every run JSON of that seed. Unmatched or
mixed-backend arms raise AnalysisRefused.

  python -m eval.analyze --results results --seeds 0,1,2,3,4
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from eval import compute_ledger as cl


class AnalysisRefused(RuntimeError):
    pass


FROZEN_ARMS = ("baseline", "awake")   # weights never change; greedy eval should be identical at every checkpoint


def frozen_eval_identical(checkpoints: list[dict]) -> bool:
    key = lambda c: (c["probe_accuracy"], c["heldout_accuracy"], c["control_accuracy"])
    return all(key(c) == key(checkpoints[0]) for c in checkpoints)


def _runs(root: Path, seed: int) -> dict[str, Path]:
    return {p.stem.rsplit("_seed", 1)[0]: p for p in sorted((root / "runs").glob(f"*_seed{seed}.json"))}


def _ledgers(root: Path, seed: int) -> dict[str, cl.Ledger]:
    out = {}
    for p in sorted((root / "ledgers").glob(f"*_seed{seed}.json")):
        L = cl.Ledger.load(p)
        out[L.arm] = L
    return out


def check_seed(root: str | Path, seed: int, tol: float = 0.05) -> dict:
    """Load ledgers, run check_matched, persist the verdict in each run JSON.
    Raises AnalysisRefused if the arms of this seed must not be compared."""
    root = Path(root)
    ledgers = _ledgers(root, seed)
    if not ledgers:
        raise AnalysisRefused(f"no ledgers for seed {seed} under {root / 'ledgers'}")
    runs = _runs(root, seed)
    backends = sorted({json.loads(p.read_text()).get("backend", "") for p in runs.values()} | {L.backend for L in ledgers.values()})
    verdict = {"seed": seed, "arms": sorted(ledgers), "backend": backends[0] if len(backends) == 1 else None,
               "tol": tol, "ok": False, "message": ""}
    try:
        if len(backends) > 1:
            raise cl.BudgetMismatch(f"runs/ledgers from different backends must never be compared: {backends}")
        cl.check_matched(ledgers, tol=tol)
        verdict["ok"] = True
        verdict["message"] = "compute budgets matched"
    except cl.BudgetMismatch as e:
        verdict["message"] = str(e)
    for arm, p in runs.items():
        d = json.loads(p.read_text())
        d["matched"] = verdict
        p.write_text(json.dumps(d, indent=1))
    if not verdict["ok"]:
        raise AnalysisRefused(f"seed {seed}: {verdict['message']}")
    return verdict


def summarize(root: str | Path, seeds: list[int], tol: float = 0.05) -> dict:
    """Per-arm criterion episodes, held-out / probe curves and forgetting
    (control at last checkpoint minus at episode 0), only after every seed
    passes check_seed."""
    root = Path(root)
    verdicts = [check_seed(root, s, tol) for s in seeds]
    arms: dict[str, dict] = {}
    for s in seeds:
        for arm, p in _runs(root, s).items():
            d = json.loads(p.read_text())
            cks = d["checkpoints"]
            a = arms.setdefault(arm, {"criterion_episodes": [], "forgetting": [], "final_heldout": [],
                                      "final_probe": [], "probe_curves": [], "seeds": []})
            a["seeds"].append(s)
            a["criterion_episodes"].append(d["criterion_episode"])
            c0 = cks[0]["control_accuracy"] if cks else None
            cl_ = cks[-1]["control_accuracy"] if cks else None
            a["forgetting"].append(None if c0 is None or cl_ is None else cl_ - c0)
            a["final_heldout"].append(cks[-1]["heldout_accuracy"] if cks else None)
            a["final_probe"].append(cks[-1]["probe_accuracy"] if cks else None)
            a["probe_curves"].append([(c["episode"], c["probe_accuracy"]) for c in cks])
            if arm in FROZEN_ARMS:
                a.setdefault("frozen_eval_identical", []).append(frozen_eval_identical(cks) if cks else None)
            # post-pilot secondary metrics (None when the run predates them)
            from eval import implicit as _imp
            a.setdefault("volatility", []).append(_imp.volatility([c["heldout_accuracy"] for c in cks]))
            last = cks[-1] if cks else {}
            for key in ("mirror_bias", "mirror_bias_chance", "short_gap", "short_structured_acc", "short_unstructured_acc",
                        "short_structured_unparsable", "short_unstructured_unparsable",
                        "late_error_rate", "early_error_rate", "control_unparsable"):
                a.setdefault(f"final_{key}", []).append(last.get(key))
    for a in arms.values():
        met = [e for e in a["criterion_episodes"] if e is not None]
        a["median_criterion_episode"] = statistics.median(met) if met else None
        a["n_censored"] = len(a["criterion_episodes"]) - len(met)
    return {"seeds_checked": seeds, "verdicts": verdicts, "arms": arms}


def report_markdown(summary: dict) -> str:
    lines = ["# doze-llm results", "", f"Seeds: {summary['seeds_checked']} (all compute-matched)", "",
             "| arm | median episode-to-criterion | censored | final held-out | final probe | forgetting (control delta) |",
             "|---|---|---|---|---|---|"]
    for arm, a in sorted(summary["arms"].items()):
        fh = [x for x in a["final_heldout"] if x is not None]
        fp = [x for x in a["final_probe"] if x is not None]
        fg = [x for x in a["forgetting"] if x is not None]
        m = lambda v: f"{statistics.mean(v):.3f}" if v else "—"
        lines.append(f"| {arm} | {a['median_criterion_episode']} | {a['n_censored']}/{len(a['seeds'])} | "
                     f"{m(fh)} | {m(fp)} | {m(fg)} |")
    lines += ["", "| arm | volatility (std held-out) | mirror_bias (chance) | short_gap (struct / unstruct) | late vs early error | GSM8K unparsable |",
              "|---|---|---|---|---|---|"]
    for arm, a in sorted(summary["arms"].items()):
        f = lambda k: [v for v in a.get(k, []) if v is not None]
        m = lambda v, d=3: f"{statistics.mean(v):.{d}f}" if v else "—"
        lines.append(f"| {arm} | {m(f('volatility'))} | {m(f('final_mirror_bias'))} ({m(f('final_mirror_bias_chance'))}) | "
                     f"{m(f('final_short_gap'))} ({m(f('final_short_structured_acc'))} / {m(f('final_short_unstructured_acc'))}) | "
                     f"{m(f('final_late_error_rate'))} vs {m(f('final_early_error_rate'))} | {m(f('final_control_unparsable'))} |")
    frozen = [(arm, a["frozen_eval_identical"]) for arm, a in sorted(summary["arms"].items()) if "frozen_eval_identical" in a]
    if frozen:
        lines += ["", "Eval-pipeline validation (frozen-arm eval identical at every checkpoint, per seed): "
                  + "; ".join(f"{arm}: {'yes' if all(v) else 'NO'} {v}" for arm, v in frozen)]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--tol", type=float, default=0.05)
    ap.add_argument("--final", action="store_true", help="PREREG §7 analysis over the records layout (<root>/seed<N>/)")
    ap.add_argument("--out", default="", help="markdown output path for --final")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    if args.final:
        fa = final_analysis(args.results, seeds, layout="per-seed", tol=args.tol)
        md = final_report_markdown(fa)
        out = Path(args.out or (Path(args.results) / ("analysis_preliminary.md" if fa["preliminary"] else "analysis_final.md")))
        out.write_text(md)
        (out.with_suffix(".json")).write_text(json.dumps(fa, indent=1, default=str))
        print(md)
        return
    summary = summarize(args.results, seeds, args.tol)       # raises before any output if unmatched
    out = Path(args.results) / "report.md"
    out.write_text(report_markdown(summary))
    (Path(args.results) / "summary.json").write_text(json.dumps(summary, indent=1))
    print(report_markdown(summary))
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        for arm, a in sorted(summary["arms"].items()):
            for curve in a["probe_curves"]:
                ax.plot([e for e, _ in curve], [v for _, v in curve], label=arm, alpha=0.6)
        ax.axhline(0.70, ls="--", c="k"); ax.set_xlabel("episode"); ax.set_ylabel("probe accuracy")
        h, l = ax.get_legend_handles_labels(); ax.legend(dict(zip(l, h)).values(), dict(zip(l, h)).keys())
        fig.savefig(Path(args.results) / "probe_curves.png", dpi=150)
    except ImportError:
        print("(matplotlib not installed; no figure)")




# =============================================================================
# Final analysis (PREREG §7) over the records layout: <root>/seed<N>/{runs,ledgers}
# =============================================================================

from eval import stats as _stats  # noqa: E402

ARMS = ("baseline", "awake", "online", "sleep", "sleep_nodream")
N_EPISODES = 600


def seed_root(root: str | Path, seed: int, layout: str) -> Path:
    root = Path(root)
    return root / f"seed{seed}" if layout == "per-seed" else root


def _seed_summary(run: dict) -> dict:
    cks = run["checkpoints"]
    held = [c["heldout_accuracy"] for c in cks]
    ctrl = [c["control_accuracy"] for c in cks]
    probe = [c["probe_accuracy"] for c in cks]
    last3 = held[-3:] if len(held) >= 3 else held
    # per-checkpoint change in control accuracy (for Sleep arms this is the post-night change)
    steps = [ctrl[i] - ctrl[i - 1] for i in range(1, len(ctrl)) if ctrl[i] is not None and ctrl[i - 1] is not None]
    return {"final_heldout": held[-1], "peak_heldout": max(held), "last3_heldout": sum(last3) / len(last3),
            "forgetting": (ctrl[-1] - ctrl[0]) if ctrl and ctrl[0] is not None and ctrl[-1] is not None else None,
            "control_min": min(c for c in ctrl if c is not None) if any(c is not None for c in ctrl) else None,
            "probe_max": max(probe), "probe_final": probe[-1], "criterion_episode": run["criterion_episode"],
            "volatility": statistics.pstdev(held) if len(held) > 1 else None,
            "post_update_control_changes": steps, "backend": run.get("backend"),
            "tokens": run["ledger"]["tokens_generated"], "steps": run["ledger"]["gradient_steps"]}


def final_analysis(root: str | Path, seeds: list[int], layout: str = "per-seed", required_seeds: int = 5,
                   tol: float = 0.05) -> dict:
    """Gate every seed (check_seed), then compute PREREG §7: H1–H3 via
    log-rank on episode-to-criterion (censored at 600), H4 via a paired
    sign-flip test on GSM8K change (Sleep − Online), plus robust per-seed
    summaries and an exploratory schedule comparison."""
    verdicts, per_seed = [], {a: {} for a in ARMS}
    for s in seeds:
        r = seed_root(root, s, layout)
        verdicts.append(check_seed(r, s, tol))
        for arm, p in _runs(r, s).items():
            run = json.loads(p.read_text())
            if run.get("partial"):
                continue
            per_seed.setdefault(arm, {})[s] = _seed_summary(run)

    def surv(arm):
        return [((per_seed[arm][s]["criterion_episode"] or N_EPISODES), per_seed[arm][s]["criterion_episode"] is not None)
                for s in sorted(per_seed.get(arm, {}))]

    def paired(arm_a, arm_b, key):
        common = sorted(set(per_seed.get(arm_a, {})) & set(per_seed.get(arm_b, {})))
        diffs = [per_seed[arm_a][s][key] - per_seed[arm_b][s][key] for s in common
                 if per_seed[arm_a][s][key] is not None and per_seed[arm_b][s][key] is not None]
        out = _stats.sign_flip_test(diffs); out["seeds"] = common; out["diffs"] = diffs
        return out

    H = {}
    # H1: Sleep reaches criterion before Baseline, Awake, Online (log-rank vs each; events needed)
    lr1 = {other: _stats.logrank(surv("sleep"), surv(other)) for other in ("baseline", "awake", "online") if per_seed.get(other)}
    any_event = any(v["events"] > 0 for v in lr1.values())
    met = sum(1 for s in per_seed.get("sleep", {}).values() if s["criterion_episode"] is not None)
    H["H1"] = {"verdict": "supported" if any_event and all(v["p"] is not None and v["p"] < 0.05 and v["observed_a"] > v["expected_a"] for v in lr1.values()) else "not supported",
               "logrank": lr1, "note": f"Sleep met the criterion in {met}/{len(per_seed.get('sleep', {}))} seeds" + ("; all arms censored at 600 — log-rank degenerate" if not any_event else "")}
    lr2 = _stats.logrank(surv("sleep"), surv("online"))
    H["H2"] = {"verdict": "supported" if (not lr2["degenerate"] and lr2["p"] < 0.05 and lr2["observed_a"] > lr2["expected_a"]) else "not supported",
               "logrank": lr2, "paired_heldout_last3": paired("sleep", "online", "last3_heldout"),
               "note": "load-bearing comparison; episode-to-criterion, censored at 600"}
    lr3 = _stats.logrank(surv("sleep"), surv("sleep_nodream"))
    H["H3"] = {"verdict": "supported" if (not lr3["degenerate"] and lr3["p"] < 0.05 and lr3["observed_a"] > lr3["expected_a"]) else "not supported",
               "logrank": lr3, "paired_heldout_last3": paired("sleep", "sleep_nodream", "last3_heldout")}
    h4 = paired("sleep", "online", "forgetting")     # predicted: Sleep forgets less -> Sleep's delta > Online's -> mean > 0
    direction = "predicted" if (h4["mean"] is not None and h4["mean"] > 0) else ("reversed" if h4["mean"] is not None and h4["mean"] < 0 else "none")
    H["H4"] = {"verdict": "supported" if (h4["p"] is not None and h4["p"] < 0.05 and direction == "predicted") else "not supported",
               "paired": h4, "direction": direction, "note": "GSM8K change ep600−ep0, Sleep − Online, exact sign-flip test"}

    paired_out = {"H4_forgetting_sleep_minus_online": h4,
                  "heldout_last3_sleep_minus_online": paired("sleep", "online", "last3_heldout"),
                  "heldout_last3_sleep_minus_nodream": paired("sleep", "sleep_nodream", "last3_heldout"),
                  "forgetting_sleep_minus_nodream": paired("sleep", "sleep_nodream", "forgetting"),
                  "forgetting_nodream_minus_online": paired("sleep_nodream", "online", "forgetting")}

    # exploratory: update schedule (Online per-step vs Sleep nightly) on control and task
    sched = {}
    for arm in ("online", "sleep", "sleep_nodream"):
        rows = per_seed.get(arm, {})
        if not rows:
            continue
        deltas = [r["forgetting"] for r in rows.values() if r["forgetting"] is not None]
        changes = [x for r in rows.values() for x in r["post_update_control_changes"]]
        sched[arm] = {"n": len(rows), "gsm8k_delta_mean": statistics.mean(deltas) if deltas else None,
                      "gsm8k_delta_per_seed": {s: r["forgetting"] for s, r in rows.items()},
                      "gsm8k_positive_seeds": sum(1 for d in deltas if d > 0),
                      "heldout_peak_mean": statistics.mean(r["peak_heldout"] for r in rows.values()),
                      "heldout_last3_mean": statistics.mean(r["last3_heldout"] for r in rows.values()),
                      "post_update_gsm8k_change_mean": statistics.mean(changes) if changes else None,
                      "post_update_gsm8k_change_min": min(changes) if changes else None,
                      "volatility_mean": statistics.mean(r["volatility"] for r in rows.values() if r["volatility"] is not None)}

    complete = all(len(per_seed.get(a, {})) >= required_seeds for a in ARMS)
    return {"seeds_checked": seeds, "verdicts": verdicts, "per_seed": per_seed, "hypotheses": H, "paired": paired_out,
            "exploratory": {"schedule": sched}, "preliminary": not complete, "required_seeds": required_seeds}


def final_report_markdown(fa: dict) -> str:
    L = ["# doze-llm — preregistered analysis (PREREG §7)", ""]
    if fa["preliminary"]:
        L += [f"**PRELIMINARY — fewer than {fa['required_seeds']} complete seeds for at least one arm. Not the final result.**", ""]
    L += [f"Seeds: {fa['seeds_checked']}; compute matching passed for every seed listed (check_seed).", ""]
    L += ["## Hypotheses", "", "| hypothesis | test | result | verdict |", "|---|---|---|---|"]
    H = fa["hypotheses"]
    def lr(v):
        return "all censored (degenerate)" if v.get("degenerate") else f"χ²={v['statistic']:.2f}, p={v['p']:.3f}, events={v['events']}"
    h1 = "; ".join(f"vs {k}: {lr(v)}" for k, v in H["H1"]["logrank"].items())
    L.append(f"| H1 insight (Sleep first to criterion) | log-rank | {h1}. {H['H1']['note']} | **{H['H1']['verdict']}** |")
    L.append(f"| H2 phase structure (Sleep vs Online) | log-rank | {lr(H['H2']['logrank'])} | **{H['H2']['verdict']}** |")
    L.append(f"| H3 dreaming (Sleep vs NoDream) | log-rank | {lr(H['H3']['logrank'])} | **{H['H3']['verdict']}** |")
    p4 = H["H4"]["paired"]
    L.append(f"| H4 retention (Sleep forgets less than Online) | paired sign-flip on GSM8K Δ | mean Sleep−Online {p4['mean']:+.3f} (n={p4['n']}, p={p4['p']}); direction {H['H4']['direction']} | **{H['H4']['verdict']}** |")
    L += ["", "## Per-arm summary (mean over seeds)", "", "| arm | n | peak held-out | last-3 held-out | GSM8K Δ | probe max | criterion met |", "|---|---|---|---|---|---|---|"]
    for arm in ARMS:
        rows = fa["per_seed"].get(arm, {})
        if not rows:
            continue
        v = list(rows.values())
        m = lambda k: statistics.mean(r[k] for r in v if r[k] is not None)
        L.append(f"| {arm} | {len(v)} | {m('peak_heldout'):.3f} | {m('last3_heldout'):.3f} | {m('forgetting'):+.3f} | {m('probe_max'):.3f} | {sum(r['criterion_episode'] is not None for r in v)}/{len(v)} |")
    L += ["", "## Paired comparisons (per seed; exact sign-flip test)", "", "| comparison | diffs | mean | p |", "|---|---|---|---|"]
    for k, v in fa["paired"].items():
        L.append(f"| {k} | {[round(x, 3) for x in v['diffs']]} | {v['mean']:+.3f} | {v['p']} |" if v["mean"] is not None else f"| {k} | — | — | — |")
    L += ["", "## Exploratory: update schedule (not preregistered)", "",
          "Online applies one verified self-training step per episode; Sleep and Sleep-NoDream apply 50 per night. Same steps, same tokens.", "",
          "| arm | n | GSM8K Δ mean | seeds with Δ>0 | mean per-checkpoint GSM8K change | worst single-checkpoint change | peak held-out | last-3 held-out | volatility |", "|---|---|---|---|---|---|---|---|---|"]
    for arm, e in fa["exploratory"]["schedule"].items():
        L.append(f"| {arm} | {e['n']} | {e['gsm8k_delta_mean']:+.3f} | {e['gsm8k_positive_seeds']}/{e['n']} | {e['post_update_gsm8k_change_mean']:+.4f} | {e['post_update_gsm8k_change_min']:+.3f} | {e['heldout_peak_mean']:.3f} | {e['heldout_last3_mean']:.3f} | {e['volatility_mean']:.3f} |")
    L += ["", "Caveats: seed 0's Sleep/Online/Awake used the pilot code (one-step dreamer, non-deterministic CUDA); seed 1 mixes GPU types (L4 for Sleep/Baseline, A16 otherwise); insight hypotheses are fully censored when no arm reaches the criterion, so the log-rank test carries no information beyond 'never met'."]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
