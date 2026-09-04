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
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
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


if __name__ == "__main__":
    main()
