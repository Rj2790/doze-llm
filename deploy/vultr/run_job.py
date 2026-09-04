"""Run one arm for one seed on a plain GPU VM (Vultr etc.). Mirrors
modal_app._run without any Modal dependency.

  python deploy/vultr/run_job.py --arm sleep --seed 1 --run-tag seed1 --results /data/results
  python deploy/vultr/run_job.py --arm awake --seed 1 --run-tag seed1 --awake-from-sleep   # budget from sleep ledger
  python deploy/vultr/run_job.py --arm baseline --seed 0 --backend scripted --n-episodes 4 --k 2 ...  # dry run

Results: <results>/<tag>/runs/<arm>_seed<seed>.json (saved every checkpoint,
resumable), <results>/<tag>/ledgers/, timing printed as a JSON line.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# the script directory must not shadow stdlib modules (a local module named like a stdlib one broke torch import)
sys.path = [p for p in sys.path if Path(p or ".").resolve() != Path(__file__).resolve().parent]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from arms import harness as hz            # noqa: E402
from arms.awake import AwakeArm           # noqa: E402
from arms.baseline import BaselineArm     # noqa: E402
from arms.online import OnlineArm         # noqa: E402
from arms.sleep import SleepArm, SleepConfig   # noqa: E402
from eval import compute_ledger as cl     # noqa: E402
from eval import orchestrate              # noqa: E402
from tasks import number_reduction as nr  # noqa: E402

TRAINABLE = ("online", "online_unfiltered", "sleep", "sleep_nodream")


def gpu_name() -> str:
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=10).stdout.strip().splitlines()[0]
    except Exception:
        return "unknown"


def build_backend(args, trainable: bool):
    if args.backend == "scripted":
        from backends.scripted import FakeTrainableBackend, ScriptedBackend
        return FakeTrainableBackend(seed=args.seed) if trainable else ScriptedBackend("literal", seed=args.seed)
    from backends.hf_backend import HFBackend
    be = HFBackend(args.model, system=nr.SYSTEM_PROMPT, lora=trainable, lr=args.lr, seed=args.seed,
                   deterministic=not args.no_deterministic)
    be.batch_size = args.batch_size
    return be


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["baseline", "awake", "online", "online_unfiltered", "sleep", "sleep_nodream"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run-tag", default="")
    ap.add_argument("--results", default=os.environ.get("DOZE_RESULTS", "results/vm"))
    ap.add_argument("--backend", choices=["hf", "scripted"], default="hf")
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--n-episodes", type=int, default=600)
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--n-probe", type=int, default=60)
    ap.add_argument("--n-heldout-eval", type=int, default=nr.N_HELDOUT)
    ap.add_argument("--control-items", type=int, default=300)
    ap.add_argument("--n-implicit", type=int, default=120)
    ap.add_argument("--awake-budget", type=int, default=0)
    ap.add_argument("--awake-from-sleep", action="store_true", help="derive the budget from <results>/<tag>/ledgers/sleep_seed<seed>.json")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--batch-size", type=int, default=8, help="eval generation batch (16 GB GPUs: 8)")
    ap.add_argument("--no-deterministic", action="store_true")
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    tag = args.run_tag or orchestrate.new_tag()
    P = orchestrate.paths(args.results, tag, args.arm, args.seed)
    trainable = args.arm in TRAINABLE
    budget = args.awake_budget
    if args.arm == "awake":
        if args.awake_from_sleep:
            budget = AwakeArm.budget_from_reference(cl.Ledger.load(orchestrate.paths(args.results, tag, "sleep", args.seed)["ledger"]),
                                                    args.n_episodes)
        if budget <= 0:
            raise SystemExit("awake needs --awake-budget > 0 or --awake-from-sleep")
    t0 = time.time()
    backend = build_backend(args, trainable)
    if args.arm == "baseline":
        arm = BaselineArm()
    elif args.arm == "awake":
        arm = AwakeArm(token_budget_per_episode=budget)
    elif args.arm in ("online", "online_unfiltered"):
        arm = OnlineArm(filtered=(args.arm == "online"))
    else:
        arm = SleepArm(SleepConfig(weight_decay=args.weight_decay), dream=(args.arm == "sleep"))
    cfg = hz.RunConfig(seed=args.seed, n_episodes=args.n_episodes, k=args.k, n_probe=args.n_probe,
                       n_heldout_eval=args.n_heldout_eval, control_items=args.control_items, n_implicit=args.n_implicit)
    print(f"gpu={gpu_name()} arm={arm.name} seed={args.seed} tag={tag} deterministic={not args.no_deterministic} "
          f"backend={backend.name} batch={args.batch_size}" + (f" awake_budget={budget}" if args.arm == "awake" else ""), flush=True)
    res = hz.run_arm(arm, backend, cfg, log=lambda m: print(m, flush=True), save_path=P["run"], resume=not args.no_resume)
    res.ledger.save(P["ledger"])
    eps = [e.seconds for e in res.episodes]
    timing = {"gpu": gpu_name(), "deterministic": not args.no_deterministic, "total_seconds": round(time.time() - t0, 1),
              "episode_seconds_mean": round(sum(eps) / len(eps), 2) if eps else None,
              "checkpoint_seconds": [round(c.seconds, 1) for c in res.checkpoints],
              "night_seconds": [c.night_seconds for c in res.checkpoints if c.night_seconds is not None],
              "resumed_from": res.resumed_from, "criterion_episode": res.criterion_episode, "ledger": res.ledger.totals()}
    print("timing " + json.dumps(timing), flush=True)
    Path(P["run"]).with_suffix(".timing.json").write_text(json.dumps(timing, indent=1))


if __name__ == "__main__":
    main()
