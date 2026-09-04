"""Run one arm for one seed and write results/runs/<arm>_seed<seed>.json.

  python run_arm.py --arm baseline --backend scripted --n-episodes 100
  python run_arm.py --arm baseline --backend mlx --model mlx-community/Qwen3-4B-4bit \
      --n-episodes 4 --k 2 --n-probe 6 --n-heldout-eval 4                      # smoke
  python run_arm.py --arm awake --awake-budget 300 --backend mlx ...

Prints the header line, a JSON summary (per-checkpoint metrics + ledger) and
`--show` raw day completions.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from arms import harness as hz
from arms.awake import AwakeArm
from arms.baseline import BaselineArm
from arms.online import OnlineArm
from arms.sleep import SleepArm, SleepConfig
from eval import compute_ledger as cl
from tasks import number_reduction as nr


TRAINABLE_ARMS = ("online", "online_unfiltered", "sleep", "sleep_nodream")


def build_backend(args):
    trainable = args.arm in TRAINABLE_ARMS
    if args.backend == "scripted":
        from backends.scripted import FakeTrainableBackend, ScriptedBackend
        if trainable:
            return FakeTrainableBackend(learn_after_steps=args.fake_learn_after or None, seed=args.seed)
        return ScriptedBackend(policy=args.policy, noise=args.noise, seed=args.seed)
    if args.backend == "mlx":
        from backends.mlx_backend import MLXBackend
        return MLXBackend(model_id=args.model, system=nr.SYSTEM_PROMPT, lora=trainable, lr=args.lr, seed=args.seed)
    raise ValueError(args.backend)


def build_arm(args):
    if args.arm == "baseline":
        return BaselineArm()
    if args.arm == "awake":
        if args.awake_reference:
            ref = cl.Ledger.load(args.awake_reference)
            budget = AwakeArm.budget_from_reference(ref, args.n_episodes)
        elif args.awake_budget:
            budget = args.awake_budget
        else:
            raise SystemExit("awake needs --awake-budget or --awake-reference <sleep ledger json>")
        return AwakeArm(token_budget_per_episode=budget)
    if args.arm in ("online", "online_unfiltered"):
        return OnlineArm(filtered=(args.arm == "online"))
    if args.arm in ("sleep", "sleep_nodream"):
        scfg = SleepConfig(weight_decay=args.weight_decay,
                           steps_per_night=args.steps_per_night or None,
                           dream_max_tokens=args.max_tokens)
        return SleepArm(scfg, dream=(args.arm == "sleep"))
    raise ValueError(args.arm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["baseline", "awake", "online", "online_unfiltered", "sleep", "sleep_nodream"],
                    required=True)
    ap.add_argument("--backend", choices=["scripted", "mlx"], default="scripted")
    ap.add_argument("--model", default="mlx-community/Qwen3-4B-bf16")
    ap.add_argument("--policy", default="literal", help="scripted only")
    ap.add_argument("--noise", type=float, default=0.0, help="scripted only")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-episodes", type=int, default=600)
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--mode", choices=["full", "work"], default="work")
    ap.add_argument("--number-digits", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--n-probe", type=int, default=60)
    ap.add_argument("--n-heldout-eval", type=int, default=nr.N_HELDOUT)
    ap.add_argument("--control-items", type=int, default=0, help="0 = skip control benchmark")
    ap.add_argument("--n-implicit", type=int, default=120, help="items per set for mirror_bias/short_gap (0 = skip)")
    ap.add_argument("--awake-budget", type=int, default=0, help="tokens per episode")
    ap.add_argument("--awake-reference", default="", help="sleep ledger json to derive the budget")
    ap.add_argument("--lr", type=float, default=1e-4, help="LoRA learning rate (tunable)")
    ap.add_argument("--weight-decay", type=float, default=0.05, help="sleep: nightly LoRA shrink (tunable)")
    ap.add_argument("--steps-per-night", type=int, default=0, help="sleep: 0 = K (matches Online)")
    ap.add_argument("--fake-learn-after", type=int, default=0, help="scripted trainable: steps until shortcut")
    ap.add_argument("--out", default="")
    ap.add_argument("--show", type=int, default=2)
    args = ap.parse_args()

    cfg = hz.RunConfig(seed=args.seed, n_episodes=args.n_episodes, k=args.k, mode=args.mode,
                       numbered=args.number_digits, max_tokens=args.max_tokens, n_probe=args.n_probe,
                       n_heldout_eval=args.n_heldout_eval, control_items=args.control_items,
                       n_implicit=args.n_implicit)
    arm = build_arm(args)          # B4: fail on a bad Awake budget before loading a model
    backend = build_backend(args)
    print(f"arm={arm.name} backend={backend.name} seed={cfg.seed} episodes={cfg.n_episodes} k={cfg.k} "
          f"mode={cfg.mode} numbered={cfg.numbered} n_probe={cfg.n_probe} n_heldout={cfg.n_heldout_eval} "
          f"control={cfg.control_items}" + (f" awake_budget={arm.budget}" if args.arm == "awake" else ""),
          flush=True)
    out = Path(args.out or f"results/runs/{arm.name}_seed{cfg.seed}.json")
    res = hz.run_arm(arm, backend, cfg, log=lambda m: print(m, flush=True), save_path=out)
    if getattr(arm, "skipped", 0) or getattr(arm, "substitutions", None):
        print(f"  online: skipped={arm.skipped} substitutions={dict(arm.substitutions)}", flush=True)
    if res.nights:
        for n in res.nights:
            print(f"  night ep={n['episode']}: kept={n['kept']} dreams={n['dreams_kept']}/{n['dreams_generated']} "
                  f"rejected={n['dreams_rejected']} replay={n['replay']} steps={n['steps']} "
                  f"train_tok={n['training_tokens']} loss={n['loss']} {n['seconds']}s", flush=True)

    if not args.out:
        res.ledger.save(f"results/ledgers/{arm.name}_seed{cfg.seed}.json")
    summary = {
        "arm": res.arm, "backend": backend.name, "seed": cfg.seed, "episodes": cfg.n_episodes,
        "criterion_episode": res.criterion_episode,
        "day_accuracy": round(sum(e.correct for e in res.episodes) / len(res.episodes), 3),
        "ledger": res.ledger.totals(), "eval_tokens": res.eval_tokens,
        "nights": [{k: n[k] for k in ("episode", "kept", "dreams_generated", "dreams_kept", "replay", "steps",
                                       "training_tokens", "loss")} for n in res.nights],
        "checkpoints": [{k: (round(v, 3) if isinstance(v, float) else v) for k, v in c.flat().items()
                         if not k.startswith("ledger_") or k in ("ledger_tokens_generated", "ledger_gradient_steps")}
                        for c in res.checkpoints],
        "out": str(out),
    }
    print(json.dumps(summary, indent=2))
    for e in res.episodes[: args.show]:
        x = nr.Instance.from_digits(e.digits)
        print(f"\n--- episode {e.episode} digits: {e.digits} gold: {nr.gold_response(x).replace(chr(10), ' | ')} "
              f"rounds={e.rounds} tokens={e.tokens}")
        print(e.text.strip()[:700])


if __name__ == "__main__":
    main()
