"""Modal entrypoint for the seeded grid (CONTEXT.md §4: all recorded numbers
come from Transformers+PEFT in the cloud). NOT YET EXECUTED.

  modal secret create huggingface HF_TOKEN=hf_...                       # once per workspace (B4)
  modal run modal_app.py::main --arm baseline --seed 0 --n-episodes 4 --k 2 --n-probe 6 --n-heldout-eval 4   # path check
  modal run modal_app.py::main --arm sleep --seed 0                                                       # full run
  modal run modal_app.py::grid --seeds 0,1,2,3,4                                                          # all arms, all seeds

Two local entrypoints exist, so always name one (`::main` or `::grid`; B2).
Results are saved after every checkpoint and the volume committed (B1).

Order inside `grid`: Sleep first per seed (its ledger sets Awake's token
budget), then Sleep-NoDream, Online and Baseline in parallel, then Awake.
Results land in the `doze-results` volume as results/runs/<arm>_seed<seed>.json
plus a ledger JSON per arm; `check_matched` runs at the end of each seed.
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

APP = "doze-llm"
MODEL_ID = "Qwen/Qwen3-4B"
GPU = "A100-80GB"
RESULTS = "/results"
HF_CACHE = "/hf-cache"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "transformers>=4.56", "peft>=0.15", "accelerate", "safetensors")   # B3: dtype= kwarg
    .env({"HF_HOME": HF_CACHE, "TOKENIZERS_PARALLELISM": "false"})
    .add_local_dir(".", remote_path="/root/doze",
                   ignore=[".venv", "__pycache__", ".pytest_cache", "results", "*.pyc", ".git"])
)
app = modal.App(APP, image=image)
results_vol = modal.Volume.from_name("doze-results", create_if_missing=True)
cache_vol = modal.Volume.from_name("doze-hf-cache", create_if_missing=True)


def _run(arm: str, seed: int, n_episodes: int, k: int, n_probe: int, n_heldout_eval: int,
         control_items: int, awake_budget: int, lr: float, weight_decay: float, online_filter: bool) -> dict:
    import sys
    sys.path.insert(0, "/root/doze")
    import os
    os.chdir("/root/doze")
    from arms import harness as hz
    from arms.awake import AwakeArm
    from arms.baseline import BaselineArm
    from arms.online import OnlineArm
    from arms.sleep import SleepArm, SleepConfig
    from backends.hf_backend import HFBackend
    from tasks import number_reduction as nr

    trainable = arm in ("online", "online_unfiltered", "sleep", "sleep_nodream")
    if arm == "awake" and awake_budget <= 0:
        raise SystemExit("awake needs --awake-budget > 0 (derive it from the Sleep ledger)")   # B4: before model load
    backend = HFBackend(MODEL_ID, system=nr.SYSTEM_PROMPT, lora=trainable, lr=lr, seed=seed)
    if arm == "baseline":
        a = BaselineArm()
    elif arm == "awake":
        a = AwakeArm(token_budget_per_episode=awake_budget)
    elif arm in ("online", "online_unfiltered"):
        a = OnlineArm(filtered=(arm == "online"))
    else:
        a = SleepArm(SleepConfig(weight_decay=weight_decay), dream=(arm == "sleep"))
    cfg = hz.RunConfig(seed=seed, n_episodes=n_episodes, k=k, n_probe=n_probe,
                       n_heldout_eval=n_heldout_eval, control_items=control_items)
    out = Path(RESULTS) / "runs" / f"{arm}_seed{seed}.json"
    res = hz.run_arm(a, backend, cfg, log=print, save_path=out, on_checkpoint=results_vol.commit)
    res.ledger.save(Path(RESULTS) / "ledgers" / f"{arm}_seed{seed}.json")
    results_vol.commit()
    return {"arm": arm, "seed": seed, "criterion_episode": res.criterion_episode,
            "ledger": res.ledger.totals(), "checkpoints": [c.flat() for c in res.checkpoints],
            "nights": res.nights, "out": str(out)}


@app.function(gpu=GPU, timeout=24 * 3600, volumes={RESULTS: results_vol, HF_CACHE: cache_vol},
              secrets=[modal.Secret.from_name("huggingface")])
def run_arm_remote(arm: str, seed: int = 0, n_episodes: int = 600, k: int = 50, n_probe: int = 60,
                   n_heldout_eval: int = 201, control_items: int = 300, awake_budget: int = 0,
                   lr: float = 1e-4, weight_decay: float = 0.05, online_filter: bool = False) -> dict:
    return _run(arm, seed, n_episodes, k, n_probe, n_heldout_eval, control_items, awake_budget, lr,
                weight_decay, online_filter)


@app.local_entrypoint()
def main(arm: str = "baseline", seed: int = 0, n_episodes: int = 600, k: int = 50, n_probe: int = 60,
         n_heldout_eval: int = 201, control_items: int = 300, awake_budget: int = 0, lr: float = 1e-4,
         weight_decay: float = 0.05, online_filter: bool = False):
    r = run_arm_remote.remote(arm, seed, n_episodes, k, n_probe, n_heldout_eval, control_items,
                              awake_budget, lr, weight_decay, online_filter)
    print(json.dumps(r, indent=1, default=str))


@app.local_entrypoint()
def grid(seeds: str = "0,1,2,3,4", n_episodes: int = 600, online_filter: bool = False):
    import sys
    sys.path.insert(0, ".")
    from arms.awake import AwakeArm
    from eval import compute_ledger as cl

    for seed in [int(s) for s in seeds.split(",")]:
        sleep = run_arm_remote.remote("sleep", seed, n_episodes, online_filter=online_filter)
        others = list(run_arm_remote.starmap([("sleep_nodream", seed, n_episodes), ("online", seed, n_episodes),
                                              ("baseline", seed, n_episodes)]))
        ref = cl.Ledger(arm="sleep", **{k: v for k, v in sleep["ledger"].items()})
        budget = AwakeArm.budget_from_reference(ref, n_episodes)
        awake = run_arm_remote.remote("awake", seed, n_episodes, awake_budget=budget)
        ledgers = {}
        for r in [sleep, awake] + others:
            L = cl.Ledger(arm=r["arm"], backend=f"hf:{MODEL_ID}")
            L.add(**r["ledger"])
            ledgers[r["arm"]] = L
        try:
            cl.check_matched(ledgers, tol=0.05)
            print(f"seed {seed}: budgets matched")
        except cl.BudgetMismatch as e:
            print(f"seed {seed}: BUDGET MISMATCH — do not compare\n{e}")
        for r in [sleep, awake] + others:
            print(json.dumps({k: r[k] for k in ("arm", "seed", "criterion_episode", "ledger")}, default=str))
