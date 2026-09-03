"""Modal entrypoint for the seeded grid (CONTEXT.md §4: all recorded numbers
come from Transformers+PEFT in the cloud). NOT YET EXECUTED.

  modal secret create huggingface HF_TOKEN=hf_...                       # once per workspace (B4)
  modal run modal_app.py::main --arm baseline --seed 0 --n-episodes 4 --k 2 --n-probe 6 --n-heldout-eval 4   # path check
  modal run modal_app.py::main --arm sleep --seed 0                                                       # full run
  modal run --detach modal_app.py::grid --seeds 0,1,2,3,4                                                 # all arms, all seeds
  modal run modal_app.py::main --gpu A100-80GB --arm baseline ...                                          # GPU override (default L4)

Always name an entrypoint (`::main`, `::grid`, `::preflight`, `::pathcheck`; B2).
Results are saved after every checkpoint and the volume committed (B1).

**Always launch the grid with `--detach`.** The grid orchestration runs in a
CPU-only Modal function (`grid_remote`), so a laptop disconnect cannot stop
the run (2026-09-02: an ephemeral app was torn down by a client disconnect
at Sleep episode 150, ~2 GPU-hours lost). Progress: `modal app logs <app>`
or the per-checkpoint run files in the `doze-results` volume; the final
summary is written to /results/grid_seed<seed>.json.

Order inside `grid`: Sleep first per seed (its ledger sets Awake's token
budget), then Sleep-NoDream, Online and Baseline in parallel, then Awake.
Results land in the `doze-results` volume as results/runs/<arm>_seed<seed>.json
plus a ledger JSON per arm; `check_matched` runs at the end of each seed.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import modal

APP = "doze-llm"
MODEL_ID = "Qwen/Qwen3-4B"
# Default GPU. Qwen3-4B bf16 (~8 GB) + LoRA r=16 at batch 1 and batched
# generation fit on an L4 (24 GB). Override per run with --gpu (entrypoint
# parameter, applied via Function.with_options) or DOZE_GPU=... in the
# environment. A100-80GB was the original unexamined default.
GPU = os.environ.get("DOZE_GPU", "L4")
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
         control_items: int, awake_budget: int, lr: float, weight_decay: float, online_filter: bool,
         run_tag: str = "") -> dict:
    import subprocess
    import sys
    import time
    sys.path.insert(0, "/root/doze")
    os.chdir("/root/doze")
    from eval import orchestrate
    t_start = time.time()
    try:
        gpu_name = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                                  capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        gpu_name = "unknown"
    from arms import harness as hz
    from arms.awake import AwakeArm
    from arms.baseline import BaselineArm
    from arms.online import OnlineArm
    from arms.sleep import SleepArm, SleepConfig
    from backends.hf_backend import HFBackend
    from tasks import number_reduction as nr

    trainable = arm in ("online", "online_unfiltered", "sleep", "sleep_nodream")
    print(f"gpu={gpu_name} arm={arm} seed={seed} episodes={n_episodes} k={k} tag={run_tag}", flush=True)
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
    tag = run_tag or "untagged"
    P = orchestrate.paths(RESULTS, tag, arm, seed)
    out = Path(P["run"])
    # resume=True: after a preemption Modal re-runs this input; we continue from the last checkpoint
    res = hz.run_arm(a, backend, cfg, log=print, save_path=out, on_checkpoint=results_vol.commit, resume=True)
    res.ledger.save(P["ledger"])
    results_vol.commit()
    eps = [e.seconds for e in res.episodes]
    timing = {"gpu": gpu_name, "total_seconds": round(time.time() - t_start, 1),
              "episode_seconds_mean": round(sum(eps) / len(eps), 2) if eps else None,
              "checkpoint_seconds": [round(c.seconds, 1) for c in res.checkpoints],
              "night_seconds": [c.night_seconds for c in res.checkpoints if c.night_seconds is not None],
              "resumed_from": res.resumed_from}
    print("timing " + json.dumps(timing), flush=True)
    return {"arm": arm, "seed": seed, "criterion_episode": res.criterion_episode,
            "ledger": res.ledger.totals(), "checkpoints": [c.flat() for c in res.checkpoints],
            "nights": res.nights, "out": str(out), "timing": timing, "tag": tag}


@app.function(gpu=GPU, timeout=24 * 3600, volumes={RESULTS: results_vol, HF_CACHE: cache_vol},
              secrets=[modal.Secret.from_name("huggingface")])
def run_arm_remote(arm: str, seed: int = 0, n_episodes: int = 600, k: int = 50, n_probe: int = 60,
                   n_heldout_eval: int = 201, control_items: int = 300, awake_budget: int = 0,
                   lr: float = 1e-4, weight_decay: float = 0.05, online_filter: bool = False,
                   run_tag: str = "") -> dict:
    return _run(arm, seed, n_episodes, k, n_probe, n_heldout_eval, control_items, awake_budget, lr,
                weight_decay, online_filter, run_tag)


@app.function(gpu=GPU, timeout=2 * 3600, volumes={RESULTS: results_vol, HF_CACHE: cache_vol},
              secrets=[modal.Secret.from_name("huggingface")])
def preflight_remote(seed: int = 0, n_items: int = 40, n_dream_seeds: int = 10) -> dict:
    """Items 1-4 of the validation list, the 40-item Baseline agreement set,
    and dream yield/length statistics (A4). One LoRA-enabled model load."""
    import os
    import sys
    import time
    sys.path.insert(0, "/root/doze"); os.chdir("/root/doze")
    from backends.hf_backend import HFBackend
    from eval import preflight as pf
    from tasks import number_reduction as nr

    t0 = time.time()
    backend = HFBackend(MODEL_ID, system=nr.SYSTEM_PROMPT, lora=True, seed=seed)
    load_s = time.time() - t0
    out = {"model_load_seconds": round(load_s, 1)}
    out["validation"] = pf.validate_backend(backend)          # ends with reset_adapter -> lora_B = 0
    t1 = time.time()
    items = pf.agreement_items(seed=seed, n=n_items)
    rows = pf.run_agreement(backend, items)
    out["agreement_rows"] = rows
    out["agreement_summary"] = pf.summarize_agreement(rows)
    out["agreement_seconds"] = round(time.time() - t1, 1)
    split = nr.make_split(seed=seed)
    # dream from kept training-set trajectories generated the same way (night conditions)
    tr_rows = pf.run_agreement(backend, split.train[:n_items])
    out["train_rows_summary"] = pf.summarize_agreement(tr_rows)
    out["dreams"] = pf.dream_stats(backend, tr_rows, split, n_seed=n_dream_seeds)
    Path(RESULTS).mkdir(exist_ok=True)
    (Path(RESULTS) / f"preflight_seed{seed}.json").write_text(json.dumps(out, indent=1, default=str))
    results_vol.commit()
    return out


@app.function(volumes={RESULTS: results_vol})
def ls_results() -> list[str]:
    import os
    out = []
    for root, _, files in os.walk(RESULTS):
        for f in files:
            p = os.path.join(root, f)
            out.append(f"{p} {os.path.getsize(p)}")
    return sorted(out)


def _gpu(fn, gpu: str):
    """Apply a per-invocation GPU override (defaults to GPU)."""
    return fn.with_options(gpu=gpu) if gpu and gpu != GPU else fn


@app.local_entrypoint()
def preflight(seed: int = 0, n_items: int = 40, n_dream_seeds: int = 10, gpu: str = GPU):
    print(f"gpu={gpu}")
    r = _gpu(preflight_remote, gpu).remote(seed, n_items, n_dream_seeds)
    Path("results").mkdir(exist_ok=True)
    Path(f"results/preflight_hf_seed{seed}.json").write_text(json.dumps(r, indent=1, default=str))
    v = r["validation"]
    print(json.dumps({k: v[k] for k in v if k.endswith("_pass") or k in ("1_describe", "2_completion_tokens", "2_count_tokens_text", "4_losses", "4_training_tokens", "4_expected_tokens", "4_norm_before_after_decay_reset", "4_decay_ratio")}, indent=1, default=str))
    print(json.dumps({"model_load_seconds": r["model_load_seconds"], "agreement": r["agreement_summary"],
                      "agreement_seconds": r["agreement_seconds"], "train_rows": r["train_rows_summary"],
                      "dreams": r["dreams"]}, indent=1, default=str))


@app.local_entrypoint()
def pathcheck(n_episodes: int = 2, gpu: str = GPU):
    """Item 5: starmap three tiny Baseline runs, then list what the volume holds."""
    print(f"gpu={gpu}")
    rs = list(_gpu(run_arm_remote, gpu).starmap([("baseline", s, n_episodes, 2, 6, 3, 3, 0, 1e-4, 0.05, False, "pathcheck") for s in (0, 1, 2)]))
    print(json.dumps([{k: r[k] for k in ("arm", "seed", "criterion_episode", "ledger", "out")} for r in rs], indent=1, default=str))
    print("\n".join(ls_results.remote()))


@app.local_entrypoint()
def main(arm: str = "baseline", seed: int = 0, n_episodes: int = 600, k: int = 50, n_probe: int = 60,
         n_heldout_eval: int = 201, control_items: int = 300, awake_budget: int = 0, lr: float = 1e-4,
         weight_decay: float = 0.05, online_filter: bool = False, gpu: str = GPU, run_tag: str = ""):
    import sys
    sys.path.insert(0, ".")
    from eval import orchestrate
    tag = run_tag or orchestrate.new_tag()
    print(f"gpu={gpu} run_tag={tag}")
    r = _gpu(run_arm_remote, gpu).remote(arm, seed, n_episodes, k, n_probe, n_heldout_eval, control_items,
                                         awake_budget, lr, weight_decay, online_filter, tag)
    print(json.dumps(r, indent=1, default=str))


@app.function(timeout=24 * 3600, volumes={RESULTS: results_vol})
def grid_remote(seeds: str, n_episodes: int, online_filter: bool, gpu: str, arms: str, run_tag: str) -> list[dict]:
    """Runs on Modal (CPU only) so the orchestration survives local disconnects,
    and is idempotent under preemption: spawned FunctionCall ids are written
    to the volume, so a restarted orchestrator re-attaches instead of
    re-spawning; complete run files are loaded, fresh partial files are
    waited on, stale ones are re-spawned with resume (eval/orchestrate.py).
    All arms of a seed run on ONE GPU type. Awake waits for Sleep's ledger."""
    import os
    import sys
    import time
    sys.path.insert(0, "/root/doze"); os.chdir("/root/doze")
    from arms.awake import AwakeArm
    from eval import compute_ledger as cl
    from eval import orchestrate

    arm_list = [a.strip() for a in arms.split(",") if a.strip()]
    assert "sleep" in arm_list, "sleep is the matching reference and must be in --arms"
    print(f"gpu={gpu} arms={arm_list} tag={run_tag}", flush=True)
    fn = _gpu(run_arm_remote, gpu)

    def run_state(path):
        p = Path(path)
        if not p.exists():
            return None
        try:
            return {"partial": json.loads(p.read_text())["partial"], "mtime": p.stat().st_mtime}
        except Exception:
            return {"partial": True, "mtime": p.stat().st_mtime}

    def result_from_file(path):
        d = json.loads(Path(path).read_text())
        return {"arm": d["arm"], "seed": d["config"]["seed"], "criterion_episode": d["criterion_episode"],
                "ledger": {k: d["ledger"][k] for k in ("tokens_generated", "gradient_steps", "training_tokens", "gpu_seconds")},
                "checkpoints": d["checkpoints"], "nights": d["nights"], "out": str(path),
                "timing": {"note": "loaded from file"}, "tag": run_tag}

    def obtain(arm, seed, calls, spawn_args):
        P = orchestrate.paths(RESULTS, run_tag, arm, seed)
        while True:
            results_vol.reload()
            action = orchestrate.plan(run_state(P["run"]), calls.get(arm), time.time())
            print(f"seed {seed} {arm}: {action}", flush=True)
            if action == "load":
                return result_from_file(P["run"])
            if action == "attach":
                try:
                    return modal.FunctionCall.from_id(calls[arm]).get()
                except Exception as e:      # call gone (e.g. expired); fall through to file logic
                    print(f"seed {seed} {arm}: attach failed ({type(e).__name__}); re-planning", flush=True)
                    calls.pop(arm, None)
                    continue
            if action == "wait":
                time.sleep(120)
                continue
            call = fn.spawn(*spawn_args)
            calls[arm] = call.object_id
            Path(P["calls"]).parent.mkdir(parents=True, exist_ok=True)
            Path(P["calls"]).write_text(json.dumps(calls))
            results_vol.commit()
            return call.get()

    summaries = []
    for seed in [int(x) for x in seeds.split(",")]:
        P0 = orchestrate.paths(RESULTS, run_tag, "sleep", seed)
        results_vol.reload()
        calls = json.loads(Path(P0["calls"]).read_text()) if Path(P0["calls"]).exists() else {}
        # spawn all non-awake arms first (idempotently), then collect
        pending = {}
        for a in arm_list:
            if a == "awake":
                continue
            P = orchestrate.paths(RESULTS, run_tag, a, seed)
            action = orchestrate.plan(run_state(P["run"]), calls.get(a), time.time())
            if action == "spawn":
                c = fn.spawn(a, seed, n_episodes, 50, 60, 201, 300, 0, 1e-4, 0.05, online_filter, run_tag)
                calls[a] = c.object_id
                pending[a] = c
                print(f"seed {seed} {a}: spawned {c.object_id}", flush=True)
        Path(P0["calls"]).parent.mkdir(parents=True, exist_ok=True)
        Path(P0["calls"]).write_text(json.dumps(calls)); results_vol.commit()
        results = {}
        for a in arm_list:
            if a == "awake":
                continue
            results[a] = pending[a].get() if a in pending else obtain(a, seed, calls, None)
        if "awake" in arm_list:
            ref = cl.Ledger(arm="sleep", **{k: v for k, v in results["sleep"]["ledger"].items()})
            budget = AwakeArm.budget_from_reference(ref, n_episodes)
            print(f"seed {seed}: awake budget {budget} tokens/episode from sleep total {ref.tokens_generated}", flush=True)
            results["awake"] = obtain("awake", seed, calls,
                                      ("awake", seed, n_episodes, 50, 60, 201, 300, budget, 1e-4, 0.05, online_filter, run_tag))
        ledgers = {}
        for a, r in results.items():
            L = cl.Ledger(arm=a, backend=f"hf:{MODEL_ID}")
            L.add(**{k: v for k, v in r["ledger"].items() if k != "gpu_seconds"}, gpu_seconds=r["ledger"].get("gpu_seconds", 0.0))
            ledgers[a] = L
        verdict = {"seed": seed, "ok": True, "message": "compute budgets matched", "arms": sorted(ledgers)}
        try:
            cl.check_matched(ledgers, tol=0.05)
            print(f"seed {seed}: budgets matched", flush=True)
        except cl.BudgetMismatch as e:
            verdict.update(ok=False, message=str(e))
            print(f"seed {seed}: BUDGET MISMATCH — do not compare\n{e}", flush=True)
        summary = {"seed": seed, "gpu": gpu, "tag": run_tag, "verdict": verdict, "arms": {}}
        for a, r in results.items():
            summary["arms"][a] = {k: r[k] for k in ("criterion_episode", "ledger", "timing", "nights")}
            summary["arms"][a]["probe_curve"] = [(c["episode"], c["probe_accuracy"]) for c in r["checkpoints"]]
            summary["arms"][a]["checkpoints"] = r["checkpoints"]
            print("RESULT " + json.dumps({k: r[k] for k in ("arm", "seed", "criterion_episode", "ledger", "timing")}, default=str), flush=True)
        Path(P0["summary"]).write_text(json.dumps(summary, indent=1, default=str))
        results_vol.commit()
        summaries.append(summary)
    return summaries


@app.local_entrypoint()
def grid(seeds: str = "0,1,2,3,4", n_episodes: int = 600, online_filter: bool = False, gpu: str = GPU,
         arms: str = "sleep,sleep_nodream,online,baseline,awake", run_tag: str = ""):
    """Launch with `modal run --detach`. Spawns grid_remote and returns its
    call id; the run then continues on Modal regardless of this client.
    Pass --run-tag <existing tag> to resume/re-attach a previous launch."""
    import sys
    sys.path.insert(0, ".")
    from eval import orchestrate
    tag = run_tag or orchestrate.new_tag()
    call = grid_remote.spawn(seeds, n_episodes, online_filter, gpu, arms, tag)
    print(f"grid_remote spawned: call_id={call.object_id} seeds={seeds} arms={arms} gpu={gpu} run_tag={tag}", flush=True)
    print(f"Files: doze-results volume under /{tag}/ (runs/, ledgers/, grid_seed<seed>.json). Resume: --run-tag {tag}", flush=True)
