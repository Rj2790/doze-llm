"""Preemption-safe orchestration helpers (pure; used by modal_app.grid_remote).

Every launch gets a run tag; all files live under <results>/<tag>/. When an
orchestrator restarts (Modal preemption re-runs it "with the same input") it
must never re-spawn a run that is complete or still running:

  plan(run_state, call_id, now) ->
    "load"   complete run file exists: read it
    "attach" a FunctionCall id was recorded: re-attach and wait on it
    "wait"   partial file is fresh but no call id: another container is writing it
    "spawn"  nothing on disk, or the partial file is stale: (re)start with resume
"""

from __future__ import annotations

from datetime import datetime, timezone

STALE_AFTER_S = 3600.0   # a run writes its file every checkpoint (~8-13 min on L4)


def new_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%MZ")


def paths(results_root: str, tag: str, arm: str, seed: int) -> dict:
    base = f"{results_root}/{tag}"
    return {"base": base, "run": f"{base}/runs/{arm}_seed{seed}.json",
            "ledger": f"{base}/ledgers/{arm}_seed{seed}.json",
            "calls": f"{base}/grid_calls_seed{seed}.json", "summary": f"{base}/grid_seed{seed}.json"}


def plan(run_state: dict | None, call_id: str | None, now: float, stale_after: float = STALE_AFTER_S) -> str:
    if run_state is not None and not run_state.get("partial", True):
        return "load"
    if call_id:
        return "attach"
    if run_state is None:
        return "spawn"
    return "wait" if (now - run_state["mtime"]) < stale_after else "spawn"
