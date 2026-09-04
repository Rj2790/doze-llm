import subprocess, sys, json
from pathlib import Path

from deploy.vultr.queue import plan_queues


def test_queues_balance_and_keep_awake_after_sleep():
    q = plan_queues(["sleep", "sleep_nodream", "online", "baseline", "awake"], 2)
    assert sorted(sum(q, [])) == sorted(["sleep", "sleep_nodream", "online", "baseline", "awake"])
    sq = next(x for x in q if "sleep" in x)
    assert sq.index("awake") > sq.index("sleep")
    q4 = plan_queues(["sleep", "sleep_nodream", "online", "baseline", "awake"], 4)
    assert len(q4) == 4 and any("awake" in x and "sleep" in x for x in q4)


def test_run_job_dry_run_scripted(tmp_path):
    out = subprocess.run([sys.executable, "deploy/vultr/run_job.py", "--arm", "sleep", "--seed", "0", "--run-tag", "t",
                          "--results", str(tmp_path), "--backend", "scripted", "--n-episodes", "4", "--k", "2",
                          "--n-probe", "6", "--n-heldout-eval", "3", "--control-items", "0", "--n-implicit", "0"],
                         capture_output=True, text=True, cwd=Path(__file__).resolve().parents[1])
    assert out.returncode == 0, out.stderr[-2000:]
    assert (tmp_path / "t" / "runs" / "sleep_seed0.json").exists() and (tmp_path / "t" / "ledgers" / "sleep_seed0.json").exists()
    timing = json.loads((tmp_path / "t" / "runs" / "sleep_seed0.timing.json").read_text())
    assert timing["ledger"]["gradient_steps"] == 4
    # awake derives its budget from the sleep ledger
    out2 = subprocess.run([sys.executable, "deploy/vultr/run_job.py", "--arm", "awake", "--seed", "0", "--run-tag", "t",
                           "--results", str(tmp_path), "--backend", "scripted", "--awake-from-sleep", "--n-episodes", "4",
                           "--k", "2", "--n-probe", "6", "--n-heldout-eval", "3", "--control-items", "0", "--n-implicit", "0"],
                          capture_output=True, text=True, cwd=Path(__file__).resolve().parents[1])
    assert out2.returncode == 0 and "awake_budget=" in out2.stdout, out2.stderr[-2000:]
