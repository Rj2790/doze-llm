"""Small local (or single-GPU) run of the transfer puzzle on a few models.

    DOZE_DEVICE=mps python -m eval.transfer_local --base ~/doze-archive/models/base/Qwen3-4B \
        --adapters-root ~/doze-archive/models/adapters --models base,online_seed4,sleep_seed4,sleep_nodream_seed4 \
        --n 10 --seed 0 --out records/transfer/variant2_local_n10.json

One base model in memory; adapters are swapped on top (eval/followup._attach_adapter).
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from eval.followup import _attach_adapter, _device
from tasks import fold_variant2 as fv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3-4B"); ap.add_argument("--adapters-root", required=True)
    ap.add_argument("--models", default="base"); ap.add_argument("--n", type=int, default=10); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True); ap.add_argument("--batch", type=int, default=2); ap.add_argument("--max-tokens", type=int, default=400)
    a = ap.parse_args()
    from backends.hf_backend import HFBackend
    items = fv.make_items(a.n, seed=a.seed)
    prompts = [fv.format_prompt(x) for x in items]
    be = HFBackend(a.base, system=fv.SYSTEM_PROMPT, lora=False, deterministic=False, device=_device())
    be.batch_size = a.batch
    out = {"task": fv.NAME, "n": a.n, "seed": a.seed, "device": _device(), "models": {}}
    root = Path(a.adapters_root)
    for name in a.models.split(","):
        adir = None if name == "base" else str(root / name)
        _attach_adapter(be, adir, name if adir else None)
        t0 = time.time()
        res = be.generate(prompts, max_tokens=a.max_tokens)
        rows = []
        for x, r in zip(items, res):
            sc = fv.score(x, r.text)
            rows.append({"digits": x.digits, "answer": x.answer, **sc, "tokens": r.completion_tokens, "text": r.text})
        acc = sum(r["correct"] for r in rows) / len(rows)
        kinds = [r["first_error_kind"] for r in rows if r["first_error_kind"]]
        summ = {"accuracy": acc, "steps_correct_mean": statistics.fmean(r["steps_correct"] for r in rows),
                "leading_correct_mean": statistics.fmean(r["leading_correct"] for r in rows),
                "first_error_rule": kinds.count("rule"), "first_error_tracking": kinds.count("tracking"), "first_error_missing": kinds.count("missing"),
                "first_error_old_rule": sum(r["first_error_old_rule"] for r in rows), "all_steps_right": sum(r["first_error_kind"] is None for r in rows),
                "tokens_mean": statistics.fmean(r["tokens"] for r in rows), "seconds": time.time() - t0}
        out["models"][name] = {"summary": summ, "rows": rows}
        print(f"{name:22s} acc={acc:.3f} leading_ok={summ['leading_correct_mean']:.1f}/11 first_err rule={summ['first_error_rule']} "
              f"tracking={summ['first_error_tracking']} missing={summ['first_error_missing']} old_rule={summ['first_error_old_rule']} "
              f"perfect={summ['all_steps_right']} tok={summ['tokens_mean']:.0f} ({summ['seconds']:.0f}s)", flush=True)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, indent=1))
    print("TRANSFER_LOCAL_DONE")


if __name__ == "__main__":
    main()
