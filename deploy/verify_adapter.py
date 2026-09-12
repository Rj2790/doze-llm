"""Load an exported PEFT adapter on top of the local base model and solve a few
held-out puzzles, printing accuracy with and without the adapter. A smoke test
that the export in deploy/export_adapter.py is loadable and behaves like the
trained arm. Needs torch + transformers + peft (see ~/doze-archive/.venv-torch).

    python deploy/verify_adapter.py --base ~/doze-archive/models/base/Qwen3-4B \
        --adapter ~/doze-archive/models/adapters/sleep_seed3 --seed 3 --n 8
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tasks import number_reduction as nr  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True); ap.add_argument("--adapter", required=True)
    ap.add_argument("--seed", type=int, default=3); ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=160)
    a = ap.parse_args()
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.bfloat16 if dev == "mps" else torch.float32
    tok = AutoTokenizer.from_pretrained(a.base)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(a.base, dtype=dtype, low_cpu_mem_usage=True).to(dev).eval()
    print(f"base loaded on {dev} in {time.time() - t0:.0f}s")
    split = nr.make_split(seed=a.seed, length=12)
    items = split.heldout[: a.n]

    def solve_all(m, label):
        correct = 0
        for inst in items:
            prompt = nr.format_prompt(inst, "work")
            msgs = [{"role": "system", "content": nr.SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            ids = tok(text, return_tensors="pt").to(dev)
            with torch.no_grad():
                out = m.generate(**ids, max_new_tokens=a.max_new, do_sample=False)
            comp = tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)
            ans = nr.parse_answer(comp)
            ok = ans == inst.answer
            correct += ok
            print(f"  [{label}] {inst.digits} truth={inst.answer} model={ans} {'OK' if ok else 'x'}")
        print(f"{label}: {correct}/{len(items)}")
        return correct

    base_c = solve_all(model, "base")
    model = PeftModel.from_pretrained(model, a.adapter).eval()
    ad_c = solve_all(model, "adapter")
    print(f"RESULT base {base_c}/{len(items)} adapter {ad_c}/{len(items)}")


if __name__ == "__main__":
    main()
