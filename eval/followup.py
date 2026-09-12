"""Post-hoc follow-up checks on the finished 4B grid (records/followup_prereg.md).

Not part of the preregistered design. Three questions, one job:

  A. GSM8K re-score with saved outputs, for the base model and every saved
     adapter, under (1) the exact prompt regime the grid used and (2) an
     unconstrained regime (a math system prompt, 1024 tokens). Decides whether
     the recorded GSM8K movements are reasoning or response-length effects.
  B. Probe instrument checks: the grid's masked probe as run; the same probe
     with the regularity stated; with working allowed; a teacher-forced
     log-probability version; and a positive control adapter trained directly
     on masked-prompt -> shortcut-answer pairs from TRAINING prefixes.
  C. Unstructured full-chain accuracy (items with no mirror structure) next to
     the structured held-out set, per model.

Model-free helpers are tested; the GPU driver is `run_all`.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from collections.abc import Sequence
from pathlib import Path

from eval import control_bench as cb
from eval import shortcut_detector as sd
from tasks import number_reduction as nr

# ---- prompt regimes (A) -------------------------------------------------------

GSM8K_REGIMES = {
    # exactly what arms/harness.py did: task system prompt + cb.format_prompt, 512 tokens
    "as_run": {"system": nr.SYSTEM_PROMPT, "max_tokens": 512},
    # a plain math regime: the control module's own (unused by the grid) system prompt, longer budget
    "math_1024": {"system": cb.SYSTEM_PROMPT, "max_tokens": 1024},
}

# ---- probe variants (B) ---------------------------------------------------------

RULE_HINT = ("Useful fact about these strings: the final result always equals the result after the "
             "first 6 digits, i.e. the answer is the 6th intermediate result.")
WORK_ALLOWED = ("You may first write the comparisons for the visible digits, one per line as "
                "'previous,digit->result', then give your guess on a line starting with 'ANSWER:'.")


def probe_prompt(inst: nr.Instance, variant: str) -> str:
    base = nr.format_probe(inst)
    if variant == "as_run":
        return base
    if variant == "rule_stated":
        return base.replace(nr.PROBE_FORMAT, RULE_HINT + "\n" + nr.PROBE_FORMAT)
    if variant == "rule_stated_work":
        return base.replace(nr.PROBE_FORMAT, RULE_HINT + "\n" + WORK_ALLOWED)
    if variant == "work_allowed":
        return base.replace(nr.PROBE_FORMAT, WORK_ALLOWED)
    raise ValueError(variant)


PROBE_VARIANTS = {"as_run": 64, "rule_stated": 64, "work_allowed": 128, "rule_stated_work": 128}


def positive_control_examples(split: nr.Split, n: int, seed: int) -> list[tuple[str, str]]:
    """(masked prompt, 'ANSWER: r6') pairs from TRAINING-prefix items only,
    answer-stratified. Never touches held-out or probe prefixes."""
    import random
    rng = random.Random(f"poscontrol-{seed}")
    items = sd.stratified_probe_items(split.train, n=n, seed=seed + 101)
    assert all(x.prefix in split.train_prefixes for x in items)
    rng.shuffle(items)
    return [(nr.format_probe(x), f"ANSWER: {nr.probe_target(x)}") for x in items]


def logprob_choice(scores: dict[str, float]) -> str:
    """Argmax over candidate answers by total log-probability of 'ANSWER: d'."""
    return max(scores, key=scores.get)


def summarize_probe(items: Sequence[nr.Instance], texts: Sequence[str]) -> dict:
    r = sd.score_probe(items, texts)
    return {"n": r.n, "correct": r.correct, "accuracy": r.accuracy, "p_value": r.p_value}


# ---- GSM8K bookkeeping (A) --------------------------------------------------------

def gsm8k_rows(items: Sequence[cb.Item], results, max_tokens: int) -> list[dict]:
    rows = []
    for x, r in zip(items, results):
        pred = cb.parse_answer(r.text)
        rows.append({"idx": x.idx, "gold": x.gold, "pred": pred, "correct": cb.exact_match(pred, x.gold),
                     "tokens": r.completion_tokens, "truncated": r.completion_tokens >= max_tokens,
                     "has_tag": bool(cb._TAGGED_RE.search(r.text)), "text": r.text})
    return rows


def gsm8k_summary(rows: Sequence[dict]) -> dict:
    n = len(rows)
    toks = [r["tokens"] for r in rows]
    return {"n": n, "accuracy": sum(r["correct"] for r in rows) / n if n else None,
            "unparsable": sum(r["pred"] is None for r in rows) / n if n else None,
            "truncated": sum(r["truncated"] for r in rows) / n if n else None,
            "has_tag": sum(r["has_tag"] for r in rows) / n if n else None,
            "tokens_mean": statistics.fmean(toks) if toks else None,
            "tokens_median": statistics.median(toks) if toks else None}


def length_accuracy_slope(summaries: Sequence[dict]) -> dict:
    """Across models within one regime: Pearson r and slope of accuracy on mean tokens."""
    xs = [s["tokens_mean"] for s in summaries]; ys = [s["accuracy"] for s in summaries]
    if len(xs) < 3:
        return {"r": None, "slope_per_10_tokens": None, "n": len(xs)}
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    sxx = sum((x - mx) ** 2 for x in xs); sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    syy = sum((y - my) ** 2 for y in ys)
    r = sxy / math.sqrt(sxx * syy) if sxx > 0 and syy > 0 else None
    return {"r": r, "slope_per_10_tokens": 10 * sxy / sxx if sxx > 0 else None, "n": len(xs)}


# ---- GPU driver ---------------------------------------------------------------------

def _load_backend(model_id: str, system: str, deterministic: bool):
    from backends.hf_backend import HFBackend
    return HFBackend(model_id, system=system, lora=False, deterministic=deterministic)


class _AdapterOff:
    """Delegate that runs the PEFT model with adapters disabled (the base model)."""
    def __init__(self, pm):
        self.pm = pm
    def eval(self):
        self.pm.eval(); return self
    def generate(self, *a, **k):
        with self.pm.disable_adapter():
            return self.pm.generate(*a, **k)
    def __call__(self, *a, **k):
        with self.pm.disable_adapter():
            return self.pm(*a, **k)
    @property
    def dtype(self):
        return self.pm.dtype


def _attach_adapter(be, adapter_dir: str | None, name: str | None = None):
    """Make `be.model` run with the given PEFT adapter, or with no adapter.
    The PEFT wrapper is created once; adapters are loaded by name on demand
    and the previous one is deleted to keep GPU memory flat."""
    import torch
    from peft import PeftModel
    if getattr(be, "_pm", None) is None:
        if adapter_dir is None:
            be.model = be.model                     # plain base until the first adapter arrives
            return
        be._pm = PeftModel.from_pretrained(be.model, adapter_dir, adapter_name=name or "a0")
        be._loaded = name or "a0"
    pm = be._pm
    if adapter_dir is None:
        be.model = _AdapterOff(pm)
    else:
        name = name or adapter_dir.rstrip("/").rsplit("/", 1)[-1]
        if be._loaded != name:
            pm.load_adapter(adapter_dir, adapter_name=name)
            pm.set_adapter(name)
            try:
                pm.delete_adapter(be._loaded)
            except Exception:
                pass
            be._loaded = name
        be.model = pm
    be.model.eval()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _score_candidates(be, prompt: str, candidates: Sequence[str]) -> dict[str, float]:
    """Teacher-forced total log-probability of each candidate completion."""
    import torch
    chat = be._chat(prompt)
    out = {}
    for cand in candidates:
        full = chat + cand
        enc_p = be.tokenizer(chat, return_tensors="pt").to(be.device)
        enc = be.tokenizer(full, return_tensors="pt").to(be.device)
        with torch.no_grad():
            logits = be.model(**enc).logits[0, :-1].float()
        lp = torch.log_softmax(logits, dim=-1)
        ids = enc["input_ids"][0, 1:]
        start = enc_p["input_ids"].shape[1] - 1
        out[cand] = float(lp[start:, :].gather(1, ids[start:, None]).sum().item())
    return out


def run_all(out_dir: str, adapters: dict[str, str], model_id: str, gpu_tag: str, seeds: Sequence[int],
            n_unstructured: int = 120, poscontrol_steps: Sequence[int] = (50, 200), deterministic: bool = True,
            only: Sequence[str] = ("A", "B", "C")) -> None:
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "gsm8k_outputs").mkdir(exist_ok=True)
    models = {"base": None, **adapters}
    control = cb.load_frozen()
    splits = {s: nr.make_split(seed=s, length=12) for s in seeds}
    log = (out / "followup.log").open("a")

    def note(msg):
        line = f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}"
        print(line, flush=True); log.write(line + "\n"); log.flush()

    be = _load_backend(model_id, nr.SYSTEM_PROMPT, deterministic)
    note(f"loaded {model_id} on {gpu_tag}; models={list(models)}")

    # ---- A: GSM8K under two regimes
    if "A" in only:
        summ_path = out / "gsm8k_summary.json"
        summ = json.loads(summ_path.read_text()) if summ_path.exists() else {}
        for regime, spec in GSM8K_REGIMES.items():
            for name, adir in models.items():
                key = f"{regime}/{name}"
                if key in summ:
                    continue
                _attach_adapter(be, adir, name); be.system = spec["system"]
                t0 = time.time()
                res = be.generate([cb.format_prompt(x) for x in control], max_tokens=spec["max_tokens"])
                rows = gsm8k_rows(control, res, spec["max_tokens"])
                (out / "gsm8k_outputs" / f"{regime}__{name}.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
                summ[key] = {**gsm8k_summary(rows), "regime": regime, "model": name, "seconds": time.time() - t0, "gpu": gpu_tag}
                summ_path.write_text(json.dumps(summ, indent=1))
                note(f"A {key}: acc={summ[key]['accuracy']:.3f} tok={summ[key]['tokens_mean']:.0f} trunc={summ[key]['truncated']:.2f} ({summ[key]['seconds']:.0f}s)")
        be.system = nr.SYSTEM_PROMPT

    # ---- B: probe instrument
    if "B" in only:
        pb_path = out / "probe_summary.json"
        pb = json.loads(pb_path.read_text()) if pb_path.exists() else {}
        for name, adir in models.items():
            seeds_for = seeds if name == "base" else [int(name.rsplit("seed", 1)[1])]
            for s in seeds_for:
                items = sd.stratified_probe_items(splits[s].probe, n=60, seed=s)
                for variant, mt in PROBE_VARIANTS.items():
                    key = f"{variant}/{name}/seed{s}"
                    if key in pb:
                        continue
                    _attach_adapter(be, adir, name)
                    res = be.generate([probe_prompt(x, variant) for x in items], max_tokens=mt)
                    pb[key] = {**summarize_probe(items, [r.text for r in res]), "variant": variant, "model": name, "seed": s,
                               "texts": [r.text for r in res][:10]}
                    pb_path.write_text(json.dumps(pb, indent=1))
                    note(f"B {key}: acc={pb[key]['accuracy']:.3f}")
                key = f"logprob/{name}/seed{s}"
                if key not in pb:
                    _attach_adapter(be, adir, name)
                    correct = 0; margins = []
                    for x in items:
                        sc = _score_candidates(be, probe_prompt(x, "as_run"), [f"ANSWER: {d}" for d in nr.DIGITS])
                        choice = logprob_choice(sc)[-1]
                        correct += choice == nr.probe_target(x)
                        tgt = sc[f"ANSWER: {nr.probe_target(x)}"]; others = max(v for k, v in sc.items() if not k.endswith(nr.probe_target(x)))
                        margins.append(tgt - others)
                    pb[key] = {"n": len(items), "correct": correct, "accuracy": correct / len(items), "mean_margin": statistics.fmean(margins),
                               "variant": "logprob", "model": name, "seed": s}
                    pb_path.write_text(json.dumps(pb, indent=1))
                    note(f"B {key}: acc={pb[key]['accuracy']:.3f} margin={pb[key]['mean_margin']:+.3f}")
    # ---- C: structured vs unstructured full-chain accuracy
    if "C" in only:
        c_path = out / "chain_summary.json"
        cs = json.loads(c_path.read_text()) if c_path.exists() else {}
        for name, adir in models.items():
            seeds_for = seeds if name == "base" else [int(name.rsplit("seed", 1)[1])]
            for s in seeds_for:
                key = f"{name}/seed{s}"
                if key in cs:
                    continue
                _attach_adapter(be, adir, name); be.system = nr.SYSTEM_PROMPT
                struct = splits[s].heldout
                unstruct = nr.unstructured_heldout(splits[s], n=n_unstructured, seed=s)
                r1 = be.generate([nr.format_prompt(x, "work") for x in struct], max_tokens=400)
                r2 = be.generate([nr.format_prompt(x, "work") for x in unstruct], max_tokens=400)
                a1 = sum(nr.score(x, r.text)["correct"] for x, r in zip(struct, r1)) / len(struct)
                a2 = sum(nr.score(x, r.text)["correct"] for x, r in zip(unstruct, r2)) / len(unstruct)
                cs[key] = {"model": name, "seed": s, "structured_heldout_acc": a1, "unstructured_acc": a2,
                           "n_structured": len(struct), "n_unstructured": len(unstruct)}
                c_path.write_text(json.dumps(cs, indent=1))
                note(f"C {key}: structured={a1:.3f} unstructured={a2:.3f}")
    # ---- B2: positive control adapters (needs a trainable backend; done last, base backend freed first)
    if "B" in only:
        pb_path = out / "probe_summary.json"
        pb = json.loads(pb_path.read_text()) if pb_path.exists() else {}
        todo = [(s, st) for s in seeds for st in poscontrol_steps if f"poscontrol{st}/base/seed{s}" not in pb]
        if todo:
            import gc
            import torch
            from backends.hf_backend import HFBackend
            del be
            gc.collect(); torch.cuda.empty_cache()
            tb = HFBackend(model_id, system=nr.SYSTEM_PROMPT, lora=True, seed=seeds[0], deterministic=deterministic)
            for s, steps in todo:
                key = f"poscontrol{steps}/base/seed{s}"
                tb.seed = s; tb.reset_adapter()
                ex = positive_control_examples(splits[s], n=min(steps, 240), seed=s)
                st = tb.train(ex, steps=steps, seed=s)
                items = sd.stratified_probe_items(splits[s].probe, n=60, seed=s)
                res = tb.generate([probe_prompt(x, "as_run") for x in items], max_tokens=64)
                pb[key] = {**summarize_probe(items, [r.text for r in res]), "variant": f"poscontrol{steps}", "model": "base", "seed": s,
                           "train_loss": st.loss, "train_examples": len(ex), "texts": [r.text for r in res][:10]}
                vis = sd.stratified_probe_items(splits[s].heldout, n=60, seed=s + 7)
                res2 = tb.generate([nr.format_prompt(x, "short") for x in vis], max_tokens=16)
                pb[key]["visible_short_accuracy"] = sum(nr.parse_answer(r.text) == x.answer for x, r in zip(vis, res2)) / len(vis)
                # and: does it still solve full chains? (sanity that training did not break the model)
                res3 = tb.generate([nr.format_prompt(x, "work") for x in splits[s].heldout[:60]], max_tokens=400)
                pb[key]["heldout_work_accuracy_60"] = sum(nr.score(x, r.text)["correct"] for x, r in zip(splits[s].heldout[:60], res3)) / 60
                pb_path.write_text(json.dumps(pb, indent=1))
                note(f"B {key}: probe acc={pb[key]['accuracy']:.3f} (loss {st.loss:.3f}); visible short={pb[key]['visible_short_accuracy']:.3f}; work60={pb[key]['heldout_work_accuracy_60']:.3f}")
    note("FOLLOWUP_DONE")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--adapters-root", required=True, help="dir containing <arm>_seedN/ PEFT folders")
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--only", default="A,B,C")
    ap.add_argument("--models", default="", help="comma list of adapter names to include (default all found)")
    ap.add_argument("--gpu-tag", default="")
    a = ap.parse_args()
    root = Path(a.adapters_root)
    adapters = {p.name: str(p) for p in sorted(root.iterdir()) if (p / "adapter_config.json").exists()}
    if a.models:
        keep = set(a.models.split(","))
        adapters = {k: v for k, v in adapters.items() if k in keep}
    seeds = [int(x) for x in a.seeds.split(",")]
    try:
        import torch
        gpu = a.gpu_tag or (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
    except Exception:
        gpu = a.gpu_tag or "?"
    run_all(a.out, adapters, a.model, gpu, seeds, only=tuple(a.only.split(",")))


if __name__ == "__main__":
    main()
