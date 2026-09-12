"""Analyse a transfer run (eval/transfer_local.py output) beyond accuracy:
where each model first goes wrong and which rule it applied instead.

    python -m eval.transfer_analysis records/transfer/variant2_zero_shot_n201.json
"""
from __future__ import annotations

import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path

from tasks import fold_variant2 as fv


def classify_first_rule_error(digits: str, text: str) -> dict | None:
    """For the first written step whose operands are right but result wrong,
    say which alternative rule the written result agrees with:
      other_parity  - the other step's table (odd/even confusion)
      old_rule      - the original puzzle's table (pairs within {1,4,9})
      copy_operand  - one of the two operands
    Returns None if there is no such step."""
    inst = fv.Instance.from_digits(digits)
    lines = fv.parse_work_lines(text)
    for k, (a, b, c) in enumerate(lines[: inst.length - 1], start=1):
        exp_a = inst.digits[0] if k == 1 else inst.responses[k - 2]
        if (a, b) != (exp_a, inst.digits[k]):
            return None                                  # first divergence is a tracking error
        if c != inst.responses[k - 1]:
            other = fv.combine(a, b, k + 1)
            return {"step": k, "written": c, "expected": inst.responses[k - 1],
                    "other_parity": c == other and other != inst.responses[k - 1],
                    "old_rule": fv.old_rule_value(a, b) == c,
                    "copy_operand": c in (a, b)}
    return None


def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    z = 1.959964; p = k / n; den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (c - h, c + h)


def analyse(path: str | Path) -> dict:
    d = json.loads(Path(path).read_text())
    out = {"task": d["task"], "n": d["n"], "seed": d["seed"], "models": {}}
    for name, m in d["models"].items():
        rows = m["rows"]; n = len(rows)
        correct = sum(r["correct"] for r in rows)
        step1 = sum(1 for r in rows if r["leading_correct"] >= 1)
        step2 = sum(1 for r in rows if r["leading_correct"] >= 2)
        kinds = Counter(r["first_error_kind"] for r in rows)
        cls = [classify_first_rule_error(r["digits"], r["text"]) for r in rows]
        cls = [c for c in cls if c]
        n_rule = len(cls)
        out["models"][name] = {
            "accuracy": correct / n, "ci95": wilson(correct, n), "n": n,
            "step1_correct": step1 / n, "step2_correct": step2 / n,
            "leading_correct_mean": statistics.fmean(r["leading_correct"] for r in rows),
            "first_error": dict(kinds),
            "first_rule_error_n": n_rule,
            "other_parity_frac": (sum(c["other_parity"] for c in cls) / n_rule) if n_rule else None,
            "old_rule_frac": (sum(c["old_rule"] for c in cls) / n_rule) if n_rule else None,
            "copy_operand_frac": (sum(c["copy_operand"] for c in cls) / n_rule) if n_rule else None,
            "first_rule_error_step_dist": dict(Counter(c["step"] for c in cls)),
            "tokens_mean": statistics.fmean(r["tokens"] for r in rows),
        }
    return out


def markdown(res: dict) -> str:
    L = [f"# Transfer run: {res['task']} — n={res['n']} items per model, seed {res['seed']}\n",
         "Chance accuracy with four symbols is 0.25. `step1/step2 ok` = fraction of items whose first / first two written steps are right. "
         "For the first *rule* error (right operands, wrong result): `other parity` = the result matches the other step-type's table "
         "(odd/even confusion); `old rule` = matches the original puzzle's table; `copy` = equals one of the operands.\n",
         "| model | accuracy (95% CI) | step1 ok | step2 ok | leading ok / 11 | first error: rule / tracking / missing / none | other parity | old rule | copy | tokens |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for name, m in res["models"].items():
        fe = m["first_error"]
        L.append(f"| {name} | {m['accuracy']:.3f} ({m['ci95'][0]:.2f}–{m['ci95'][1]:.2f}) | {m['step1_correct']:.2f} | {m['step2_correct']:.2f} | "
                 f"{m['leading_correct_mean']:.1f} | {fe.get('rule',0)} / {fe.get('tracking',0)} / {fe.get('missing',0)} / {fe.get(None,0)} | "
                 f"{(m['other_parity_frac'] or 0):.2f} | {(m['old_rule_frac'] or 0):.2f} | {(m['copy_operand_frac'] or 0):.2f} | {m['tokens_mean']:.0f} |")
    return "\n".join(L) + "\n"


def main() -> None:
    path = sys.argv[1]
    res = analyse(path)
    md = markdown(res)
    Path(path).with_suffix(".analysis.md").write_text(md)
    Path(path).with_suffix(".analysis.json").write_text(json.dumps(res, indent=1))
    print(md)


if __name__ == "__main__":
    main()
