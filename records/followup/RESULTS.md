# Follow-up results (parts B and C complete 2026-09-12 18:32 IST; part A in progress)

Decision rules were fixed in `../followup_prereg.md` before running. GPU: NVIDIA L4 (Modal), deterministic kernels.

## B. Probe instrument — verdict: the grid's probe could not be passed by any model at this scale

Mean accuracy over seeds (chance 0.333, grid criterion 0.70; base n=5 seeds, adapters n=4–5):

| variant | base | Online | Sleep | Sleep-NoDream |
|---|---|---|---|---|
| as_run (grid's probe, answer only) | 0.340 | 0.346 | 0.367 | 0.317 |
| rule_stated (regularity written in the prompt, answer only) | 0.347 | 0.292 | 0.350 | 0.293 |
| work_allowed (may write the visible comparisons) | 0.383 | 0.675 | 0.446 | 0.550 |
| rule_stated_work | 0.340 | **0.783** | 0.542 | 0.633 |
| logprob (teacher-forced argmax over 1/4/9) | 0.327 | 0.333 | 0.333 | 0.333 |
| poscontrol50 (base + 50 steps on masked→r6 pairs) | 0.327 | — | — | — |
| poscontrol200 | 0.333 | — | — | — |

- **Telling the model the rule does nothing when it must answer without working** (rule_stated = chance for every model, including the trained ones). Allowing it to write the visible comparisons is what moves the number (Online 0.78 with the rule stated, 0.88/0.83/0.75/0.67 per seed). This is the pre-registered second branch: *the probe was blind by construction; H1–H3 were unobservable as designed.*
- **Direct training on the probe task fails.** 200 LoRA steps on masked-prompt → "ANSWER: r6" pairs (training loss plateaus ≈ 0.21, i.e. the model cannot fit it) leaves probe accuracy at 0.32–0.35 and slightly damages full-chain accuracy (0.32–0.53 on 60 held-out vs ≈ 0.43 for base). A 4B model cannot learn to produce r6 silently in this budget, regardless of regime. The probe was unreachable, not merely unreached.
- **No latent signal either**: teacher-forced log-probability picks r6 at chance for all 14 models; the margin of r6 over the best alternative is negative everywhere (base −2.6, adapters −0.5 to −0.8 nats).
- Caveat on `work_allowed`: with the visible prefix of 7 digits, the last written result *is* r6, so writing the visible chain and reporting its end is correct by construction. That variant measures "can compute r6 in writing", not shortcut knowledge. Base scores 0.38 on it because it tends to continue through the masked positions and drift.

## C. Structured vs unstructured full-chain accuracy — verdict: general execution learning

| model | structured held-out (201) | unstructured (120) | gap |
|---|---|---|---|
| base (5 seeds) | 0.436 | 0.372 | +0.064 |
| Online (4) | 0.913 | 0.844 | +0.069 (seeds 1–2: +0.16; seeds 3–4: ≈0) |
| Sleep (4) | 0.888 | 0.840 | +0.048 |
| Sleep-NoDream (5) | 0.925 | 0.887 | +0.038 |

The base model itself is 0.06 better on structured strings, so the adapters' gaps of 0.04–0.07 are the item-set difference, not specialisation. By the pre-set rule (gap < 0.05 general, ≥ 0.15 specific) the trained arms learned general execution; two Online seeds sit in between. The structured held-out numbers reproduce the grid's recorded values on a different GPU to within 0.01–0.02.

## A. GSM8K re-score — partial

`as_run` (task system prompt, 512 tokens) reproduces the grid exactly: base 0.587, Online seed 1 0.760, Sleep seed 4 0.780, Sleep seed 3 0.537 (grid: 0.587/0.760/0.777/0.520). Mean response length already tracks accuracy across models (54 tokens at 0.54; 80 tokens at 0.78).

**Under the math system prompt with 1024 tokens the base model scores 0.910 (mean 131 tokens, 0% truncated), against 0.587 under the grid's regime.** Pre-set rule: base ≥ 0.70 → the grid's GSM8K baseline was a suppressed-reasoning artefact. Adapter results under this regime are pending; they decide whether the recorded Online +0.155 and Sleep dips survive at all.

Files: `probe_summary.json`, `chain_summary.json`, `gsm8k_summary.json` (growing), `gsm8k_outputs/*.jsonl` (per-item texts), `followup.log`.
