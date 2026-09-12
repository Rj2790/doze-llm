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

## A. GSM8K re-score — verdict: the grid's GSM8K movements were a prompt-regime artefact

Both regimes on the same L4, outputs saved in `gsm8k_outputs/`. `as_run` = the grid's regime (task system prompt "…no explanation…", 512 tokens); `math_1024` = math system prompt ("think step by step briefly…"), 1024 tokens.

| model | as_run | grid recorded (ep 600) | tokens | math_1024 | tokens | truncated |
|---|---|---|---|---|---|---|
| base | 0.587 | 0.583 | 64 | **0.910** | 131 | 0 |
| Online s1 / s2 / s3 / s4 | 0.760 / 0.693 / 0.787 / 0.740 | 0.760 / 0.680 / 0.777 / 0.753 | 63–78 | 0.907 / 0.927 / 0.907 / 0.897 | 118–132 | 0 |
| Sleep s1 / s2 / s3 / s4 | 0.493 / 0.647 / 0.537 / 0.780 | 0.497 / 0.630 / 0.520 / 0.777 | 53–80 | 0.930 / 0.920 / 0.933 / 0.917 | 130–133 | 0 |
| NoDream s0 / s1 / s2 / s3 / s4 | 0.827 / 0.563 / 0.623 / 0.683 / 0.653 | 0.827 / 0.553 / 0.603 / 0.680 / 0.650 | 56–93 | 0.913 / 0.927 / 0.923 / 0.917 / 0.927 | 129–136 | 0 |

- **Reproduction.** `as_run` matches the grid's recorded endpoints to within 0.02 on every model (different GPU, deterministic kernels).
- **Base under a normal math prompt: 0.910** vs 0.587 in the grid's regime. Pre-set rule met (≥ 0.70): the grid's GSM8K baseline was a suppressed-reasoning artefact.
- **Adapter − base, mean per arm:** Online +0.158 → **−0.001**; Sleep-NoDream +0.083 → **+0.011**; Sleep +0.027 → **+0.015** (as_run → math_1024). Every adapter is within ±0.02 of base under the math prompt (per-seed range −0.013 to +0.023). Pre-set rule met (|mean| < 0.05 for every arm): the recorded GSM8K movements, including the H4 reversal, were regime/verbosity effects and carry no information about arithmetic ability.
- **Length coupling.** Across the 14 models, accuracy vs mean answer length: r = 0.925 (slope +0.085 per 10 tokens) in `as_run`; r = 0.22 (slope +0.005) in `math_1024`. In the no-explanation regime the models that wrote a little more scored more; in the normal regime length no longer matters. 100% of answers carry the ANSWER tag in both regimes; nothing was truncated.

## Overall reading of the follow-up

1. H1–H3 (insight): unobservable as designed — the probe cannot be passed at this scale even by a model told the rule or trained on the probe itself.
2. H4 (retention): the grid's control benchmark was administered in a no-explanation regime; under a normal math prompt all 14 models score 0.90–0.93. No trained arm forgot anything measurable on GSM8K, and none gained.
3. What the grid did establish, and this confirms: verified self-distillation teaches general execution of the stated procedure (0.84–0.89 on strings without the practice structure vs 0.37 for base), equally under all three schedules, with dreams adding nothing.
