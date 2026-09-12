# Transfer study, variant 2 — zero-shot results (2026-09-12, local MPS, 201 items per model)

Task: `tasks/fold_variant2.py` — four symbols (1, 4, 7, 9), left-to-right fold, and the rule alternates
by step: odd steps take the smaller of the two uninvolved symbols, even steps the larger. Same output
format as the original puzzle. No hidden structure. Design note: `../transfer_prereg.md`. Models: base
Qwen3-4B and the 13 saved adapters, no further training. Chance accuracy = 0.25. Per-item texts in
`variant2_zero_shot_n201.json`.

## By arm (means over seeds)

| arm | n models | accuracy | step 1 right | step 2 right | leading correct steps / 11 | first rule error = other step's table | = old puzzle's table | = copies an operand |
|---|---|---|---|---|---|---|---|---|
| base | 1 | 0.254 | 0.90 | 0.34 | 1.48 | 0.47 | 0.13 | 0.53 |
| online | 4 | 0.274 | 0.61 | 0.24 | 1.04 | 0.60 | 0.12 | 0.40 |
| sleep | 4 | 0.233 | 0.79 | 0.29 | 1.28 | 0.45 | 0.12 | 0.55 |
| sleep_nodream | 5 | 0.240 | 0.84 | 0.31 | 1.40 | 0.45 | 0.15 | 0.55 |

## Per model


| model | accuracy (95% CI) | step1 ok | step2 ok | leading ok / 11 | first error: rule / tracking / missing / none | other parity | old rule | copy | tokens |
|---|---|---|---|---|---|---|---|---|---|
| base | 0.254 (0.20–0.32) | 0.90 | 0.34 | 1.5 | 188 / 13 / 0 / 0 | 0.47 | 0.13 | 0.53 | 95 |
| online_seed1 | 0.244 (0.19–0.31) | 0.67 | 0.25 | 1.2 | 187 / 14 / 0 / 0 | 0.87 | 0.19 | 0.13 | 94 |
| online_seed2 | 0.343 (0.28–0.41) | 0.42 | 0.17 | 0.8 | 196 / 5 / 0 / 0 | 0.76 | 0.11 | 0.24 | 102 |
| online_seed3 | 0.249 (0.19–0.31) | 0.56 | 0.22 | 0.9 | 193 / 8 / 0 / 0 | 0.47 | 0.09 | 0.53 | 96 |
| online_seed4 | 0.259 (0.20–0.32) | 0.81 | 0.30 | 1.3 | 200 / 1 / 0 / 0 | 0.33 | 0.09 | 0.68 | 98 |
| sleep_nodream_seed0 | 0.259 (0.20–0.32) | 0.80 | 0.32 | 1.4 | 194 / 7 / 0 / 0 | 0.64 | 0.19 | 0.36 | 96 |
| sleep_nodream_seed1 | 0.234 (0.18–0.30) | 0.95 | 0.33 | 1.5 | 199 / 2 / 0 / 0 | 0.35 | 0.11 | 0.65 | 98 |
| sleep_nodream_seed2 | 0.229 (0.18–0.29) | 0.79 | 0.30 | 1.5 | 198 / 2 / 0 / 1 | 0.57 | 0.19 | 0.43 | 96 |
| sleep_nodream_seed3 | 0.259 (0.20–0.32) | 0.90 | 0.32 | 1.3 | 197 / 4 / 0 / 0 | 0.19 | 0.07 | 0.81 | 98 |
| sleep_nodream_seed4 | 0.219 (0.17–0.28) | 0.77 | 0.27 | 1.3 | 199 / 2 / 0 / 0 | 0.50 | 0.18 | 0.50 | 98 |
| sleep_seed1 | 0.199 (0.15–0.26) | 0.95 | 0.35 | 1.6 | 193 / 7 / 0 / 1 | 0.52 | 0.17 | 0.48 | 98 |
| sleep_seed2 | 0.249 (0.19–0.31) | 0.70 | 0.26 | 1.1 | 197 / 3 / 0 / 1 | 0.51 | 0.12 | 0.49 | 98 |
| sleep_seed3 | 0.264 (0.21–0.33) | 0.70 | 0.27 | 1.1 | 196 / 5 / 0 / 0 | 0.36 | 0.06 | 0.64 | 98 |
| sleep_seed4 | 0.219 (0.17–0.28) | 0.84 | 0.30 | 1.3 | 200 / 1 / 0 / 0 | 0.41 | 0.14 | 0.59 | 96 |

## Reading

1. **Everyone is at chance.** All 14 models score 0.20–0.34 with 95% intervals that include 0.25.
   The alternating rule is beyond zero-shot reach for this model: the base model gets the first step
   (odd table) right 90% of the time and the second step (even table) 34% of the time; no model ever
   completes more than ~1.5 correct leading steps on average.
2. **No positive transfer, and a small negative one.** The trained adapters are *worse* at step 1
   than the base model (Online 0.61, Sleep 0.79, NoDream 0.84 vs base 0.90; per-model intervals are
   about ±0.07, so the Online deficit is real across all four seeds). Their first rule errors come
   earlier (Online seed 3: 88 of 193 at step 1 vs base 21 of 188).
3. **What leaks is the habit, not the table.** Only 6–19% of first rule errors reproduce the original
   puzzle's table, no more for adapters than for base. The dominant errors are applying the other
   step's table (45–60%) or writing one of the operands as the result (40–55%). 600 steps of training
   on a single fixed rule made the models more likely to apply *a* single rule to every step, which is
   exactly wrong here.
4. **Length is not a confound**: 94–102 tokens per attempt for every model.

## What this means for the transfer question

- "Do the trained models figure out new rules faster or perform better?" — zero-shot, no. On a rule
  that changes with position they perform at chance, like the base model, and slightly worse on the
  very first step.
- Because every model is on the floor, this variant cannot show positive transfer even if it existed.
  A fairer zero-shot test needs a rule the base model can partly execute (variant 1: one new fixed
  table, four symbols). The learning-to-learn question (train on the new rule from base vs from an
  adapter) remains open and is the only way to answer "faster".
- The negative-transfer finding is the useful one: verified self-distillation on one procedure
  entrenched "one rule, every step", measurable as a 6–29 point drop in first-step accuracy on a
  rule that alternates.
