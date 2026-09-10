# doze-llm — preregistered analysis (PREREG §7)

**PRELIMINARY — fewer than 5 complete seeds for at least one arm. Not the final result.**

Seeds: [0, 1, 2, 3, 4]; compute matching passed for every seed listed (check_seed).

## Hypotheses

| hypothesis | test | result | verdict |
|---|---|---|---|
| H1 insight (Sleep first to criterion) | log-rank | vs baseline: all censored (degenerate); vs awake: all censored (degenerate); vs online: all censored (degenerate). Sleep met the criterion in 0/5 seeds; all arms censored at 600 — log-rank degenerate | **not supported** |
| H2 phase structure (Sleep vs Online) | log-rank | all censored (degenerate) | **not supported** |
| H3 dreaming (Sleep vs NoDream) | log-rank | all censored (degenerate) | **not supported** |
| H4 retention (Sleep forgets less than Online) | paired sign-flip on GSM8K Δ | mean Sleep−Online -0.123 (n=5, p=0.125); direction reversed | **not supported** |

## Per-arm summary (mean over seeds)

| arm | n | peak held-out | last-3 held-out | GSM8K Δ | probe max | criterion met |
|---|---|---|---|---|---|---|
| baseline | 2 | 0.425 | 0.425 | +0.000 | 0.392 | 0/2 |
| awake | 2 | 0.428 | 0.428 | +0.000 | 0.325 | 0/2 |
| online | 5 | 0.963 | 0.889 | +0.155 | 0.377 | 0/5 |
| sleep | 5 | 0.937 | 0.894 | +0.031 | 0.417 | 0/5 |
| sleep_nodream | 4 | 0.942 | 0.901 | +0.082 | 0.383 | 0/4 |

## Paired comparisons (per seed; exact sign-flip test)

| comparison | diffs | mean | p |
|---|---|---|---|
| H4_forgetting_sleep_minus_online | [-0.077, -0.257, -0.05, -0.257, 0.023] | -0.123 | 0.125 |
| heldout_last3_sleep_minus_online | [0.095, -0.043, 0.041, -0.095, 0.025] | +0.005 | 1.0 |
| heldout_last3_sleep_minus_nodream | [-0.041, -0.05, 0.093, -0.096] | -0.024 | 0.625 |
| forgetting_sleep_minus_nodream | [-0.18, -0.05, 0.027, -0.16] | -0.091 | 0.25 |
| forgetting_nodream_minus_online | [0.103, -0.207, -0.077, -0.097] | -0.069 | 0.5 |

## Exploratory: update schedule (not preregistered)

Online applies one verified self-training step per episode; Sleep and Sleep-NoDream apply 50 per night. Same steps, same tokens.

| arm | n | GSM8K Δ mean | seeds with Δ>0 | mean per-checkpoint GSM8K change | worst single-checkpoint change | peak held-out | last-3 held-out | volatility |
|---|---|---|---|---|---|---|---|---|
| online | 5 | +0.155 | 5/5 | +0.0129 | -0.207 | 0.963 | 0.889 | 0.176 |
| sleep | 5 | +0.031 | 3/5 | +0.0026 | -0.087 | 0.937 | 0.894 | 0.168 |
| sleep_nodream | 4 | +0.082 | 3/4 | +0.0068 | -0.110 | 0.942 | 0.901 | 0.163 |

Caveats: seed 0's Sleep/Online/Awake used the pilot code (one-step dreamer, non-deterministic CUDA); seed 1 mixes GPU types (L4 for Sleep/Baseline, A16 otherwise); insight hypotheses are fully censored when no arm reaches the criterion, so the log-rank test carries no information beyond 'never met'.
