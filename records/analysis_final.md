# doze-llm — preregistered analysis (PREREG §7)

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
| baseline | 5 | 0.421 | 0.421 | +0.000 | 0.340 | 0/5 |
| awake | 5 | 0.419 | 0.419 | +0.000 | 0.310 | 0/5 |
| online | 5 | 0.963 | 0.889 | +0.155 | 0.377 | 0/5 |
| sleep | 5 | 0.937 | 0.894 | +0.031 | 0.417 | 0/5 |
| sleep_nodream | 5 | 0.950 | 0.917 | +0.079 | 0.383 | 0/5 |

## Paired comparisons (per seed; exact sign-flip test)

| comparison | diffs | mean | p |
|---|---|---|---|
| H4_forgetting_sleep_minus_online | [-0.077, -0.257, -0.05, -0.257, 0.023] | -0.123 | 0.125 |
| heldout_last3_sleep_minus_online | [0.095, -0.043, 0.041, -0.095, 0.025] | +0.005 | 1.0 |
| heldout_last3_sleep_minus_nodream | [-0.041, -0.05, 0.093, -0.096, -0.02] | -0.023 | 0.5 |
| forgetting_sleep_minus_nodream | [-0.18, -0.05, 0.027, -0.16, 0.127] | -0.047 | 0.375 |
| forgetting_nodream_minus_online | [0.103, -0.207, -0.077, -0.097, -0.103] | -0.076 | 0.3125 |

## Exploratory: update schedule (not preregistered)

Online applies one verified self-training step per episode; Sleep and Sleep-NoDream apply 50 per night. Same steps, same tokens.

| arm | n | GSM8K Δ mean | seeds with Δ>0 | mean per-checkpoint GSM8K change | worst single-checkpoint change | peak held-out | last-3 held-out | volatility |
|---|---|---|---|---|---|---|---|---|
| online | 5 | +0.155 | 5/5 | +0.0129 | -0.207 | 0.963 | 0.889 | 0.176 |
| sleep | 5 | +0.031 | 3/5 | +0.0026 | -0.087 | 0.937 | 0.894 | 0.168 |
| sleep_nodream | 5 | +0.079 | 4/5 | +0.0066 | -0.110 | 0.950 | 0.917 | 0.163 |

Caveats: seed 0's Sleep/Online/Awake used the pilot code (one-step dreamer, non-deterministic CUDA); seed 1 mixes GPU types (L4 for Sleep/Baseline, A16 otherwise); insight hypotheses are fully censored when no arm reaches the criterion, so the log-rank test carries no information beyond 'never met'.
