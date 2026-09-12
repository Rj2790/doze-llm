# doze-llm results

Seeds: [4] (all compute-matched)

| arm | median episode-to-criterion | censored | final held-out | final probe | forgetting (control delta) |
|---|---|---|---|---|---|
| awake | None | 1/1 | 0.398 | 0.267 | 0.000 |
| baseline | None | 1/1 | 0.398 | 0.250 | 0.000 |
| online | None | 1/1 | 0.925 | 0.317 | 0.170 |
| sleep | None | 1/1 | 0.965 | 0.450 | 0.193 |
| sleep_nodream | None | 1/1 | 0.985 | 0.350 | 0.067 |

| arm | volatility (std held-out) | mirror_bias (chance) | short_gap (struct / unstruct) | late vs early error | GSM8K unparsable |
|---|---|---|---|---|---|
| awake | 0.000 | 0.351 (0.320) | -0.042 (0.342 / 0.383) | 0.481 vs 0.376 | 0.000 |
| baseline | 0.000 | 0.365 (0.322) | -0.042 (0.342 / 0.383) | 0.486 vs 0.371 | 0.000 |
| online | 0.191 | 0.444 (0.389) | 0.042 (0.367 / 0.325) | 0.000 vs 0.030 | 0.000 |
| sleep | 0.173 | 0.143 (0.393) | -0.017 (0.333 / 0.350) | 0.010 vs 0.007 | 0.000 |
| sleep_nodream | 0.162 | 0.273 (0.409) | 0.058 (0.367 / 0.308) | 0.000 vs 0.012 | 0.000 |

Eval-pipeline validation (frozen-arm eval identical at every checkpoint, per seed): awake: yes [True]; baseline: yes [True]
