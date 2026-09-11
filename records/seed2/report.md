# doze-llm results

Seeds: [2] (all compute-matched)

| arm | median episode-to-criterion | censored | final held-out | final probe | forgetting (control delta) |
|---|---|---|---|---|---|
| awake | None | 1/1 | 0.408 | 0.333 | 0.000 |
| baseline | None | 1/1 | 0.428 | 0.317 | 0.000 |
| online | None | 1/1 | 0.856 | 0.333 | 0.097 |
| sleep | None | 1/1 | 0.930 | 0.300 | 0.047 |
| sleep_nodream | None | 1/1 | 0.861 | 0.317 | 0.020 |

| arm | volatility (std held-out) | mirror_bias (chance) | short_gap (struct / unstruct) | late vs early error | GSM8K unparsable |
|---|---|---|---|---|---|
| awake | 0.000 | 0.333 (0.344) | 0.083 (0.392 / 0.308) | 0.486 vs 0.431 | 0.000 |
| baseline | 0.000 | 0.319 (0.346) | 0.083 (0.392 / 0.308) | 0.454 vs 0.424 | 0.000 |
| online | 0.200 | 0.354 (0.314) | 0.033 (0.358 / 0.325) | 0.071 vs 0.096 | 0.000 |
| sleep | 0.195 | 0.240 (0.340) | 0.142 (0.400 / 0.258) | 0.024 vs 0.029 | 0.000 |
| sleep_nodream | 0.149 | 0.279 (0.338) | 0.108 (0.417 / 0.308) | 0.065 vs 0.063 | 0.000 |

Eval-pipeline validation (frozen-arm eval identical at every checkpoint, per seed): awake: yes [True]; baseline: yes [True]
