# doze-llm results

Seeds: [1] (all compute-matched)

| arm | median episode-to-criterion | censored | final held-out | final probe | forgetting (control delta) |
|---|---|---|---|---|---|
| awake | None | 1/1 | 0.453 | 0.350 | 0.000 |
| baseline | None | 1/1 | 0.478 | 0.417 | 0.000 |
| online | None | 1/1 | 0.900 | 0.367 | 0.177 |
| sleep | None | 1/1 | 0.811 | 0.367 | -0.080 |
| sleep_nodream | None | 1/1 | 0.886 | 0.383 | -0.030 |

| arm | volatility (std held-out) | mirror_bias (chance) | short_gap (struct / unstruct) | late vs early error | GSM8K unparsable |
|---|---|---|---|---|---|
| awake | 0.000 | 0.327 (0.332) | 0.042 (0.367 / 0.325) | 0.412 vs 0.425 | 0.000 |
| baseline | 0.000 | 0.330 (0.338) | 0.000 (0.342 / 0.342) | 0.405 vs 0.401 | 0.000 |
| online | 0.147 | 0.167 (0.333) | -0.033 (0.342 / 0.375) | 0.020 vs 0.062 | 0.000 |
| sleep | 0.161 | 0.270 (0.343) | 0.033 (0.392 / 0.358) | 0.036 vs 0.118 | 0.000 |
| sleep_nodream | 0.150 | 0.333 (0.343) | 0.133 (0.433 / 0.300) | 0.046 vs 0.084 | 0.000 |

Eval-pipeline validation (frozen-arm eval identical at every checkpoint, per seed): awake: yes [True]; baseline: yes [True]
