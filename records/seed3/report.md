# doze-llm results

Seeds: [3] (all compute-matched)

| arm | median episode-to-criterion | censored | final held-out | final probe | forgetting (control delta) |
|---|---|---|---|---|---|
| awake | None | 1/1 | 0.433 | 0.300 | 0.000 |
| baseline | None | 1/1 | 0.428 | 0.350 | 0.000 |
| online | None | 1/1 | 0.960 | 0.333 | 0.193 |
| sleep | None | 1/1 | 0.876 | 0.283 | -0.063 |
| sleep_nodream | None | 1/1 | 0.905 | 0.317 | 0.097 |

| arm | volatility (std held-out) | mirror_bias (chance) | short_gap (struct / unstruct) | late vs early error | GSM8K unparsable |
|---|---|---|---|---|---|
| awake | 0.000 | 0.317 (0.322) | 0.133 (0.392 / 0.258) | 0.450 vs 0.414 | 0.000 |
| baseline | 0.000 | 0.322 (0.322) | 0.133 (0.392 / 0.258) | 0.492 vs 0.422 | 0.000 |
| online | 0.153 | 0.500 (0.333) | 0.000 (0.350 / 0.350) | 0.000 vs 0.017 | 0.000 |
| sleep | 0.143 | 0.338 (0.324) | 0.067 (0.400 / 0.333) | 0.131 vs 0.023 | 0.000 |
| sleep_nodream | 0.176 | 0.310 (0.336) | 0.000 (0.333 / 0.333) | 0.069 vs 0.021 | 0.000 |

Eval-pipeline validation (frozen-arm eval identical at every checkpoint, per seed): awake: yes [True]; baseline: yes [True]
