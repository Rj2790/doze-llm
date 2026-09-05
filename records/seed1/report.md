# doze-llm results

Seeds: [1] (all compute-matched)

| arm | median episode-to-criterion | censored | final held-out | final probe | forgetting (control delta) |
|---|---|---|---|---|---|
| baseline | None | 1/1 | 0.478 | 0.417 | 0.000 |
| sleep | None | 1/1 | 0.811 | 0.367 | -0.080 |

| arm | volatility (std held-out) | mirror_bias (chance) | short_gap (struct / unstruct) | late vs early error | GSM8K unparsable |
|---|---|---|---|---|---|
| baseline | 0.000 | 0.330 (0.338) | 0.000 (0.342 / 0.342) | 0.405 vs 0.401 | 0.000 |
| sleep | 0.161 | 0.270 (0.343) | 0.033 (0.392 / 0.358) | 0.036 vs 0.118 | 0.000 |

Eval-pipeline validation (frozen-arm eval identical at every checkpoint, per seed): baseline: yes [True]
