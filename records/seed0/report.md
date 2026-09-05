# doze-llm results

Seeds: [0] (all compute-matched)

| arm | median episode-to-criterion | censored | final held-out | final probe | forgetting (control delta) |
|---|---|---|---|---|---|
| awake | None | 1/1 | 0.403 | 0.300 | 0.000 |
| baseline | None | 1/1 | 0.373 | 0.367 | 0.000 |
| online | None | 1/1 | 0.667 | 0.317 | 0.137 |
| sleep | None | 1/1 | 0.930 | 0.317 | 0.060 |
| sleep_nodream | None | 1/1 | 0.970 | 0.267 | 0.240 |

| arm | volatility (std held-out) | mirror_bias (chance) | short_gap (struct / unstruct) | late vs early error | GSM8K unparsable |
|---|---|---|---|---|---|
| awake | 0.000 | — (—) | — (— / —) | — vs — | — |
| baseline | 0.000 | 0.353 (0.356) | 0.000 (0.317 / 0.317) | 0.512 vs 0.431 | 0.000 |
| online | 0.188 | — (—) | — (— / —) | — vs — | — |
| sleep | 0.167 | — (—) | — (— / —) | — vs — | — |
| sleep_nodream | 0.174 | 0.294 (0.324) | 0.000 (0.000 / 0.000) | 0.010 vs 0.020 | 0.000 |

Eval-pipeline validation (frozen-arm eval identical at every checkpoint, per seed): awake: yes [True]; baseline: yes [True]
