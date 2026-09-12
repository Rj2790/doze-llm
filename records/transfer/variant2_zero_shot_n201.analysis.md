# Transfer run: fold_variant2 — n=201 items per model, seed 0

Chance accuracy with four symbols is 0.25. `step1/step2 ok` = fraction of items whose first / first two written steps are right. For the first *rule* error (right operands, wrong result): `other parity` = the result matches the other step-type's table (odd/even confusion); `old rule` = matches the original puzzle's table; `copy` = equals one of the operands.

| model | accuracy (95% CI) | step1 ok | step2 ok | leading ok / 11 | first error: rule / tracking / missing / none | other parity | old rule | copy | tokens |
|---|---|---|---|---|---|---|---|---|---|
| base | 0.254 (0.20–0.32) | 0.90 | 0.34 | 1.5 | 188 / 13 / 0 / 0 | 0.47 | 0.13 | 0.53 | 95 |
| online_seed1 | 0.244 (0.19–0.31) | 0.67 | 0.25 | 1.2 | 187 / 14 / 0 / 0 | 0.87 | 0.19 | 0.13 | 94 |
| online_seed2 | 0.343 (0.28–0.41) | 0.42 | 0.17 | 0.8 | 196 / 5 / 0 / 0 | 0.76 | 0.11 | 0.24 | 102 |
