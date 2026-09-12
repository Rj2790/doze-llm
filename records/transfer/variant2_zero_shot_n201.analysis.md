# Transfer run: fold_variant2 — n=201 items per model, seed 0

Chance accuracy with four symbols is 0.25. `step1/step2 ok` = fraction of items whose first / first two written steps are right. For the first *rule* error (right operands, wrong result): `other parity` = the result matches the other step-type's table (odd/even confusion); `old rule` = matches the original puzzle's table; `copy` = equals one of the operands.

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
