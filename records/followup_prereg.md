# Follow-up checks on the 4B grid — pre-analysis note (written before running, 2026-09-12)

Post-hoc and exploratory; nothing here changes PREREG verdicts. One job on
one GPU instance (`eval/followup.py`), base model + the 13 saved adapters.

## A. GSM8K re-score with saved outputs

**Fact established while preparing this:** the grid's control benchmark was
generated under the *task* system prompt ("You solve digit-combination
puzzles. Output only the requested lines… no explanation…") because the
harness backend carries one system prompt per run; `control_bench.SYSTEM_PROMPT`
("think step by step briefly") exists but was never used. So the recorded
GSM8K numbers are a "no-explanation" regime. Recorded in DEVIATIONS.md and
CONTEXT §5m.

Regimes: (1) `as_run` — task system prompt, 512 tokens, should reproduce the
recorded base 0.58 and adapter endpoints up to GPU numerics; (2) `math_1024`
— the math system prompt, 1024 tokens. Per item we save text, tokens,
truncation, tag presence, parsed answer, correctness.

Decision rules (stated in advance):
- If base `math_1024` ≥ 0.70, the grid's 0.58 baseline was a suppressed-
  reasoning artefact of the prompt regime.
- If adapter−base differences under `math_1024` shrink to |mean| < 0.05 for
  every arm, the recorded GSM8K movements (Online +0.155, Sleep dips) were
  regime/verbosity effects and the H4 axis carries no information about
  arithmetic ability. Report per-model tokens and the accuracy–length slope
  in both regimes.
- If Online's advantage persists under `math_1024` (mean ≥ +0.05 over base,
  positive in ≥ 3 of 4 seeds) with token counts within ±15% of base, it is
  a genuine transfer effect and deserves the broader benchmark panel.
- Anything in between: report as unresolved; the panel decides.

## B. Probe instrument

Variants on each seed's 60 probe items: `as_run` (the grid's probe);
`rule_stated` (regularity written into the prompt, answer only);
`work_allowed` (may write the visible comparisons first, 128 tokens);
`rule_stated_work`; `logprob` (teacher-forced total log-probability of
"ANSWER: d" for d ∈ {1,4,9}, argmax vs r6, plus mean margin); and
`poscontrol50` / `poscontrol200` (fresh LoRA trained 50 / 200 steps on
masked-prompt → "ANSWER: r6" pairs drawn from TRAINING prefixes only, then
scored on the seed's probe items; also its answer-only accuracy on visible
held-out items).

Decision rules:
- `rule_stated` ≥ 0.70 for base → the probe detects explicit knowledge
  without working; the grid's null then means no arm acquired it explicitly.
- `rule_stated` ≈ 1/3 but `rule_stated_work` ≥ 0.70 → the probe was blind
  by construction (silent computation required); H1–H3 unobservable as
  designed. Any adapter whose `work_allowed` probe exceeds base's by ≥ 0.15
  would be the first sign of shortcut use.
- `poscontrol200` ≥ 0.70 → a model can be taught to pass this probe; the
  adapters' ~1/3 then means they did not internalise the mapping.
  `poscontrol200` ≈ 1/3 → 200 steps cannot make a 4B model do it silently;
  the probe is unreachable at this scale regardless of regime.
- `logprob` accuracy ≥ 0.45 for adapters vs ≈ 1/3 for base → latent signal
  the generation probe missed.

## C. Structured vs unstructured full-chain accuracy

201 structured held-out (reproduces recorded numbers) and 120 unstructured
held-out-prefix strings, work format, per model. Rule: unstructured within
0.05 of structured → the adapters learned general execution; a gap ≥ 0.15 →
learning was specific to the structured subset.

## Reporting

`records/followup/` (summaries, per-item outputs, log), a results section
appended to `analysis_outsider.md`, CONTEXT §5n. All 14 models on one GPU
type; the GPU is recorded with every number.
