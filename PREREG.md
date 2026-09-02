# Preregistration: Phased offline consolidation ("sleep") in LLMs

**Status:** v1.0 — frozen 2026-09-02 at commit `6cfc31923d180b79cb204f199c13bb29098f84c9` ("Freeze PREREG v1.0").
**Author:** Rushil
**Date frozen:** 2026-09-02

## 1. Question

**Public freeze record:** https://gist.github.com/Rj2790/5bfb6765d5491802c3e139d65afe3dc8 (freeze hash, date, and
this document as of the freeze commit).

Does alternating a small language model between a *day* phase (attempting
problems with fixed weights) and a *night* phase (offline replay, synthetic
"dreaming", and a parameter-efficient weight update) produce

(a) faster discovery of hidden task structure ("insight"), and
(b) less forgetting of unrelated capability,

than spending the **same compute** on either extended inference-time
reasoning or continuous online updates?

The biological analogy (Wagner et al., 2004; sleep-dependent insight on the
Number Reduction Task) motivates the design but is not itself the claim.

## 2. Hypotheses

- **H1 (insight).** The Sleep arm reaches the shortcut-discovery criterion
  (§6.1) in fewer episodes than Baseline, Awake, and Online.
- **H2 (phase structure).** Sleep beats Online at matched gradient budget.
  This is the load-bearing comparison. Beating Baseline only shows that
  fine-tuning works.
- **H3 (dreaming).** Sleep beats Sleep-NoDream at matched gradient budget.
  If H3 fails, replay alone carries the effect.
- **H4 (retention).** Sleep shows a smaller drop on the control benchmark
  than Online.

Each hypothesis can fail independently. All four are reported regardless.

## 3. Model

- Primary: `Qwen/Qwen3-4B` (dense, Apache 2.0), bf16, LoRA r=16 on all
  attention and MLP projections. No quantisation in the primary run.
- Confirmation (only if H2 holds at 4B): `Qwen/Qwen3-8B`, identical config.
- Scale sweep (only if the 4B result is clean): `Qwen3-1.7B`, `4B`, `8B`.
- Thinking mode **off** for all arms; reasoning happens in visible tokens
  so it can be counted.
- **Fully seeded.** The run seed fixes the instance sequence, the split,
  dream sampling, LoRA initialisation and every sampler draw. Bit-identical
  on MLX; seeded on CUDA (A1).

## 4. Tasks

### 4.1 Number Reduction (primary)

Twelve-digit strings over {1, 4, 9} (Wagner used 8; see below). Pairwise
rule: same→same, different→the third digit. Applied left-to-right to
produce eleven intermediate responses r1..r11; the answer is r11.

Instances are constructed so that the last three responses mirror the
preceding three: **r11 = r6**, r10 = r7, r9 = r8. A solver that has
discovered this can answer after six comparisons instead of eleven, using
only the first seven digits.

Why 12 not 8: the constraint leaves 3^(L−3) structured instances. At L=8
that is 243, with 27 distinct 3-digit prefixes each mapping to a single
answer — a lookup table a 4B model would memorise in one night. At L=12
there are 19,683 instances over 2,187 seven-digit prefixes.

- **Splits are by prefix.** 20% of prefixes (437) are held out; no
  held-out prefix appears in any training instance. Probe items are drawn
  only from held-out prefixes, so beating chance requires the rule, not a
  memorised prefix→answer map.
- Train: 801 instances from training prefixes. Held-out eval: 201 from
  held-out prefixes. Probe: 60 further held-out-prefix instances, disjoint
  from the 201 eval items. Each of the three sets is sampled stratified by
  answer so its answer distribution is exactly uniform over {1,4,9}
  (sizes are multiples of 3 for that reason; A6, C7).
- Probe: digits 8..12 masked. The literal rule reaches only r6; the
  answer is recoverable above chance (1/3) only via r11 = r6.
- Implementation and tests: `tasks/number_reduction.py`,
  `eval/shortcut_detector.py`, `tests/`. Frozen at commit `6cfc31923d180b79cb204f199c13bb29098f84c9`.

### 4.2 String Grammar (secondary)

A procedurally generated rewrite grammar with a hidden invariant (defined
in `tasks/string_grammar.py`; frozen before the first run). Included so
the result is not specific to one task. Same probe logic.

### 4.3 Control benchmark

A fixed 300-item sample from GSM8K test, scored exact-match. Measures
forgetting. Never trained on by any arm.

## 5. Arms

All arms see the same sequence of training instances (same seed) and are
evaluated at the same checkpoints: episode 0 (before any day episode; the
forgetting baseline and the untrained probe) and then every K=50 episodes
(A5).

| Arm | Weights | Per-episode compute | Notes |
|---|---|---|---|
| Baseline | frozen | 1 attempt, short answer format | floor |
| Awake | frozen | matched **token** budget spent on chain-of-thought + self-critique | inference-time control |
| Online | LoRA | 1 gradient step after **every** episode on a *kept* trajectory (keep rule below): the current one if kept, else a random one from today's kept pool, else the most recently kept ever, else skip and log. Steps = K per K episodes. | continuous-update control |
| Online-Unfiltered | LoRA | 1 gradient step on every trajectory as written, right or wrong | optional; separates phase structure from filtering |
| Sleep | LoRA | 0 during the day; every K episodes a night: filter → dream → replay-interleave → gradient steps → weight-decay pass | treatment |
| Sleep-NoDream | LoRA | as Sleep, no synthetic generation | ablation |

**Compute matching.**
- Awake's total generated tokens = Sleep's total generated tokens
  (day attempts + night dreams), ±5%.
- Online's total gradient steps and total training tokens = Sleep's, ±5%.
- `eval/compute_ledger.py` logs both per arm. `eval/analyze.py` loads the
  per-arm ledgers, persists the matching verdict per seed into every run
  JSON, and refuses to produce any figure or table for unmatched or
  mixed-backend arms (A3).

**Night procedure (Sleep).**
1. Collect the day's K trajectories.
2. Keep trajectories whose final answer is correct **and** ≥ 9/11
   intermediate responses are correct. Wrong-answer near-misses are not
   kept. The same keep rule is used by Online; dreams must be fully correct
   (C3).
3. Dream: prompt the current model to generate 2 variations per kept
   trajectory; keep only those that verify against the ground-truth solver.
4. Interleave 1:1 with a uniform sample from the replay buffer of all
   previous nights.
5. Train LoRA for S steps (S fixed so the ledger matches Online).
6. One pass of weight decay on LoRA params (the pruning analogue).
7. Append the night's kept *real* trajectories to the replay buffer.
   Dreams are trained on once and never replayed (C2).

## 6. Metrics

### 6.1 Shortcut discovery (primary)

At each checkpoint, run the 60 probe items (digits 8–12 masked; see §4.1):
stratified by target, from held-out prefixes, disjoint from the eval items
(C7). The Awake arm runs its critique rounds at probe time as well, with a
probe-specific critique prompt; this is a deliberate property of that arm,
since inference-time compute is what it tests (C4). Criterion:
probe accuracy ≥ 0.70 for two consecutive checkpoints. Report the episode
index at which the criterion is first met (censored at the end of training
if never met).

Secondary insight signal: median generated tokens per correct answer on
the held-out set; a collapse indicates the model has stopped simulating
the full chain.

### 6.2 Task accuracy

Held-out accuracy at each checkpoint.

### 6.3 Forgetting

Control benchmark accuracy at each checkpoint minus accuracy at episode 0.

### 6.4 Cost

Tokens generated, gradient steps, training tokens, GPU-seconds, per arm.

## 7. Analysis

- 5 seeds per arm. Seeds fix the instance sequence, dream sampling, and
  LoRA init.
- H1/H2/H3: compare episode-to-criterion distributions with a log-rank
  test (censoring handled). Report medians and full curves.
- H4: paired comparison of forgetting at the final checkpoint.
- A hypothesis is "supported" at p < 0.05 with effect in the predicted
  direction across the primary task; the secondary task is reported as
  replication, not pooled.
- No arms, metrics, or stopping rules are added after the freeze date.
  Deviations are listed in a `DEVIATIONS.md`.

## 8. Stopping rule

Each run trains for 600 episodes (12 nights) or until all arms have met
the criterion, whichever comes first. No peeking-based extension.

## 8a. Amendments before freeze (2026-09-02)

Applied after a fresh-session review of the implementation, before the
freeze, so they are amendments rather than deviations: A1 full seeding
(§3); A2 filtered Online + optional Online-Unfiltered (§5); A3 persisted
matching verdict and analysis gate (§5, §7); A5 episode-0 checkpoint (§5);
A6 answer-stratified splits, 801/201/60 (§4.1); C2 dreams never replayed
(§5 step 7); C3 keep rule = correct answer and ≥ 9/11 steps (§5 step 2);
C4 Awake critique at probe time (§6.1); C7 probe disjoint from eval (§4.1,
§6.1). Infrastructure and hygiene items from the same review (B1 24 h
Modal timeout, per-checkpoint saving, batched evaluation; B2 named
entrypoints; B3 dtype assertion and transformers pin; B4 prerequisites
checked before model load; C1 leak check before any gradient step; C5 pure
temperature sampling on both backends; C6 repo hygiene) are applied too.
The review itself is `REVIEW.md`. A4 (estimate Online vs Sleep training-
token drift from the Baseline run before launching trainable arms) is a
pre-launch check, listed in `CONTEXT.md` §8.

## 9. Known limitations (stated up front)

- LoRA on a 4B model is not the same intervention as full fine-tuning on a
  frontier model. The result may not transfer.
- The tasks are synthetic and small; a positive result shows a mechanism,
  not a product.
- "Dreaming" here is verified self-generation; unverified dreams are a
  separate, riskier condition not tested in v0.1.

## 10. Outputs

Everything is released regardless of result: code, seeds, raw logs, per-arm
ledgers, plots, and a write-up. A null result on H2 is a publishable
contribution.

## Appendix A. Calibration record (added 2026-09-02, before freeze)

Untrained `mlx-community/Qwen3-4B-bf16` (MLX, Mac), L=12, held-out items,
work-mode prompt (one `prev,digit->result` line per comparison, then
`STEPS:` and `ANSWER:`), 400-token budget, thinking off, greedy.

| n | full accuracy | all 11 steps exact | probe (n=120, stratified) | binomial p vs 1/3 |
|---|---|---|---|---|
| 40 | 0.475 | 0.175 | 0.325 | 0.61 |

Chance on the full task is 1/3. The compact format (STEPS line only, no
per-comparison lines) sat at chance (0.30, 0 exact chains) in both 4-bit
and bf16 and was replaced by work mode; a numbered-digit variant (0.525,
0 exact chains, errors at steps 1–3) was rejected. Prompt wording is
tunable and is not part of the frozen design; the task construction, the
prefix-disjoint split and the probe (mask digits 8–12, target r6) are.
Full record: `CONTEXT.md` §5, `results/calibration*.json`.

**Addendum 2026-09-02 (documentation correction, see DEVIATIONS.md).** The
40 calibration items above were the first 40 held-out items of the
pre-amendment split (before A6 introduced answer-stratified sampling and
the 801/201/60 sizes). They are not the first 40 held-out items of the
frozen split; the two sets share no items. On the frozen split's first 40
held-out items the same untrained model, prompt and decoding give
accuracy 0.25 and 0.075 exact chains (MLX bf16, `results/
agreement_mlx_bf16_seed0.json`). Repeated-digit statistics of the two sets
are indistinguishable (mean adjacent-equal positions 3.55 vs 3.48; mean
longest run 3.1 vs 2.9). The calibration verdict ("learnable regime") was
based on the pre-A6 sample; the pooled estimate over both 40-item samples
is 0.36. The Baseline arm's episode-0 checkpoint on 201 items is the
authoritative untrained accuracy.
