# A fresh-eyes reading of the experiment (2026-09-12)

Written deliberately without the hypothesis lens. The question is not "did
the sleep cycle produce insight" but "here is an experiment someone ran and
all of its data; what did it actually show, what would a skeptical reader
want to know, and what is worth anything?" Numbers come from
`analysis_final.json`, `analysis_deep.json` and direct reads of the run
files. A second, independently written reading by a fresh session that had
no access to this conversation is appended at the end.

## 1. What was actually done, stated plainly

- **Model.** Qwen3-4B, bf16, thinking mode off, greedy decoding. Trainable
  arms use LoRA r=16 on all attention and MLP projections (33M params),
  Adam 1e-4, batch 1.
- **Task.** A deterministic string-reduction puzzle: 12 digits from {1,4,9}
  are folded left to right with a fixed ternary rule, 11 steps, the answer
  is the last result. The model is told the rule and shown a worked example,
  and must write all 11 steps, a STEPS line and an ANSWER line (~97 tokens).
  Every puzzle in the training and evaluation sets is drawn from a subset of
  strings in which the 6th and 11th intermediate results coincide, so the
  answer is always available five steps early. Nothing in the prompt says
  so.
- **Regimes** (five, each run for 600 puzzles on the same puzzle sequence per
  seed):
  1. *Frozen.* Attempt each puzzle; never update.
  2. *Frozen + self-critique.* Two to three rounds of "check your work" at
     roughly twice the tokens; never update.
  3. *Per-episode fine-tuning.* After each attempt, if the answer is right
     and at least 9 of 11 steps are right, take one gradient step on that
     attempt. ~1 step per puzzle, 582–600 steps total.
  4. *Batched fine-tuning with synthetic variants.* Same filter, but keep
     the passing attempts in a buffer and, every 50 puzzles, (a) have the
     model propose new digit strings by editing the last 1–3 of the final
     five digits of buffered puzzles, solve them, and keep those it solves
     correctly; (b) train 50 steps on new keeps + accepted variants + an
     equal number of replayed old keeps; (c) shrink the LoRA weights by 5%.
     600 steps total.
  5. *Batched fine-tuning without variants.* As 4 minus step (a).
- **Measures at 13 checkpoints (every 50 puzzles, including before the
  first).** Accuracy on 201 unseen puzzles from the same subset; accuracy on
  60 puzzles with digits 8–12 hidden, answerable only via the coincidence
  (chance 1/3); exact-match on a fixed 300-item GSM8K sample; answer-only
  (no working) accuracy on structured vs unstructured strings; error
  position statistics; token counts.
- **Matching.** Generated tokens matched between the two frozen regimes and
  between 2 and 4; gradient steps and training tokens matched between 3, 4
  and 5. All within ±3%.
- **Replication.** 5 seeds = 5 puzzle sequences and splits. Seed 0's
  regimes 2–4 ran an earlier code version (one-step variant generator,
  non-deterministic kernels). Seed 1 ran two arms on a different GPU model.

## 2. What the data show, in order of robustness

**(a) Fine-tuning on the model's own verified solutions teaches the
procedure, quickly and reliably.** 15 of 15 trained runs go from 0.37–0.48
held-out accuracy to 0.81–0.99 (mean of last three checkpoints 0.89–0.92).
The first checkpoint at or above 0.80 comes after 200–450 puzzles. The
untrained model's errors are almost never rule errors; they are
digit-tracking slips (it combines with the wrong next digit, e.g. reading
position 5 as 1 when it is 9). Training removes most of those slips and
pushes the survivors later in the chain (first-error mode 5–6 → 7–8). This
is the largest effect in the dataset by an order of magnitude and it is not
surprising: rejection-sampled self-distillation with a perfect verifier on a
narrow algorithmic task is known to work.

**(b) Nothing was learned that works without the chain.** Answer-only
accuracy on the same kind of puzzle stays between 0.25 and 0.39 for every
regime at every checkpoint, trained or not. The hidden-digit items stay at
chance: pooled over 3,900 evaluations per trained regime, 0.32–0.34 with
±0.015 intervals; 0.31–0.34 for the frozen regimes. The +0.04 answer-only
advantage of structured over unstructured strings is already present in the
untrained model and does not change with training, so it is a property of
the item sets, not evidence of anything learned. Generated length never
shrinks (97 tokens per attempt throughout). In plain terms: the models got
good at executing the recipe and never once skipped it.

**(c) The update schedule does not change the endpoint on the task.**
Per-episode vs batched-every-50: last-three held-out 0.889 vs 0.894 vs
0.917 (with variants / without), paired differences 0.005–0.023, all noise.
The batched regimes lag by roughly one night mid-run (block-6 day accuracy
0.70 / 0.76 vs 0.84) and their held-out curves are smoother (max drawdown
0.09–0.11 vs 0.18).

**(d) Training on the puzzle moved the GSM8K score, a lot, fast, and in a
regime-dependent direction.** Per-episode: +0.08 after the first 50 steps,
+0.155 at the end, positive in 5 of 5 seeds (+0.10 to +0.19). Batched with
variants: −0.06 after the first night, a trough around −0.05 through night
5, then partial recovery to +0.03 (range −0.08 to +0.19). Batched without
variants: +0.06 after the first night, +0.08 at the end (−0.03 to +0.24).
These are 300-item scores (standard error ±0.028), so single-checkpoint
wiggles of ±0.05 are noise, but the per-episode rise and the early trough
of the variant regime are not.

**(e) Self-critique without external feedback did nothing.** 0.410 vs 0.405
day accuracy at 1.9× tokens; accuracy by number of rounds 0.40 / 0.41 / 0.42.

**(f) The synthetic variants added nothing measurable and are low quality.**
Yield 14% → 32% across nights; ~40% of proposals were exact duplicates of
their source; by the last nights variants were ~40% of each night's new
training items. Regime 4 vs 5: task −0.023, GSM8K −0.047, neither
significant. Zero variants touched a held-out or masked-item prefix.

## 3. What a skeptical reader would press on

1. **The hidden-digit probe has no positive control.** Nothing in the record
   shows that a model which *does* know the coincidence passes this probe.
   The probe also changes the prompt format (the example loses its steps,
   the instruction asks for a guess), so a failure conflates "does not know
   the regularity" with "cannot express it in a format it has never seen".
   The answer-only gap is a partial second instrument and agrees, but a
   30-minute check (state the regularity in the prompt, or fine-tune 50
   steps on masked→answer pairs, and see whether the probe goes to ~1.0)
   would have made the null interpretable. Without it, "the probe never
   moved" is solid; "no regime induced the regularity" is an inference.
2. **The training signal never rewards the shortcut.** Every trained regime
   fine-tunes on full correct chains. That reinforces chain execution and is
   agnostic to, or mildly against, skipping it. A reader unfamiliar with the
   framing would take "no shortcut emerged" as the expected outcome, not a
   finding. What would have been informative is a regime that could benefit
   from the shortcut (a length penalty, answer-only training, or a reward
   for early answers) as a positive control for discoverability.
3. **The GSM8K effect is the most interesting number and the least
   diagnosable.** A +0.08 change after 50 LoRA steps of batch size 1 on a
   digit puzzle is too fast to be a change in mathematical ability. The
   parsimonious candidates are behavioural: answer formatting (the puzzle
   training drills a terminal `ANSWER:` line, which the GSM8K scorer also
   looks for first), verbosity and truncation within the 512-token budget,
   and hedging. None of this can be checked because the control-benchmark
   outputs were not saved, only the score and the unparsable rate (0%
   throughout, which rules out the crudest version of the format story but
   not truncation or last-number extraction errors). This is the single
   biggest instrumentation gap in the study.
4. **The early trough is specific to the variant regime.** Regime 5 rises
   after night 1; regime 4 falls. The two differ only in the presence of
   synthetic variants in the night's train set (both have replay and the 5%
   decay). Early nights train 50 steps on ~36 items, ≈1.4 passes; in regime
   4 roughly 40% of those items are near-duplicate edits of each other. The
   most concrete lead in the data is therefore: *near-duplicate synthetic
   data in small early batches damaged an unrelated benchmark while doing
   nothing for the task.* Five seeds cannot confirm it (Δ −0.047, p=0.375).
5. **Regime confounds.** The batched regimes carry two extra interventions
   that the per-episode regime lacks, 1:1 replay and a 5% LoRA shrink each
   night, so "batched vs online" is really "batched + replay + decay vs
   online". Optimizer moments also evolve on different timescales. None of
   this matters for the task endpoint (identical) but it does for the
   control-benchmark story.
6. **Heterogeneous seeds.** Seed 0's trained arms are earlier code; seed 1
   mixes GPUs. Restricting to seeds 2–4 changes nothing qualitative
   (per-episode GSM8K +0.153, batched +0.059 / +0.061; held-out
   0.90–0.91 across the three), which is reassuring, but the pooled n=5
   tests are more generous than the data strictly earn.
7. **Evaluation noise vs claimed effects.** 60-item probe: ±0.06 per
   checkpoint, so the 0.70 criterion needs a true rate near 0.55+ to be
   passable and a single 0.47 reading (seed 1) is unremarkable. 201-item
   held-out at 0.9: ±0.02. 300-item GSM8K: ±0.03. Several per-seed
   statements in the project narrative (e.g. individual seeds "regressing")
   are inside these bands.
8. **Generalisation is within-distribution.** Held-out puzzles are drawn
   from the same structured subset as training. Whether the trained model
   handles arbitrary 12-digit strings equally well is not reported as an
   accuracy (the unstructured items were used only for error-bias
   statistics). It probably does, since the errors are digit-tracking not
   rule errors, but it is an inference.
9. **Duration.** 600 steps at lr 1e-4, batch 1, r=16 is a light touch. "No
   regularity emerged in 600 steps" is a much weaker statement than "cannot
   emerge", and the human analogue this design echoes involves far more
   trials.

## 4. What is worth something, framing aside

- **A clean demonstration that verified self-distillation fixes an
  execution-limited skill without inducing any abstraction of it**, with the
  abstraction measured three ways (masked items, answer-only accuracy,
  output length) and all three flat over 15 runs × 13 checkpoints. That is a
  crisp, if unglamorous, negative about what this kind of training does.
- **The transfer of digit-puzzle fine-tuning to GSM8K**, if it survives
  output inspection and a broader benchmark, is the only surprising
  positive result. Its regime dependence (steady per-episode updates help;
  small batched updates with near-duplicate synthetic data hurt early) is a
  practical continual-learning observation.
- **Batched consolidation buys smoothness, not endpoints**, and its early
  data-poor phase is where its costs live. If anyone builds the sleep idea
  again, the nights should scale their step count to the buffer and skip
  the first few.
- **The harness itself**: matched compute, resumable runs, per-item rows,
  full trajectories, and an audit trail that let every headline number be
  recomputed from raw rows. That is reusable and rarer than it should be.

## 5. What I would run next, cheapest first

1. **Re-score GSM8K with saved outputs** for the base model and the 13
   adapters, same prompt, same GPU (local or ~$3 on an L4). Then read the
   outputs: answer-tag presence, length, truncation, last-number
   extraction. This decides whether (d) is a reasoning effect or a format
   effect before anything else is built on it.
2. **Probe positive control** (~30 GPU-minutes): (i) base model with the
   regularity stated in the prompt; (ii) 50-step LoRA on masked→answer
   pairs. If neither reaches high probe accuracy, the probe was never able
   to detect what it was built for and the H1–H3 nulls say less than they
   appear to.
3. **Unstructured full-chain accuracy** for the 13 adapters (local, free):
   settles the generalisation question in §3.8.
4. **Broader retention panel** (full GSM8K, ARC-Challenge, an MMLU sample)
   on base + adapters, one GPU type (~$10): turns (d) from one benchmark
   into a profile.
5. Only after 1–4: a redesigned batched regime (steps ∝ buffer, no
   variants, no decay confound) if the retention story still looks real.

## 6. Bottom line

Stripped of its framing, this is a well-instrumented, compute-matched study
of rejection-sampled self-distillation on a synthetic algorithmic task with
a 4B model. It shows that such training reliably repairs an
execution-limited skill within a few hundred steps, that it induces no
abstraction of the task by any of three measures, that batching the updates
changes the path but not the destination, that self-generated variants
were low-diversity filler, and that the training has a fast, regime-
dependent side effect on an unrelated math benchmark that the study was
not instrumented to explain. The hypothesis-driven reading ("sleep did not
produce insight") is true but thin, because no regime was given a reason
to find the shortcut and the probe was never shown to detect it. The
empirically interesting thread is the GSM8K transfer and its dependence on
how the updates were scheduled and what was mixed into them; that is where
a cheap next experiment would pay off.

## Appendix A. Independent reading by a fresh session (no access to this conversation)

*Produced by a separate session given only the repository and the records folder, with instructions to analyse the experiment without regard to its stated hypotheses. Reproduced verbatim; the reconciliation follows in Appendix B.*

### What was actually done

**Task.** "Number Reduction": a 12-digit string over {1,4,9}; combine left-to-right with a 3-element rule (same→same, different→third). The answer is the 11th intermediate result. Every training/eval instance is constructed so r9=r8, r10=r7, r11=r6 ("mirror" structure), so the answer equals r6. Splits are by 7-digit prefix (801 train / 201 held-out / 60 probe, answer-stratified). A "probe" masks digits 8–12 and asks for the final answer in answer-only format (64 tokens); its target is r6.

**Model & regimes.** Qwen3-4B bf16, LoRA r=16/α=32 on all projections, Adam lr 1e-4, one persistent optimizer per run, single-example steps. Five regimes, 600 episodes each, evaluated at episode 0 and every 50: *Baseline* (frozen, one attempt); *Awake* (frozen; critique-and-revise rounds, token-matched to Sleep — in practice two rounds on 77% of attempts); *Online* (one SFT step after every episode on a verified own trajectory: correct answer and ≥9/11 steps); *Sleep* (no updates by day; every 50 episodes 50 steps on that day's kept trajectories + model-generated "dream" variants verified by the solver + a 1:1 replay draw, then multiply `lora_B` by 0.95); *Sleep-NoDream* (same without dreams). Steps and training tokens matched to ±5%. Measures: held-out accuracy (201 items, work format), probe accuracy (60), GSM8K exact match (300, 512-token cap), and post-pilot secondary metrics (answer-only accuracy on 120 structured/120 unstructured items, mirror bias, etc.). Five seeds; seed 0's Sleep/Online/Awake ran pilot code (no dream de-duplication, non-deterministic CUDA, no per-item rows), seed 1 mixed L4/A16 GPUs.

### 1. Robust effects, ranked

1. **All three LoRA regimes learn the task; Baseline and Awake do not.** Final held-out: Online 0.862±0.115, Sleep 0.902±0.060, NoDream 0.921±0.054 vs Baseline 0.421±0.039, Awake 0.419±0.023 (5 seeds each; every trained-minus-frozen paired difference is positive in 5/5 seeds, exact sign-flip p=0.0625, the minimum attainable). Awake−Baseline: −0.002 (5 seeds; 0.399/0.412/0.416 accuracy by number of critique rounds). Nothing separates the three trained arms: last-3-checkpoint Sleep−Online +0.005, NoDream−Sleep +0.023, Online−NoDream −0.028 (all p≥0.5).

2. **What was learned is digit tracking, not the rule.** I split each attempt's work lines into rule-application errors (`a,b->c` with wrong `c`) and tracking errors (operand sequence ≠ the digits, or chain misaligned). Pooled over seeds, episodes 1–100: rule-only errors are 5–13 of 500 attempts; tracking errors 330–413. Episodes 501–600 in the trained arms: rule errors 0–2, tracking errors 114–159, and 191 of 194 single-digit skips sit next to an equal digit. The base model already knows the rule; fine-tuning taught it to keep its place in runs like `9999`. Per-step held-out accuracy at ep600 still degrades left to right (step 1 = 1.00 → step 11 = 0.69–0.80).

3. **The probe never moved, in any arm.** Pooled over checkpoints 1–12: Online 1150/3600 = 0.319, Sleep 1209/3600 = 0.336, NoDream 1139/3600 = 0.316; final-checkpoint pooling 0.33–0.34 (n=300 each); Baseline 0.340, Awake 0.310 (n=300). The 180 trained-arm checkpoint values have SD 0.055 against a binomial SD of 0.061 at n=60 — i.e., they look like pure noise. Mirror bias on unstructured items at ep600 is at or below its chance level (Online 0.329 vs 0.323; Sleep 0.282 vs 0.339; NoDream 0.303 vs 0.341).

4. **GSM8K rose under Online in 5/5 seeds** (Δ final = +0.137, +0.177, +0.097, +0.193, +0.170; mean +0.155; mean over ck1–12 +0.133) and was mixed under Sleep (+0.031 mean; 3/5 positive) and NoDream (+0.079; 4/5). Sleep−Online −0.123 (p=0.125). Restricting to the clean seeds 2–4 gives the same picture.

5. **A Sleep-specific first-night dip on GSM8K.** After the first 50 steps, Sleep is down in 5/5 seeds (−0.06, −0.04, −0.09, −0.08, −0.05), NoDream up in 4/5 (+0.15, +0.02, +0.08, −0.08, +0.12), Online up in 3/5. The Sleep−NoDream twin difference at ck1 is −0.21/−0.06/−0.16/0.00/−0.17 on GSM8K and −0.10/−0.05/−0.08/+0.05/−0.09 on held-out, though the two runs differ at that point only by 2–4 dream examples in a ~12-item train set plus the RNG path. Sleep's GSM8K is below its own ep0 through ck1–4 in 5/5 seeds and recovers later (ck9–12 mean +0.044).

6. **Dreams are inert.** Yield 22–39% of proposals; 40–50% rejected as verbatim copies; 2–8 structured dreams per seed out of 163–267 accepted (seeds 1–4); ~45–53k extra generated tokens per seed for no measurable task or control effect.

### 2. What deserves a skeptic's scrutiny

**The "forgetting" metric is almost certainly measuring response length.** The per-checkpoint `eval_tokens` field, minus the (constant) held-out/probe/implicit share, gives an estimate of GSM8K tokens per item. Across the 156 trained-arm checkpoints of seeds 1–4 this correlates r = 0.91 (r² = 0.83) with GSM8K accuracy; within-arm r is 0.73 (Online), 0.96 (Sleep), 0.89 (NoDream); slope ≈ +0.09 accuracy per +10 tokens/item. Baseline sits at ~99 tokens/item and 0.58, Online at ~109 and 0.72, Sleep runs with 0.50 at ~77 tokens. The system prompt says "think step by step *briefly*", and 0.58 is far below what Qwen3-4B normally scores on GSM8K, so the ep-0 number looks like an under-reasoning artefact and the "gains" and "losses" look like shifts in how much the model writes before answering. Whether it lengthens or shortens GSM8K answers is sensitive to tiny run differences (point 5 above). None of this can be confirmed because no GSM8K outputs were saved — only accuracy and an unparsable fraction that is 0 everywhere (the parser falls back to "last number in text", so a truncated or unfinished answer is scored wrong, not unparsable).

**The probe has no demonstrated sensitivity, and the design's own secondary metric shows why.** Answering the probe requires computing r6 through six comparisons *without writing work* and then applying r11=r6. But answer-only accuracy on **unmasked, fully visible** structured items never leaves chance for any model: Baseline 0.357, and the trained arms 0.354/0.381/0.310 at ep600 (max at any checkpoint 0.44–0.47, n=120). A model with perfect explicit knowledge of the shortcut would still score ~1/3 on this probe. There is no positive control (e.g., a model trained on `masked prompt → r6`), no probe texts, and no logit-level test. The probe null is therefore not evidence against shortcut discovery; it is evidence that the instrument cannot measure it. (Copy heuristics like "answer the last visible digit" also give exactly 1/3 on r6, so chance is well specified — that part is fine.)

**The training objective never rewards the shortcut.** All updates are SFT on complete 11-line chains; tokens per attempt are fixed by format (97–98 throughout), so the preregistered "token collapse" signal could not fire by construction. In Wagner et al. the payoff was speed; here there is none.

**Regime confounds.** Only the Sleep arms apply the 0.95 shrink of `lora_B` (0.95¹² ≈ 0.54 cumulative on early components); a weight-shrinkage story for "closer to base on GSM8K" (Sleep's ck1–12 mean GSM8K 0.578 ≈ base 0.583 vs Online 0.717) is indistinguishable from a schedule story. `cycle_examples` truncates each night to 50 presentations, so after night ~4 a night sees a random 50 of a 60–190-item pool and the 1:1 replay/dream interleave is largely notional. Kept targets are noisy: 33–53% of block-1 kept trajectories contain 1–2 wrong intermediate steps, still 15–22% in block 12. Online's single-example steps produce large swings (seed 0 final 0.667 after 0.965; seed 3 0.950→0.766→0.940), far beyond the 201-item SE (~0.02–0.035).

**Noise.** 300-item GSM8K SE ≈ 0.028 but observed checkpoint-to-checkpoint SD is 0.05–0.06, so run-level chaos dominates item noise; effective n is seeds. Seed 1's ep-0 held-out differs across GPU types by 3/201 items with greedy decoding. With n=5 the sign-flip test can only ever reach p=0.0625.

### 3. Design critiques and claim calibration

Missing: saved GSM8K and probe texts (or at least answer distributions and lengths); a probe positive control; a log-prob probe (P(r6) vs alternatives under the masked prompt, teacher-forced); an "insight-eliciting" condition where the shortcut is rewarded (answer-only or length-penalised format); adapter norms per checkpoint; the optional Online-Unfiltered arm; and more seeds (the authors' own power estimate for H4 is ~9).

Relative to CONTEXT §5k/§5l: "Nobody found the shortcut" is over-claimed — the correct statement is "the shortcut, if learned, is undetectable with this probe and this training format". "Schedule matters for the control benchmark" is over-claimed: NoDream shares the schedule and decay yet shows no first-night dip, and the whole GSM8K axis is confounded by verbosity. "The probe null is tight" is arithmetically right and inferentially empty. Under-claimed: the tracking-versus-rule decomposition (the clearest mechanistic result in the data), and the striking length–accuracy coupling on GSM8K, which turns H4 into a different and more interesting question about behavioural transfer of "show your work".

### 4. What is interesting regardless of framing; follow-ups, cheapest first

Interesting: (i) 600 single-example LoRA steps of verified self-training suffice to fix a 4B model's digit-tracking failure on 12-symbol strings, identically whether interleaved or batched; (ii) inference-time critique at 2× tokens does nothing on this failure mode; (iii) narrow SFT on terse step-by-step arithmetic changes how long the model reasons on an unrelated benchmark, and accuracy tracks that length almost perfectly — a clean, cheap-to-study case of format transfer.

Follow-ups: (1) Re-score the base model on the 300 GSM8K items with "briefly" removed and with 1024 tokens; if it lands near 0.75, H4 is closed as an artefact (CPU/MPS, an hour). (2) Load the 13 archived adapters and re-run GSM8K saving texts; regress accuracy on length directly. (3) Probe positive control and log-prob probe with the same adapters; also a masked prompt with work allowed ("compute as far as visible, then guess"). (4) Train one adapter on answer-only structured items (the only condition where the shortcut pays) and see whether the probe or unstructured-vs-structured gap moves. (5) Only then revisit schedules, with NoDream and Online run at a matched shrink schedule.

### 5. Bottom line

Across 25 runs, the data robustly show one thing: verified self-training with ~600 LoRA steps takes Qwen3-4B from 0.42 to ~0.9 on this task by fixing digit-position tracking (rule errors were already rare), and it does so equally under continuous or batched updating, while inference-time critique is inert. Nothing in the data speaks to "insight": the training format never rewards the shortcut and the masked probe would return chance even for a model that knew it, since answer-only accuracy on fully visible items also stays at 1/3. The GSM8K "retention" result is confounded by response length (r = 0.91 with a token-count proxy) and by a Sleep-only first-night dip that NoDream does not share, so neither the direction nor the interpretation of H4 should be carried forward. This is a well-executed null on its own hypotheses, a useful negative result on Awake-style critique, and an accidental positive lead on format transfer — worth reporting as such, with the probe-validity and verbosity caveats stated up front.

## Appendix B. Reconciling the two readings

The two readings were written without contact. They agree on the ranking of
effects (self-training fixes execution; nothing transfers to answer-only or
masked items; schedule changes the path not the endpoint; critique inert;
variants inert). The independent reading goes further in three places, and
each was checked against the run files before being accepted here.

**B1. GSM8K accuracy tracks how much the model writes (verified).**
`eval_tokens` per checkpoint, divided by the 300 control items, is a proxy
for GSM8K response length because the other evaluation components are
fixed-format. Over the 156 trained-arm checkpoints of seeds 1–4 it
correlates r = 0.908 with GSM8K accuracy (Online 0.71, Sleep 0.96, NoDream
0.89; slope ≈ +0.09 accuracy per +10 tokens per item). The base model's
0.58 under a "think briefly" system prompt with a 512-token cap is well
below Qwen3-4B's usual GSM8K level, which fits an under-reasoning baseline
whose score moves with verbosity. Nothing in the record can separate
"better arithmetic" from "writes more before answering"; the outputs were
not saved. **Consequence:** the H4 result and the "retention" narrative in
§5k/§5l should be read as a statement about response-length transfer until
the outputs are re-scored, not about forgetting.

**B2. The masked probe could not have detected the shortcut (accepted).**
The probe demands an answer with no working. Answer-only accuracy on fully
visible structured puzzles is 0.31–0.39 for every regime at every
checkpoint, so no model in this study can compute six steps in its head; a
model that knew r11 = r6 perfectly would still have to compute r6 silently
and would still score ~1/3. The pooled probe null is therefore arithmetic
without inference: it bounds the *probe rate*, not shortcut knowledge. The
first reading called this a missing positive control; the second shows the
control would have failed. **Consequence:** §5k's "nobody found the
shortcut" should read "the instrument could not have shown it if they had".
The auxiliary measures (mirror bias, output length) remain valid and are
also flat, so the weaker statement "no sign of shortcut use in the written
chains" stands.

**B3. The first-night dip is Sleep-only and the interleave is partly
notional (verified).** After the first night, Sleep's GSM8K falls in 5/5
seeds (−0.04 to −0.09), NoDream rises in 4/5, Online rises in 3/5. The
twins differ at that point only by a handful of dream examples in a ~12-item
train set and the RNG path, which is too little to carry a causal story at
n=5; combined with B1 it may be a verbosity swing. Separately,
`cycle_examples` presents exactly 50 examples per night, so once the night's
train set exceeds 50 items (from about night 4) each night sees a random
subset and the 1:1 replay/new interleave is a property of the pool, not of
what the model was trained on. This does not affect matching (steps and
tokens are counted as executed) but it weakens the design description in
§5k.

**Where the readings differ in emphasis.** The first reading treats the
early-night trough as the most actionable lead; the second treats it as
likely an artefact of B1. Given r = 0.91, the second is the safer prior.
Both agree the cheapest decisive experiments are the same: re-score GSM8K
with saved outputs (base + 13 adapters, a "briefly"-free prompt and a
longer cap as a second condition), and a probe positive control including a
teacher-forced log-probability version that does not require silent
computation.

**Net effect on the project's written conclusions.** H1–H3 remain "not
supported" but the reason changes from "no insight occurred" to "the study
could not have observed it"; H4 remains "not supported" but the reversed
direction should not be carried forward as a retention finding. The robust,
reportable results are: verified self-distillation repairs digit tracking
in ~200–400 steps regardless of schedule; inference-time critique without
feedback does nothing; self-generated variants added nothing; and narrow
SFT shifts the verbosity, and thereby the scored accuracy, of an unrelated
benchmark in a schedule-dependent way.
