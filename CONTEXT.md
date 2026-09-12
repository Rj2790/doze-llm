# doze-llm — project context

Handoff document. Everything decided so far, why it was decided, and what is
next. Written 2026-09-02. Keep this file in the repo root and update it when
decisions change; it is the memory that survives between sessions.

---

## 1. The question

Does giving an LLM a *sleep cycle* — alternating a **day** phase (attempting
problems with fixed weights) with a **night** phase (offline replay, verified
self-generated "dreams", and a small LoRA weight update) — produce

(a) faster discovery of hidden task structure ("insight"), and
(b) less forgetting of unrelated capability,

than spending the **same compute** on either (i) extended inference-time
reasoning or (ii) continuous online weight updates?

The biological analogy motivates the design but is not the claim. Human sleep
does roughly four things: replays episodes and consolidates them, prunes
synapses back down, clears waste, and does offline recombination that
produces measurable insight (Wagner et al. 2004: people were ~3x more likely
to discover a hidden shortcut in the Number Reduction Task after sleep).
Fragments of each exist for LLMs — experience replay in RL, memory systems
that consolidate conversation into notes, wake-sleep (Hinton 1995),
distillation/pruning at deployment — but nobody has assembled them into a
scheduled phase and *measured* whether the phase structure itself matters.

**The load-bearing comparison is Sleep vs Online at matched gradient budget.**
Beating a frozen baseline only shows fine-tuning works. Beating continuous
updates shows the phase structure does something.

## 2. Literature check (done 2026-09-02)

- **Letta, "Sleep-time compute" (arXiv 2504.13171, Apr 2025).** Offline
  pre-processing of *context* with frozen weights to cut test-time cost. Not
  what we are building; it is the "diary" track. Cite, don't replicate.
- **"Let Them Sleep: Adaptive LLM Agents via a Sleep Cycle" (Medium,
  ~Dec 2025).** Proposes day/night with LoRA overlay on a frozen base. An
  architecture proposal, apparently no experiments. Idea is in the air;
  nobody has measured it.
- **Replay + LoRA for forgetting is well-trodden**: ERI-LoRA (2025), SSR
  self-synthesized rehearsal (2024), FOREVER (2026), Hybrid-CASR (2026),
  etc. "Replay prevents forgetting" is not a finding anyone will care about.
- **Therefore the novelty lives in two things**: phased vs continuous
  updates at matched compute (H2), and the insight effect on hidden-structure
  tasks (H1). Keep the design pointed at exactly those.

## 3. Design (frozen in PREREG.md — read that file for the authoritative version)

### Hypotheses
- H1 insight: Sleep reaches the shortcut criterion in fewer episodes than
  Baseline, Awake, Online.
- H2 phase structure: Sleep beats Online at matched gradient budget.
- H3 dreaming: Sleep beats Sleep-NoDream at matched gradient budget.
- H4 retention: Sleep forgets less than Online on the control benchmark.
All reported regardless of outcome. A null on H2 is publishable.

### Arms (same instance sequence per seed, same checkpoints every K=50 episodes)
| Arm | Weights | Compute | Role |
|---|---|---|---|
| Baseline | frozen | 1 attempt, work format | floor |
| Awake | frozen | matched **token** budget on extra CoT / self-critique | inference-time control |
| Online | LoRA | 1 gradient step after every episode on its own trajectory | continuous-update control |
| Sleep | LoRA | 0 by day; every K episodes: filter → dream → replay-interleave → S steps → weight-decay pass | treatment |
| Sleep-NoDream | LoRA | as Sleep, no synthetic generation | ablation |

Compute matching: Awake tokens = Sleep tokens (day + dreams) ±5%; Online
gradient steps and training tokens = Sleep's ±5%. `eval/compute_ledger.py`
logs both and the analysis refuses to compare arms outside tolerance.

Night procedure: keep correct and near-miss (≥9/11 steps) trajectories;
dream = prompt current model for 2 variations per kept trajectory, keep only
those that verify against the ground-truth solver; interleave 1:1 with a
uniform sample from the replay buffer of all prior nights; train LoRA for S
steps; one weight-decay pass on LoRA params (pruning analogue); append to
buffer.

### Primary task: Number Reduction, L=12
Digits over {1,4,9}. Rule: same→same, different→third digit. Applied
left-to-right giving r1..r11; answer is r11. Instances constructed so the
last three responses mirror the preceding three: **r11 = r6**, r10 = r7,
r9 = r8. The shortcut needs only the first 7 digits.

Why 12 not Wagner's 8: structured instances number 3^(L−3). At L=8 that is
243 with 27 three-digit prefixes each mapping to one answer — a lookup table
a 4B model memorises in one night. At L=12: 19,683 instances, 2,187 prefixes.

**Splits are by prefix.** 20% of prefixes (437) held out; no held-out prefix
appears in training. Probe items come only from held-out prefixes, so
passing the probe requires the rule r11 = r6, not a memorised map.
Train 801, held-out eval 201, probe 60 (disjoint from eval); each sampled
stratified by answer so exactly uniform (A6, C7; 800/200 are not multiples
of 3).

**Probe**: digits 8–12 masked. Literal rule reaches only r6; answering r11
above chance (1/3) requires the shortcut. Checkpoints at episode 0 and every
K (A5). Scripted tests confirm a literal
solver sits at chance and a shortcut solver scores 1.0.

### Secondary task: string grammar with a hidden invariant (not yet written).
### Control benchmark: fixed 300-item GSM8K test sample, exact match, never trained on.

### Metrics
- Shortcut criterion (primary): probe accuracy ≥ 0.70 on two consecutive
  checkpoints (60 items each). Report episode of first hit, censored.
- Secondary insight signal: median tokens per correct held-out answer
  (collapse = stopped simulating the chain).
- Held-out accuracy; control-benchmark delta; tokens, grad steps, GPU-s.

### Analysis
5 seeds per arm. Log-rank test on episode-to-criterion (censoring handled).
Paired comparison for forgetting. p<0.05 in predicted direction on the
primary task; secondary task is replication, not pooled. 600 episodes (12
nights) per run or until all arms meet criterion. Deviations go in
`DEVIATIONS.md`.

## 4. Decisions log (with reasons)

**Model: Qwen3-4B, bf16, LoRA r=16 on all attention+MLP projections,
thinking OFF.** Strongest small family, Apache 2.0, dense, boring PEFT path.
Thinking off so all reasoning is in visible, countable tokens. Avoid Qwen 3.5
(hybrid Mamba/attention complicates LoRA). Gemma 4 E4B is the alternate but
its licence is more restrictive, which matters for open-sourcing.

**Start at 4B, don't start big.** Small first because: ceiling effects (a
70B may spot the rule in-context by episode 3, leaving Sleep nothing to add);
statistics (5 seeds at 4B vs 1 at 70B); forgetting dynamics (bigger models
forget less per update, so the forgetting metric loses discrimination).
Plan: full protocol at 4B → sweep 1.7B/4B/8B with same seeds → if trend is
clean, one Qwen3-32B headline run (fits one 80GB GPU with LoRA bf16). Whether
the Sleep advantage shrinks or grows with scale is the most interesting
possible result. Stated limitation: LoRA on 4B ≠ full FT on a frontier model.

**Infra: MacBook M3 Pro 18GB for development; Modal for the seeded grid.**
4B bf16 (~8GB) fits for inference and LoRA training at batch 2–4. But the
full grid (5 arms × 5 seeds + ablation, Awake generating several× the
tokens) is over a week of laptop time vs an afternoon on Modal for tens of
dollars. bf16 throughput observed locally: ~8 s per ~100-token work-mode
completion. **All reported numbers come from the cloud backend
(Transformers+PEFT).** MLX and PEFT have different LoRA numerics; never mix
backends in a plot. Local 4-bit is fine for iteration only.

**Backend abstraction.** `backends/base.py` Protocol (`generate`,
`count_tokens`; training methods added when arms land). `backends/scripted.py`
model-free solvers (literal / shortcut, with noise) for harness tests.
`backends/mlx_backend.py` for the Mac.

**Prompt format: `work` mode** (one line per comparison `prev,digit->result`,
then `STEPS:` and `ANSWER:`), with a system prompt forbidding preamble and a
6-digit worked example (too short to carry the mirror structure, so it cannot
leak). History: compact format at max_tokens=96 → 100% unparsable (model
wrote Markdown preamble and got cut off); compact format at 400 tokens →
30% = chance, 33/40 chains wrong at step 1–2, model emitting eleven
plausible digits without computing; `work` mode → 47.5%, 7/40 chains exact,
errors spread over steps 3–11. **Prompt wording is tunable; task construction,
splits and probe target are frozen.**

**4-bit vs bf16:** no difference in compact mode (both at floor). Precision
was not the problem; the missing scratchpad was. Develop in bf16 (primary
precision anyway).

**Remaining failure mode is positional, not rule errors**: the model skips a
repeated digit or misreads one in runs of identical digits. This is noise
unrelated to the hypothesis and it corrupts r6 when it happens early. The
numbered-digit variant (`1:1 2:4 3:9 ...`) was tested 2026-09-02 (run 6):
full_acc 0.525 but steps_exact 0.00 and first errors moved to steps 1–3
(21/40) — the model copies position labels into its work lines and starts
misapplying the rule itself. Outside the 0.60–0.80 acceptance band and
worse chain quality than plain work mode. **Rejected. Final format: work
mode, unnumbered, 400-token budget.** The `--number-digits` flag stays in
the code as a documented negative.

**Probe statistics fixed (2026-09-02).** Probe items are now sampled
stratified by target (n/3 per digit; `eval/shortcut_detector.py:
stratified_probe_items`) and `above_chance` is a one-sided exact binomial
test vs 1/3 at p<0.01. Re-run of bf16 work mode with n_probe=120: 0.325,
p=0.61 — the earlier 0.50 was the artifact predicted here. The untrained
model has no shortcut. The PREREG criterion (0.70, two consecutive) is
unchanged. The old fixed-margin test had also been hiding a lucky random
guesser (53/120) in the model-free test suite.

**Doc fix done:** PREREG §6.1 now reads "digits 8–12 masked (see §4.1)".

**Online arm, decided (A2, 2026-09-02).** One gradient step after every
episode on a *kept* trajectory: the current one if kept, else a random one
from today's kept pool, else the most recently kept ever, else skip and
log. Steps therefore equal K per K episodes once anything has been kept.
The literal reading (train on every trajectory as written) survives as the
optional arm `online_unfiltered`, matched to Sleep like Online.

**Keep rule (C3).** Correct final answer AND ≥ 9/11 steps correct, shared
by Sleep's night filter, Online and (stricter: fully correct) the dreamer.
Wrong-answer near-misses are out.

**Frozen arms are re-evaluated at every checkpoint (decided 2026-09-03).**
Baseline and Awake have frozen weights and greedy decoding, so their 13
checkpoints should be identical (Awake ep 0 == ep 50 exactly). Skipping the
repeats would save ~5 h / ~$4.4 per Awake run and ~$20 across the grid,
but was rejected on output quality: measured flatness is evidence, copied
flatness is an assumption, eval determinism has only been checked twice and
can break across GPU types (GSM8K 172 vs 176/300 on A100 vs L4), and the
frozen arms are the controls for H1. `eval/analyze.py` now reports
`frozen_eval_identical` per frozen arm and seed as an eval-pipeline
validation.

**Dream instruction (tunable wording, 2026-09-02).** Dreams keep the
first seven digits of the source and change 1–3 of the last five; dreams
whose prefix differs from the source are rejected (`prefix_changed`). Held-
out rejections are impossible by construction (source = training
instance). Digit-7 boundary: digits 1–7 fix r6 (the shortcut target), so
these dreams vary r11 while r6 stays put; most accepted dreams are
unstructured strings solved literally. Logged per night: generated,
rejected-wrong, rejected-heldout, rejected-prefix-changed, accepted, and
the fraction of accepted dreams that are mirror-structured. Dream
verification = shared C3 keep rule (`filters.keep_scores`), aligned
2026-09-02 (it had been an exact-chain check; DEVIATIONS.md doc note).

**Dreams are never replayed (C2).** The replay buffer holds kept real
trajectories only; dreams are trained on in the night they were generated.

**Seeding (A1).** `backend.set_seed(seed)` is called by the harness; MLX
seeds `mx.random` (LoRA init and sampling), HF seeds torch/CUDA/random.
Scripted same-seed reproducibility is tested; the MLX 4-episode check
(§5) is bit-identical.

**Analysis gate (A3).** `eval/analyze.py` loads ledgers with
`Ledger.load`, runs `check_matched`, writes the verdict into every run JSON
of that seed (`"matched"`) and raises before producing any table or figure
if arms are unmatched or from different backends.

**Compute matching, as implemented.** Sleep trains S = K = 50 steps per
night (batch 1), so gradient steps match Online (1/episode) by
construction. Training tokens match only if day trajectories and dreams
have similar lengths; same format, same L, so they should, but this is
checked post hoc by `check_matched`, not forced. Awake accrues its token
budget cumulatively across day episodes (overshoot in one episode is
repaid in the next) so its total lands within one completion of Sleep's.
Evaluation tokens (checkpoint probe / held-out / control) are recorded
separately and are not part of the matched quantity; Awake uses its
critique loop at eval too, with a probe-specific critique prompt at probe
time (C4), otherwise it would be identical to Baseline on every metric.

**Dreams are verified, nothing more.** A dream enters training only if
every work line, every STEPS entry and the ANSWER agree with the solver,
and its prefix is not held out. We deliberately do NOT filter dreams for
the mirror structure: that would inject the experimenter's knowledge of
the hidden rule into the treatment.

**Weight-decay pass** (night step 6) shrinks lora_b by 5% (`weight_decay`
in `SleepConfig`, tunable). Kept near-misses (correct answer, ≤ 2 wrong
steps) are trained on as the model wrote them.

**Open-sourcing plan.** Don't design the package before there is a result.
If the weights track shows signal → a library (`sleep` scheduler wrapping any
open-weight fine-tuning setup; replay buffer, dream generator, forgetting
monitor as swappable parts). The text-memory track (Letta-style, works with
closed models too) → a connector/MCP server, worth shipping regardless. Two
artifacts, two audiences. Write up either way; preregistering publicly is
what makes a positive result credible.

## 5. Calibration record (untrained Qwen3-4B, L=12, n=40)

| run | precision | mode | n_probe | full_acc | steps_exact | unparsable | median tok | probe (p) |
|---|---|---|---|---|---|---|---|---|
| 1 | 4-bit | compact, 96 tok | 40 | 0.00 | 0.00 | 1.00 | — | 0.00 (cut off) |
| 2 | 4-bit | compact, 400 tok | 40 | 0.30 | 0.00 | 0.00 | 31 | 0.375 |
| 3 | bf16 | compact, 400 tok | 40 | 0.30 | 0.00 | 0.05 | 31 | 0.50 (artifact) |
| 4 | bf16 | **work**, 400 tok | 40 | **0.475** | **0.175** | 0.00 | 97 | 0.50 (artifact) |
| 5 | bf16 | work, 400 tok | 120 strat. | 0.475 | 0.175 | 0.00 | 97 | **0.325 (p=0.61)** |
| 6 | bf16 | work + numbered | 120 strat. | 0.525 | 0.000 | 0.00 | 97 | 0.333 (p=0.53) |

Run 4/5 first-error histogram: {3:4, 4:3, 5:6, 6:3, 7:6, 8:1, 9:6, 10:2, 11:1}.
Run 6: {1:7, 2:5, 3:9, 4:5, 5:5, 6:3, 7:3, 9:2}.
Runs 1–4 used the first 40 held-out items for the probe (targets 17/15/8);
runs 5–6 use the stratified sampler. Files: `results/calibration*.json`.
Verdict: in the learnable regime; no pretrained shortcut; format decision
made (work, unnumbered).

**Calibration items predate A6 (2026-09-02; PREREG Appendix A addendum,
DEVIATIONS.md doc correction).** Runs 1–6 used the first 40 held-out items
of the pre-A6 split. The frozen split's first 40 held-out items (no overlap)
give 0.25 / 0.075 exact chains with the same model, prompt and decoding
(`results/agreement_mlx_bf16_seed0.json`). Re-running the pre-A6 items on
the frozen code reproduces 0.475 / 0.175 exactly
(`results/agreement_mlx_bf16_preA6_items.json`), so the Baseline path is
unchanged and the gap is item-sample variance (two 40-item draws, pooled
0.36, difference ≈ 2 SE). Repeated-digit statistics of the two sets are
indistinguishable (runs 2.2 vs 2.3, longest run 3.1 vs 2.9, adjacent-equal
3.55 vs 3.48). Per-gold-answer accuracy on the new set is flat (0.25 / 0.29
/ 0.20), so it is not answer bias. Untrained accuracy is therefore
~0.3–0.4 rather than ~0.475; still above the 0.15 floor. The Baseline
episode-0 checkpoint on all 201 held-out items is the authoritative number.

**Harness smoke against the real model (4-bit, 2026-09-02, not a recorded
number).** `run_arm.py --arm sleep --n-episodes 4 --k 2 --n-probe 6
--n-heldout-eval 2`: day attempts, night after episode 2 and 4 (before the
checkpoint), LoRA training on MLX (2 steps, loss 2.05, 196 training
tokens), dream generation (4 dreams: 2 rejected as wrong, 2 rejected for
landing on a held-out prefix — the leak guard fires in practice), weight
decay, checkpoints and JSON output all worked. Night 2 kept nothing (both
day trajectories wrong) and trained 0 steps: an **empty night** is
possible at K=2 but at K=50 with ~47% day accuracy is not expected; the
ledger check would flag the resulting step deficit against Online. About
30 s per 4-bit checkpoint of 6 probe + 2 held-out items; a full local
600-episode run is not practical (as §4 says: cloud for grids).

**A1 reproducibility check (4-bit, 2026-09-02).** Two Sleep runs, seed 0,
4 episodes, K=2, dreams at temperature 0.8: episodes, checkpoints, night
statistics and the training loss (0.15914291076478548) are bit-identical.
Observation: across the smokes so far 4 of the 6 parsable dreams were
rejected for landing on a held-out prefix (20% of prefixes are held out).
**Guard verified 2026-09-02 (`tests/test_dream_guard.py`):** on 2,000
uniformly random 12-digit strings the guard rejects 0.203 (expected 0.20 ±
0.03), and the split the dreamer checks against is the same object the
harness built for the seed (one `make_split` call per run). The 4/6 result
is therefore attributable to the model's choice of which digits to change,
not to the guard. Dream yield (kept / generated, with the rejection
breakdown) is logged per night in the run JSON and is a yield metric, not a
validity concern.

### 5a. Modal path check and first-execution validation (2026-09-02)

**Path check** `modal_app.py::main --arm baseline --seed 0 --n-episodes 4 --k 2
--n-probe 6 --n-heldout-eval 4` (control 300), A100-80GB (the pre-L4
default): three checkpoints (0, 2, 4) with identical results, as required
for frozen weights + greedy: probe 1/6, held-out 3/4, GSM8K 0.573 (172/300).
Run file complete in the volume (`partial: false`, backend `hf:Qwen/Qwen3-4B`).

| validation item | result | observed |
|---|---|---|
| 1 dtype / adapters / trainable | pass | model bf16; adapter params fp32; 33,030,144 trainable; 36 layers |
| 2 template / stop / tokens | pass | `<think></think>` emitted; stopped at 100 tokens (< 400 budget); completion_tokens 100 vs count_tokens 99 |
| 3 greedy determinism | pass | two calls byte-identical; three checkpoints identical |
| 4 training mechanics | pass | losses 0.391, 0.003, 0.0001, 0.192, 0.0007, 0.0008 (finite, one bump); training_tokens 588 = 6×(97+1); lora_norm 0 → 1.305 → 1.240 (ratio 0.94999983) → 0 after reset |
| 5 starmap / volume / cache | pass | 3/3 results (seeds 0–2); volume holds runs/ + ledgers/ + preflight; no model download in later apps (HF cache reused; 0 "Fetching" lines vs 5 in the first app); model load 24.9 s from cache |
| 6 wall-clock | pass | identical 4-episode workload: A100 5.9 s/episode, checkpoints 262/254/253 s; L4 6.3 s/episode, checkpoints 276/270/270 s (60 probe + 4 held-out + 300 GSM8K each) |

**GPU choice: L4 (decided 2026-09-02).** Same workload, Modal list prices
(A100-80GB $0.000694/s, L4 $0.000222/s):

| GPU | 4-episode path check (4 episodes + 3 full checkpoints) | cost | projected 600-episode Baseline (13 full checkpoints with 201 held-out) | cost |
|---|---|---|---|---|
| A100-80GB | 793 s | $0.55 | ≈ 2.4 h | ≈ $6.0 |
| L4 | 842 s (+6%) | $0.19 | ≈ 2.6 h | ≈ $2.1 |

Generation at batch 1 (day episodes, dreams) and LoRA steps at batch 1 are
latency-bound, so the A100 buys almost no time; the L4 costs ~2.9× less.
Trainable arms add nights (50 steps ≈ 1–2 min) and Awake adds critique
rounds; a full 5-arm × 5-seed grid on L4 is on the order of $60–80 and
~70 GPU-hours (parallel across seeds). 24 h per-function timeout is ample.

**Cross-GPU numerics.** Same weights, prompts, greedy decoding: GSM8K
control accuracy at episode 0 was 172/300 on A100 and 176/300 on L4; probe
and held-out were identical. "Seeded on CUDA" does not mean bit-identical
across GPU types. **Rule: every arm of a seed runs on the same GPU type**
(the grid entrypoint takes one `--gpu` for all arms), and the GPU name is
written to each run JSON's timing block.

**Backend agreement (frozen Baseline, 40 held-out items, greedy).** MLX bf16
0.25 / 0.075 exact chains (kept 3); HF bf16 0.275 / 0.05 (kept 4). 30/40
completions byte-identical, 31/40 same answer, 39/40 same exact-chain flag,
both-correct 8, only-MLX 2, only-HF 3. All 10 divergences start at work
line 6–11 (histogram {6:1, 7:2, 10:3, 11:4}), i.e. deep in a greedy chain
where logits are near-tied — numeric noise across backends, no flag.
Files: `results/agreement_mlx_bf16_seed0.json`,
`results/preflight_hf_seed0.json`, `results/agreement_mlx_vs_hf_seed0.json`.

**Dream yield on the untrained model is zero.** From 7 kept training
trajectories (HF; keep rate 7/40 = 0.175 on training items), 14 dreams:
0 verified, 9 wrong, 5 on held-out prefixes. Expected: a dream must be a
fully correct 11-step chain on a new string, and the untrained model's
exact-chain rate is 5–8%, so ≈ 1 of 14 was the expectation. The held-out
rejection rate is again above 20% (5/14; cumulative 9/20 across all
smokes), consistent with the model preferring to change early digits.
Consequence: in the first nights Sleep ≈ Sleep-NoDream; dreaming can only
contribute once the model's chain accuracy rises. **Decision needed
before the grid** (see §8).

**A4 estimate.** Kept-trajectory completion tokens (HF, training items):
mean 97.1, median 98 (held-out kept: 94.5 / 95). Dream generations average
119 tokens including the ~27-token Digits line → completion ≈ 92. Projected
Sleep/Online training-token ratio: 1.000 at yield 0, 0.997 at 0.07, 0.988
at 0.30, 0.975 at 1.0 — within ±5% at every yield, because dreams and
trajectories share the format. No `steps_per_night` change; DEVIATIONS.md
unchanged. Secondary observation: at keep rate 0.175 a night has ~9 kept
trajectories for 50 steps (≈5 passes each); Online substitutes the same few
trajectories for its non-kept episodes. Both arms overfit the same small
pool early; this is symmetric and inherent to the design at this accuracy.

### 5b. Seed-0 run, attempt 1 (2026-09-02/03, L4) — lost to a client disconnect

`modal run modal_app.py::grid --seeds 0 --arms sleep,online,awake` ran
Sleep and Online concurrently. The ephemeral app was torn down when the
local client disconnected ("Stopping app - local client disconnected") at
Sleep episode 150 / Online episode 100; ~2 GPU-hours (~$1.7) lost. LoRA
state is not resumable, so the run restarts. Fix: orchestration moved into
the CPU-only Modal function `grid_remote`, launched with `modal run
--detach`; the laptop is no longer in the loop.

What the partial run showed (single seed, not results):

| ep | Sleep probe / held-out / GSM8K | Online probe / held-out / GSM8K |
|---|---|---|
| 0 | 0.367 / 0.373 / 0.587 | same (identical untrained weights) |
| 50 | 0.350 / 0.493 / 0.540 | 0.333 / 0.363 / 0.653 (43 steps: 7 skipped before first kept) |
| 100 | 0.317 / 0.527 / 0.477 | 0.317 / 0.428 / 0.677 (93 steps) |
| 150 | 0.350 / 0.597 / 0.443 | — |

Sleep nights: kept 8 → 11 → ?, dreams 16 → 22, accepted 3 → 6, wrong 9 → 6,
unparsable 4 → 10, held-out and prefix-changed 0 → 0, structured fraction
1.00 → 0.33, mean night loss 0.068 → 0.020. Timing on L4: Sleep 7.8–8.3 s
per episode, nights 34/43/55 s, full checkpoint 440–515 s; Online 11.4–11.7
s per episode (train step each), checkpoint 590–633 s.

Observations to carry forward (all logged now, none acted on):
- The 3/3 structured accepted dreams on night 1 were probably verbatim
  copies of the source string. Accepted dreams are now recorded with
  digits, source, structured and duplicate flags, and duplicates are
  counted per night; they are still accepted (PREREG says "variations";
  rejecting copies is a decision, not yet taken).
- Unparsable dreams rose after the first night (4 → 10 of 22): training on
  solution-only completions pulls the model away from the "Digits: …"
  dream format. Logged as its own category.
- Online's GSM8K rose (0.587 → 0.677) while Sleep's fell (0.587 → 0.443)
  over 100 steps; Sleep's held-out rose faster (0.597 vs 0.428 at the
  same step count ±7). One seed; noted, not interpreted.
- Online skipped 7 steps before its first kept trajectory; the deficit is
  fixed and ends at 593/600 (1.2%), inside tolerance.

### 5c. Seed-0 run, attempt 2 (2026-09-03, L4, detached) — orchestrator preempted

Sleep and Online ran concurrently under `grid_remote`. After ~4 h Modal
preempted the orchestrator container ("Container terminated due to
preemption. Your Function will be restarted with the same input"). The
restarted orchestrator re-spawned Sleep and Online from episode 0 while the
original GPU containers (then at episode 300) kept running; five tasks were
active and both copies wrote to the same `runs/<arm>_seed0.json` paths.
Stopping individual containers only reschedules their inputs, so the
duplicates could not be cancelled separately.

Decision (human, 2026-09-03): salvage — let everything run, capture the
original runs' final files from the volume in the window before the
duplicates overwrite them (`results/final/`), then stop the app and launch
Awake separately with the budget from Sleep's ledger. Accepted waste ≈ $3.

**Preemption safety, built 2026-09-03 (tests in `tests/test_resume.py`):**
(1) every launch has a run tag (`YYYYMMDDTHHMMZ`); all files live under
`/results/<tag>/{runs,ledgers}/` and `grid_seed<seed>.json` — restarts can
never clobber another launch; (2) `grid_remote` is idempotent: spawned
FunctionCall ids are written to `<tag>/grid_calls_seed<seed>.json`, and on
restart it re-attaches to them, loads complete run files, waits on fresh
partial files and only re-spawns stale ones (`eval/orchestrate.plan`); (3)
GPU runs are resumable: at every checkpoint the harness saves backend
state (adapter params, optimizer, torch/CUDA/python RNG) and arm state
(Awake budget accounting, Online most-recent-kept, Sleep replay buffer +
night count) to `<run>.json.state/`, and `run_arm(resume=True)` continues
from the last checkpoint, redoing the partial block. Model-free test: a run
crashed mid-block and resumed reproduces the uninterrupted run exactly for
all four arm types. On real hardware resume is exact only up to CUDA
non-determinism. `modal run --detach ... --run-tag <tag>` re-attaches a
launch.

Also observed in attempt 2: run-to-run non-determinism at fixed seed on
CUDA after the first update (Sleep ep-50 probe 0.417 vs 0.350 in attempt 1;
Online ep-50 held-out 0.527 vs 0.363). Consider
`torch.use_deterministic_algorithms(True)` (tunable) for the grid.

### 5d. Seed 0 results — Sleep, Online, Awake (complete 2026-09-04)

Files: `results/final/{runs,ledgers}/<arm>_seed0.json`, `results/final/report.md`,
`summary.json` (local; results/ is gitignored — the volume holds the originals:
Sleep/Online under `/runs/` from the salvaged attempt-2 containers, Awake under
`/seed0-awake/`). Backend hf:Qwen/Qwen3-4B, NVIDIA L4 for all three arms,
PREREG v1.0, K=50, 600 episodes, 13 checkpoints (60 probe / 201 held-out /
300 GSM8K).

**Compute matching — `check_seed` verdict persisted in all three run JSONs:
MATCHED.** Awake 141,628 generated tokens vs Sleep 141,499 (+0.09%); Online
593 steps vs 600 (−1.17%), 58,088 training tokens vs 57,270 (+1.43%). Awake
budget 236 tokens/episode, realised 236.0; mean 2.44 critique rounds per day
episode.

**Wall-clock / cost (L4):** Sleep 3.40 h $2.72; Online 3.22 h $2.57; Awake
8.43 h $6.74 (checkpoints 5.97 h: its critique rounds triple eval cost).
Useful compute $12.03; failures cost ~$5.5 more (§5b–5c, two Awake
false starts).

**Primary metric (probe, chance 0.33, criterion 0.70 × 2):** no arm met the
criterion; all three censored at 600. Curves:

| ep | 0 | 50 | 100 | 150 | 200 | 250 | 300 | 350 | 400 | 450 | 500 | 550 | 600 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Sleep | .37 | .42 | .32 | .27 | .33 | .35 | .28 | .30 | .37 | .32 | .25 | .33 | .32 |
| Online | .37 | .32 | .37 | .35 | .35 | .33 | .33 | .35 | .27 | .30 | .32 | .32 | .32 |
| Awake | .30 | .30 | .30 | .30 | .30 | .30 | .30 | .30 | .30 | .30 | .30 | .30 | .30 |

Median tokens per correct held-out answer: 98 (Sleep, Online) and 294
(Awake, ~3 rounds) at every checkpoint — no token collapse anywhere.

**Task accuracy (held-out, 201):** Sleep 0.373 → 0.930; Online 0.373 →
0.667 (peak 0.965 at ep 550); Awake 0.403 flat (frozen; critique adds +0.03
over single-shot). Day accuracy over 600 episodes: Sleep 0.718, Online
0.748, Awake 0.413.

**Forgetting (GSM8K, 300, ep 600 − ep 0):** Sleep +0.060 (min 0.477 at ep
100), Online +0.137 (never below ep 0), Awake 0.000 (frozen; 0.597). No arm
forgot; the trainable arms improved. H4's predicted direction (Sleep < Online
forgetting) is reversed on this seed.

**Eval-pipeline validation:** Awake's 13 checkpoints are identical
(`frozen_eval_identical: [True]`), i.e. greedy evaluation is deterministic
across 8.4 h on one L4.

**Dreams (12 nights):** 714 generated, 152 accepted (58 verbatim copies, 94
real variations, 4 of them mirror-structured), 279 wrong, 283 unparsable.
Real yield 13%. Held-out / prefix-changed rejections 0.

**Reading of seed 0 (one seed; not a result):** H1 and H2 null — no
insight in any arm, Sleep and Online equal at chance on the probe; H4
reversed; H3 untested (no Sleep-NoDream yet). Both LoRA arms learn the
literal 11-step rule to >0.9 held-out without ever using the r11 = r6
shortcut, which is the regime the task was designed to produce. The dream
stream as constructed (digits 1–7 fixed) is almost entirely unstructured
and cannot carry the mirror statistic. Cross-attempt non-determinism on
CUDA (§5c) is of the same order as between-arm differences on held-out and
GSM8K, so seeds 1–4 are required before any comparison is read.

### 5e. Post-pilot changes and retro-computation (2026-09-04)

Code: duplicate-dream rejection, decoupled dreamer (digits proposal at
T=0.8 → solve with the day prompt, greedy → C3), 1–3 changed digits
enforced (`out_of_spec`), deterministic CUDA default, secondary metrics
(mirror_bias with chance, short_gap, late_vs_early, volatility), GSM8K
parse rate, per-item held-out rows, final adapter state. DEVIATIONS.md and
PREREG Appendix B record them. 114 model-free tests.

**Two-step dreamer on the real model (4-bit MLX, untrained, 4 seeds × 2):**
8/8 proposals parsable (pilot: 40% unparsable), prefix kept 8/8, 1
duplicate caught, model sometimes changed all five trailing digits (now
rejected as out_of_spec), 2/8 solves passed C3.

**Deterministic-mode slowdown:** see DEVIATIONS.md entry 4: ≈ +50% per
episode and +90% per checkpoint for LoRA arms, none for frozen arms.

**Pilot retro-computation.** Possible only from saved data or for frozen
arms (no adapters / eval texts were saved for Sleep and Online).

| metric | Baseline (untrained base) | Awake (base + critique) | Sleep | Online |
|---|---|---|---|---|
| (a) mirror_bias (chance) | 0.353 (0.356), n=201 error steps | 0.358 (0.348), n=204 | n/a | n/a |
| (b) short_gap (struct / unstruct) | 0.000 (0.317 / 0.317) | 0.000 (0.317 / 0.317) | n/a | n/a |
| (c) late vs early error, held-out | 0.512 vs 0.431 | 0.488 vs 0.426 | proxy below | proxy below |
| (d) volatility (std held-out) | 0.000 | 0.000 | 0.167 | 0.188 |
| GSM8K unparsable | 0.0% | 0.0% | n/a | n/a |

(c) proxy from day trajectories (training items) by 100-episode block,
"late | steps 1–8 correct" vs "early | steps 1–2 correct": Sleep 0.61/0.44 →
0.25/0.25 → 0.02/0.18 → 0.07/0.07 → 0.14/0.12 → 0.01/0.02; Online 0.09/0.35 →
0.13/0.22 → 0.03/0.15 → 0.01/0.10 → 0.05/0.07 → 0.01/0.04; Awake ≈ 0.45/0.45
throughout. Both LoRA arms drive late-step errors near zero while early-step
errors stay several times higher in the middle blocks — consistent with
learning the literal rule, not a late-step shortcut, but a proxy on training
items, not held-out.

**Answers.** The untrained base shows no mirror bias (0.353 vs chance 0.356)
and no short-mode gap, so the implicit metrics start at their null values.
GSM8K parse rate is 100% for the base and for Awake, so Online's GSM8K gain
(0.587 → 0.723) cannot be a parse-rate effect. Trained-arm values for (a),
(b) and GSM8K parse rate will exist from the Sleep-NoDream/Baseline runs
onward (adapters and rows are now saved).

### 5f. Seed 0 complete — all five arms (Sleep-NoDream and Baseline added 2026-09-04)

Files: `results/final/{runs,ledgers}/` (all five), `report.md`, `summary.json`;
volume `/seed0-nodream-base/`. Caveat: Sleep, Online and Awake are the pilot
runs (non-deterministic CUDA, one-step dreamer); Sleep-NoDream and Baseline
ran with the post-pilot code (deterministic CUDA, secondary metrics). Same
protocol, same GPU type; the dreamer change does not touch these two arms.

**check_seed over all five arms: MATCHED.** Sleep-NoDream 600 steps /
57,252 training tokens vs Sleep 600 / 57,270 (−0.03%).

| arm | gen tokens | steps | wall-clock | cost (L4) | probe curve (13 ckpts) | held-out 0→600 | GSM8K 0→600 | volatility |
|---|---|---|---|---|---|---|---|---|
| Baseline | 58,084 | 0 | 2.70 h | $2.16 | 0.37 flat | 0.373 → 0.373 | 0.587 → 0.587 | 0.000 |
| Awake | 141,628 | 0 | 8.43 h | $6.74 | 0.30 flat | 0.403 → 0.403 | 0.597 → 0.597 | 0.000 |
| Online | 58,806 | 593 | 3.22 h | $2.57 | 0.27–0.37 | 0.373 → 0.667 (max 0.965) | 0.587 → 0.723 | 0.188 |
| Sleep | 141,499 | 600 | 3.40 h | $2.72 | 0.25–0.42 | 0.373 → 0.930 | 0.587 → 0.647 (min 0.477) | 0.167 |
| Sleep-NoDream | 57,302 | 600 | 5.68 h (det.) | $4.54 | 0.18–0.37 | 0.373 → 0.970 (max 0.980) | 0.587 → 0.827 | 0.174 |

Primary metric: no arm met the criterion; all five censored at 600. Sleep-
NoDream's probe sat *below* chance for nine consecutive checkpoints
(0.18–0.25), the opposite direction of insight. Frozen-arm eval identical
at all 13 checkpoints for both Baseline and Awake (validation passes).

H3 on this seed (Sleep vs Sleep-NoDream, matched budget): dreams did not
help — NoDream ends higher on held-out (0.970 vs 0.930) and far higher on
GSM8K (+0.240 vs +0.060), with the same null probe. H4: the arm with the
least forgetting is Sleep-NoDream, then Online, then Sleep; all improved.

**Sleep-NoDream secondary metrics.** mirror_bias at chance throughout while
n was adequate (ep 50–200: 0.38/0.40/0.35/0.39 vs chance 0.37/0.35/0.34/
0.36, n=91–124 error steps); from ep 250 the error count collapses (n=8–60)
as held-out accuracy passes 0.9 and the metric becomes uninformative.
late_vs_early: untrained 0.51 vs 0.43; trained 0.01–0.19 late vs 0.02–0.33
early — late-step errors fall faster than early-step errors, consistent
with literal-rule mastery, not a late-step shortcut. short_gap: −0.10 to
−0.12 at ep 50–150 (no structured advantage), then both short-mode
accuracies collapse to ~0 from ep 200 because the trained model ignores the
answer-only instruction and writes work lines that the 16-token budget cuts
off — a measurement artefact, now logged as short-mode unparsable
fractions (post-run fix). GSM8K unparsable 0.0% at every checkpoint: the
control gains are real correctness.

Nights: kept 8 → 48 of 50; replay 1:1; loss 0.044 → ~0.006–0.013 (heavy
repetition of a small pool early). Cost of seed 0 in total: $18.73 useful
+ ≈ $6.3 failures.

**Reading (one seed):** H1, H2, H3 null; H4 reversed; both LoRA arms reach
> 0.9 on the literal task without any implicit-shortcut signal on the probe,
mirror_bias or late/early metrics. The dreams (pilot version) were, if
anything, a small negative for task and control accuracy. Seeds 1–4 are
needed before any of this is a result; the implicit-probe numbers above are
the first for a trained arm.

### 5g. Compute move: Modal → Vultr GPU VM (decided 2026-09-05)

Modal wallet exhausted (~$1 left); the remaining grid (seeds 1–4, ~29 L4
GPU-hours per seed) would cost ~$93 on Modal. Options weighed: ComputeGPU
serverless (L4 $0.53/h; an inference-endpoint product with 30 s sync calls,
undocumented job limits, storage "coming soon") — rejected for multi-hour
batch jobs; Vultr with $250 free credit — chosen. Plan
`vcg-a16-12c-128g-32vram` (2 × NVIDIA A16 16 GB, 12 vCPU, 128 GB RAM,
700 GB disk, $0.942/h; sjc/sgp/blr). The 8 GB A16 slice cannot hold Qwen3-4B
bf16 (8 GB of weights). Footprint per arm ≈ 12 GB (weights 8, LoRA+Adam
0.5, activations/KV 2–3) → one arm per GPU, eval batch 8. Expected speed:
A16 is ~1.5× slower than L4 at batch 1 and 3–6× on batched work; a seed ≈
60–90 A16-GPU-hours ≈ 30–45 h wall on two GPUs, $28–42 of credit. VMs are
not preempted; resume remains the safety net.

Deployment (`deploy/vultr/`, no Modal dependency): `setup.sh` (driver if
missing, py3.11 venv, cu121 torch, deps; HF cache and results on /data),
`run_job.py` (one arm/seed/tag; `--awake-from-sleep`; `--backend scripted`
dry run; tested), `runner.sh` (per-GPU sequential queue, resumable, logs
under /data/results/<tag>/logs), `launch_seed.sh` (tmux sessions per GPU
from `queue.plan_queues`, Awake after Sleep on the same GPU). Repo reaches
the VM by rsync from the laptop (no GitHub credentials on the VM); results
come back by rsync. Same validation list as Modal before seed 1 (§8 item 9),
plus A16 timing.

### 5h. Lightning AI Studios (2026-09-05) — seed 1 Sleep; idle-credit incident

Vultr GPU plans require a support approval (ticket pending). Lightning AI
(user rushiljain2001, teamspace general, 30 org credits) used instead:
Studio `doze-llm`, L4, torch 2.8 cu128, Python 3.12; repo uploaded as a
tarball (zero-byte files break single-file upload; `deploy/vultr/queue.py`
shadowed stdlib `queue` and broke torch import → renamed `gpu_queue.py`).
Validation (4-episode Baseline): same numbers as Modal's L4 path check,
4.9 s/episode, 40 s small checkpoint. `L4_X_2` could not be started on this
account (HTTP 400) — single L4 only.

**Incident:** after disabling auto-sleep for the planned launch, the network
dropped mid-sequence and the Studio idled on an L4 for ~8 h with nothing
running: ~14 of 30 credits lost. Implied L4 rate ≈ 1.3 credits/h. Guards
added: `runner.sh` stops the Studio when its queue finishes
(`DOZE_STOP_WHEN_DONE=1`), and the monitor stops it on QUEUE_DONE. Rule:
never leave a Studio with auto-sleep off unless a job is running on it.

**Seed 1 on Lightning (complete 2026-09-05):** Sleep 04:38–09:03 UTC
(4.42 h, 600 steps, 101,136 generated / 58,424 training tokens) then
Baseline 09:04–11:38 UTC (2.57 h, 57,978 tokens), chained on one L4; Studio
stopped itself. `check_seed(records/seed1)`: MATCHED (Sleep reference).
Files: `records/seed1/`. ≈ 9 credits; ≈ 7 left.

| arm | probe curve | held-out 0→600 | GSM8K 0→600 | volatility |
|---|---|---|---|---|
| Sleep | .42 .38 .33 .38 .42 .42 .38 .40 .35 .38 .47 .33 .37 | 0.478 → 0.811 (peak 0.896) | 0.577 → 0.497 (−0.08) | 0.161 |
| Baseline | .42 flat (13 identical checkpoints) | 0.478 | 0.577 | 0.000 |

Sleep seed 1 secondary metrics at ep 600: mirror_bias 0.270 vs chance
0.343 (n=89); short_gap +0.03 with 0% short-mode unparsable; late/early
error 0.036 / 0.118; GSM8K unparsable 0%. Dreams (two-step): 642
proposals, 191 accepted (real yield 30%), 300 source copies rejected, 68
wrong, 7 unparsable (1%), 69 out of spec, 20 within-night repeats, 5
structured. Unlike seed 0's Sleep (+0.06 on GSM8K), seed 1's Sleep forgot
(−0.08): cross-seed variance on H4. Probe null again.

Still missing for seed 1: Online, Sleep-NoDream, Awake (Vultr when
approved, or new credits).

### 5i. Vultr grid launch (2026-09-08)

Vultr GPU access approved 09-07 (Trust & Safety). Instance
`doze-llm-a16-sgp` (id 8b8aa0a1…, `vcg-a16-12c-128g-32vram`, 2 × NVIDIA
A16-16Q, Ubuntu 22.04.5, driver preinstalled, IP 207.148.68.114, Singapore;
a Bangalore instance could not be powered on — host-side "Unable to start
server" for 40 min — and was destroyed; the Singapore create needed three
tries because the deleted instance still counted toward the monthly fee
cap). Setup: py3.11 venv, torch 2.5.1+cu121, transformers 5.16.1, peft
0.20.0; 118 tests pass on the VM. Validation (4-episode Baseline, GPU 0):
identical results to L4/Modal; 6.5 s/episode (1.3× L4), 96 s per small
checkpoint (2.4× L4); 8.8 GB used at eval batch 8.

Launched 07:59 UTC via `deploy/vultr/run_all.sh` (nohup on the VM, resumable,
immune to laptop disconnects) with `plan_multi` queues:
- gpu0: 2:sleep 4:sleep 1:sleep_nodream 2:online 3:online 4:online 2:awake 4:awake
- gpu1: 3:sleep 1:awake 1:online 2:sleep_nodream 3:sleep_nodream 4:sleep_nodream 2:baseline 3:baseline 4:baseline 3:awake
(≈ 53 and 55 L4-hour-equivalents each). Seed 1's Sleep and Baseline files
(Lightning) were placed under /data/results/seed1/ so 1:awake derives its
budget from them. Estimate at 2.2–2.5× L4: 120–135 h per queue in parallel,
5–6 days, ≈ $120–130 of the $300 credit. Results: /data/results/seed<N>/
(runs/, ledgers/, logs/); pulled into records/seed<N>/ as they complete.
Teardown: destroy the instance when both queues report QUEUE_DONE (a
stopped instance still bills).

**Progress 2026-09-09 03:50 UTC (20 h in):** Sleep seeds 2, 3, 4 complete
(8.9–9.0 h each on A16; all probe curves at chance; held-out 0.876–0.965;
GSM8K deltas +0.05 / −0.06 / +0.19; dreams two-step, real yield ~30%).
Running: seed 1 Awake (ep 350; Awake checkpoints take ~71 min on A16 →
~19 h per Awake arm) and seed 1 Sleep-NoDream (ep 100). No errors, no
OOM (peak 14.0 GB on the NoDream process). Remaining ≈ 71 h (gpu0) / 84 h
(gpu1) → finish ≈ 2026-09-12 midday UTC; spent ≈ $19 so far.

**Caveat recorded 2026-09-09:** seed 1 mixes GPU types (Sleep, Baseline on
Lightning L4; Online, Sleep-NoDream, Awake on Vultr A16); untrained ep-0
evaluations differ slightly across the two (0.417/0.478/0.577 vs
0.400/0.463/0.583). Seeds 2–4 are entirely on the A16. Details and other
analysis caveats: `records/STATUS.md`.

### 5j. Seed 1 complete (2026-09-10 00:30 IST)

`check_seed(records/seed1)`: MATCHED across all five arms. Sleep (Lightning
L4) and Baseline (Lightning L4); Online, Sleep-NoDream, Awake (Vultr A16).

| arm | probe range | held-out 0→600 | GSM8K 0→600 | volatility |
|---|---|---|---|---|
| Baseline | 0.42 flat | 0.478 | 0.577 | 0 |
| Awake | 0.35 flat | 0.453 | 0.587 | 0 |
| Online | 0.28–0.40 | 0.463 → 0.900 (max 0.920) | 0.583 → 0.760 (+0.177) | 0.147 |
| Sleep | 0.33–0.47 | 0.478 → 0.811 (max 0.896) | 0.577 → 0.497 (−0.080) | 0.161 |
| Sleep-NoDream | 0.33–0.47 | 0.463 → 0.886 (max 0.930) | 0.583 → 0.553 (−0.030) | 0.150 |

No arm met the criterion. Same shape as seed 0: Online ends highest on the
task and gains most on GSM8K; Sleep is lowest of the three trained arms on
both; dreams (two-step, 30% real yield) did not help versus NoDream (H3
direction negative again). Secondary metrics at ep 600: mirror_bias Online
0.167 (chance 0.333, n=30), Sleep 0.270 (0.343, n=89), NoDream 0.333
(0.343); late/early error Online 0.020/0.062, Sleep 0.036/0.118, NoDream
0.046/0.084; GSM8K unparsable 0% everywhere. Files: `records/seed1/`.

### 5k. Grid complete — final preregistered analysis (2026-09-12 10:37 IST)

> **Read with §5m.** Two conclusions below are over-claimed: the probe could
> not have detected the shortcut, and the GSM8K movements track response
> length (r = 0.91). Verdicts stand; interpretations are revised in §5m.

All 25 arms (5 seeds × 5 arms) finished. Last arm: Awake seed 3 on Vultr
gpu1 at 10:35 IST. `check_seed` MATCHED for every seed (largest drift in any
seed: Online gradient steps −3.0% in seed 2; everything else within ±2.3%).
Grid wall-clock on the Vultr 2×A16 VM: 08 Sep 13:29 IST → 12 Sep 10:35 IST
(3 d 21 h, 181.8 GPU-hours across 18 arms; seed 1 Sleep/Baseline on
Lightning L4; seed 0 on Modal L4). Vultr pending charges at teardown:
$89.50 of the $300 credit. Files: `records/analysis_final.md` (+ `.json`),
per-seed `records/seedN/report.md`, run/ledger/log/timing files per seed.
Command: `python -m eval.analyze --final --results records --seeds 0,1,2,3,4`.

**Verdicts (PREREG §7):**

| hypothesis | result | verdict |
|---|---|---|
| H1 insight: Sleep first to criterion | criterion (probe ≥ 0.70 at two consecutive checkpoints) met by **0 of 25 arms**; all censored at 600; log-rank degenerate | not supported |
| H2 phase structure: Sleep vs Online | all censored | not supported |
| H3 dreaming: Sleep vs NoDream | all censored | not supported |
| H4 retention: Sleep forgets less than Online (GSM8K Δ) | paired diffs Sleep−Online [−0.077, −0.257, −0.050, −0.257, +0.023], mean −0.123, exact sign-flip p = 0.125; direction **reversed** | not supported |

The preregistered outcome is a clean null on the insight hypotheses and a
reversed-direction null on retention. PREREG §10 says a null on H2 is
publishable; this is that case, with the caveats below.

**Per-arm summary (mean over 5 seeds; probe = 60 masked items, chance 1/3):**

| arm | peak held-out | last-3 held-out | GSM8K Δ (0→600) | seeds Δ>0 | probe max (any ckpt) | volatility |
|---|---|---|---|---|---|---|
| Baseline | 0.421 | 0.421 | 0 | — | 0.340 | 0 |
| Awake | 0.419 | 0.419 | 0 | — | 0.310 | 0 |
| Online | 0.963 | 0.889 | **+0.155** | 5/5 | 0.377 | 0.176 |
| Sleep | 0.937 | 0.894 | +0.031 | 3/5 | 0.417 | 0.168 |
| Sleep-NoDream | 0.950 | 0.917 | +0.079 | 4/5 | 0.383 | 0.163 |

Per-seed final held-out / GSM8K Δ (trained arms):

| seed | Online | Sleep | Sleep-NoDream |
|---|---|---|---|
| 0 (pilot code for Sleep/Online) | 0.667 (peak 0.965) / +0.137 | 0.930 / +0.060 | 0.970 / +0.240 |
| 1 (Sleep on L4) | 0.900 / +0.177 | 0.811 / −0.080 | 0.886 / −0.030 |
| 2 | 0.856 (peak 0.965) / +0.097 | 0.930 (peak 0.995) / +0.047 | 0.861 / +0.020 |
| 3 | 0.960 / +0.193 | 0.876 / −0.063 | 0.905 / +0.097 |
| 4 | 0.925 (peak 1.000) / +0.170 | 0.965 / +0.193 | 0.985 / +0.067 |

Other paired comparisons (exact sign-flip, n=5): held-out last-3 Sleep−Online
+0.005 (p=1.0); Sleep−NoDream −0.023 (p=0.5); GSM8K Δ Sleep−NoDream −0.047
(p=0.375); NoDream−Online −0.076 (p=0.31). Nothing separates the three
trained arms on the task; they differ only on the control benchmark, where
Online gains in every seed.

**What the data say, plainly.**
1. Nobody found the shortcut. Probe accuracy stayed at chance in all 25 arms
   at all 13 checkpoints (best single checkpoint 0.47, Sleep seed 1). Held-out
   accuracy going to 0.9+ with the probe flat means the model learned to
   execute the 11-step procedure reliably, not to skip it. Mirror-bias at
   ep 600 sits at or below its chance level in every trained arm (e.g. Online
   0.17 vs 0.33, Sleep 0.27 vs 0.34 in seed 1; Sleep 0.14 vs 0.39 in seed 4),
   so there is no sign of the r11 = r6 regularity being used implicitly either.
2. Self-training on verified own solutions works on the task regardless of
   schedule: same gradient steps, same tokens, same filter → same held-out
   endpoint (0.86–0.99) whether the steps are spread one per episode (Online)
   or batched 50 per night (Sleep, NoDream).
3. Schedule matters for the control benchmark. Online improved GSM8K in 5/5
   seeds (+0.10 to +0.19); nightly batches of 50 steps produced a mix of
   gains and regressions (Sleep −0.08 to +0.19). Exploratory, not
   preregistered: mean per-checkpoint GSM8K change Online +0.013 vs Sleep
   +0.003; worst single-checkpoint drop Online −0.207 vs Sleep −0.087.
   A plausible reading is that 50 consecutive LoRA steps on a narrow buffer
   with 5% decay perturbs general ability more than 50 steps interleaved with
   nothing else — but the design cannot distinguish that from noise at n=5.
4. Dreams did not help. Post-pilot two-step dreamer: yield 22–33% of
   proposals verified (seeds 1–4), duplicates the dominant rejection
   (300–442 of 640–820 proposals per seed), held-out-prefix rejections 0 in
   every seed (watchlist item confirmed clean), structured-dream fraction
   ~1%. Sleep vs NoDream: held-out −0.023, GSM8K −0.047, both n.s.; dreams
   cost 42–53k extra generated tokens per seed for no measurable return.
5. Inference-time reasoning at matched tokens did nothing. Awake (critique
   rounds, 2× the generated tokens of Baseline) sits at 0.419 held-out vs
   Baseline 0.421 and the same probe chance level.

**Caveats (carry into any write-up).** Seed 0 Sleep/Online/Awake ran the
pilot code (one-step dreamer, non-deterministic CUDA) — see §5d–5f and
DEVIATIONS.md; seed 1 mixes L4 (Sleep, Baseline) and A16 (others) — see
records/STATUS.md; n=5 seeds gives the sign-flip test a minimum attainable
p of 0.0625, so only unanimous effects can reach conventional significance;
the insight hypotheses are fully censored and the log-rank test is
uninformative beyond "never met" — a 4B model in 600 episodes of this task
does not discover the shortcut under any arm, so the experiment cannot
speak to whether sleep would accelerate a discovery that never happens.

### 5l. Deep exploratory analysis and local archive (2026-09-12)

`python -m eval.deep_analysis --results records --seeds 0,1,2,3,4` writes
`records/analysis_deep.md` (+ `.json`, `records/figures/fig1–4`); the human
reading is `records/analysis_deep_notes.md`. Not preregistered. Headlines:

- Probe pooled over 3,900 item-evaluations per trained arm: 0.32–0.34,
  CI ±0.015; conservative final-checkpoint pooling (n=300/arm) 0.33–0.34.
  Upper bound on shortcut use: a few percent. Working length never shrank
  (97 → 97 tokens/attempt).
- Trained arms learned execution from the left: correct leading steps per
  day attempt 6.6 → ~10/11; first-error position moved from mid-chain
  (mode 5–6) to late (7–8). Only 3/978 held-out item-evaluations unsolved by
  every trained arm at ep 600.
- Sleep lags Online mid-run (block-6 day accuracy 0.70 vs 0.84) and catches
  up by the end. Batched schedule is smoother on the task (held-out max
  drawdown 0.11 vs Online 0.18) but worse on GSM8K, and the damage is early:
  Sleep's GSM8K change −0.013/night over nights 1–4 (train sets ~36 items,
  ≈1.4 passes, ~40% dreams) vs +0.010 over nights 5–12; NoDream shows no
  early dip. Night GSM8K change correlates −0.27 with night loss, +0.21 with
  train-set size (n=60 nights). Hypothesis: data-poor early nights, not
  consolidation per se, drive the H4 reversal.
- Dreams: yield 0.14 → 0.32 across nights, duplicates ~40% of proposals
  throughout, 0 held-out/probe touches in ~3,660 proposals; ~40% of night-12
  new items were dreams; no measurable return for +80% generated tokens.
- Awake: 2 critique rounds on 77% of attempts; accuracy by rounds
  0.40/0.41/0.42; inert.
- Buffers: 336–454 distinct puzzles, 42–57% of train, balanced answers,
  zero leakage — watchlist confirmed from artefacts.
- Held-out accuracy recomputed from saved per-item rows matches the
  reported number for all 22 runs with rows.
- Power: H4 needs ~9 seeds at the observed effect.

Local archive (`~/doze-archive/`, see `records/README.md` for the layout):
13 trained adapters exported to standard PEFT folders
(`models/adapters/<arm>_seedN/`, via `deploy/export_adapter.py`), raw resume
states, the Vultr tar, the Lightning seed-1 tar (fetched over SSH after
starting the Studio on CPU; Studio stopped again), the full Modal volume,
and the base model (7.5 GB). Export verified with `deploy/verify_adapter.py`:
Sleep seed 3 adapter on the local base (MPS, bf16) solves 6/6 held-out puzzles
vs 2/6 for the bare base. The seed 0 pilot Sleep/Online adapters were never saved
by the pilot code and cannot be recovered. No remote compute is running.

### 5m. Fresh-eyes review: two over-claims found (2026-09-12)

`records/analysis_outsider.md`: a hypothesis-blind reading of the whole
experiment, plus an independent reading by a fresh session with no access to
this conversation (Appendix A) and a reconciliation with verifications
(Appendix B). Both readings agree on the effect ranking. Three findings were
checked against the run files and accepted:

1. **GSM8K accuracy tracks response length.** `eval_tokens`/300 vs GSM8K
   accuracy over 156 trained checkpoints: r = 0.908 (Sleep 0.96, NoDream
   0.89, Online 0.71), slope +0.09 per +10 tokens/item. Base 0.58 under a
   "think briefly" prompt with a 512-token cap is an under-reasoning
   baseline. Control outputs were not saved, so "better arithmetic" vs
   "writes more" cannot be separated. **H4 and the retention story are a
   response-length story until re-scored.**
2. **The masked probe could not have detected the shortcut.** Answer-only
   accuracy on fully visible structured puzzles is 0.31–0.39 for every
   regime at every checkpoint: no model here can do six steps silently, so a
   model that knew r11 = r6 would still score ~1/3 on the probe. §5k's
   "nobody found the shortcut" → "the instrument could not have shown it".
   Mirror bias and output length (both flat) still support "no sign of
   shortcut use in written chains".
3. **First-night GSM8K dip is Sleep-only** (5/5 seeds down; NoDream 4/5 up;
   Online 3/5 up) on a ~12-item train set differing by a few dreams; and
   nights present exactly 50 examples, so from ~night 4 the 1:1 interleave
   describes the pool, not the training batch.

Also from the independent reading: the base model's errors are digit
tracking, not rule errors (rule errors 5–13 of 500 early attempts vs
330–413 tracking errors; 191/194 single-digit skips sit next to an equal
digit); kept trajectories contain 1–2 wrong intermediate steps in 33–53% of
block-1 keeps (C3 allows ≥9/11); Online's single-example steps swing
held-out far beyond item noise.

**What this changes.** PREREG verdicts unchanged (H1–H4 not supported). The
*reasons* change: H1–H3 are unobservable with this probe and training
format rather than observed-negative; H4's reversed direction should not be
reported as forgetting. Robust, reportable: verified self-distillation
repairs digit tracking in 200–400 steps regardless of schedule; critique
without feedback inert; self-generated variants inert; narrow SFT shifts
verbosity, and thereby scored accuracy, on an unrelated benchmark in a
schedule-dependent way.

**Addendum (same day):** the control benchmark ran under the *task* system
prompt ("no explanation"), not `control_bench.SYSTEM_PROMPT`; the harness
backend has one system prompt per run. Recorded in DEVIATIONS.md. This makes
the length coupling in (1) expected rather than surprising: the recorded
GSM8K is a suppressed-reasoning score. Follow-up plan with decision rules
written before running: `records/followup_prereg.md`; code `eval/followup.py`.

**Cheapest decisive follow-ups (before any larger model):**
(a) re-score GSM8K on base + 13 adapters with saved outputs, plus a second
condition without "briefly" and with a 1024-token cap (local MPS or ~$3 on
L4); (b) probe positive control: state the regularity in the prompt / 50-step
LoRA on masked→answer pairs / teacher-forced log-prob of r6 vs alternatives,
so the instrument's sensitivity is known; (c) unstructured full-chain
accuracy for the adapters; (d) only then a broader benchmark panel.

### 5n. Follow-up results (2026-09-12, Modal L4; `records/followup/RESULTS.md`)

Decision rules were fixed in `records/followup_prereg.md` before running.

- **A — GSM8K.** The grid's regime reproduces to ±0.02 on all 14 models.
  Under a normal math prompt with 1024 tokens: base **0.910** (grid regime
  0.587); every adapter 0.897–0.933, i.e. within ±0.02 of base (Online
  −0.001, NoDream +0.011, Sleep +0.015 mean). Accuracy–length r = 0.925 in
  the grid regime, 0.22 in the math regime. **The H4 axis measured
  verbosity under a no-explanation prompt; no arm forgot or gained.**
- **B — probe.** Answer-only probe stays at chance for every model even
  with the rule stated in the prompt (0.29–0.35); 200 steps of direct
  training on masked→r6 pairs also stays at chance (loss floor ≈ 0.21) and
  costs chain accuracy; teacher-forced log-prob argmax is at chance with
  negative margins everywhere. With working allowed and the rule stated,
  Online reaches 0.78 (0.67–0.88 per seed) — but there the last written
  visible result *is* r6, so that variant measures computation, not
  shortcut knowledge. **H1–H3 were unobservable by construction and
  unreachable at this scale.**
- **C — generalisation.** Unstructured strings (no mirror structure):
  base 0.372, Online 0.844, Sleep 0.840, NoDream 0.887 (structured
  0.436 / 0.913 / 0.888 / 0.925; base's own gap 0.06). **General execution
  learning, not specialisation to the practice structure.**

**Standing conclusions of the 4B study after the follow-up.** Verified
self-distillation on the model's own correct chains repairs digit tracking
and teaches general execution of the stated rule within a few hundred
steps, under any of the three schedules, with self-generated variants and
inference-time critique adding nothing. The study cannot speak to insight
(probe unpassable) and its retention result was an artefact of the control
prompt. Both instrument problems are now documented with evidence, and any
redesign must (i) let the probe be answered in writing with a shortcut that
is checkable against a no-structure control set, and (ii) run the control
benchmark under its own prompt with outputs saved.

**Transfer (variant 2: four symbols, alternating rules; local MPS).** n=10
smoke: base 5/10, Online 2, Sleep 3, NoDream 2 (all ~chance for four
symbols; every model derails by step 3–4; no old-rule leakage). Full
zero-shot run, 14 models × 201 items, launched locally 19:39 IST
(`records/transfer/variant2_zero_shot_n201.*`). Design note:
`records/transfer_prereg.md`; task `tasks/fold_variant2.py`.

## 6. Repo state

```
doze-llm/
  PREREG.md                 preregistration (authoritative design) + calibration appendix
  README.md
  CONTEXT.md                this file
  CLAUDE.md                 working conventions for future sessions
  requirements.txt          pytest; mlx-lm on darwin
  calibrate.py              untrained-model difficulty check (--mode, --number-digits, stratified probe)
  run_arm.py                run one arm / one seed -> results/runs/*.json
  backends/  base.py            Backend + TrainableBackend protocols, GenResult, TrainStats
             scripted.py        ScriptedBackend (literal/shortcut) + FakeTrainableBackend
             mlx_backend.py     MLX generation + LoRA r=16 training (Mac, dev only)
             hf_backend.py      Transformers+PEFT (Modal, recorded numbers) — not yet run
             training_utils.py  LoRA targets, loss-mask conventions, example cycling
  tasks/     number_reduction.py  generator, prefix-disjoint splits, prompts (full/work/short, numbered), parsing, scoring
  eval/      orchestrate.py        run tags, preemption-safe orchestration decisions
             shortcut_detector.py stratified probe + binomial test, token collapse, CriterionTracker
             compute_ledger.py    per-arm Ledger, check_matched (+-5%, never mixes backends)
             analyze.py           A3 gate: check_seed persists verdicts, summarize/report only when matched
             control_bench.py     300 fixed GSM8K test items (ids frozen in control_bench_ids.json)
  arms/      harness.py  episode loop, nights after episode K, checkpoints, RunResult
             baseline.py awake.py online.py sleep.py
  sleep/     filters.py replay_buffer.py dreamer.py consolidate.py
  data/      gsm8k_test.jsonl (1319 rows, original OpenAI release)
  tests/     102 model-free tests
  results/   git-ignored working area (calibration*.json, final/, vm/)
  records/   COMMITTED artefacts: seed0/ (all arms + report), seed1/, calibration/, retro/, infra/, TIMELINE.md
  .gitignore                .venv/, results/, data/*.jsonl, __pycache__/
  modal_app.py              Modal entrypoint (used for seed 0); deploy/vultr/ for the VM path
```
Remote: https://github.com/Rj2790/doze-llm (private; `main`; freeze commit
6cfc319 pushed 2026-09-02).

Not yet written: `tasks/string_grammar.py`; log-rank test and figures
beyond `eval/analyze.py`'s probe-curve plot. `REVIEW.md` = the
fresh-session review (2026-09-02); `DEVIATIONS.md` created at freeze.

## 7. Frozen vs tunable

**Frozen (PREREG §4.1, §5, §6, §7):** L=12; mirror construction; prefix-
disjoint split with 20% held-out prefixes; probe = mask digits 8–12, target
r6; the five arms and their compute-matching rule; the 0.70 / two-checkpoint
criterion; 5 seeds; K=50; 600 episodes; analysis plan.
**Tunable:** prompt wording, system prompt, worked example, token budgets,
whether digits are numbered, probe sample size and statistics, MLX vs PEFT
implementation details, Modal configuration.
Any change to a frozen item goes in `DEVIATIONS.md` with a reason.

## 8. Immediate next steps, in order

Steps 1–14 are done: calibration, freeze, review amendments, path check,
seed 0 (§5d–5f), seeds 1–4 on Lightning/Vultr (§5h–5j), the five-seed
preregistered analysis (§5k, `records/analysis_final.md`). The Vultr VM is
destroyed (2026-09-12); trained-arm adapter states for seeds 1–4 are archived
off-repo at `~/doze-archive/vultr-final/trained_arm_states_seeds1-4.tar`.

16. **Decision (user):** what follows the 4B null — see §5m first: the two
    cheap checks (GSM8K re-score with outputs; probe positive control) should
    precede any model-scale decision, because they determine whether the
    4B study measured what it meant to. Options on the table:
    (a) stop and write up the null (PREREG §10); (b) an 8B dense
    confirmation with the same design, mainly to test whether the shortcut
    is discoverable at all; (c) the preregistered larger-model extension
    (Qwen3.8-27B was the candidate, §4 decisions log) — only worth the cost
    if (b) or a cheaper probe shows the shortcut is reachable. The 4B result
    says the task, not the schedule, is the binding constraint: no arm ever
    left chance on the probe.
17. Write-up of the 4B result (task, arms, matching, verdicts, exploratory
    schedule finding, caveats). `tasks/string_grammar.py` replication only
    if the follow-up route calls for it.

## 9. Working conventions

- Run everything from the repo root with the venv active. Python 3.14.
- Tests are model-free and must stay green; run `python -m pytest -q` before
  and after every change.
- Never put shell comments containing apostrophes on copy-paste lines (zsh
  will open a quote and hang on `quote>`).
- Prefer editing files in place over creating copies; the stale-copy problem
  has already cost one wasted calibration cycle.
- Report results as the header line + JSON + one or two raw completions.
- Don't announce routine actions; do flag anything that touches a frozen
  item, any inconsistency between code and PREREG, and any result that
  contradicts an expectation stated in this file.
- Cheap over clever: 4-bit for smoke tests, bf16 for numbers, cloud for grids.

