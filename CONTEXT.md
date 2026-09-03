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

**Preemption-safety work required before any further seed** (not yet
built): unique run directory per launch; an idempotent orchestrator that,
on restart, waits on an existing in-progress run file instead of
re-spawning; resumable GPU runs (save adapter + optimizer + replay buffer +
arm state at every checkpoint; resume from the last one). Preemption is a
standing risk for 3–8 h functions and a 25-run grid will hit it.

Also observed in attempt 2: run-to-run non-determinism at fixed seed on
CUDA after the first update (Sleep ep-50 probe 0.417 vs 0.350 in attempt 1;
Online ep-50 held-out 0.527 vs 0.363). Consider
`torch.use_deterministic_algorithms(True)` (tunable) for the grid.

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
  eval/      shortcut_detector.py stratified probe + binomial test, token collapse, CriterionTracker
             compute_ledger.py    per-arm Ledger, check_matched (+-5%, never mixes backends)
             analyze.py           A3 gate: check_seed persists verdicts, summarize/report only when matched
             control_bench.py     300 fixed GSM8K test items (ids frozen in control_bench_ids.json)
  arms/      harness.py  episode loop, nights after episode K, checkpoints, RunResult
             baseline.py awake.py online.py sleep.py
  sleep/     filters.py replay_buffer.py dreamer.py consolidate.py
  data/      gsm8k_test.jsonl (1319 rows, original OpenAI release)
  tests/     81 model-free tests
  results/   calibration*.json; runs/ (arm runs); ledgers/ (per-arm ledgers, read by analyze.py)
  .gitignore                .venv/, results/, data/*.jsonl, __pycache__/
  modal_app.py              Modal entrypoint for the grid — not yet run
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

1. ~~Finish the three in-flight items~~ done 2026-09-02: stray copy deleted;
   stratified probe + binomial test with tests; `--number-digits` variant
   run and rejected (§4, §5 runs 5–6).
2. ~~Pick the final prompt format~~ work mode, unnumbered. PREREG §6.1
   fixed; calibration appendix added to PREREG. Frozen 2026-09-02 as
   PREREG v1.0 (hash in PREREG header); `DEVIATIONS.md` created empty.
3. ~~`eval/compute_ledger.py`~~ done, with tests (incl. backend-mixing guard).
4. ~~`eval/control_bench.py`~~ done; 300 ids frozen; not yet run on a model.
5. ~~Arms: Baseline and Awake~~ written and tested model-free; 4-bit smoke
   against the real model in progress (see §5 / results/runs).
6. `backends/hf_backend.py` and `modal_app.py` written but **never
   executed** (no torch/modal locally). First cloud action: run the
   Baseline arm for 4 episodes on Modal to validate the path.
7. ~~`sleep/` components + Online, Sleep, Sleep-NoDream arms~~ written and
   tested against a fake trainable backend; MLX LoRA path smoke-tested
   locally (4-bit).
8. ~~Fresh-session review~~ done 2026-09-02 (`REVIEW.md`); all A/B/C items
   applied (PREREG §8a). Frozen: see PREREG header for the commit hash.
9. ~~Modal path check~~ done 2026-09-02, all six validation items pass
   (§5a). Original list: `modal run modal_app.py::main --arm baseline
   --seed 0 --n-episodes 4 --k 2 --n-probe 6 --n-heldout-eval 4`, with the
   first-execution validation list from REVIEW.md: (1) model.dtype bf16,
   adapter params fp32, ~33M trainable; (2) chat template emits the empty
   think block, generation stops at <|im_end|>, completion_tokens ≈
   count_tokens(text) ±1; (3) two greedy calls identical; (4) train on one
   repeated example: finite decreasing loss, training_tokens = completion
   length + 1, lora_norm grows, decay_adapter(0.05) shrinks it by exactly
   5%, reset_adapter returns it to zero; (5) starmap returns three results,
   volume commit persists /results/runs, HF cache reused; (6) wall-clock
   per episode and per checkpoint.
10. ~~Pre-launch (A4)~~ done (§5a): ratio 1.000–0.975 across yields, no
    adjustment. **Open decision before any trainable arm:** dream yield is
    0 on the untrained model (§5a). Options: (a) run as designed and treat
    yield-per-night as a reported metric (H3 may be uninformative if yield
    stays ~0); (b) tunable prompt work on the dream instruction (e.g. ask
    for changes only in the last five digits, which also removes the
    held-out-prefix rejections) — prompt wording is tunable, but steering
    which digits change is close to the hidden structure and must be
    weighed; (c) defer. Builder recommendation: (a) for the first seed,
    log yield per night, decide (b) with data.
11. 5-seed grid on Modal; `eval/analyze.py`; `tasks/string_grammar.py` as
    replication; write-up.

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

