# DEVIATIONS

No deviations from PREREG v1.0 as of 2026-09-02.

## Documentation corrections (not design deviations)

- **2026-09-02 — Appendix A calibration items predate A6.** The 40
  calibration items in PREREG Appendix A were drawn from the pre-amendment
  split (first 40 held-out items before answer-stratified sampling). On the
  frozen split's first 40 held-out items the untrained model scores 0.25 /
  0.075 exact chains (vs 0.475 / 0.175 recorded). No design item changes;
  Appendix A carries a dated addendum. Authoritative untrained accuracy is
  the Baseline episode-0 checkpoint on the 201 frozen held-out items.

- **2026-09-02 — §5 night step 2 wording: dream verification.** The frozen
  text said "dreams must be fully correct (C3)". The decided C3 rule
  (correct final answer AND ≥ 9/11 steps) applies to Sleep, Online and the
  dreamer alike; "fully correct" was the builder's mis-wording, and the
  code was stricter than decided (exact chain plus work-line check). Both
  are aligned to C3. Not a design change.

## Post-pilot changes (2026-09-04) — after seed 0 (Sleep, Online, Awake); PREREG §8a design unchanged

Evidence from the pilot (CONTEXT.md §5d): 714 dreams, 152 accepted of which
58 were verbatim copies of the source; 283 unparsable (40%, rising with
training); 4 of 94 real variations mirror-structured; same-seed attempts
diverged after the first LoRA update (Online ep-50 held-out 0.363 vs 0.527).

1. **Reject verbatim dream copies** (`duplicate`). A copy of the source is
   not a variation (PREREG §5 step 3) and inflated accepted-dream counts by
   38% in the pilot. Tunable implementation detail; recorded because it
   changes the dream stream after seed 0.
2. **Decoupled dreamer.** Step 1: the model proposes new digits only
   (temperature 0.8). Step 2: the proposed string is solved with the
   ordinary day prompt (greedy) and verified with the C3 keep rule. Reason:
   the one-step "Digits: + solution" format drifted as training targets
   never contained a Digits line (283/714 unparsable). The solve step now
   uses exactly the format the model is trained on. Dream tokens = proposal
   + solve tokens.
3. **Variation restricted to digits 8–12** (prefix kept). Introduced before
   the pilot as tunable wording; recorded here with its consequence: held-out
   and prefix-changed rejections were 0/714, and only 4/94 real variations
   were mirror-structured (digits 1–7 fix r6, so dreams vary r11). Kept.
4. **Deterministic CUDA algorithms** (`torch.use_deterministic_algorithms(
   True)`, cuDNN deterministic, CUBLAS_WORKSPACE_CONFIG=:4096:8), default on
   for all runs after the pilot. Reason: attempt 1 vs attempt 2 of the same
   seed diverged by up to 16 points on held-out after 43 steps. Slowdown
   measured on a 4-episode Online run: see CONTEXT.md §5e (filled when
   measured).

**Secondary insight metrics added after the pilot; the primary metric
(PREREG §6.1 probe criterion) is unchanged.** Computed at every checkpoint
and by `eval/analyze.py`:
- (a) `mirror_bias`: on 120 held-out UNSTRUCTURED strings (unseen prefixes,
  answer-uniform), the fraction of erroneous steps 9–11 whose value equals
  the mirror value (r9↔r8, r10↔r7, r11↔r6). Reported with its chance level
  (½ per erroneous step when the mirror value differs from the truth, 0
  when it coincides).
- (b) `short_gap`: short-mode (answer-only) accuracy on 120 structured
  held-out items minus on the 120 unstructured items; both reported.
- (c) `late_vs_early`: from the work-mode held-out completions already
  generated at each checkpoint, error rate at steps 9–11 given steps 1–8
  correct, vs error rate at steps 3–8 given steps 1–2 correct.
- (d) `volatility`: population std of held-out accuracy across checkpoints.
Also added: GSM8K unparsable fraction per checkpoint; per-item held-out
rows (digits, parsed steps, answer) saved in the run file; final adapter
state saved at run end for retro-analysis. Pilot retro-computation: only
(d) and a day-trajectory proxy for (c) are possible for Sleep/Online (no
adapters or eval texts were saved); (a)–(c) and GSM8K parse rate are
computed for the frozen arms from the base model (CONTEXT.md §5e).
