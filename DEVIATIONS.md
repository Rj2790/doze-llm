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
