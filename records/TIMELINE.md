# Run timeline, failures and costs

All dates 2026. GPU costs are list prices at the time (Modal L4 $0.80/h, A100-80GB $2.50/h; Lightning L4 ≈ 1.3 credits/h measured).

| date | what | where | outcome | cost |
|---|---|---|---|---|
| 09-02 | Calibration runs 1–6 (compact vs work prompt, 4-bit vs bf16, numbered digits, stratified probe) | MLX, Mac | work mode chosen; numbered rejected; probe artifact fixed | $0 |
| 09-02 | PREREG v1.0 frozen (commit 6cfc319); fresh-session review; amendments A1–A6, B1–B4, C1–C7 | — | — | — |
| 09-02 | Modal path check, 4-episode Baseline | A100-80GB | pass; 5.9 s/episode, 255 s/full checkpoint | $0.55 |
| 09-02 | Pre-flight validation (items 1–4), 40-item agreement, dream stats; starmap check; L4 timing run | L4 | all pass; L4 6% slower, 3× cheaper than A100 | ~$1 |
| 09-02/03 | Seed 0 attempt 1: Sleep+Online | L4 | lost at ep 150/100 — client disconnect killed the ephemeral app | ~$1.7 wasted |
| 09-03 | Seed 0 attempt 2 (detached, remote orchestrator): Sleep+Online | L4 | orchestrator preempted at ~ep 300 → duplicates spawned; originals salvaged from the volume (complete) | $2.72 + $2.57 useful, ~$3 wasted |
| 09-03 | Awake seed 0, attempts 1–2 | L4 | attempt 1 unbatched eval (would take ~30 h) stopped; attempt 2 cancelled by client exit (.remote) | ~$0.75 wasted |
| 09-03/04 | Awake seed 0, attempt 3 (spawned, batched eval) | L4 | complete; 8.43 h | $6.74 |
| 09-04 | Sleep-NoDream + Baseline seed 0 (deterministic, secondary metrics) | L4 | complete; all five arms MATCHED | $4.54 + $2.16 |
| 09-04 | det-on/off timing pair; frozen-arm retro | L4 | det. ≈ +50%/+90% for LoRA arms, none for frozen; base model has no mirror bias, 0% GSM8K unparsable | ~$1.5 |
| 09-04 | Post-pilot changes: duplicate rejection, decoupled dreamer, 1–3 digit rule, deterministic default, secondary metrics, resume/preemption safety | — | 117 tests | — |
| 09-05 | Modal wallet ~$1; Vultr GPU plan gated (support ticket); Lightning AI chosen for seed 1 | — | — | — |
| 09-05 | Lightning validation (4-episode Baseline) | L4 Studio | pass; 4.9 s/episode | ~0.3 credits |
| 09-05 | Idle Studio after failed L4_X_2 switch + laptop network drop | L4 Studio | ~8 h idle, ~14 of 30 credits lost; guards added (runner stops Studio; monitor) | −14 credits |
| 09-05 | Seed 1 Sleep (two-step dreamer) then Baseline, chained | L4 Studio | complete; Sleep 4.42 h, Baseline 2.57 h; MATCHED; Studio stopped itself | ≈ 9 credits |

Seed-0 useful compute: $18.73; seed-0 waste from failures: ≈ $6.3. Remaining plan: Online, Sleep-NoDream, Awake for seed 1 and seeds 2–4 on Vultr (approval pending) or new credits.
| 09-07 | Vultr GPU access approved after Trust & Safety ticket | — | — | — |
| 09-08 | Vultr `vcg-a16-12c-128g-32vram`: blr instance unbootable (host timeout, destroyed); sgp instance created; setup + validation pass (6.5 s/episode, 96 s small checkpoint) | 2×A16 | grid launched 07:59 UTC: seed 1 (online, nodream, awake) + seeds 2–4 all arms; ~5–6 days | ≈ $120–130 of $300 credit (projected) |
- 2026-09-11 17:01 IST — Baseline seed 4 done (18,710 s, exit 0). gpu1 starts Awake seed 3 (final queued arm). 23/25 complete.
- 2026-09-12 06:07 IST — Awake seed 4 done (63,575 s, exit 0); gpu0 QUEUE_DONE. Seed 4 five-arm check MATCHED. 24/25 complete.
- 2026-09-12 10:35 IST — Awake seed 3 done (63,288 s, exit 0); gpu1 QUEUE_DONE. Seed 3 five-arm check MATCHED. **25/25 complete.** Grid wall-clock 3 d 21 h on 2×A16.
- 2026-09-12 10:37 IST — Final analysis run (`analysis_final.md`): H1–H3 all censored (0/25 arms met criterion), H4 reversed (Sleep−Online GSM8K Δ −0.123, p=0.125). Trained-arm adapter states archived to `~/doze-archive/vultr-final/`.
- 2026-09-12 11:02 IST — Vultr instance 8b8aa0a1 (sgp, 2×A16) destroyed after archive verified (tar 4.37 GB, 40 entries, byte count matches VM). Pending charges at teardown $89.50. No compute running anywhere.
- 2026-09-12 12:30 IST — All artefacts consolidated locally: Vultr tar extracted to `~/doze-archive/models/`, Modal volume (16 GB) downloaded, seed 1 Sleep state fetched from the Lightning Studio over SSH (Studio started on CPU, stopped after), 13 adapters exported to PEFT format. Deep exploratory analysis written (`analysis_deep*.md`, `figures/`).
- 2026-09-12 13:05 IST — Base Qwen3-4B downloaded to the local archive; exported adapter smoke test (sleep_seed3, 6 held-out items): base 2/6, adapter 6/6. Local archive complete.
- 2026-09-12 14:10 IST — Hypothesis-blind review (`analysis_outsider.md`, own + independent session). Verified: GSM8K accuracy vs eval-token proxy r=0.908 (n=156); probe insensitive by construction (answer-only accuracy on visible items 0.31–0.39 in all arms); Sleep-only first-night dip 5/5. CONTEXT §5m records the revised reading.
- 2026-09-12 16:55 IST — Follow-up launched on Modal L4 (two detached containers: A; B,C) after a second blr A16 instance failed to start (destroyed). Pre-analysis note `followup_prereg.md`; code `eval/followup.py` smoke-tested end to end on MPS.
- 2026-09-12 19:52 IST — Follow-up part A complete. Base GSM8K 0.910 under a math prompt (0.587 in the grid regime); all 13 adapters within ±0.02 of base. H4 axis was a prompt-regime artefact. B/C complete 18:32 IST: probe unpassable at this scale; general execution learning confirmed. CONTEXT §5n.
- 2026-09-12 19:39 IST — Variant-2 zero-shot transfer (14 models × 201) running locally on MPS.
- 2026-09-12 22:05 IST — Variant-2 zero-shot transfer done locally: all 14 models at chance; adapters worse than base at step 1 (negative transfer of the one-rule habit, not the old table). `records/transfer/RESULTS.md`, CONTEXT §5n.
