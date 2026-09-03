"""The night (PREREG §5, Sleep arm), steps 1-7:

1. take the day's K trajectories        2. keep correct + near-miss
3. dream (verified) unless disabled     4. interleave 1:1 with a uniform
                                           replay-buffer sample
5. train LoRA for S steps (S = K by default, so gradient steps match Online
   by construction)                     6. one weight-decay pass on LoRA params
7. append the night's kept REAL trajectories to the replay buffer (dreams
   are trained on once and never replayed; amendment C2)
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass

from arms.harness import Episode, RunConfig
from backends.base import TrainableBackend
from eval import compute_ledger as cl
from sleep import dreamer, filters
from sleep.replay_buffer import ReplayBuffer
from tasks import number_reduction as nr


@dataclass
class SleepConfig:
    n_variations: int = 2            # PREREG §5 step 3
    steps_per_night: int | None = None   # None -> K (matches Online: 1 step/episode)
    weight_decay: float = 0.05       # tunable; the pruning analogue (step 6)
    dream_max_tokens: int = 400
    dream_temperature: float = dreamer.DREAM_TEMPERATURE


def run_night(day: Sequence[Episode], backend: TrainableBackend, ledger: cl.Ledger, cfg: RunConfig,
              split: nr.Split, scfg: SleepConfig, buffer: ReplayBuffer, night_index: int,
              dream_enabled: bool) -> dict:
    t0 = time.time()
    kept_eps = filters.filter_day(day)
    kept = [filters.to_example(e, cfg.mode, cfg.numbered) for e in kept_eps]
    buffer.check(kept)   # day episodes are training instances by construction; verify anyway

    dreams, dstats = [], {"generated": 0, "tokens": 0, "kept": 0, "rejected": {}, "structured": 0, "structured_frac": None,
                          "duplicates": 0, "accepted": []}
    if dream_enabled and kept:
        dreams, dstats = dreamer.dream(backend, kept, scfg.n_variations, split, cfg.mode, cfg.numbered,
                                       max_tokens=scfg.dream_max_tokens, temperature=scfg.dream_temperature)
        ledger.add(tokens_generated=dstats["tokens"])
    new = kept + dreams
    buffer.check(new)    # C1: defence in depth — nothing with a held-out prefix reaches backend.train

    replay = buffer.sample(len(new), seed=cfg.seed * 1000 + night_index)
    train_set = dreamer.interleave(new, replay)

    steps = scfg.steps_per_night if scfg.steps_per_night is not None else cfg.k
    if train_set:
        st = backend.train([(e.prompt, e.completion) for e in train_set], steps=steps,
                           seed=cfg.seed * 1000 + night_index)
    else:
        from backends.base import TrainStats
        st = TrainStats(steps=0, training_tokens=0)
    backend.decay_adapter(scfg.weight_decay)
    buffer.add(kept, night=night_index)   # C2: real trajectories only
    ledger.add(gradient_steps=st.steps, training_tokens=st.training_tokens, gpu_seconds=time.time() - t0)

    return {"night": night_index, "day_episodes": len(day), "kept": len(kept), "new": len(new),
            "dreams_generated": dstats["generated"], "dreams_kept": dstats["kept"],
            "dreams_rejected": dstats["rejected"], "dream_tokens": dstats["tokens"],
            "dreams_rejected_wrong": dstats["rejected"].get("wrong", 0),
            "dreams_rejected_heldout": dstats["rejected"].get("heldout_prefix", 0),
            "dreams_rejected_prefix_changed": dstats["rejected"].get("prefix_changed", 0),
            "dreams_structured": dstats["structured"], "dreams_structured_frac": dstats["structured_frac"],
            "dreams_rejected_unparsable": dstats["rejected"].get("unparsable", 0),
            "dreams_duplicates_of_source": dstats["duplicates"], "dreams_accepted": dstats["accepted"],
            "replay": len(replay), "train_set": len(train_set), "steps": st.steps,
            "training_tokens": st.training_tokens, "loss": st.loss, "buffer_size": len(buffer),
            "weight_decay": scfg.weight_decay, "seconds": round(time.time() - t0, 1),
            "empty_night": not train_set}
