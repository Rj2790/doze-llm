"""Episode harness shared by all arms (PREREG §5, §6).

An arm sees the same seeded sequence of training instances as every other
arm. Episodes are 1-based. After every K-th episode (50, 100, ..., 600) the
arm's `night` hook runs (no-op for frozen arms), then a checkpoint is taken:
probe accuracy (stratified held-out items), held-out task accuracy, median
tokens per correct answer, the control benchmark (if configured) and a
ledger snapshot. The shortcut criterion is tracked with CriterionTracker.

Compute accounting: day-episode tokens and night tokens go into the ledger
(the quantity matched across arms). Evaluation tokens are recorded
separately in `RunResult.eval_tokens` and are never matched.
"""

from __future__ import annotations

import json
import random
import statistics
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol

from backends.base import Backend, GenResult
from eval import compute_ledger as cl
from eval import control_bench as cb
from eval import shortcut_detector as sd
from tasks import number_reduction as nr


@dataclass
class RunConfig:
    seed: int = 0
    n_episodes: int = 600
    k: int = 50                     # night / checkpoint period
    length: int = nr.DEFAULT_LENGTH
    mode: str = "work"              # prompt format for attempts (tunable)
    numbered: bool = False          # digits as position:digit (tunable)
    max_tokens: int = 400
    probe_max_tokens: int = 64
    n_probe: int = nr.N_PROBE       # PREREG §6.1; drawn from split.probe (disjoint from heldout)
    n_heldout_eval: int = nr.N_HELDOUT   # PREREG §4.1 held-out eval size
    criterion_threshold: float = 0.70
    control_items: int = 0          # 0 = skip control benchmark (tests / local)
    control_max_tokens: int = 512


@dataclass
class Episode:
    episode: int
    digits: str
    text: str
    answer: str | None
    correct: bool
    steps_correct: int
    tokens: int
    rounds: int = 1
    seconds: float = 0.0            # wall-clock for this episode (attempt + after_episode)


@dataclass
class Checkpoint:
    episode: int
    probe: sd.ProbeResult
    heldout_accuracy: float
    heldout_n: int
    median_tokens_correct: float | None
    control_accuracy: float | None
    ledger: dict
    eval_tokens: int
    seconds: float                  # wall-clock of this checkpoint evaluation
    day_seconds_mean: float | None = None   # mean episode wall-clock since the previous checkpoint
    night_seconds: float | None = None

    def flat(self) -> dict:
        d = {"episode": self.episode, "probe_accuracy": self.probe.accuracy,
             "day_seconds_mean": self.day_seconds_mean, "night_seconds": self.night_seconds,
             "probe_p_value": self.probe.p_value, "probe_above_chance": self.probe.above_chance,
             "heldout_accuracy": self.heldout_accuracy, "heldout_n": self.heldout_n,
             "median_tokens_correct": self.median_tokens_correct,
             "control_accuracy": self.control_accuracy, "eval_tokens": self.eval_tokens,
             "seconds": round(self.seconds, 1)}
        d.update({f"ledger_{k}": v for k, v in self.ledger.items() if k != "episode"})
        return d


@dataclass
class RunResult:
    arm: str
    config: RunConfig
    episodes: list[Episode]
    checkpoints: list[Checkpoint]
    ledger: cl.Ledger
    criterion_episode: int | None
    eval_tokens: int
    nights: list[dict] = field(default_factory=list)
    matched: dict | None = None     # A3: check_matched verdict, written by eval/analyze.py
    partial: bool = False           # B1: True while the run is still in progress

    def save(self, path: str | Path) -> None:
        d = {"arm": self.arm, "backend": self.ledger.backend, "config": asdict(self.config),
             "criterion_episode": self.criterion_episode, "eval_tokens": self.eval_tokens, "matched": self.matched,
             "partial": self.partial,
             "ledger": asdict(self.ledger), "checkpoints": [c.flat() for c in self.checkpoints],
             "nights": self.nights, "episodes": [asdict(e) for e in self.episodes]}
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(d, indent=1))


def load_run(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


class Arm(Protocol):
    name: str

    def attempt(self, prompt: str, backend: Backend, max_tokens: int,
                phase: str = "day") -> tuple[GenResult, int]:
        """Solve one prompt. Returns (final result, rounds). Total tokens the
        arm generated for this prompt must be in result.completion_tokens.
        phase is "day" (ledger-matched) or "eval" (checkpoint, not matched)."""
        ...

    def attempt_many(self, prompts: Sequence[str], backend: Backend, max_tokens: int,
                     phase: str = "eval") -> list[tuple[GenResult, int]]:
        """Batched attempt (checkpoint evaluation). Single-shot arms send one
        backend.generate call; Awake loops per prompt (B1)."""
        ...

    def after_episode(self, ep: Episode, backend: Backend, ledger: cl.Ledger, cfg: RunConfig) -> None:
        """Hook right after an episode (Online trains here). Others: no-op."""
        ...

    def night(self, episodes: Sequence[Episode], backend: Backend, ledger: cl.Ledger,
              cfg: RunConfig, split: nr.Split) -> dict:
        """Offline phase after every K episodes. Frozen arms return {}."""
        ...


def batched_attempt_many(prompts: Sequence[str], backend: Backend, max_tokens: int) -> list[tuple[GenResult, int]]:
    """Default attempt_many for single-shot arms: one batched generate call."""
    if not prompts:
        return []
    return [(r, 1) for r in backend.generate(list(prompts), max_tokens=max_tokens)]


def is_checkpoint(episode: int, k: int) -> bool:
    """1-based episode index; nights fall after episodes k, 2k, ..."""
    return episode > 0 and episode % k == 0


def episode_sequence(split: nr.Split, cfg: RunConfig) -> list[nr.Instance]:
    """Seeded sequence of training instances, identical across arms.
    Samples with replacement from the 800 training instances when
    n_episodes > len(train)."""
    rng = random.Random(f"episodes-{cfg.seed}")
    train = list(split.train)
    if cfg.n_episodes <= len(train):
        rng.shuffle(train)
        return train[: cfg.n_episodes]
    return [rng.choice(train) for _ in range(cfg.n_episodes)]


def eval_items(split: nr.Split, cfg: RunConfig) -> tuple[list[nr.Instance], list[nr.Instance]]:
    """(probe items, held-out eval items): both from held-out prefixes only,
    probe from split.probe (disjoint from split.heldout, C7), stratified."""
    probe = sd.stratified_probe_items(split.probe, n=cfg.n_probe, seed=cfg.seed)
    heldout = list(split.heldout[: cfg.n_heldout_eval])
    assert all(x.prefix in split.heldout_prefixes for x in probe + heldout)
    assert not {x.digits for x in probe} & {x.digits for x in heldout}
    return probe, heldout


def _checkpoint(arm: Arm, backend: Backend, cfg: RunConfig, episode: int,
                probe_items: Sequence[nr.Instance], heldout_items: Sequence[nr.Instance],
                control: Sequence[cb.Item], ledger: cl.Ledger) -> Checkpoint:
    t0 = time.time()
    used = 0

    def gen_many(prompts: list[str], max_tokens: int) -> list[GenResult]:
        nonlocal used
        rs = [r for r, _ in arm.attempt_many(prompts, backend, max_tokens, phase="eval")]
        used += sum(r.completion_tokens for r in rs)
        return rs

    probe_out = gen_many([nr.format_probe(x, numbered=cfg.numbered) for x in probe_items], cfg.probe_max_tokens)
    probe = sd.score_probe(probe_items, [r.text for r in probe_out])
    held_out = gen_many([nr.format_prompt(x, cfg.mode, numbered=cfg.numbered) for x in heldout_items], cfg.max_tokens)
    tok_correct, n_correct = [], 0
    for x, r in zip(heldout_items, held_out):
        if nr.score(x, r.text)["correct"]:
            n_correct += 1
            tok_correct.append(r.completion_tokens)
    control_acc = None
    if control:
        ctrl_out = gen_many([cb.format_prompt(x) for x in control], cfg.control_max_tokens)
        control_acc = cb.score(control, [r.text for r in ctrl_out]).accuracy
    return Checkpoint(
        episode=episode, probe=probe,
        heldout_accuracy=n_correct / len(heldout_items) if heldout_items else float("nan"),
        heldout_n=len(heldout_items),
        median_tokens_correct=statistics.median(tok_correct) if tok_correct else None,
        control_accuracy=control_acc, ledger=ledger.checkpoint(episode),
        eval_tokens=used, seconds=time.time() - t0)


def run_arm(arm: Arm, backend: Backend, cfg: RunConfig,
            control: Sequence[cb.Item] | None = None, log=None,
            save_path: str | Path | None = None, on_checkpoint=None) -> RunResult:
    """Run one arm for one seed. If save_path is given the (partial) result is
    written after every checkpoint (B1) and `on_checkpoint()` is called
    afterwards (e.g. to commit a volume)."""
    if hasattr(backend, "set_seed"):
        backend.set_seed(cfg.seed)                    # A1: sampling / init generators follow the run seed
    split = nr.make_split(seed=cfg.seed, length=cfg.length)
    seq = episode_sequence(split, cfg)
    probe_items, heldout_items = eval_items(split, cfg)
    if control is None and cfg.control_items:
        control = cb.load_frozen()[: cfg.control_items]
    control = list(control or [])
    ledger = cl.Ledger(arm=arm.name, backend=backend.name)
    tracker = sd.CriterionTracker(threshold=cfg.criterion_threshold)
    episodes: list[Episode] = []
    checkpoints: list[Checkpoint] = []
    nights: list[dict] = []
    eval_tokens = 0

    def result(partial: bool) -> RunResult:
        return RunResult(arm=arm.name, config=cfg, episodes=episodes, checkpoints=checkpoints, ledger=ledger,
                         criterion_episode=tracker.first_met_at, eval_tokens=eval_tokens, nights=nights,
                         partial=partial)

    def persist(partial: bool) -> None:
        if save_path is not None:
            result(partial).save(save_path)
            if on_checkpoint:
                on_checkpoint()

    # A5: episode-0 checkpoint before any day episode (forgetting baseline, untrained probe)
    ck0 = _checkpoint(arm, backend, cfg, 0, probe_items, heldout_items, control, ledger)
    eval_tokens += ck0.eval_tokens
    checkpoints.append(ck0)
    tracker.update(0, ck0.probe)
    if log:
        log(f"[{arm.name} seed={cfg.seed}] ep=0 probe={ck0.probe.accuracy:.3f} heldout={ck0.heldout_accuracy:.3f} "
            f"control={ck0.control_accuracy} {ck0.seconds:.0f}s")
    persist(partial=True)

    for i, inst in enumerate(seq, start=1):
        t0 = time.time()
        r, rounds = arm.attempt(nr.format_prompt(inst, cfg.mode, numbered=cfg.numbered), backend, cfg.max_tokens,
                                phase="day")
        s = nr.score(inst, r.text)
        ledger.add(tokens_generated=r.completion_tokens, gpu_seconds=time.time() - t0)
        episodes.append(Episode(episode=i, digits=inst.digits, text=r.text, answer=s["answer"],
                                correct=s["correct"], steps_correct=s["steps_correct"],
                                tokens=r.completion_tokens, rounds=rounds))
        arm.after_episode(episodes[-1], backend, ledger, cfg)
        episodes[-1].seconds = time.time() - t0
        if is_checkpoint(i, cfg.k):
            day = episodes[i - cfg.k: i]
            assert len(day) == cfg.k and day[0].episode == i - cfg.k + 1 and day[-1].episode == i
            tn = time.time()
            night_info = arm.night(day, backend, ledger, cfg, split) or {}
            night_s = time.time() - tn
            if night_info:
                nights.append({"episode": i, **night_info})
            ck = _checkpoint(arm, backend, cfg, i, probe_items, heldout_items, control, ledger)
            ck.day_seconds_mean = statistics.mean(e.seconds for e in day)
            ck.night_seconds = night_s
            eval_tokens += ck.eval_tokens
            checkpoints.append(ck)
            tracker.update(i, ck.probe)
            if log:
                log(f"[{arm.name} seed={cfg.seed}] ep={i} probe={ck.probe.accuracy:.3f} "
                    f"heldout={ck.heldout_accuracy:.3f} control={ck.control_accuracy} "
                    f"tokens={ledger.tokens_generated} steps={ledger.gradient_steps} "
                    f"| episode {ck.day_seconds_mean:.1f}s night {night_s:.0f}s checkpoint {ck.seconds:.0f}s")
            persist(partial=(i < cfg.n_episodes))
    res = result(partial=False)
    if save_path is not None:
        res.save(save_path)
    return res
