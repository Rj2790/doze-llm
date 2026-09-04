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
from eval import implicit
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
    n_implicit: int = 120           # items per set for mirror_bias / short_gap (0 = skip); post-pilot
    short_max_tokens: int = 16


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
    # post-pilot secondary metrics (DEVIATIONS 2026-09-04)
    mirror: dict | None = None
    short: dict | None = None
    late_early: dict | None = None
    control_unparsable: float | None = None
    heldout_rows: list[dict] = field(default_factory=list)   # per held-out item: digits, parsed steps, answer

    def flat(self) -> dict:
        m, sh, le = self.mirror or {}, self.short or {}, self.late_early or {}
        d = {"episode": self.episode, "probe_accuracy": self.probe.accuracy,
             "probe_n": self.probe.n, "probe_correct": self.probe.correct,
             "day_seconds_mean": self.day_seconds_mean, "night_seconds": self.night_seconds,
             "mirror_bias": m.get("mirror_bias"), "mirror_bias_chance": m.get("chance"), "mirror_bias_n": m.get("n_error_steps"),
             "short_structured_acc": sh.get("structured_acc"), "short_unstructured_acc": sh.get("unstructured_acc"),
             "short_gap": sh.get("gap"),
             "late_error_rate": le.get("late_error_rate"), "early_error_rate": le.get("early_error_rate"),
             "late_n": le.get("n_late"), "early_n": le.get("n_early"),
             "control_unparsable": self.control_unparsable, "heldout_rows": self.heldout_rows,
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
    resumed_from: int | None = None # checkpoint episode this run resumed from (preemption safety)

    def save(self, path: str | Path) -> None:
        d = {"arm": self.arm, "backend": self.ledger.backend, "config": asdict(self.config),
             "criterion_episode": self.criterion_episode, "eval_tokens": self.eval_tokens, "matched": self.matched,
             "partial": self.partial, "resumed_from": self.resumed_from,
             "ledger": asdict(self.ledger), "checkpoints": [c.flat() for c in self.checkpoints],
             "nights": self.nights, "episodes": [asdict(e) for e in self.episodes]}
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(d, indent=1))


def load_run(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def state_dir(save_path: str | Path) -> Path:
    return Path(str(save_path) + ".state")


def _checkpoint_from_flat(c: dict) -> Checkpoint:
    probe = sd.ProbeResult(n=c["probe_n"], correct=c["probe_correct"], accuracy=c["probe_accuracy"],
                           above_chance=c["probe_above_chance"], p_value=c["probe_p_value"])
    ledger = {"episode": c["episode"], **{k[len("ledger_"):]: v for k, v in c.items() if k.startswith("ledger_")}}
    mirror = {"mirror_bias": c.get("mirror_bias"), "chance": c.get("mirror_bias_chance"), "n_error_steps": c.get("mirror_bias_n")}
    short = {"structured_acc": c.get("short_structured_acc"), "unstructured_acc": c.get("short_unstructured_acc"), "gap": c.get("short_gap")}
    le = {"late_error_rate": c.get("late_error_rate"), "early_error_rate": c.get("early_error_rate"),
          "n_late": c.get("late_n"), "n_early": c.get("early_n")}
    return Checkpoint(episode=c["episode"], probe=probe, heldout_accuracy=c["heldout_accuracy"],
                      heldout_n=c["heldout_n"], median_tokens_correct=c["median_tokens_correct"],
                      control_accuracy=c["control_accuracy"], ledger=ledger, eval_tokens=c["eval_tokens"],
                      seconds=c["seconds"], day_seconds_mean=c.get("day_seconds_mean"),
                      night_seconds=c.get("night_seconds"), mirror=mirror, short=short, late_early=le,
                      control_unparsable=c.get("control_unparsable"), heldout_rows=c.get("heldout_rows", []))


def _restore(d: dict) -> tuple[list[Episode], list[Checkpoint], cl.Ledger, list[dict], int]:
    """Rebuild in-memory state from a partial run file, truncated to its last
    checkpoint (backend/arm state on disk corresponds to that checkpoint)."""
    ck_ep = d["checkpoints"][-1]["episode"]
    episodes = [Episode(**e) for e in d["episodes"] if e["episode"] <= ck_ep]
    checkpoints = [_checkpoint_from_flat(c) for c in d["checkpoints"]]
    ledger = cl.Ledger(**d["ledger"])
    nights = [n for n in d["nights"] if n["episode"] <= ck_ep]
    return episodes, checkpoints, ledger, nights, ck_ep


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


def implicit_items(split: nr.Split, cfg: RunConfig) -> tuple[list[nr.Instance], list[nr.Instance]]:
    """(structured, unstructured) item sets for short_gap / mirror_bias; both
    from held-out prefixes, n_implicit each (0 -> empty)."""
    if cfg.n_implicit <= 0:
        return [], []
    structured = sd.stratified_probe_items(split.heldout, n=cfg.n_implicit, seed=cfg.seed + 7)
    unstructured = nr.unstructured_heldout(split, n=cfg.n_implicit, seed=cfg.seed)
    return structured, unstructured


def _checkpoint(arm: Arm, backend: Backend, cfg: RunConfig, episode: int,
                probe_items: Sequence[nr.Instance], heldout_items: Sequence[nr.Instance],
                control: Sequence[cb.Item], ledger: cl.Ledger,
                implicit_sets: tuple[Sequence[nr.Instance], Sequence[nr.Instance]] = ((), ())) -> Checkpoint:
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
    heldout_rows = [{"digits": x.digits, "steps": "".join(nr.parse_steps(r.text) or []), "answer": nr.parse_answer(r.text)}
                    for x, r in zip(heldout_items, held_out)]
    late_early = implicit.late_vs_early(heldout_items, [r.text for r in held_out]) if heldout_items else None
    control_acc, control_unp = None, None
    if control:
        ctrl_out = gen_many([cb.format_prompt(x) for x in control], cfg.control_max_tokens)
        br = cb.score(control, [r.text for r in ctrl_out])
        control_acc, control_unp = br.accuracy, br.unparsable_frac
    mirror = short = None
    s_items, u_items = implicit_sets
    if s_items and u_items:
        u_work = gen_many([nr.format_prompt(x, cfg.mode, numbered=cfg.numbered) for x in u_items], cfg.max_tokens)
        mirror = implicit.mirror_bias(u_items, [r.text for r in u_work])
        s_short = gen_many([nr.format_prompt(x, "short", numbered=cfg.numbered) for x in s_items], cfg.short_max_tokens)
        u_short = gen_many([nr.format_prompt(x, "short", numbered=cfg.numbered) for x in u_items], cfg.short_max_tokens)
        short = implicit.short_gap(s_items, [r.text for r in s_short], u_items, [r.text for r in u_short])
    return Checkpoint(
        episode=episode, probe=probe,
        heldout_accuracy=n_correct / len(heldout_items) if heldout_items else float("nan"),
        heldout_n=len(heldout_items),
        median_tokens_correct=statistics.median(tok_correct) if tok_correct else None,
        control_accuracy=control_acc, ledger=ledger.checkpoint(episode),
        eval_tokens=used, seconds=time.time() - t0, mirror=mirror, short=short, late_early=late_early,
        control_unparsable=control_unp, heldout_rows=heldout_rows)


def run_arm(arm: Arm, backend: Backend, cfg: RunConfig,
            control: Sequence[cb.Item] | None = None, log=None,
            save_path: str | Path | None = None, on_checkpoint=None, resume: bool = False) -> RunResult:
    """Run one arm for one seed. If save_path is given the (partial) result is
    written after every checkpoint (B1) together with backend + arm state in
    <save_path>.state/, and `on_checkpoint()` is called afterwards (e.g. to
    commit a volume). With resume=True and an existing partial file, the run
    continues from its last checkpoint (preemption safety); a complete file
    is returned as is."""
    if hasattr(backend, "set_seed"):
        backend.set_seed(cfg.seed)                    # A1: sampling / init generators follow the run seed
    split = nr.make_split(seed=cfg.seed, length=cfg.length)
    seq = episode_sequence(split, cfg)
    probe_items, heldout_items = eval_items(split, cfg)
    implicit_sets = implicit_items(split, cfg)
    if control is None and cfg.control_items:
        control = cb.load_frozen()[: cfg.control_items]
    control = list(control or [])
    ledger = cl.Ledger(arm=arm.name, backend=backend.name)
    tracker = sd.CriterionTracker(threshold=cfg.criterion_threshold)
    episodes: list[Episode] = []
    checkpoints: list[Checkpoint] = []
    nights: list[dict] = []
    eval_tokens = 0
    resumed_from: int | None = None
    start = 1

    if resume and save_path is not None and Path(save_path).exists():
        d = load_run(save_path)
        if not d["partial"]:
            episodes, checkpoints, ledger, nights, _ = _restore(d)
            for c in checkpoints:
                tracker.update(c.episode, c.probe)
            if log:
                log(f"[{arm.name} seed={cfg.seed}] complete run found at {save_path}; nothing to do")
            return RunResult(arm=arm.name, config=cfg, episodes=episodes, checkpoints=checkpoints, ledger=ledger,
                             criterion_episode=tracker.first_met_at, eval_tokens=d["eval_tokens"], nights=nights,
                             matched=d.get("matched"), partial=False, resumed_from=d.get("resumed_from"))
        episodes, checkpoints, ledger, nights, ck_ep = _restore(d)
        eval_tokens = d["eval_tokens"]
        for c in checkpoints:
            tracker.update(c.episode, c.probe)
        sdir = state_dir(save_path)
        if hasattr(backend, "load_state"):
            backend.load_state(sdir)
        if hasattr(arm, "load_state"):
            arm.load_state(sdir, episodes, cfg, split)
        resumed_from = ck_ep
        start = ck_ep + 1
        if log:
            log(f"[{arm.name} seed={cfg.seed}] resumed from checkpoint {ck_ep} ({len(episodes)} episodes kept)")

    def result(partial: bool) -> RunResult:
        return RunResult(arm=arm.name, config=cfg, episodes=episodes, checkpoints=checkpoints, ledger=ledger,
                         criterion_episode=tracker.first_met_at, eval_tokens=eval_tokens, nights=nights,
                         partial=partial, resumed_from=resumed_from)

    def persist(partial: bool) -> None:
        if save_path is not None:
            sdir = state_dir(save_path)          # saved at every checkpoint AND at the end (final adapter for retro-analysis)
            sdir.mkdir(parents=True, exist_ok=True)
            if hasattr(backend, "save_state"):
                backend.save_state(sdir)
            if hasattr(arm, "save_state"):
                arm.save_state(sdir)
            result(partial).save(save_path)
            if on_checkpoint:
                on_checkpoint()

    if start == 1:
        # A5: episode-0 checkpoint before any day episode (forgetting baseline, untrained probe)
        ck0 = _checkpoint(arm, backend, cfg, 0, probe_items, heldout_items, control, ledger, implicit_sets)
        eval_tokens += ck0.eval_tokens
        checkpoints.append(ck0)
        tracker.update(0, ck0.probe)
        if log:
            log(f"[{arm.name} seed={cfg.seed}] ep=0 probe={ck0.probe.accuracy:.3f} heldout={ck0.heldout_accuracy:.3f} "
                f"control={ck0.control_accuracy} {ck0.seconds:.0f}s")
        persist(partial=True)

    for i, inst in enumerate(seq, start=1):
        if i < start:
            continue
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
            ck = _checkpoint(arm, backend, cfg, i, probe_items, heldout_items, control, ledger, implicit_sets)
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
    persist(partial=False)
    return result(partial=False)
