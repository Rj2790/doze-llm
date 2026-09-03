"""Awake arm (PREREG §5): frozen weights; a matched *token* budget spent on
extra chain-of-thought and self-critique at inference time.

Per prompt: first attempt, then repeated critique-and-revise rounds until the
tokens generated for this prompt reach `token_budget_per_episode` (or
`max_rounds`). The last round's text is the answer. The budget is derived
from the Sleep arm's ledger: total Sleep tokens (day + dreams) / episodes,
so the ledger check in eval/compute_ledger.py passes within +-5%.

Critique wording is tunable. The rule text and digits are restated so the
model re-derives rather than copies.
"""

from __future__ import annotations

from collections.abc import Sequence

from arms.harness import Episode, RunConfig
from backends.base import Backend, GenResult
from eval import compute_ledger as cl
from tasks import number_reduction as nr

CRITIQUE_PROBE = ("Below is a previous best guess for the puzzle above, in which some digits are hidden. "
                  "Work through the comparisons as far as the visible digits allow, then give your best "
                  "guess for the final result on a line starting with 'ANSWER:'. Output only that line.\n\n"
                  "Previous guess:\n")

CRITIQUE = ("Below is a previous attempt at the problem above. Check every comparison against "
            "the rule, one by one, from the first pair. If any line is wrong, redo the solution "
            "from that line onward. Then write the full corrected solution in the requested "
            "format (all lines, then STEPS:, then ANSWER:). Output only the solution.\n\n"
            "Previous attempt:\n")


def critique_prompt(prompt: str, previous: str) -> str:
    """Probe prompts (hidden digits) get a probe-specific critique (C4);
    full prompts get the step-by-step recheck."""
    text = CRITIQUE_PROBE if nr.PROBE_FORMAT in prompt else CRITIQUE
    return f"{prompt}\n\n{text}{previous.strip()}"


class AwakeArm:
    name = "awake"

    def __init__(self, token_budget_per_episode: int, max_rounds: int = 8):
        if token_budget_per_episode <= 0:
            raise ValueError("token budget must be positive")
        self.budget = token_budget_per_episode
        self.max_rounds = max_rounds
        self.spent = 0       # day tokens generated so far
        self.allowed = 0     # day tokens allowed so far (episodes * budget)

    @staticmethod
    def budget_from_reference(reference: cl.Ledger, n_episodes: int) -> int:
        """Per-episode token budget that reproduces the reference arm's total."""
        return max(1, round(reference.tokens_generated / n_episodes))

    def attempt(self, prompt: str, backend: Backend, max_tokens: int,
                phase: str = "day") -> tuple[GenResult, int]:
        """Day: the budget accrues per episode and rounds continue while
        cumulative spend < cumulative allowance, so overshoot in one episode
        is repaid in the next and the total lands within one completion of
        n_episodes * budget. Eval: rounds until this prompt alone has used
        `budget` tokens (same procedure, not ledger-matched)."""
        if phase == "day":
            self.allowed += self.budget
            more = lambda total: self.spent + total < self.allowed
        else:
            more = lambda total: total < self.budget
        r = backend.generate([prompt], max_tokens=max_tokens)[0]
        total = r.completion_tokens
        rounds = 1
        while more(total) and rounds < self.max_rounds:
            nxt = backend.generate([critique_prompt(prompt, r.text)], max_tokens=max_tokens)[0]
            total += nxt.completion_tokens
            rounds += 1
            if nr.parse_answer(nxt.text) is not None:   # keep the last parsable answer
                r = nxt
        if phase == "day":
            self.spent += total
        return GenResult(text=r.text, prompt_tokens=r.prompt_tokens, completion_tokens=total), rounds

    def save_state(self, sdir) -> None:
        import json
        from pathlib import Path
        Path(sdir, "awake.json").write_text(json.dumps({"spent": self.spent, "allowed": self.allowed}))

    def load_state(self, sdir, episodes, cfg, split) -> None:
        import json
        from pathlib import Path
        d = json.loads(Path(sdir, "awake.json").read_text())
        self.spent, self.allowed = d["spent"], d["allowed"]

    def attempt_many(self, prompts: Sequence[str], backend: Backend, max_tokens: int,
                     phase: str = "eval") -> list[tuple[GenResult, int]]:
        """Eval: round-synchronous batching. All first attempts in one batch,
        then one batched critique round for every prompt still under its
        budget, until none remain or max_rounds. Per-prompt semantics are
        identical to attempt(phase="eval"). Day attempts stay sequential
        because the budget accrues per episode."""
        if phase == "day" or not prompts:
            return [self.attempt(p, backend, max_tokens, phase=phase) for p in prompts]
        prompts = list(prompts)
        best = backend.generate(prompts, max_tokens=max_tokens)
        totals = [r.completion_tokens for r in best]
        rounds = [1] * len(prompts)
        active = [i for i in range(len(prompts)) if totals[i] < self.budget and rounds[i] < self.max_rounds]
        while active:
            outs = backend.generate([critique_prompt(prompts[i], best[i].text) for i in active], max_tokens=max_tokens)
            for i, o in zip(active, outs):
                totals[i] += o.completion_tokens
                rounds[i] += 1
                if nr.parse_answer(o.text) is not None:
                    best[i] = o
            active = [i for i in active if totals[i] < self.budget and rounds[i] < self.max_rounds]
        return [(GenResult(text=best[i].text, prompt_tokens=best[i].prompt_tokens, completion_tokens=totals[i]), rounds[i])
                for i in range(len(prompts))]

    def after_episode(self, ep: Episode, backend: Backend, ledger: cl.Ledger, cfg: RunConfig) -> None:
        return None

    def night(self, episodes: Sequence[Episode], backend: Backend, ledger: cl.Ledger,
              cfg: RunConfig, split: nr.Split) -> dict:
        return {}
