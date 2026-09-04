"""Control benchmark (PREREG §4.3): a fixed 300-item sample of the GSM8K test
set, scored by exact match on the final number. Measures forgetting. No arm
ever trains on it.

Data: `data/gsm8k_test.jsonl` is the original OpenAI release (1319 rows,
fields question/answer, gold after '####'). The 300 row indices are frozen
in `eval/control_bench_ids.json` so every arm, seed and backend scores the
same items.
"""

from __future__ import annotations

import json
import random
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

DATA_PATH = Path("data/gsm8k_test.jsonl")
IDS_PATH = Path(__file__).with_name("control_bench_ids.json")
N_ITEMS = 300
GSM8K_TEST_ROWS = 1319

SYSTEM_PROMPT = ("You solve grade-school math word problems. Think step by step briefly, then give "
                 "the final number on the last line as 'ANSWER: <number>'.")


@dataclass(frozen=True)
class Item:
    idx: int
    question: str
    gold: str


@dataclass
class BenchResult:
    n: int
    correct: int
    accuracy: float
    unparsable: int = 0

    @property
    def unparsable_frac(self) -> float | None:
        return (self.unparsable / self.n) if self.n else None


_GOLD_RE = re.compile(r"####\s*(-?[\d,]*\.?\d+)")
_TAGGED_RE = re.compile(r"(?:ANSWER|####)\W{0,6}\$?\s*(-?[\d,]*\.?\d+)", re.IGNORECASE)
_NUM_RE = re.compile(r"-?\$?[\d,]*\.?\d+")


def _norm(s: str) -> str:
    s = s.replace(",", "").replace("$", "").strip().rstrip(".")
    if re.fullmatch(r"-?\d+\.0+", s):
        s = s.split(".")[0]
    return s


def gold_answer(answer_field: str) -> str:
    m = _GOLD_RE.search(answer_field)
    if not m:
        raise ValueError(f"no #### answer in: {answer_field[:80]!r}")
    return _norm(m.group(1))


def parse_answer(text: str) -> str | None:
    """Prefer an explicit ANSWER:/#### tag (last one wins); else the last
    number in the text."""
    tagged = _TAGGED_RE.findall(text)
    if tagged:
        return _norm(tagged[-1])
    nums = _NUM_RE.findall(text)
    nums = [n for n in nums if re.search(r"\d", n)]
    return _norm(nums[-1]) if nums else None


def exact_match(pred: str | None, gold: str) -> bool:
    return pred is not None and _norm(pred) == _norm(gold)


def _read_rows(path: str | Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def sample_ids(n_rows: int, n: int, seed: int) -> list[int]:
    return sorted(random.Random(seed).sample(range(n_rows), n))


def frozen_ids() -> list[int]:
    return json.loads(IDS_PATH.read_text())


def load_sample(path: str | Path = DATA_PATH, n: int = N_ITEMS, seed: int = 0,
                ids: Sequence[int] | None = None) -> list[Item]:
    """Deterministic sample of n rows. Pass ids=frozen_ids() for the PREREG
    set; the (n, seed) path exists for tests on small fixtures."""
    rows = _read_rows(path)
    if ids is None:
        ids = sample_ids(len(rows), n, seed)
    return [Item(idx=i, question=rows[i]["question"], gold=gold_answer(rows[i]["answer"])) for i in ids]


def load_frozen(path: str | Path = DATA_PATH) -> list[Item]:
    items = load_sample(path, ids=frozen_ids())
    assert len(items) == N_ITEMS
    return items


def format_prompt(item: Item) -> str:
    return f"{item.question}\n\nEnd with a line 'ANSWER: <number>'."


def score(items: Sequence[Item], texts: Sequence[str]) -> BenchResult:
    assert len(items) == len(texts)
    preds = [parse_answer(t) for t in texts]
    correct = sum(exact_match(p, x.gold) for x, p in zip(items, preds))
    return BenchResult(n=len(items), correct=correct, accuracy=correct / len(items) if items else 0.0,
                       unparsable=sum(p is None for p in preds))


def evaluate(generate: Callable[[str], str], items: Sequence[Item]) -> BenchResult:
    return score(items, [generate(format_prompt(x)) for x in items])


def _fixture_answer(prompt: str) -> str:
    """Test helper: fixture questions are 'Qi: what is i + i?'."""
    i = int(re.search(r"Q(\d+):", prompt).group(1))
    return "1234" if i == 3 else str(2 * i)
