"""Backend interface.

All arms talk to a model only through this. Two implementations are planned:
MLX (Apple silicon, development) and Transformers+PEFT (Modal, reported
results). Numbers from different backends are never mixed in a plot.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence


@dataclass
class GenResult:
    text: str
    prompt_tokens: int
    completion_tokens: int


class Backend(Protocol):
    name: str

    def generate(self, prompts: Sequence[str], max_tokens: int = 128,
                 temperature: float = 0.0) -> list[GenResult]:
        """Chat-formatted generation, thinking disabled. One result per prompt."""
        ...

    def count_tokens(self, text: str) -> int: ...


@dataclass
class TrainStats:
    steps: int
    training_tokens: int      # completion tokens under the loss, summed over steps
    seconds: float = 0.0
    loss: float | None = None


class TrainableBackend(Backend, Protocol):
    """LoRA-trainable backend (Online / Sleep arms). Batch size 1 per step;
    `train` cycles through `examples` in order for `steps` steps so gradient
    steps and training tokens are ledger-countable and matchable."""

    def train(self, examples: Sequence[tuple[str, str]], steps: int, seed: int) -> TrainStats: ...

    def decay_adapter(self, factor: float) -> None:
        """One weight-decay pass on LoRA params: W_lora *= (1 - factor)."""
        ...

    def reset_adapter(self) -> None: ...

    def save_adapter(self, path: str) -> None: ...
