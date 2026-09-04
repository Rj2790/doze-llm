"""Per-GPU work queues for a multi-GPU VM (pure; tested).

Arms of one seed are split across GPUs so that Awake follows Sleep on the
same GPU (its token budget comes from Sleep's ledger) and total expected
hours per GPU are balanced. Expected hours are L4-relative weights; the
absolute time on an A16 is 2-6x longer but the ratios hold.
"""

from __future__ import annotations

EXPECTED_HOURS = {"sleep": 6.0, "sleep_nodream": 6.0, "online": 6.0, "awake": 8.5, "baseline": 2.7}


def plan_queues(arms: list[str], n_gpus: int) -> list[list[str]]:
    """Greedy longest-first assignment; 'awake' is appended to the queue that
    holds 'sleep' (dependency), after it."""
    arms = list(arms)
    if "awake" in arms and "sleep" not in arms:
        raise ValueError("awake needs sleep in the same run (budget)")
    queues: list[list[str]] = [[] for _ in range(n_gpus)]
    loads = [0.0] * n_gpus
    order = sorted((a for a in arms if a != "awake"), key=lambda a: -EXPECTED_HOURS.get(a, 3.0))
    for a in order:
        i = loads.index(min(loads))
        queues[i].append(a)
        loads[i] += EXPECTED_HOURS.get(a, 3.0)
    if "awake" in arms:
        i = next(i for i, q in enumerate(queues) if "sleep" in q)
        queues[i].append("awake")
    return queues
