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


def plan_multi(items: list[tuple[int, str]], n_gpus: int) -> list[list[tuple[int, str]]]:
    """Assign (seed, arm) items to n_gpus queues, longest-first, balanced by
    EXPECTED_HOURS. A seed's 'awake' runs on the queue holding that seed's
    'sleep', after it (its budget comes from the sleep ledger), and the
    sleep is weighted with the awake's hours at assignment time so the
    balance accounts for it. An awake whose sleep is not among the items
    (already done elsewhere) is placed like any other item."""
    items = list(items)
    have_sleep = {s for s, a in items if a == "sleep"}
    paired_awakes = {s for s, a in items if a == "awake" and s in have_sleep}
    def weight(it):
        s, a = it
        w = EXPECTED_HOURS.get(a, 3.0)
        if a == "sleep" and s in paired_awakes:
            w += EXPECTED_HOURS["awake"]
        return w
    queues: list[list[tuple[int, str]]] = [[] for _ in range(n_gpus)]
    loads = [0.0] * n_gpus
    free = [it for it in items if not (it[1] == "awake" and it[0] in paired_awakes)]
    for it in sorted(free, key=lambda it: -weight(it)):
        i = loads.index(min(loads))
        queues[i].append(it)
        loads[i] += weight(it)
    for s in sorted(paired_awakes):
        i = next(i for i, q in enumerate(queues) if (s, "sleep") in q)
        queues[i].append((s, "awake"))
    return queues
