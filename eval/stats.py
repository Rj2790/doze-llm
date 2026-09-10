"""Small-sample statistics for the preregistered analysis (PREREG §7), with
no scipy dependency: an exact sign-flip permutation test for paired
differences, a two-sample log-rank test with right-censoring, and the
chi-square(1) survival function it needs.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from itertools import product


def sign_flip_test(diffs: Sequence[float]) -> dict:
    """Exact two-sided sign-flip (permutation) test of mean(diffs) = 0.
    With n paired differences there are 2^n sign assignments; p is the
    fraction whose |mean| >= the observed |mean|. n <= 20."""
    d = [float(x) for x in diffs]
    n = len(d)
    if n == 0:
        return {"n": 0, "mean": None, "sd": None, "p": None}
    mean = sum(d) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in d) / (n - 1)) if n > 1 else None
    if n == 1 or all(x == 0 for x in d):
        return {"n": n, "mean": mean, "sd": sd, "p": 1.0}
    obs = abs(mean)
    count = 0
    for signs in product((1, -1), repeat=n):
        m = sum(s * x for s, x in zip(signs, d)) / n
        if abs(m) >= obs - 1e-12:
            count += 1
    return {"n": n, "mean": mean, "sd": sd, "p": count / 2 ** n}


def chi2_sf_1df(x: float) -> float:
    """P(chi2_1 > x) = erfc(sqrt(x/2))."""
    if x <= 0:
        return 1.0
    return math.erfc(math.sqrt(x / 2.0))


def logrank(a: Sequence[tuple[int, bool]], b: Sequence[tuple[int, bool]]) -> dict:
    """Two-sample log-rank test. Each observation is (time, event) where
    event=False means censored at `time` (e.g. criterion never met by 600).
    Returns the chi-square(1) statistic and p, or degenerate=True when there
    are no events at all."""
    obs = [(t, e, 0) for t, e in a] + [(t, e, 1) for t, e in b]
    events = sum(1 for _, e, _ in obs if e)
    if events == 0:
        return {"statistic": 0.0, "p": None, "events": 0, "degenerate": True, "n_a": len(a), "n_b": len(b)}
    times = sorted({t for t, e, _ in obs if e})
    O_a = E_a = V = 0.0
    for t in times:
        at_risk = [o for o in obs if o[0] >= t]
        n = len(at_risk); n_a = sum(1 for o in at_risk if o[2] == 0)
        d = sum(1 for o in at_risk if o[0] == t and o[1]); d_a = sum(1 for o in at_risk if o[0] == t and o[1] and o[2] == 0)
        if n == 0:
            continue
        E_a += d * n_a / n
        O_a += d_a
        if n > 1:
            V += d * (n_a / n) * (1 - n_a / n) * (n - d) / (n - 1)
    stat = (O_a - E_a) ** 2 / V if V > 0 else 0.0
    return {"statistic": stat, "p": chi2_sf_1df(stat), "events": events, "degenerate": False,
            "observed_a": O_a, "expected_a": E_a, "n_a": len(a), "n_b": len(b)}
