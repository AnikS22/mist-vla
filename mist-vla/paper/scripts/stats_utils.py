"""Shared statistics helpers for paper tables and run_stat_tests.py (Wilson CI, power, Holm)."""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import numpy as np
from scipy import stats


def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """95% Wilson score interval for binomial proportion k/n (in [0,1])."""
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    rad = z * math.sqrt(max((p * (1.0 - p) / n + z2 / (4.0 * n * n)), 0.0)) / denom
    return (max(0.0, center - rad), min(1.0, center + rad))


def wilson_ci_pp(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson interval in percentage points."""
    lo, hi = wilson_ci(k, n, z)
    return (100.0 * lo, 100.0 * hi)


def required_n_per_group_two_proportion(
    p1: float,
    p2: float,
    power: float = 0.8,
    alpha: float = 0.05,
    two_sided: bool = True,
) -> int:
    """
    Approximate sample size per group (equal n) for two-sample z-test on proportions.
    Uses normal approximation with pooled variance under H0.
    Returns -1 if inputs invalid or effect is zero.
    """
    if abs(p1 - p2) < 1e-12:
        return -1
    if not (0 <= p1 <= 1 and 0 <= p2 <= 1):
        return -1
    delta = abs(p1 - p2)
    p_bar = 0.5 * (p1 + p2)
    z_a = stats.norm.ppf(1 - alpha / 2) if two_sided else stats.norm.ppf(1 - alpha)
    z_b = stats.norm.ppf(power)
    se2 = 2 * p_bar * (1 - p_bar)
    if se2 <= 0:
        return -1
    n_float = se2 * ((z_a + z_b) ** 2) / (delta**2)
    return int(math.ceil(max(n_float, 1.0)))


def posthoc_power_two_proportion_z(
    s1: int,
    n1: int,
    s2: int,
    n2: int,
    alpha: float = 0.05,
    two_sided: bool = True,
) -> float:
    """
    Common normal-approx post-hoc power: non-centrality = |p1-p2|/SE under pooled H0 SE.
    Interpret cautiously; prefer reporting CIs and required-N.
    """
    if min(n1, n2) == 0:
        return float("nan")
    p1, p2 = s1 / n1, s2 / n2
    p_pool = (s1 + s2) / (n1 + n2)
    se_pool = math.sqrt(max(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2), 1e-15))
    z_crit = stats.norm.ppf(1 - alpha / 2) if two_sided else stats.norm.ppf(1 - alpha)
    ncp = abs(p1 - p2) / se_pool
    return float(stats.norm.cdf(ncp - z_crit) + stats.norm.cdf(-ncp - z_crit))


def tost_two_proportion(
    s1: int,
    n1: int,
    s2: int,
    n2: int,
    delta: float = 0.02,
    alpha: float = 0.05,
) -> dict:
    """
    Two One-Sided Tests (TOST) for equivalence of two proportions.
    H0: |p1 - p2| >= delta  vs  H1: |p1 - p2| < delta
    Returns dict with p_tost (max of two one-sided p-values), and whether
    equivalence is established at the given alpha.
    """
    if min(n1, n2) == 0:
        return {"p_tost": float("nan"), "equivalent": False, "delta": delta}
    p1 = s1 / n1
    p2 = s2 / n2
    diff = p1 - p2
    se = math.sqrt(max((p1 * (1 - p1) / n1) + (p2 * (1 - p2) / n2), 1e-15))

    # Upper test: H0: p1 - p2 >= delta
    z_upper = (diff - delta) / se
    p_upper = float(stats.norm.cdf(z_upper))

    # Lower test: H0: p1 - p2 <= -delta
    z_lower = (diff + delta) / se
    p_lower = float(1.0 - stats.norm.cdf(z_lower))

    p_tost = max(p_upper, p_lower)
    return {
        "p_tost": p_tost,
        "equivalent": p_tost < alpha,
        "delta": delta,
        "diff": diff,
        "z_upper": z_upper,
        "z_lower": z_lower,
        "p_upper": p_upper,
        "p_lower": p_lower,
    }


def holm_adjusted_pvalues(p_values: Sequence[float]) -> List[float]:
    """Holm step-down adjusted p-values (two-sided family)."""
    m = len(p_values)
    if m == 0:
        return []
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    sorted_p = p[order]
    adj_sorted = np.zeros(m)
    for i in range(m):
        factors = [(m - j) * sorted_p[j] for j in range(i + 1)]
        adj_sorted[i] = min(1.0, max(factors))
    for i in range(m - 2, -1, -1):
        adj_sorted[i] = max(adj_sorted[i], adj_sorted[i + 1])
    out = np.zeros(m)
    out[order] = adj_sorted
    return out.tolist()
