"""Utility statistical functions for metrics and confidence intervals.

Copyright (c) 2025 Concha Labra. All Rights Reserved.
Private and confidential.
"""

from __future__ import annotations
import math

def fisher_z_ci(r: float, n: int, confidence: float = 0.95):
    """Compute Fisher z-transform based confidence interval for a Pearson correlation.

    Parameters
    ----------
    r : float
        Sample correlation coefficient.
    n : int
        Number of independent samples underlying r (sequence pairs). Must be >= 4 for CI.
    confidence : float, default 0.95
        Two-sided confidence level.

    Returns
    -------
    (low, high) : tuple[float|None, float|None]
        Lower and upper bounds of the confidence interval. (None, None) if n < 4 or |r|≈1.
    """
    if n < 4 or r is None or math.isnan(r) or abs(r) >= 0.999999:
        return None, None
    # Fisher z transform
    try:
        z = 0.5 * math.log((1 + r) / (1 - r + 1e-12))
    except ValueError:
        return None, None
    # Standard error
    se = 1.0 / math.sqrt(n - 3)
    # z critical value (normal approximation)
    # Inverse normal critical value computed via rational approximation; no external deps.
    alpha = 1 - confidence
    # Inverse CDF for standard normal via approximation (avoid scipy dependency)
    # Using rational approximation for inverse error function
    def _norm_ppf(p: float) -> float:
        # Abramowitz and Stegun formula 26.2.23 refinement (sufficient for CI usage)
        if p <= 0 or p >= 1:
            raise ValueError("p must be in (0,1)")
        # Convert to 2-sided tail parameter
        # For central interval we need z_(1-alpha/2)
        # We'll call with p = 1 - alpha/2
        a1 = -39.69683028665376
        a2 = 220.9460984245205
        a3 = -275.9285104469687
        a4 = 138.3577518672690
        a5 = -30.66479806614716
        a6 = 2.506628277459239
        b1 = -54.47609879822406
        b2 = 161.5858368580409
        b3 = -155.6989798598866
        b4 = 66.80131188771972
        b5 = -13.28068155288572
        c1 = -7.784894002430293e-03
        c2 = -0.3223964580411365
        c3 = -2.400758277161838
        c4 = -2.549732539343734
        c5 = 4.374664141464968
        c6 = 2.938163982698783
        d1 = 7.784695709041462e-03
        d2 = 0.3224671290700398
        d3 = 2.445134137142996
        d4 = 3.754408661907416
        plow = 0.02425
        phigh = 1 - plow
        if p < plow:
            q = math.sqrt(-2 * math.log(p))
            return (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / (
                (((d1 * q + d2) * q + d3) * q + d4)
            )
        if p > phigh:
            q = math.sqrt(-2 * math.log(1 - p))
            return -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / (
                (((d1 * q + d2) * q + d3) * q + d4)
            )
        q = p - 0.5
        r2 = q * q
        return (((((a1 * r2 + a2) * r2 + a3) * r2 + a4) * r2 + a5) * r2 + a6) * q / (
            (((((b1 * r2 + b2) * r2 + b3) * r2 + b4) * r2 + b5) * r2 + 1)
        )
    zcrit = _norm_ppf(1 - alpha / 2)
    z_low = z - zcrit * se
    z_high = z + zcrit * se
    def inv(zv: float) -> float:
        ez = math.exp(2 * zv)
        return (ez - 1) / (ez + 1)
    return inv(z_low), inv(z_high)

__all__ = ["fisher_z_ci"]