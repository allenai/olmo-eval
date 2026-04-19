"""CLT-based power helpers for paired evaluation scores."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.stats import norm


def _validate_alpha_power(alpha: float, power: float) -> None:
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if not 0 < power < 1:
        raise ValueError(f"power must be in (0, 1), got {power}")


def _validate_k(k_a: int, k_b: int) -> None:
    if k_a < 1 or k_b < 1:
        raise ValueError(f"k_a and k_b must be >= 1, got k_a={k_a}, k_b={k_b}")


def _validate_variances(omega2: float, sigma2_a: float, sigma2_b: float) -> None:
    if omega2 < 0 or sigma2_a < 0 or sigma2_b < 0:
        raise ValueError(
            f"variance components must be non-negative, got "
            f"omega2={omega2}, sigma2_a={sigma2_a}, sigma2_b={sigma2_b}"
        )


def _z_combined(alpha: float, power: float) -> float:
    """Return ``z_{alpha/2} + z_{power}`` for a two-sided test."""
    return float(norm.ppf(1.0 - alpha / 2.0) + norm.ppf(power))


def required_sample_size(
    mde: float,
    omega2: float,
    sigma2_a: float = 0.0,
    sigma2_b: float = 0.0,
    k_a: int = 1,
    k_b: int = 1,
    alpha: float = 0.05,
    power: float = 0.80,
) -> int:
    """Return the paired sample size needed to detect ``mde``.

    n = (z_{alpha/2} + z_beta)^2 * (omega^2 + sigma_A^2/K_A + sigma_B^2/K_B) / delta^2
    """
    _validate_alpha_power(alpha, power)
    _validate_k(k_a, k_b)
    _validate_variances(omega2, sigma2_a, sigma2_b)
    if mde <= 0:
        raise ValueError(f"mde must be > 0, got {mde}")

    z = _z_combined(alpha, power)
    variance = omega2 + sigma2_a / k_a + sigma2_b / k_b
    return math.ceil((z**2) * variance / (mde**2))


def minimum_detectable_effect(
    n: int,
    omega2: float,
    sigma2_a: float = 0.0,
    sigma2_b: float = 0.0,
    k_a: int = 1,
    k_b: int = 1,
    alpha: float = 0.05,
    power: float = 0.80,
) -> float:
    """Return the smallest detectable effect at ``n`` paired questions.

    delta = (z_{alpha/2} + z_beta) * sqrt((omega^2 + sigma_A^2/K_A + sigma_B^2/K_B) / n)
    """
    _validate_alpha_power(alpha, power)
    _validate_k(k_a, k_b)
    _validate_variances(omega2, sigma2_a, sigma2_b)
    if n <= 0:
        raise ValueError(f"n must be > 0, got {n}")

    z = _z_combined(alpha, power)
    variance = omega2 + sigma2_a / k_a + sigma2_b / k_b
    return z * math.sqrt(variance / n)


def estimate_variance_components(
    score_a: np.ndarray | list[float],
    score_b: np.ndarray | list[float],
    cluster_ids: np.ndarray | list[Any] | None = None,
    k_a: int = 1,
    k_b: int = 1,
    binary: bool = False,
) -> dict[str, Any]:
    """Estimate the variance terms used by the power helpers.

    If ``cluster_ids`` is set, use a cluster-robust paired-difference variance.
    If ``binary`` is set, estimate conditional variance as ``mean(s * (1 - s))``.
    """
    _validate_k(k_a, k_b)

    a = np.asarray(score_a, dtype=float)
    b = np.asarray(score_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: score_a {a.shape} vs score_b {b.shape}")
    if a.ndim != 1:
        raise ValueError(f"scores must be 1-D, got {a.ndim}D")
    n = a.shape[0]
    if n < 2:
        raise ValueError(f"need at least 2 paired observations, got {n}")

    d = a - b

    n_clusters: int | None = None
    if cluster_ids is not None:
        cluster_arr = np.asarray(cluster_ids)
        if cluster_arr.shape != (n,):
            raise ValueError(f"cluster_ids length must be {n}, got {cluster_arr.shape}")
        d_demean = d - d.mean()
        unique = np.unique(cluster_arr)
        n_clusters = int(unique.shape[0])
        cluster_sum_sq = 0.0
        for c in unique:
            cluster_sum_sq += float(d_demean[cluster_arr == c].sum()) ** 2
        # Scale Var(mean(d)) back to a per-question term.
        var_paired_diff = cluster_sum_sq / n
    else:
        var_paired_diff = float(np.var(d, ddof=1))

    if binary:
        sigma2_a = float(np.mean(a * (1.0 - a)))
        sigma2_b = float(np.mean(b * (1.0 - b)))
    else:
        sigma2_a = 0.0
        sigma2_b = 0.0

    omega2 = max(0.0, var_paired_diff - sigma2_a / k_a - sigma2_b / k_b)

    correlation = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else 0.0

    return {
        "omega2": omega2,
        "sigma2_a": sigma2_a,
        "sigma2_b": sigma2_b,
        "var_paired_diff": var_paired_diff,
        "correlation": correlation,
        "n": int(n),
        "n_clusters": n_clusters,
    }


def power_summary(
    score_a: np.ndarray | list[float],
    score_b: np.ndarray | list[float],
    cluster_ids: np.ndarray | list[Any] | None = None,
    k_a: int = 1,
    k_b: int = 1,
    binary: bool = False,
    alpha: float = 0.05,
    power: float = 0.80,
) -> str:
    """Format the estimated variance terms and sizing tables."""
    vc = estimate_variance_components(
        score_a,
        score_b,
        cluster_ids=cluster_ids,
        k_a=k_a,
        k_b=k_b,
        binary=binary,
    )
    n_suffix = f" ({vc['n_clusters']} clusters)" if vc["n_clusters"] is not None else ""
    lines: list[str] = [
        f"n = {vc['n']}{n_suffix}",
        f"correlation(a, b) = {vc['correlation']:+.4f}",
        f"Var(d_i) = {vc['var_paired_diff']:.6f}",
        f"  omega^2   = {vc['omega2']:.6f}",
        f"  sigma_A^2 = {vc['sigma2_a']:.6f}  (K_A = {k_a})",
        f"  sigma_B^2 = {vc['sigma2_b']:.6f}  (K_B = {k_b})",
        "",
        f"required n for target MDE (alpha={alpha}, power={power}):",
    ]
    for mde in (0.01, 0.02, 0.03, 0.05, 0.10):
        required = required_sample_size(
            mde=mde,
            omega2=vc["omega2"],
            sigma2_a=vc["sigma2_a"],
            sigma2_b=vc["sigma2_b"],
            k_a=k_a,
            k_b=k_b,
            alpha=alpha,
            power=power,
        )
        lines.append(f"  MDE {mde:>5.0%}  ->  n = {required:>7,}")

    lines.append("")
    lines.append(f"MDE at given n (alpha={alpha}, power={power}):")
    for n_sample in (100, 250, 500, 1000, 2000, 5000):
        mde_val = minimum_detectable_effect(
            n=n_sample,
            omega2=vc["omega2"],
            sigma2_a=vc["sigma2_a"],
            sigma2_b=vc["sigma2_b"],
            k_a=k_a,
            k_b=k_b,
            alpha=alpha,
            power=power,
        )
        lines.append(f"  n = {n_sample:>5,}  ->  MDE = {mde_val:.1%}")

    return "\n".join(lines)
