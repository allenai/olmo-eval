"""Tests for the eval_power module."""

from __future__ import annotations

import numpy as np
import pytest

from olmo_eval.analysis.eval_power import (
    estimate_variance_components,
    minimum_detectable_effect,
    power_summary,
    required_sample_size,
)


class TestRequiredSampleSize:
    def test_paper_worked_example(self) -> None:
        n = required_sample_size(mde=0.03, omega2=1 / 9)
        assert n == 969

    def test_monotone_in_mde(self) -> None:
        n_small = required_sample_size(mde=0.01, omega2=1 / 9)
        n_large = required_sample_size(mde=0.05, omega2=1 / 9)
        assert n_small > n_large

    def test_monotone_in_variance(self) -> None:
        n_low = required_sample_size(mde=0.03, omega2=0.05)
        n_high = required_sample_size(mde=0.03, omega2=0.20)
        assert n_high > n_low

    def test_resamples_reduce_n(self) -> None:
        n_k1 = required_sample_size(mde=0.05, omega2=0.1, sigma2_a=0.2, sigma2_b=0.2, k_a=1, k_b=1)
        n_k10 = required_sample_size(
            mde=0.05, omega2=0.1, sigma2_a=0.2, sigma2_b=0.2, k_a=10, k_b=10
        )
        assert n_k10 < n_k1

    def test_validates_mde(self) -> None:
        with pytest.raises(ValueError):
            required_sample_size(mde=0.0, omega2=0.1)
        with pytest.raises(ValueError):
            required_sample_size(mde=-0.01, omega2=0.1)

    def test_validates_alpha_power(self) -> None:
        with pytest.raises(ValueError):
            required_sample_size(mde=0.01, omega2=0.1, alpha=0.0)
        with pytest.raises(ValueError):
            required_sample_size(mde=0.01, omega2=0.1, power=1.0)

    def test_validates_k(self) -> None:
        with pytest.raises(ValueError):
            required_sample_size(mde=0.01, omega2=0.1, k_a=0)

    def test_validates_variances(self) -> None:
        with pytest.raises(ValueError):
            required_sample_size(mde=0.01, omega2=-0.1)


class TestMinimumDetectableEffect:
    def test_paper_mde_k1(self) -> None:
        mde = minimum_detectable_effect(
            n=198, omega2=1 / 9, sigma2_a=1 / 6, sigma2_b=1 / 6, k_a=1, k_b=1
        )
        assert mde == pytest.approx(0.132, abs=1e-3)

    def test_paper_mde_k10(self) -> None:
        mde = minimum_detectable_effect(
            n=198, omega2=1 / 9, sigma2_a=1 / 6, sigma2_b=1 / 6, k_a=10, k_b=10
        )
        assert mde == pytest.approx(0.075, abs=1e-3)

    def test_round_trip(self) -> None:
        orig = 0.05
        n = required_sample_size(mde=orig, omega2=0.2, sigma2_a=0.1, sigma2_b=0.1)
        recovered = minimum_detectable_effect(n=n, omega2=0.2, sigma2_a=0.1, sigma2_b=0.1)
        assert recovered <= orig
        assert recovered == pytest.approx(orig, rel=0.01)

    def test_validates_n(self) -> None:
        with pytest.raises(ValueError):
            minimum_detectable_effect(n=0, omega2=0.1)


class TestEstimateVarianceComponents:
    def test_basic_unclustered(self) -> None:
        a = np.array([1, 1, 0, 0, 1, 0, 1, 0, 1, 1], dtype=float)
        b = np.array([1, 0, 0, 1, 1, 0, 0, 0, 1, 1], dtype=float)
        vc = estimate_variance_components(a, b, binary=False)
        d = a - b
        assert vc["n"] == 10
        assert vc["n_clusters"] is None
        assert vc["var_paired_diff"] == pytest.approx(float(np.var(d, ddof=1)))
        assert vc["sigma2_a"] == 0.0
        assert vc["sigma2_b"] == 0.0
        assert vc["omega2"] == pytest.approx(vc["var_paired_diff"])

    def test_binary_flag_yields_zero_on_bernoulli(self) -> None:
        a = np.array([1, 0, 1, 0, 1, 0], dtype=float)
        b = np.array([0, 1, 1, 0, 0, 1], dtype=float)
        vc = estimate_variance_components(a, b, binary=True)
        assert vc["sigma2_a"] == 0.0
        assert vc["sigma2_b"] == 0.0

    def test_binary_flag_on_fractional_scores(self) -> None:
        a = np.array([0.8, 0.6, 0.5, 0.3])
        b = np.array([0.4, 0.5, 0.5, 0.2])
        vc = estimate_variance_components(a, b, binary=True)
        assert vc["sigma2_a"] == pytest.approx(float(np.mean(a * (1 - a))))
        assert vc["sigma2_b"] == pytest.approx(float(np.mean(b * (1 - b))))

    def test_omega2_floored_at_zero(self) -> None:
        a = np.array([1, 1, 0, 0, 1, 0, 1, 1], dtype=float)
        b = a.copy()
        vc = estimate_variance_components(a, b, binary=True)
        assert vc["var_paired_diff"] == 0.0
        assert vc["omega2"] == 0.0

    def test_clustered_inflates_variance_with_correlated_within_cluster(self) -> None:
        a = np.array([0.8, 0.8, 0.8, 0.8, 0.2, 0.2, 0.2, 0.2])
        b = np.array([0.3, 0.3, 0.3, 0.3, 0.7, 0.7, 0.7, 0.7])
        cluster_ids = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        vc_cluster = estimate_variance_components(a, b, cluster_ids=cluster_ids)
        vc_plain = estimate_variance_components(a, b)
        assert vc_cluster["var_paired_diff"] > vc_plain["var_paired_diff"]

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            estimate_variance_components([1.0, 0.0], [1.0, 0.0, 1.0])

    def test_single_observation_raises(self) -> None:
        with pytest.raises(ValueError):
            estimate_variance_components([1.0], [0.0])

    def test_reports_correlation(self) -> None:
        a = np.array([1.0, 2.0, 3.0, 4.0])
        b = np.array([1.0, 2.0, 3.0, 4.0])
        vc = estimate_variance_components(a, b)
        assert vc["correlation"] == pytest.approx(1.0)


class TestPowerSummary:
    def test_contains_expected_sections(self) -> None:
        rng = np.random.default_rng(0)
        a = rng.uniform(size=50)
        b = rng.uniform(size=50)
        out = power_summary(a, b)
        assert "Var(d_i)" in out
        assert "omega^2" in out
        assert "required n for target MDE" in out
        assert "MDE at given n" in out
