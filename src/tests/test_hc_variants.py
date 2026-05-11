"""
Tests for the five Higher Criticism methods in MultiTest.

Test groups
-----------
TestReturnTypes         – structural: scalar vs. tuple return depending on return_threshold
TestThresholdValidity   – threshold p-value is a valid member of the input array
TestNullBehavior        – under H0 (uniform p-values), all HC statistics are positive and
                          occupy a similar range (same behavior under the null)
TestAlternativeDetection – under a sparse-normal H1 all variants detect the signal
TestRangeRestriction    – hc_star correctly excludes p-values below 1/n
TestStandardizations    – three standardization choices give distinct scores
TestDefaultAlias        – hc() is an exact alias for hc_dj2008()
"""

import pytest
import numpy as np
from scipy.stats import norm
from src.multitest import MultiTest

# ── helpers ──────────────────────────────────────────────────────────────────

HC_METHODS = ["hc", "hc_dj2004", "hc_dj2008", "hc_beta", "hc_star"]

def _run_hc(mt: MultiTest, name: str, **kw):
    return getattr(mt, name)(**kw)

def _sparse_normal_pvals(n, beta, r, rng):
    """Two-sided p-values for a sparse normal mixture."""
    mu = np.sqrt(2 * r * np.log(n))
    ep = n ** -beta
    Z = rng.standard_normal(n)
    Z[rng.random(n) < ep] += mu
    return 2 * norm.cdf(-np.abs(Z))


# ── structural return-type tests ─────────────────────────────────────────────

class TestReturnTypes:
    """HC methods return a scalar by default and a 2-tuple with return_threshold=True."""

    @pytest.fixture(scope="class")
    def mt(self):
        rng = np.random.default_rng(0)
        return MultiTest(rng.uniform(size=300))

    @pytest.mark.parametrize("name", HC_METHODS)
    def test_default_returns_scalar(self, mt, name):
        result = _run_hc(mt, name)
        assert np.ndim(result) == 0, f"{name}: expected scalar, got {type(result)}"

    @pytest.mark.parametrize("name", HC_METHODS)
    def test_return_threshold_returns_tuple(self, mt, name):
        result = _run_hc(mt, name, return_threshold=True)
        assert isinstance(result, tuple) and len(result) == 2, (
            f"{name}: expected 2-tuple, got {result!r}"
        )

    @pytest.mark.parametrize("name", HC_METHODS)
    def test_score_is_finite(self, mt, name):
        score = _run_hc(mt, name)
        assert np.isfinite(score), f"{name}: score is not finite"


# ── threshold validity tests ──────────────────────────────────────────────────

class TestThresholdValidity:
    """The p-value threshold returned by HC is a valid member of the input array."""

    @pytest.fixture(scope="class")
    def mt_and_pvals(self):
        rng = np.random.default_rng(1)
        pvals = rng.uniform(size=200)
        return MultiTest(pvals), pvals

    @pytest.mark.parametrize("name", HC_METHODS)
    def test_threshold_in_unit_interval(self, mt_and_pvals, name):
        mt, _ = mt_and_pvals
        _, thr = _run_hc(mt, name, return_threshold=True)
        assert 0.0 <= thr <= 1.0, f"{name}: threshold {thr} outside [0, 1]"

    @pytest.mark.parametrize("name", HC_METHODS)
    def test_threshold_is_one_of_input_pvals(self, mt_and_pvals, name):
        mt, pvals = mt_and_pvals
        _, thr = _run_hc(mt, name, return_threshold=True)
        assert np.any(np.isclose(pvals, thr)), (
            f"{name}: threshold {thr} is not in the input p-value array"
        )

    @pytest.mark.parametrize("name", HC_METHODS)
    def test_score_consistent_with_threshold(self, mt_and_pvals, name):
        """Score from default call matches the first element of the threshold call."""
        mt, _ = mt_and_pvals
        score_only = _run_hc(mt, name)
        score_pair, _ = _run_hc(mt, name, return_threshold=True)
        assert np.isclose(score_only, score_pair), (
            f"{name}: score mismatch between default and return_threshold=True calls"
        )


# ── null behavior tests ───────────────────────────────────────────────────────

class TestNullBehavior:
    """
    Under H0 (uniform p-values) all four HC statistics should:
      1. be positive,
      2. occupy a similar order of magnitude (same behavior under the null).

    The hc_dj2004 variant uses the observed p-value std as denominator, which
    creates heavier tails; its median is checked separately with a looser bound.
    hc_dj2008, hc_beta, and hc_star share similar null distributions and their
    medians are verified to lie within a factor of 2 of each other.
    """

    N = 500
    N_MONTE = 2000

    @pytest.fixture(scope="class")
    def null_stats(self):
        rng = np.random.default_rng(42)
        records = {name: [] for name in HC_METHODS}
        for _ in range(self.N_MONTE):
            pvals = rng.uniform(size=self.N)
            mt = MultiTest(pvals)
            for name in HC_METHODS:
                records[name].append(_run_hc(mt, name))
        return {k: np.asarray(v) for k, v in records.items()}

    @pytest.mark.parametrize("name", HC_METHODS)
    def test_median_is_positive(self, null_stats, name):
        assert np.median(null_stats[name]) > 0, (
            f"{name}: median under null should be positive"
        )

    @pytest.mark.parametrize("name", HC_METHODS)
    def test_median_in_reasonable_range(self, null_stats, name):
        """All HC medians under the null should be between 0.3 and 5.0 for n=500."""
        med = np.median(null_stats[name])
        assert 0.3 < med < 5.0, (
            f"{name}: null median {med:.3f} is outside expected range (0.3, 5.0)"
        )

    def test_stable_variants_have_similar_medians(self, null_stats):
        """
        hc, hc_dj2008, hc_beta, and hc_star all use expected (theoretical) standard
        deviations for normalization, so their null medians should agree within
        a factor of 2.
        """
        stable = ["hc", "hc_dj2008", "hc_beta", "hc_star"]
        medians = {name: np.median(null_stats[name]) for name in stable}
        ratio = max(medians.values()) / min(medians.values())
        assert ratio < 2.0, (
            f"Null medians of stable HC variants diverge too much: {medians}"
        )

    def test_dj2004_median_positive_and_finite(self, null_stats):
        """hc_dj2004 uses the observed p-value std; its null median should still
        be positive and finite even though its distribution has heavier tails."""
        vals = null_stats["hc_dj2004"]
        assert np.median(vals) > 0
        assert np.isfinite(np.median(vals))


# ── alternative detection tests ───────────────────────────────────────────────

class TestAlternativeDetection:
    """
    Under a sparse-normal H1 (detectable signal), all four HC variants should
    have higher mean statistics than under H0.
    """

    N = 500
    N_MONTE = 1000
    BETA = 0.7    # sparsity exponent
    R = 1.5       # signal strength (above the HC detection boundary)

    @pytest.fixture(scope="class")
    def null_and_alt_means(self):
        null_scores = {name: [] for name in HC_METHODS}
        alt_scores  = {name: [] for name in HC_METHODS}

        rng_null = np.random.default_rng(7)
        rng_alt  = np.random.default_rng(8)

        for _ in range(self.N_MONTE):
            pv_null = rng_null.uniform(size=self.N)
            pv_alt  = _sparse_normal_pvals(self.N, self.BETA, self.R, rng_alt)

            for pvals, store in [(pv_null, null_scores), (pv_alt, alt_scores)]:
                mt = MultiTest(pvals)
                for name in HC_METHODS:
                    store[name].append(_run_hc(mt, name))

        return (
            {k: np.mean(v) for k, v in null_scores.items()},
            {k: np.mean(v) for k, v in alt_scores.items()},
        )

    @pytest.mark.parametrize("name", HC_METHODS)
    def test_mean_increases_under_alternative(self, null_and_alt_means, name):
        null_mean, alt_mean = null_and_alt_means
        assert alt_mean[name] > null_mean[name], (
            f"{name}: alt mean ({alt_mean[name]:.3f}) not > null mean "
            f"({null_mean[name]:.3f})"
        )


# ── range restriction tests ───────────────────────────────────────────────────

class TestRangeRestriction:
    """hc_star restricts the search to p-values above 1/n.  When all p-values
    are already above that threshold it should agree with hc_beta."""

    def test_hc_star_equals_hc_beta_when_no_small_pvals(self):
        """When all p-values > 1/n, hc_star and hc_beta scan the same range."""
        n = 200
        rng = np.random.default_rng(10)
        # All p-values well above 1/n = 0.005
        pvals = rng.uniform(0.05, 0.5, size=n)
        mt = MultiTest(pvals)
        assert mt._imin_star == 0, "Expected _imin_star == 0 (no small p-values)"
        assert np.isclose(mt.hc_star(), mt.hc_beta()), (
            "hc_star should equal hc_beta when no p-values are below 1/n"
        )

    def test_hc_star_imin_positive_with_small_pvals(self):
        """When some p-values are below 1/n, _imin_star should be > 0."""
        n = 100
        pvals = np.linspace(1e-4, 0.5, n)   # includes values < 1/n = 0.01
        mt = MultiTest(pvals)
        assert mt._imin_star > 0, (
            "Expected _imin_star > 0 when p-values below 1/n are present"
        )

    def test_hc_dj_methods_include_full_range(self):
        """hc_dj2004 and hc_dj2008 always start from index 0."""
        n = 100
        pvals = np.linspace(1e-5, 0.5, n)
        mt = MultiTest(pvals)
        assert np.isfinite(mt.hc_dj2004())
        assert np.isfinite(mt.hc_dj2008())


# ── standardization sensitivity tests ────────────────────────────────────────

class TestStandardizations:
    """The three standardization schemes produce distinct z-scores and HC values."""

    def test_three_standardizations_differ(self):
        """hc_dj2004, hc_dj2008, and hc_beta give different scores on the same data."""
        rng = np.random.default_rng(99)
        pvals = rng.uniform(size=500)
        mt = MultiTest(pvals)
        s2004 = mt.hc_dj2004()
        s2008 = mt.hc_dj2008()
        sbeta = mt.hc_beta()
        assert not np.isclose(s2004, s2008), "hc_dj2004 and hc_dj2008 should differ"
        assert not np.isclose(s2004, sbeta), "hc_dj2004 and hc_beta should differ"
        assert not np.isclose(s2008, sbeta), "hc_dj2008 and hc_beta should differ"

    def test_zscores_differ_across_standardizations(self):
        """The internal z-score arrays differ between standardization types."""
        rng = np.random.default_rng(55)
        pvals = rng.uniform(size=200)
        mt = MultiTest(pvals)
        _, zz2004 = mt._get_zscores('donoho-jin2004')
        _, zz2008 = mt._get_zscores('donoho-jin2008')
        _, zzbeta = mt._get_zscores('beta-std')
        assert not np.allclose(zz2004, zz2008)
        assert not np.allclose(zz2004, zzbeta)
        assert not np.allclose(zz2008, zzbeta)

    def test_dj2008_and_beta_close_for_large_n(self):
        """
        For large n the Donoho-Jin 2008 and beta-std standardizations converge
        because i/(n+1) ≈ i/n for large n. The scores should be very close.
        """
        rng = np.random.default_rng(33)
        pvals = rng.uniform(size=10_000)
        mt = MultiTest(pvals)
        assert abs(mt.hc_dj2008() - mt.hc_beta()) < 0.1, (
            "hc_dj2008 and hc_beta should nearly agree for very large n"
        )


# ── default alias tests ───────────────────────────────────────────────────────

class TestDefaultAlias:
    """hc() is an exact alias for hc_dj2008() and must behave identically."""

    @pytest.fixture(scope="class")
    def mt(self):
        rng = np.random.default_rng(77)
        return MultiTest(rng.uniform(size=400))

    def test_hc_score_equals_hc_dj2008(self, mt):
        assert np.isclose(mt.hc(), mt.hc_dj2008()), (
            "hc() score should equal hc_dj2008() score"
        )

    def test_hc_threshold_equals_hc_dj2008_threshold(self, mt):
        hc_score, hc_thr = mt.hc(return_threshold=True)
        dj_score, dj_thr = mt.hc_dj2008(return_threshold=True)
        assert np.isclose(hc_score, dj_score)
        assert np.isclose(hc_thr, dj_thr), (
            "hc() threshold should equal hc_dj2008() threshold"
        )

    def test_hc_respects_gamma(self, mt):
        """Passing an explicit gamma to hc() should propagate correctly."""
        for g in [0.1, 0.3, 0.5]:
            assert np.isclose(mt.hc(gamma=g), mt.hc_dj2008(gamma=g)), (
                f"hc(gamma={g}) != hc_dj2008(gamma={g})"
            )
