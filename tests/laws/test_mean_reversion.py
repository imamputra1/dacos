"""
MEAN REVERSION TESTS (THE PHYSICS VALIDATOR)
Location: tests/laws/test_mean_reversion.py
Paradigm: Strict Mathematical Validation, Guard Clause Verification.
"""

from __future__ import annotations

import numpy as np
import pytest
from dacos.laws import (
    _kernel_hurst_exponent,
    _kernel_ou_half_life,
    compute_adf_test_safe,
    compute_engle_arch_test_safe,
    compute_half_life_safe,
    compute_hurst_exponent_safe,
)

# ============================================================================
# FIXTURES (SYNTHETIC MARKET DATA GENERATORS)
# ============================================================================


@pytest.fixture
def mean_reverting_spread() -> np.ndarray:
    """
    Generates a strongly mean-reverting series using an AR(1) process.
    y[t] = theta * y[t-1] + noise (where theta < 1)
    """
    np.random.seed(42)
    n_samples = 500
    y = np.zeros(n_samples)
    theta = 0.5  # Mean reversion speed
    for t in range(1, n_samples):
        y[t] = theta * y[t - 1] + np.random.normal(0, 1)
    return y


@pytest.fixture
def random_walk_spread() -> np.ndarray:
    """
    Generates a pure Random Walk (Geometric Brownian Motion equivalent).
    Should not be mean-reverting. Hurst should be ~0.5.
    """
    np.random.seed(42)
    return np.cumsum(np.random.normal(0, 1, 500))


@pytest.fixture
def trending_spread() -> np.ndarray:
    """
    Generates a perfectly trending linear series.
    Hurst should be ~1.0. Beta in OU will be >= 0 (diverging).
    """
    return np.linspace(0, 100, 500)


@pytest.fixture
def homoskedastic_spread() -> np.ndarray:
    """
    Generates a series with constant volatility (no ARCH effects).
    """
    np.random.seed(42)
    return np.random.normal(0, 1, 500)


@pytest.fixture
def heteroskedastic_spread() -> np.ndarray:
    """
    Generates a series with volatility clustering (ARCH effects).
    Simulated by combining blocks of low variance and high variance.
    """
    np.random.seed(42)
    low_volatility = np.random.normal(0, 1, 200)
    high_volatility = np.random.normal(0, 10, 100)
    low_volatility_again = np.random.normal(0, 1, 200)
    return np.concatenate([low_volatility, high_volatility, low_volatility_again])


# ============================================================================
# KERNEL TESTS: MATHEMATICAL CORRECTNESS
# ============================================================================


def test_kernel_ou_half_life_returns_valid_float_for_mean_reverting(mean_reverting_spread: np.ndarray) -> None:
    """Tests that a mean-reverting spread yields a positive, finite half-life."""
    half_life = _kernel_ou_half_life(mean_reverting_spread)
    assert not np.isnan(half_life)
    assert half_life > 0.0


def test_kernel_ou_half_life_returns_nan_for_trending_series(trending_spread: np.ndarray) -> None:
    """Tests that a trending series (non-stationary, diverging) is cut off and returns NaN."""
    half_life = _kernel_ou_half_life(trending_spread)
    assert np.isnan(half_life)


def test_kernel_hurst_exponent_classifies_regimes_correctly(
    mean_reverting_spread: np.ndarray,
    random_walk_spread: np.ndarray,
    trending_spread: np.ndarray,
) -> None:
    """
    Tests the mathematical boundaries of the Hurst Exponent.
    H < 0.5 = Mean Reverting
    H ~ 0.5 = Random Walk
    H > 0.5 = Trending
    """
    max_lag = 20

    hurst_mr = _kernel_hurst_exponent(mean_reverting_spread, max_lag)
    hurst_rw = _kernel_hurst_exponent(random_walk_spread, max_lag)
    hurst_tr = _kernel_hurst_exponent(trending_spread, max_lag)

    # Mean reverting should be strictly less than 0.5
    assert hurst_mr < 0.5, f"Expected H < 0.5 for mean-reverting, got {hurst_mr}"

    # Random walk should be around 0.5
    assert 0.4 <= hurst_rw <= 0.6, f"Expected H ~ 0.5 for random walk, got {hurst_rw}"

    # Trending should be close to 1.0 (or at least significantly > 0.5)
    assert hurst_tr > 0.8, f"Expected H > 0.8 for trending, got {hurst_tr}"


# ============================================================================
# SAFE WRAPPER TESTS: GUARD CLAUSES & ERROR HANDLING
# ============================================================================


def test_compute_adf_test_safe_detects_stationarity(
    mean_reverting_spread: np.ndarray, random_walk_spread: np.ndarray
) -> None:
    """Tests ADF Test correctly identifies stationary vs non-stationary series."""
    # Mean Reverting (Stationary -> p-value < 0.05)
    result_mr = compute_adf_test_safe(mean_reverting_spread)
    assert result_mr.is_ok()
    assert result_mr.unwrap()["p_value"] < 0.05

    # Random Walk (Non-Stationary -> p-value > 0.05)
    result_rw = compute_adf_test_safe(random_walk_spread)
    assert result_rw.is_ok()
    assert result_rw.unwrap()["p_value"] > 0.05


def test_compute_adf_test_safe_returns_err_on_invalid_input() -> None:
    """Tests guard clauses for ADF Test."""
    # 2D Array
    assert compute_adf_test_safe(np.zeros((25, 2))).is_err()
    # Not enough samples (< 20)
    assert compute_adf_test_safe(np.zeros(15)).is_err()


def test_compute_half_life_safe_returns_ok_for_valid_data(mean_reverting_spread: np.ndarray) -> None:
    """Tests half-life wrapper success path."""
    result = compute_half_life_safe(mean_reverting_spread)
    assert result.is_ok()
    assert result.unwrap() > 0.0


def test_compute_half_life_safe_returns_err_on_invalid_data(trending_spread: np.ndarray) -> None:
    """Tests half-life wrapper gracefully catches kernel NaN output (divergence)."""
    result = compute_half_life_safe(trending_spread)
    assert result.is_err()
    assert "not mean-reverting" in str(result.unwrap_err())


def test_compute_half_life_safe_guard_clauses() -> None:
    """Tests guard clauses for Half-Life computation."""
    # 2D Array
    assert compute_half_life_safe(np.zeros((10, 2))).is_err()
    # Too few samples (< 3)
    assert compute_half_life_safe(np.array([1.0, 2.0])).is_err()


def test_compute_hurst_exponent_safe_returns_ok_for_valid_data(mean_reverting_spread: np.ndarray) -> None:
    """Tests Hurst wrapper success path."""
    result = compute_hurst_exponent_safe(mean_reverting_spread, max_lag=20)
    assert result.is_ok()
    assert result.unwrap() < 0.5


def test_compute_hurst_exponent_safe_guard_clauses() -> None:
    """Tests guard clauses for Hurst computation."""
    # 2D Array
    assert compute_hurst_exponent_safe(np.zeros((50, 2)), max_lag=20).is_err()
    # Insufficient data for the given lag
    assert compute_hurst_exponent_safe(np.zeros(15), max_lag=20).is_err()


def test_compute_engle_arch_test_safe_detects_no_arch(homoskedastic_spread: np.ndarray) -> None:
    """Tests ARCH test returns high p-value for series with constant variance."""
    result = compute_engle_arch_test_safe(homoskedastic_spread)
    assert result.is_ok()

    # p-value > 0.05 means we fail to reject null hypothesis (No ARCH effects)
    assert result.unwrap()["p_value"] > 0.05


def test_compute_engle_arch_test_safe_detects_arch(heteroskedastic_spread: np.ndarray) -> None:
    """Tests ARCH test returns low p-value for series with volatility clustering."""
    result = compute_engle_arch_test_safe(heteroskedastic_spread)
    assert result.is_ok()

    # p-value < 0.05 means we reject null hypothesis (ARCH effects exist)
    assert result.unwrap()["p_value"] < 0.05


def test_compute_engle_arch_test_safe_guard_clauses() -> None:
    """Tests guard clauses for Engle's ARCH test computation."""
    # 2D Array
    assert compute_engle_arch_test_safe(np.zeros((50, 2))).is_err()
    # Insufficient data (max_lag=5 requires at least 15 samples based on our guard clause)
    assert compute_engle_arch_test_safe(np.zeros(10)).is_err()
