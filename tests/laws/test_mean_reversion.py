"""
tests/laws/test_mean_reversion.py

Unit tests for mean reversion statistical functions.
Tests cover Hurst exponent, ADF p-value, half-life, and error handling.
"""

import numpy as np

from dacos import (
    calculate_adf_pvalue,
    calculate_halflife,
    calculate_hurst,
    is_err,
    is_ok,
)

# ============================================================================
# Test 1: Hurst Exponent (mean-reverting vs trending)
# ============================================================================

def test_hurst_mean_reverting() -> None:
    """Hurst exponent for a mean-reverting AR(1) process should be < 0.5."""
    np.random.seed(42)
    n = 10000
    series = np.zeros(n)
    phi = 0.1  # positive but <1, mean-reverting
    series[0] = np.random.normal()
    for i in range(1, n):
        series[i] = phi * series[i-1] + np.random.normal(0, 1)
    result = calculate_hurst(series)
    assert is_ok(result)
    hurst = result.ok()
    assert hurst < 0.5, f"Hurst {hurst} should be < 0.5 for mean-reverting"


def test_hurst_trending() -> None:
    """Hurst exponent for a quadratic trend should be > 0.5."""
    # Quadratic trend: var(diff) increases with lag
    series = np.linspace(0, 100, 1000) ** 2
    result = calculate_hurst(series)
    assert is_ok(result)
    hurst = result.ok()
    assert hurst > 0.5, f"Hurst {hurst} should be > 0.5 for trending"


# ============================================================================
# Test 2: ADF p-value (stationary vs non-stationary)
# ============================================================================

def test_adf_stationary() -> None:
    """ADF p-value for white noise should be < 0.05."""
    np.random.seed(42)
    series = np.random.normal(0, 1, 1000)
    result = calculate_adf_pvalue(series)
    assert is_ok(result)
    pvalue = result.ok()
    assert pvalue < 0.05, f"p-value {pvalue} should be < 0.05 for stationary"


def test_adf_nonstationary() -> None:
    """ADF p-value for random walk should be > 0.05."""
    np.random.seed(42)
    steps = np.random.normal(0, 1, 1000)
    series = np.cumsum(steps)
    result = calculate_adf_pvalue(series)
    assert is_ok(result)
    pvalue = result.ok()
    assert pvalue > 0.05, f"p-value {pvalue} should be > 0.05 for non-stationary"


# ============================================================================
# Test 3: Half-Life (mean-reverting vs trending)
# ============================================================================

def test_halflife_mean_reverting() -> None:
    """Half-life for mean-reverting AR(1) should be positive."""
    np.random.seed(42)
    n = 10000
    series = np.zeros(n)
    phi = 0.1
    series[0] = np.random.normal()
    for i in range(1, n):
        series[i] = phi * series[i-1] + np.random.normal(0, 1)
    result = calculate_halflife(series)
    assert is_ok(result)
    half_life = result.ok()
    assert half_life > 0, f"Half-life {half_life} should be positive"


def test_halflife_trending() -> None:
    """Half-life for quadratic trend should return Err (no mean reversion)."""
    series = np.linspace(0, 100, 1000) ** 2
    result = calculate_halflife(series)
    assert is_err(result), "Trending series should not have mean-reversion half-life"


# ============================================================================
# Test 4: Error handling (short series, NaNs)
# ============================================================================

def test_hurst_short_series() -> None:
    """Very short series should return Err."""
    series = np.array([1.0, 2.0])
    result = calculate_hurst(series)
    assert is_err(result)


def test_adf_short_series() -> None:
    """Very short series should return Err."""
    series = np.array([1.0, 2.0, 3.0])
    result = calculate_adf_pvalue(series)
    assert is_err(result)


def test_halflife_short_series() -> None:
    """Very short series should return Err."""
    series = np.array([1.0, 2.0, 3.0])
    result = calculate_halflife(series)
    assert is_err(result)


def test_hurst_with_nan() -> None:
    """Series containing NaN should return Err."""
    series = np.array([1.0, np.nan, 2.0, 3.0])
    result = calculate_hurst(series)
    assert is_err(result)


def test_adf_with_nan() -> None:
    """Series containing NaN should return Err."""
    series = np.array([1.0, np.nan, 2.0, 3.0])
    result = calculate_adf_pvalue(series)
    assert is_err(result)


def test_halflife_with_nan() -> None:
    """Series containing NaN should return Err."""
    series = np.array([1.0, np.nan, 2.0, 3.0])
    result = calculate_halflife(series)
    assert is_err(result)
