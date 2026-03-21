from __future__ import annotations

import numpy as np
from numba import njit
from statsmodels.stats.diagnostic import het_arch
from statsmodels.tsa.stattools import adfuller

from dacos.utils import Err, Ok, Result

# ============================================================================
# ⚙️ LEVEL 1: HARDWARE KERNELS (C-LEVEL EXECUTION via NUMBA)
# ============================================================================


@njit(cache=True, fastmath=True)
def _kernel_ou_half_life(spread_array: np.ndarray) -> float:
    """
    Computes Ornstein-Uhlenbeck Half-Life using highly optimized 1D OLS.
    Equation: delta_y = beta * y_lag + alpha
    Half-Life = -ln(2) / beta
    """
    total_samples = spread_array.shape[0]
    if total_samples < 3:
        return np.nan

    delta_y = np.empty(total_samples - 1, dtype=np.float64)
    y_lag = np.empty(total_samples - 1, dtype=np.float64)

    for i in range(1, total_samples):
        delta_y[i - 1] = spread_array[i] - spread_array[i - 1]
        y_lag[i - 1] = spread_array[i - 1]

    mean_x = np.mean(y_lag)
    mean_y = np.mean(delta_y)

    cov_xy = 0.0
    var_x = 0.0
    for i in range(total_samples - 1):
        diff_x = y_lag[i] - mean_x
        diff_y = delta_y[i] - mean_y
        cov_xy += diff_x * diff_y
        var_x += diff_x * diff_x

    if var_x == 0.0:
        return np.nan

    beta = cov_xy / var_x

    if beta >= -1e-8:
        return np.nan

    half_life = -np.log(2.0) / beta
    return half_life


@njit(cache=True, fastmath=True)
def _kernel_hurst_exponent(spread_array: np.ndarray, max_lag: int) -> float:
    """
    Computes the Hurst Exponent (H) using Mean Squared Displacement (MSD).
    E[(z_t - z_{t-tau})^2] is proportional to tau^(2H).
    OLS of log(MSD) on log(tau) yields a slope of 2H.
    """
    total_samples = spread_array.shape[0]
    if total_samples < max_lag + 1:
        return np.nan

    lags = np.arange(2, max_lag + 1)
    num_lags = lags.shape[0]

    tau_log_array = np.empty(num_lags, dtype=np.float64)
    var_log_array = np.empty(num_lags, dtype=np.float64)

    for i in range(num_lags):
        current_lag = lags[i]
        tau_log_array[i] = np.log(float(current_lag))

        diff_len = total_samples - current_lag
        sum_sq_diff = 0.0

        for j in range(diff_len):
            diff = spread_array[j + current_lag] - spread_array[j]
            sum_sq_diff += diff * diff

        msd = sum_sq_diff / float(diff_len)

        if msd <= 1e-15:
            var_log_array[i] = np.nan
        else:
            var_log_array[i] = np.log(msd)

    mean_x = np.nanmean(tau_log_array)
    mean_y = np.nanmean(var_log_array)

    cov_xy = 0.0
    var_x = 0.0
    for i in range(num_lags):
        if not np.isnan(var_log_array[i]):
            dx = tau_log_array[i] - mean_x
            dy = var_log_array[i] - mean_y
            cov_xy += dx * dy
            var_x += dx * dx

    if var_x == 0.0:
        return np.nan

    slope = cov_xy / var_x
    hurst_exponent = slope / 2.0

    return hurst_exponent


# ============================================================================
# 🛡️ LEVEL 2: SAFE MONADIC WRAPPERS (PYTHON RUNTIME PROTECTOR)
# ============================================================================


def compute_adf_test_safe(spread_series: np.ndarray) -> Result[dict[str, float], ValueError]:
    """
    Performs Augmented Dickey-Fuller (ADF) test using statsmodels.

    Args:
        spread_series: 1D Numpy array of the spread.

    Returns:
        Ok(dict) containing 'adf_statistic' and 'p_value', or Err(ValueError).
    """
    if spread_series.ndim != 1:
        return Err(ValueError(f"ADF test requires 1D array, got {spread_series.ndim}D."))

    if spread_series.shape[0] < 20:
        return Err(
            ValueError(f"Insufficient data for ADF test. Got {spread_series.shape[0]} samples, minimum 20 required.")
        )

    try:
        adf_result = adfuller(spread_series, maxlag=1, autolag=None)

        results = {
            "adf_statistic": float(adf_result[0]),
            "p_value": float(adf_result[1]),
            "critical_value_1pct": float(adf_result[4]["1%"]),
            "critical_value_5pct": float(adf_result[4]["5%"]),
            "critical_value_10pct": float(adf_result[4]["10%"]),
        }
        return Ok(results)
    except Exception as computation_error:
        return Err(ValueError(f"ADF Test failed: {computation_error}"))


def compute_half_life_safe(spread_series: np.ndarray) -> Result[float, ValueError]:
    """
    Computes the Ornstein-Uhlenbeck mean-reversion half-life.

    Args:
        spread_series: 1D Numpy array of the spread.

    Returns:
        Ok(half_life) in ticks/periods, or Err(ValueError) if divergent/invalid.
    """
    if spread_series.ndim != 1:
        return Err(ValueError(f"Half-life requires 1D array, got {spread_series.ndim}D."))

    if spread_series.shape[0] < 3:
        return Err(ValueError("Half-life requires at least 3 samples."))

    try:
        safe_array = spread_series.astype(np.float64)
        half_life = _kernel_ou_half_life(safe_array)

        if np.isnan(half_life):
            return Err(ValueError("Spread is not mean-reverting (beta >= 0). Half-life is infinite/undefined."))

        return Ok(half_life)
    except Exception as computation_error:
        return Err(ValueError(f"Half-life computation failed: {computation_error}"))


def compute_hurst_exponent_safe(spread_series: np.ndarray, max_lag: int = 20) -> Result[float, ValueError]:
    """
    Computes the Hurst Exponent to measure long-term memory of the spread.
    H < 0.5: Mean Reverting
    H = 0.5: Geometric Brownian Motion (Random Walk)
    H > 0.5: Trending / Momentum

    Args:
        spread_series: 1D Numpy array of the spread.
        max_lag: Maximum lag window for variance testing.

    Returns:
        Ok(hurst_exponent), or Err(ValueError).
    """
    if spread_series.ndim != 1:
        return Err(ValueError(f"Hurst calculation requires 1D array, got {spread_series.ndim}D."))

    if spread_series.shape[0] < max_lag + 1:
        return Err(ValueError(f"Insufficient data. Need at least {max_lag + 1} samples for max_lag={max_lag}."))

    try:
        safe_array = spread_series.astype(np.float64)
        hurst = _kernel_hurst_exponent(safe_array, int(max_lag))

        if np.isnan(hurst):
            return Err(ValueError("Hurst Exponent resulted in NaN (likely constant array or zero variance)."))

        return Ok(float(hurst))
    except Exception as computation_error:
        return Err(ValueError(f"Hurst Exponent computation failed: {computation_error}"))


def compute_engle_arch_test_safe(spread_series: np.ndarray, max_lag: int = 5) -> Result[dict[str, float], ValueError]:
    """
    Performs Engle's Test for Autoregressive Conditional Heteroskedasticity (ARCH).
    Validates if the spread exhibits volatility clustering.

    Args:
        spread_series: 1D Numpy array of the spread.
        max_lag: Maximum number of lags to include in the test.

    Returns:
        Ok(dict) containing 'lm_statistic', 'p_value', 'f_statistic', and 'f_pvalue',
        or Err(ValueError).
    """
    if spread_series.ndim != 1:
        return Err(ValueError(f"ARCH test requires 1D array, got {spread_series.ndim}D."))

    if spread_series.shape[0] < max_lag + 10:
        return Err(ValueError(f"Insufficient data for ARCH test. Need at least {max_lag + 10} samples."))

    try:
        residuals = spread_series - np.mean(spread_series)

        arch_result = het_arch(residuals, nlags=max_lag)

        results = {
            "lm_statistic": float(arch_result[0]),
            "p_value": float(arch_result[1]),
            "f_statistic": float(arch_result[2]),
            "f_pvalue": float(arch_result[3]),
        }
        return Ok(results)
    except Exception as computation_error:
        return Err(ValueError(f"ARCH Test failed: {computation_error}"))


__all__ = [
    "_kernel_ou_half_life",
    "_kernel_hurst_exponent",
    "compute_adf_test_safe",
    "compute_half_life_safe",
    "compute_hurst_exponent_safe",
    "compute_engle_arch_test_safe",
]
