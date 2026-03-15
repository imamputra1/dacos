"""
Stastical test for mean reversion: Hurst Exponent, ADF P-Value, and half-life.
All functions are decoreted with @safe to Result[float, Exceptions].
"""

import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

from dacos.utils import safe


@safe
def calculate_hurst(series: np.ndarray) -> float:
    """
    Calculate the Hurst exponent using the variance of lagged differences method.

    Args:
        series: 1D numpy array of price (or log_price).

    Returns:
        Hurst exponent (float). Interpretation:
        - H < 0.5: mean-reversion
        - H = 0.5: random walk
        - H > 0.5: trending

    Raise:
        ValueError: if series is too short or variance becomes zero.
    """
    if len(series) < 20:
        raise ValueError(f"Series to short (len={len(series)}), need at least 20 observations")

    lags = range(2, min(20, len(series) // 2))
    if len(lags) < 2:
        raise ValueError("Insufficient lags for Hurst calculation")

    variances = []
    for lag in lags:
        diff = series[lag:] - series [:-lag]
        var = np.var(diff)
        if var == 0:
            raise ValueError("Series is constant (zero variance)")
        variances.append(var)

    log_lags = np.log(lags)
    log_var = np.log(variances)

    slope, _ = np.polyfit(log_lags, log_var, 1)

    hurst = slope / 2.0
    return hurst

@safe
def calculate_adf_pvalue(series: np.ndarray) -> float:
    """
    Calculate the P-Value of the Augmented Dickey-Fuller test for stationarity.

    Args:
        series: 1D numpy array of prices (or log_price)

    Returns:
        P-Value (float): Typically, p < 0.05 indicates stationarity.

    Raises:
        ValueError: if series is too short (ADF requires at least some observations)
    """
    if len(series) < 10:
        raise ValueError(f"Series too short (len={len(series)}), need at least 10 observations")

    result = adfuller(series, maxlag=1, autolag=None)  # type: ignore[arg-type]
    pvalue = result[1]
    return pvalue

@safe
def calculate_halflife(series: np.ndarray) -> float:
    """
    Calculate the half-life of mean reversion using ornstein-uhlenback process.

    step:
        1. Form lagged price (y_prev = series[:-1] and delta (dy = series[1:] - y_prev)).
        2. Regress dy on y_prev with constant.
        3. Extract lambda (coefiesient of y_prev).
        4. Half-Life = -ln(-2) / lambda.

    Args:
        Series: 1D numpy array of price (or log_price)

    Returns:
        Half-Life in the same time units as the series (e.g., minutes if series is minutely)

    Raise:
        ValueError: if series is too short, lambda >= 0 (no mean reversion), or OLS fails.
    """
    if len(series) < 10:
        raise ValueError(f"Series too short (len={len(series)}), need at least 10 observations")

    y_prev = series[:-1]
    dy = series[1:] - y_prev

    X = sm.add_constant(y_prev)
    model = sm.OLS(dy, X).fit()

    lambda_val = model.params[1]

    if lambda_val >= 0:
        raise ValueError(f"Lambda = {lambda_val} >= 0, series is not mean-reversion")

    half_life = -np.log(2) / lambda_val
    return half_life
