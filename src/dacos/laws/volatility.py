from __future__ import annotations

import numpy as np
from numba import njit

from dacos.utils import Err, Ok, Result


@njit(cache=True, fastmath=True)
def _kernel_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, window: int) -> np.ndarray:
    n = high.shape[0]
    atr = np.empty(n, dtype=np.float64)
    atr[:] = np.nan

    if n <= window:
        return atr

    tr = np.empty(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = np.abs(high[i] - close[i - 1])
        lc = np.abs(low[i] - close[i - 1])
        tr[i] = max(hl, hc, lc)

    sum_tr = 0.0
    for i in range(1, window + 1):
        sum_tr += tr[i]
    atr[window] = sum_tr / float(window)

    for i in range(window + 1, n):
        atr[i] = (atr[i - 1] * (window - 1) + tr[i]) / float(window)

    return atr


@njit(cache=True, fastmath=True)
def _kernel_donchian_channels(high: np.ndarray, low: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    n = high.shape[0]
    upper = np.empty(n, dtype=np.float64)
    lower = np.empty(n, dtype=np.float64)
    upper[:] = np.nan
    lower[:] = np.nan

    if n < window:
        return upper, lower

    for i in range(window - 1, n):
        max_h = high[i - window + 1]
        min_l = low[i - window + 1]
        for j in range(1, window):
            idx = i - window + 1 + j
            if high[idx] > max_h:
                max_h = high[idx]
            if low[idx] < min_l:
                min_l = low[idx]
        upper[i] = max_h
        lower[i] = min_l

    return upper, lower


@njit(cache=True, fastmath=True)
def _kernel_garman_klass(
    open_p: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray, window: int, ann_factor: float
) -> np.ndarray:
    n = open_p.shape[0]
    gk_vol = np.empty(n, dtype=np.float64)
    gk_vol[:] = np.nan

    if n < window:
        return gk_vol

    log_hl = np.log(high / low)
    log_co = np.log(close / open_p)
    rs = 0.5 * (log_hl * log_hl) - (2.0 * np.log(2.0) - 1.0) * (log_co * log_co)

    for i in range(window - 1, n):
        sum_rs = 0.0
        for j in range(window):
            sum_rs += rs[i - j]
        variance = sum_rs / float(window)
        gk_vol[i] = np.sqrt(variance * ann_factor)

    return gk_vol


@njit(cache=True, fastmath=True)
def _kernel_yang_zhang(
    open_p: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray, window: int, ann_factor: float
) -> np.ndarray:
    n = open_p.shape[0]
    yz_vol = np.empty(n, dtype=np.float64)
    yz_vol[:] = np.nan

    if n < window + 1:
        return yz_vol

    log_ho = np.log(high / open_p)
    log_lo = np.log(low / open_p)
    log_co = np.log(close / open_p)

    log_oc = np.empty(n, dtype=np.float64)
    log_oc[0] = 0.0
    for i in range(1, n):
        log_oc[i] = np.log(open_p[i] / close[i - 1])

    rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)
    k = 0.34 / (1.34 + (window + 1.0) / (window - 1.0))

    for i in range(window, n):
        mean_oc = 0.0
        mean_co = 0.0
        for j in range(window):
            mean_oc += log_oc[i - j]
            mean_co += log_co[i - j]
        mean_oc /= float(window)
        mean_co /= float(window)

        var_oc = 0.0
        var_co = 0.0
        var_rs = 0.0
        for j in range(window):
            diff_oc = log_oc[i - j] - mean_oc
            var_oc += diff_oc * diff_oc

            diff_co = log_co[i - j] - mean_co
            var_co += diff_co * diff_co

            var_rs += rs[i - j]

        var_oc /= (window - 1.0)
        var_co /= (window - 1.0)
        var_rs /= float(window)

        variance = var_oc + k * var_co + (1.0 - k) * var_rs
        yz_vol[i] = np.sqrt(variance * ann_factor)

    return yz_vol


def compute_atr_safe(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, window: int = 14
) -> Result[np.ndarray, ValueError]:
    """
    Args:
        high: 1D array of high prices.
        low: 1D array of low prices.
        close: 1D array of close prices.
        window: Lookback period.

    Returns:
        Ok(np.ndarray) containing ATR, or Err(ValueError).
    """
    if not (high.ndim == low.ndim == close.ndim == 1):
        return Err(ValueError("All input arrays must be 1D."))
    if not (high.shape[0] == low.shape[0] == close.shape[0]):
        return Err(ValueError("Input arrays must have the same length."))
    if window < 2:
        return Err(ValueError("Window must be at least 2."))

    try:
        return Ok(_kernel_atr(high.astype(np.float64), low.astype(np.float64), close.astype(np.float64), window))
    except Exception as e:
        return Err(ValueError(f"ATR computation failed: {e}"))


def compute_donchian_channels_safe(
    high: np.ndarray, low: np.ndarray, window: int = 20
) -> Result[tuple[np.ndarray, np.ndarray], ValueError]:
    """
    Args:
        high: 1D array of high prices.
        low: 1D array of low prices.
        window: Lookback period.

    Returns:
        Ok((upper_channel, lower_channel)), or Err(ValueError).
    """
    if not (high.ndim == low.ndim == 1):
        return Err(ValueError("All input arrays must be 1D."))
    if high.shape[0] != low.shape[0]:
        return Err(ValueError("Input arrays must have the same length."))
    if window < 2:
        return Err(ValueError("Window must be at least 2."))

    try:
        return Ok(_kernel_donchian_channels(high.astype(np.float64), low.astype(np.float64), window))
    except Exception as e:
        return Err(ValueError(f"Donchian Channels computation failed: {e}"))


def compute_garman_klass_safe(
    open_p: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray, window: int = 20, ann_factor: float = 365.0
) -> Result[np.ndarray, ValueError]:
    """
    Args:
        open_p: 1D array of open prices.
        high: 1D array of high prices.
        low: 1D array of low prices.
        close: 1D array of close prices.
        window: Lookback period.
        ann_factor: Annualization factor.

    Returns:
        Ok(np.ndarray) containing Garman-Klass volatility, or Err(ValueError).
    """
    if not (open_p.ndim == high.ndim == low.ndim == close.ndim == 1):
        return Err(ValueError("All input arrays must be 1D."))
    if not (open_p.shape[0] == high.shape[0] == low.shape[0] == close.shape[0]):
        return Err(ValueError("Input arrays must have the same length."))
    if window < 2:
        return Err(ValueError("Window must be at least 2."))

    try:
        return Ok(
            _kernel_garman_klass(
                open_p.astype(np.float64),
                high.astype(np.float64),
                low.astype(np.float64),
                close.astype(np.float64),
                window,
                float(ann_factor),
            )
        )
    except Exception as e:
        return Err(ValueError(f"Garman-Klass computation failed: {e}"))


def compute_yang_zhang_safe(
    open_p: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray, window: int = 20, ann_factor: float = 365.0
) -> Result[np.ndarray, ValueError]:
    """
    Args:
        open_p: 1D array of open prices.
        high: 1D array of high prices.
        low: 1D array of low prices.
        close: 1D array of close prices.
        window: Lookback period.
        ann_factor: Annualization factor.

    Returns:
        Ok(np.ndarray) containing Yang-Zhang volatility, or Err(ValueError).
    """
    if not (open_p.ndim == high.ndim == low.ndim == close.ndim == 1):
        return Err(ValueError("All input arrays must be 1D."))
    if not (open_p.shape[0] == high.shape[0] == low.shape[0] == close.shape[0]):
        return Err(ValueError("Input arrays must have the same length."))
    if window < 2:
        return Err(ValueError("Window must be at least 2."))

    try:
        return Ok(
            _kernel_yang_zhang(
                open_p.astype(np.float64),
                high.astype(np.float64),
                low.astype(np.float64),
                close.astype(np.float64),
                window,
                float(ann_factor),
            )
        )
    except Exception as e:
        return Err(ValueError(f"Yang-Zhang computation failed: {e}"))


__all__ = [
    "_kernel_atr",
    "_kernel_yang_zhang",
    "_kernel_donchian_channels",
    "_kernel_garman_klass",
    "compute_atr_safe",
    "compute_donchian_channels_safe",
    "compute_garman_klass_safe",
    "compute_yang_zhang_safe",
]
