from __future__ import annotations

import numpy as np
import pytest
from dacos.laws import (
    compute_atr_safe,
    compute_donchian_channels_safe,
    compute_garman_klass_safe,
    compute_yang_zhang_safe,
)


@pytest.fixture
def ohlc_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
        tuple: (open, high, low, close) 1D arrays of size 100.
    """
    np.random.seed(42)
    n = 100
    open_p = np.random.uniform(100, 110, n)
    high = open_p + np.random.uniform(1, 5, n)
    low = open_p - np.random.uniform(1, 5, n)
    close = open_p + np.random.uniform(-2, 2, n)
    return open_p, high, low, close


def test_compute_atr_safe_valid(ohlc_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> None:
    _, high, low, close = ohlc_data
    window = 14
    result = compute_atr_safe(high, low, close, window)

    assert result.is_ok()
    atr = result.unwrap()
    assert atr.shape == high.shape
    assert np.isnan(atr[: window - 1]).all()
    assert not np.isnan(atr[window:]).any()
    assert (atr[window:] > 0).all()


def test_compute_donchian_channels_safe_valid(ohlc_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> None:
    _, high, low, _ = ohlc_data
    window = 20
    result = compute_donchian_channels_safe(high, low, window)

    assert result.is_ok()
    upper, lower = result.unwrap()
    assert upper.shape == high.shape
    assert lower.shape == low.shape
    assert np.isnan(upper[: window - 1]).all()
    assert np.isnan(lower[: window - 1]).all()
    assert not np.isnan(upper[window - 1 :]).any()
    assert not np.isnan(lower[window - 1 :]).any()
    assert (upper[window - 1 :] >= lower[window - 1 :]).all()


def test_compute_garman_klass_safe_valid(ohlc_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> None:
    open_p, high, low, close = ohlc_data
    window = 20
    result = compute_garman_klass_safe(open_p, high, low, close, window)

    assert result.is_ok()
    gk = result.unwrap()
    assert gk.shape == open_p.shape
    assert np.isnan(gk[: window - 1]).all()
    assert not np.isnan(gk[window - 1 :]).any()
    assert (gk[window - 1 :] >= 0).all()


def test_compute_yang_zhang_safe_valid(ohlc_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> None:
    open_p, high, low, close = ohlc_data
    window = 20
    result = compute_yang_zhang_safe(open_p, high, low, close, window)

    assert result.is_ok()
    yz = result.unwrap()
    assert yz.shape == open_p.shape
    assert np.isnan(yz[:window]).all()
    assert not np.isnan(yz[window:]).any()
    assert (yz[window:] >= 0).all()


def test_guard_clauses_invalid_dimensions(ohlc_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> None:
    open_p, high, low, close = ohlc_data
    high_2d = high.reshape(-1, 1)

    assert compute_atr_safe(high_2d, low, close).is_err()
    assert compute_donchian_channels_safe(high_2d, low).is_err()
    assert compute_garman_klass_safe(open_p, high_2d, low, close).is_err()
    assert compute_yang_zhang_safe(open_p, high_2d, low, close).is_err()


def test_guard_clauses_mismatched_lengths(ohlc_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> None:
    open_p, high, low, close = ohlc_data
    short_low = low[:-5]

    assert compute_atr_safe(high, short_low, close).is_err()
    assert compute_donchian_channels_safe(high, short_low).is_err()
    assert compute_garman_klass_safe(open_p, high, short_low, close).is_err()
    assert compute_yang_zhang_safe(open_p, high, short_low, close).is_err()


def test_guard_clauses_invalid_window(ohlc_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> None:
    open_p, high, low, close = ohlc_data

    assert compute_atr_safe(high, low, close, window=1).is_err()
    assert compute_donchian_channels_safe(high, low, window=1).is_err()
    assert compute_garman_klass_safe(open_p, high, low, close, window=1).is_err()
    assert compute_yang_zhang_safe(open_p, high, low, close, window=1).is_err()
