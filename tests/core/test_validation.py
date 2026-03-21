from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from dacos.core.validation import (
    _kernel_detect_flatline,
    _kernel_detect_spikes,
    validate_market_integrity,
)


@pytest.fixture
def valid_market_dataframe() -> pl.DataFrame:
    """Creates a DataFrame with normal, healthy price movements."""
    return pl.DataFrame(
        {
            "symbol": ["BTC-USDT"] * 5,
            "close": [100.0, 101.0, 100.5, 102.0, 101.5],
        }
    )


@pytest.fixture
def flatline_market_dataframe() -> pl.DataFrame:
    """Creates a DataFrame where the price stagnates for 4 ticks."""
    return pl.DataFrame(
        {
            "symbol": ["ETH-USDT"] * 5,
            "close": [2000.0, 2000.0, 2000.0, 2000.0, 2005.0],
        }
    )


@pytest.fixture
def spike_market_dataframe() -> pl.DataFrame:
    """Creates a DataFrame with a 60% price jump in one tick."""
    return pl.DataFrame(
        {
            "symbol": ["SOL-USDT"] * 4,
            "close": [100.0, 102.0, 163.2, 160.0],  # 102 -> 163.2 is exactly a +60% spike
        }
    )


def test_kernel_detect_flatline_returns_true_for_clean_array() -> None:
    """Tests that the Numba flatline kernel allows normal price arrays."""
    prices = np.array([10.0, 11.0, 10.5, 11.5], dtype=np.float64)
    result = _kernel_detect_flatline(prices, max_consecutive_flat=3)
    assert result is True


def test_kernel_detect_flatline_returns_false_for_stagnant_array() -> None:
    """Tests that the Numba flatline kernel catches repeating identical prices."""
    prices = np.array([10.0, 10.0, 10.0, 11.5], dtype=np.float64)
    result = _kernel_detect_flatline(prices, max_consecutive_flat=3)
    assert result is False


def test_kernel_detect_spikes_returns_true_for_clean_array() -> None:
    """Tests that the Numba spike kernel allows normal percentage changes."""
    prices = np.array([100.0, 110.0, 105.0], dtype=np.float64)  # Max jump is 10%
    result = _kernel_detect_spikes(prices, max_spike_pct=0.15)
    assert result is True


def test_kernel_detect_spikes_returns_false_for_extreme_jump_array() -> None:
    """Tests that the Numba spike kernel catches percentage jumps above the threshold."""
    prices = np.array([100.0, 105.0, 160.0], dtype=np.float64)  # Jump from 105 to 160 is ~52.3%
    result = _kernel_detect_spikes(prices, max_spike_pct=0.50)
    assert result is False


def test_validate_market_integrity_returns_ok_for_clean_data(valid_market_dataframe: pl.DataFrame) -> None:
    """Tests that the main validation wrapper accepts healthy Polars DataFrames."""
    result = validate_market_integrity(
        valid_market_dataframe,
        price_column="close",
        symbol_column="symbol",
        max_flatline_ticks=3,
        max_spike_pct=0.50,
    )

    assert result.is_ok()
    assert result.unwrap().height == 5


def test_validate_market_integrity_returns_err_when_price_column_is_missing(
    valid_market_dataframe: pl.DataFrame,
) -> None:
    """Tests guard clause for missing price column."""
    bad_dataframe = valid_market_dataframe.rename({"close": "wrong_price_name"})

    result = validate_market_integrity(
        bad_dataframe,
        price_column="close",
        symbol_column="symbol",
    )

    assert result.is_err()
    assert "Missing required column: close" in str(result.unwrap_err())


def test_validate_market_integrity_returns_err_when_symbol_column_is_missing(
    valid_market_dataframe: pl.DataFrame,
) -> None:
    """Tests guard clause for missing symbol column."""
    bad_dataframe = valid_market_dataframe.rename({"symbol": "ticker"})

    result = validate_market_integrity(
        bad_dataframe,
        price_column="close",
        symbol_column="symbol",
    )

    assert result.is_err()
    assert "Missing required column: symbol" in str(result.unwrap_err())


def test_validate_market_integrity_returns_err_when_flatline_detected(flatline_market_dataframe: pl.DataFrame) -> None:
    """Tests that the pipeline correctly propagates flatline detection as an Err(ValueError)."""
    result = validate_market_integrity(
        flatline_market_dataframe,
        max_flatline_ticks=3,  # Data has 4 consecutive identical prices
    )

    assert result.is_err()
    error = result.unwrap_err()
    assert isinstance(error, ValueError)
    assert "Flatline detected" in str(error)
    assert "ETH-USDT" in str(error)


def test_validate_market_integrity_returns_err_when_spike_detected(spike_market_dataframe: pl.DataFrame) -> None:
    """Tests that the pipeline correctly propagates extreme spike detection as an Err(ValueError)."""
    result = validate_market_integrity(
        spike_market_dataframe,
        max_spike_pct=0.50,  # Data has a 60% jump
    )

    assert result.is_err()
    error = result.unwrap_err()
    assert isinstance(error, ValueError)
    assert "Extreme price spike detected" in str(error)
    assert "SOL-USDT" in str(error)
