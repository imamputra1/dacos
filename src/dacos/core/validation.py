from __future__ import annotations

import numpy as np
import polars as pl
from numba import njit

from dacos.protocols import DataFrame
from dacos.utils import Err, Ok, Result


@njit(cache=True, fastmath=True)
def _kernel_detect_flatline(prices: np.ndarray, max_consecutive_flat: int) -> bool:
    """
    Args:
        prices: 1D Numpy array of float64 price data.
        max_consecutive_flat: Maximum allowed identical consecutive prices.

    Returns:
        bool: True if data is clean, False if a flatline violation is detected.
    """
    total_rows = prices.shape[0]
    if total_rows < 2:
        return True

    # BEDAH LOGIKA: Streak counter dimulai dari 1.
    # Harga pertama yang muncul sudah dihitung sebagai "1".
    flatline_counter = 1
    for i in range(1, total_rows):
        if prices[i] == prices[i - 1]:
            flatline_counter += 1
            if flatline_counter >= max_consecutive_flat:
                return False
        else:
            flatline_counter = 1

    return True


@njit(cache=True, fastmath=True)
def _kernel_detect_spikes(prices: np.ndarray, max_spike_pct: float) -> bool:
    """
    Args:
        prices: 1D Numpy array of float64 price data.
        max_spike_pct: Maximum allowed percentage change between consecutive ticks.

    Returns:
        bool: True if data is clean, False if an extreme spike is detected.
    """
    total_rows = prices.shape[0]
    if total_rows < 2:
        return True

    for i in range(1, total_rows):
        previous_price = prices[i - 1]
        if previous_price == 0.0:
            continue

        current_price = prices[i]
        percentage_change = abs((current_price - previous_price) / previous_price)

        if percentage_change > max_spike_pct:
            return False

    return True


def validate_market_integrity(
    data: DataFrame,
    price_column: str = "close",
    symbol_column: str = "symbol",
    max_flatline_ticks: int = 10,
    max_spike_pct: float = 0.50,
) -> Result[DataFrame, ValueError]:
    """
    Args:
        data: Polars DataFrame containing market data.
        price_column: Name of the column containing price data.
        symbol_column: Name of the column containing asset symbols.
        max_flatline_ticks: Limit for stagnant price ticks before declaring data stale.
        max_spike_pct: Limit for extreme price jumps (e.g., 0.50 = 50% jump).

    Returns:
        Ok(DataFrame) if valid, Err(ValueError) if corrupted data is found.
    """
    if price_column not in data.columns:
        return Err(ValueError(f"Missing required column: {price_column}"))

    if symbol_column not in data.columns:
        return Err(ValueError(f"Missing required column: {symbol_column}"))

    partitions = data.partition_by(symbol_column, as_dict=True)

    for symbol_tuple, partition in partitions.items():
        symbol_name = str(symbol_tuple[0]) if isinstance(symbol_tuple, tuple) else str(symbol_tuple)

        prices_series = partition.get_column(price_column)

        # BEDAH API: zero_copy_only diganti menjadi allow_copy sesuai Polars modern
        prices_array = prices_series.cast(pl.Float64).to_numpy(allow_copy=True)

        if not _kernel_detect_flatline(prices_array, max_flatline_ticks):
            return Err(
                ValueError(
                    f"Validation Failed [{symbol_name}]: Flatline detected exceeding {max_flatline_ticks} ticks."
                )
            )

        if not _kernel_detect_spikes(prices_array, max_spike_pct):
            return Err(
                ValueError(
                    f"Validation Failed [{symbol_name}]: Extreme price spike detected exceeding {max_spike_pct * 100}%."
                )
            )

    return Ok(data)


__all__ = [
    "validate_market_integrity",
]
