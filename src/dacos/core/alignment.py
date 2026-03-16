from __future__ import annotations

import numpy as np
from numba import njit

from dacos.utils import Err, Ok, Result


@njit(cache=True, fastmath=True)
def _kernel_align_and_forward_fill_strict(
    master_timestamps: np.ndarray,
    target_timestamps: np.ndarray,
    target_prices: np.ndarray,
    max_ffill_tolerance_ms: int,
) -> np.ndarray:
    """
    Numba-accelerated Two-Pointer kernel to align target prices to a master time grid.
    
    Args:
        master_timestamps: 1D Numpy array of int64 representing the master time grid.
        target_timestamps: 1D Numpy array of int64 representing the target asset's time grid.
        target_prices: 1D Numpy array of float64 representing the target asset's prices.
        max_ffill_tolerance_ms: Maximum allowed milliseconds to forward-fill stale data.

    Returns:
        np.ndarray: 1D Numpy array of float64 containing aligned prices (or NaN if stale/missing).
    """
    master_length = master_timestamps.shape[0]
    target_length = target_timestamps.shape[0]

    aligned_prices = np.empty(master_length, dtype=np.float64)
    aligned_prices[:] = np.nan

    target_idx = 0

    for master_idx in range(master_length):
        current_master_time = master_timestamps[master_idx]

        # Advance the target pointer to the most recent timestamp <= current_master_time
        while target_idx < target_length - 1 and target_timestamps[target_idx + 1] <= current_master_time:
            target_idx += 1

        # Extract data if the target pointer is valid and represents a time <= master
        if target_idx < target_length and target_timestamps[target_idx] <= current_master_time:
            time_difference = current_master_time - target_timestamps[target_idx]

            # Forward Fill Cutoff: Guard against stale data
            if time_difference <= max_ffill_tolerance_ms:
                aligned_prices[master_idx] = target_prices[target_idx]

    return aligned_prices


def synchronize_asset_to_master_grid_strict(
    master_time_grid: np.ndarray,
    asset_timestamps: np.ndarray,
    asset_prices: np.ndarray,
    max_stale_data_ms: int = 300_000,
) -> Result[np.ndarray, ValueError]:
    """
    Synchronizes an asset's price vector to a master time grid using a safe Monadic wrapper.
    
    Args:
        master_time_grid: 1D Numpy array of int64 timestamps (the anchor timeline).
        asset_timestamps: 1D Numpy array of int64 timestamps (the target asset).
        asset_prices: 1D Numpy array of float64 prices corresponding to asset_timestamps.
        max_stale_data_ms: Maximum milliseconds a price is valid before it's considered dead (default: 5 mins).

    Returns:
        Ok(np.ndarray) containing strictly aligned float64 prices, or Err(ValueError) on validation/kernel panic.
    """
    # Guard Clauses: Dimensionality and Sanity Checks
    if master_time_grid.size == 0 or asset_timestamps.size == 0:
        return Err(ValueError("Alignment Failed: One or both timestamp arrays are empty."))

    if asset_timestamps.shape[0] != asset_prices.shape[0]:
        return Err(
            ValueError(
                f"Alignment Failed: Timestamp length ({asset_timestamps.shape[0]}) "
                f"does not match prices length ({asset_prices.shape[0]})."
            )
        )

    if max_stale_data_ms < 0:
        return Err(ValueError(f"Alignment Failed: Tolerance cannot be negative. Got {max_stale_data_ms}."))

    # O(1) Endpoint Monotonicity Check (Polars ETL should guarantee strict sorting, this is a fast safety net)
    if asset_timestamps.shape[0] > 1 and asset_timestamps[0] > asset_timestamps[-1]:
         return Err(ValueError("Alignment Failed: Asset timestamps are not chronologically sorted (endpoints inverted)."))

    try:
        # Strict Casting for C-Level Numba compatibility
        safe_master_grid = master_time_grid.astype(np.int64)
        safe_target_grid = asset_timestamps.astype(np.int64)
        safe_target_prices = asset_prices.astype(np.float64)

        aligned_price_vector = _kernel_align_and_forward_fill_strict(
            master_timestamps=safe_master_grid,
            target_timestamps=safe_target_grid,
            target_prices=safe_target_prices,
            max_ffill_tolerance_ms=int(max_stale_data_ms),
        )

        # Post-Processing Guard: Complete overlap failure
        if np.isnan(aligned_price_vector).all():
            return Err(
                ValueError(
                    "Alignment Failed: 100% of aligned data resulted in NaN. "
                    "Time grids likely do not overlap or all data exceeds stale tolerance."
                )
            )

        return Ok(aligned_price_vector)

    except Exception as kernel_exception:
        return Err(ValueError(f"Alignment Kernel Panic: {kernel_exception}"))


__all__ = [
    "synchronize_asset_to_master_grid_strict",
]
