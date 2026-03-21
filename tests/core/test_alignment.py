from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from dacos.core.alignment import (
    _kernel_align_and_forward_fill_strict,
    synchronize_asset_to_master_grid_strict,
)


@pytest.fixture
def master_grid() -> np.ndarray:
    """Master time grid: 5 ticks, 1 second apart (1000 ms)."""
    return np.array([1000, 2000, 3000, 4000, 5000], dtype=np.int64)


@pytest.fixture
def target_timestamps() -> np.ndarray:
    """
    Target timestamps with missing/delayed ticks.
    Matches at 1000. Misses 2000 (last was 1000). Updates at 2500. Misses 4000 & 5000.
    """
    return np.array([1000, 2500, 3500], dtype=np.int64)


@pytest.fixture
def target_prices() -> np.ndarray:
    """Corresponding prices for the target asset."""
    return np.array([100.0, 105.0, 102.0], dtype=np.float64)


def test_kernel_perfect_alignment() -> None:
    """Tests the kernel with 1:1 perfect time matching."""
    master = np.array([10, 20, 30], dtype=np.int64)
    target_time = np.array([10, 20, 30], dtype=np.int64)
    target_price = np.array([1.1, 1.2, 1.3], dtype=np.float64)

    result = _kernel_align_and_forward_fill_strict(master, target_time, target_price, 5000)

    assert_array_equal(result, np.array([1.1, 1.2, 1.3]))


def test_kernel_forward_fill_within_tolerance(
    master_grid: np.ndarray, target_timestamps: np.ndarray, target_prices: np.ndarray
) -> None:
    """Tests the kernel successfully forward-fills missing data if within tolerance."""
    # Tolerance 2000ms.
    # At master 2000, last target was 1000. Diff = 1000 <= 2000. Price = 100.0
    # At master 3000, last target was 2500. Diff = 500 <= 2000. Price = 105.0
    # At master 4000, last target was 3500. Diff = 500 <= 2000. Price = 102.0
    # At master 5000, last target was 3500. Diff = 1500 <= 2000. Price = 102.0
    result = _kernel_align_and_forward_fill_strict(
        master_grid, target_timestamps, target_prices, max_ffill_tolerance_ms=2000
    )

    expected = np.array([100.0, 100.0, 105.0, 102.0, 102.0])
    assert_array_equal(result, expected)


def test_kernel_forward_fill_exceeds_tolerance_generates_nan(
    master_grid: np.ndarray, target_timestamps: np.ndarray, target_prices: np.ndarray
) -> None:
    """Tests the kernel cuts off stale data and produces NaN when tolerance is exceeded."""
    # Tolerance 800ms.
    # At master 2000, last target was 1000. Diff = 1000 > 800. Price = NaN
    # At master 5000, last target was 3500. Diff = 1500 > 800. Price = NaN
    result = _kernel_align_and_forward_fill_strict(
        master_grid, target_timestamps, target_prices, max_ffill_tolerance_ms=800
    )

    # We must use np.isnan to check for NaNs because NaN != NaN in numpy
    assert result[0] == 100.0
    assert np.isnan(result[1])  # Exceeded tolerance
    assert result[2] == 105.0
    assert result[3] == 102.0
    assert np.isnan(result[4])  # Exceeded tolerance


def test_wrapper_returns_ok_on_success(
    master_grid: np.ndarray, target_timestamps: np.ndarray, target_prices: np.ndarray
) -> None:
    """Tests the safe wrapper returns an Ok Result on valid inputs."""
    result = synchronize_asset_to_master_grid_strict(
        master_grid, target_timestamps, target_prices, max_stale_data_ms=2000
    )

    assert result.is_ok()
    assert len(result.unwrap()) == len(master_grid)


def test_wrapper_returns_err_on_empty_arrays() -> None:
    """Tests guard clause for empty arrays."""
    master = np.array([], dtype=np.int64)
    target = np.array([], dtype=np.int64)
    prices = np.array([], dtype=np.float64)

    result = synchronize_asset_to_master_grid_strict(master, target, prices)

    assert result.is_err()
    assert "empty" in str(result.unwrap_err())


def test_wrapper_returns_err_on_dimension_mismatch(master_grid: np.ndarray) -> None:
    """Tests guard clause when target timestamps and prices have different lengths."""
    target_time = np.array([1000, 2000], dtype=np.int64)
    target_price = np.array([100.0], dtype=np.float64)  # Mismatched length

    result = synchronize_asset_to_master_grid_strict(master_grid, target_time, target_price)

    assert result.is_err()
    assert "does not match" in str(result.unwrap_err())


def test_wrapper_returns_err_on_negative_tolerance(
    master_grid: np.ndarray, target_timestamps: np.ndarray, target_prices: np.ndarray
) -> None:
    """Tests guard clause preventing negative time tolerance."""
    result = synchronize_asset_to_master_grid_strict(
        master_grid, target_timestamps, target_prices, max_stale_data_ms=-100
    )

    assert result.is_err()
    assert "cannot be negative" in str(result.unwrap_err())


def test_wrapper_returns_err_on_unsorted_timestamps(master_grid: np.ndarray, target_prices: np.ndarray) -> None:
    """Tests guard clause for non-chronological target timestamps."""
    unsorted_target_timestamps = np.array([3000, 1000, 2000], dtype=np.int64)

    result = synchronize_asset_to_master_grid_strict(master_grid, unsorted_target_timestamps, target_prices)

    assert result.is_err()
    assert "not chronologically sorted" in str(result.unwrap_err())


def test_wrapper_returns_err_on_complete_overlap_failure() -> None:
    """Tests post-processing guard clause when grids do not overlap at all (100% NaN)."""
    master = np.array([10000, 20000], dtype=np.int64)
    target_time = np.array([100, 200], dtype=np.int64)
    target_price = np.array([1.0, 2.0], dtype=np.float64)

    # Tolerance is 1000ms. Difference between master and target is ~9800ms. All will be NaN.
    result = synchronize_asset_to_master_grid_strict(master, target_time, target_price, max_stale_data_ms=1000)

    assert result.is_err()
    assert "100% of aligned data resulted in NaN" in str(result.unwrap_err())
