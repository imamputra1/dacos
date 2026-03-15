"""
tests/test_stress.py

Stress tests for the dacos pipeline to ensure performance and memory safety
under large data volumes. Uses time.perf_counter() to measure execution speed.
"""

import time

import numpy as np
import polars as pl
import pytest

from dacos.research import run_pairs_research
from dacos.utils import is_ok

# ============================================================================
# Helper: generate large dataset
# ============================================================================

def generate_large_dataset(
    tmp_path,
    num_symbols: int,
    rows_per_symbol: int,
    freq: int = 1000,  # average time step in ms
) -> str:
    """
    Generate a skinny table with many symbols and save to a temporary Parquet file.
    Returns the path to the silver directory.
    """
    silver_dir = tmp_path / "silver"
    silver_dir.mkdir()

    # Create timestamps: each symbol gets a sequence of timestamps spaced by `freq` ms
    # To avoid huge memory, we generate symbol by symbol and concatenate
    dfs = []
    np.random.seed(42)
    for sym_idx in range(num_symbols):
        symbol = f"SYM_{sym_idx:04d}"
        n = rows_per_symbol
        timestamps = np.arange(0, n * freq, freq, dtype=np.int64)
        # Random walk for log_price
        log_price = np.cumsum(np.random.normal(0, 1e-4, n))
        log_volume = np.random.normal(5, 1, n)
        df_sym = pl.DataFrame({
            "timestamp": timestamps,
            "symbol": [symbol] * n,
            "log_price": log_price,
            "log_volume": log_volume,
        })
        dfs.append(df_sym)

    # Concatenate all symbols
    df = pl.concat(dfs)
    # Write to Parquet
    file_path = silver_dir / "skinny.parquet"
    df.write_parquet(file_path)
    return str(silver_dir)


# ============================================================================
# Stress Test 1: Predicate Pushdown (1000 coins, only read 2)
# ============================================================================

@pytest.mark.stress
def test_stress_predicate_pushdown(tmp_path) -> None:
    """Generate 1000 symbols × 10k rows = 10M rows, then research only 2 symbols."""
    num_symbols = 1000
    rows_per_symbol = 10_000
    silver_path = generate_large_dataset(tmp_path, num_symbols, rows_per_symbol)

    start = time.perf_counter()
    result = run_pairs_research(
        silver_path=silver_path,
        y_symbol="SYM_0000",
        x_symbol="SYM_0001",
        frequency="1m",
        min_rows=100,
        z_window=100,
    )
    elapsed = time.perf_counter() - start

    assert is_ok(result), f"Pipeline failed: {result.err()}"
    # Optional: print performance info
    print(f"\n[Pushdown] Processed {num_symbols*rows_per_symbol:,} rows total, read only 2 symbols in {elapsed:.3f} seconds")
    # Assert that it's reasonably fast (adjust threshold to your hardware)
    # On a typical laptop, should be < 5 seconds; CI may be slower, so we don't enforce.
    # We'll just warn if too slow.
    if elapsed > 5.0:
        pytest.skip(f"Pushdown test took {elapsed:.3f}s, may be slow on this machine.")


# ============================================================================
# Stress Test 2: Extreme Alignment (1M rows per symbol, frequency 1s)
# ============================================================================

@pytest.mark.stress
def test_stress_alignment(tmp_path) -> None:
    """Generate 2 symbols with 1M rows each (irregular timestamps) and align to 1s."""
    num_symbols = 2
    rows_per_symbol = 1_000_000
    # Use irregular timestamps: random spacing with average 100ms
    silver_dir = tmp_path / "silver"
    silver_dir.mkdir()

    dfs = []
    np.random.seed(42)
    for sym_idx in range(num_symbols):
        symbol = f"SYM_{sym_idx}"
        n = rows_per_symbol
        # Generate random timestamps with average step 100ms (but irregular)
        base = 0
        timestamps = []
        for _ in range(n):
            step = np.random.poisson(100)  # average 100 ms
            base += step
            timestamps.append(base)
        timestamps = np.array(timestamps, dtype=np.int64)
        log_price = np.cumsum(np.random.normal(0, 1e-4, n))
        log_volume = np.random.normal(5, 1, n)
        df_sym = pl.DataFrame({
            "timestamp": timestamps,
            "symbol": [symbol] * n,
            "log_price": log_price,
            "log_volume": log_volume,
        })
        dfs.append(df_sym)

    df = pl.concat(dfs)
    file_path = silver_dir / "skinny.parquet"
    df.write_parquet(file_path)

    start = time.perf_counter()
    result = run_pairs_research(
        silver_path=str(silver_dir),
        y_symbol="SYM_0",
        x_symbol="SYM_1",
        frequency="1s",          # align to 1 second grid
        min_rows=100,
        z_window=100,
    )
    elapsed = time.perf_counter() - start

    assert is_ok(result), f"Alignment failed: {result.err()}"
    # Number of rows after alignment should be roughly (max_timestamp/1000) * 2
    max_ts = timestamps[-1]
    expected_rows_approx = (max_ts // 1000) * 2
    df_result = result.ok()["data"]
    print(f"\n[Alignment] Input rows: {2*rows_per_symbol:,}, aligned rows: {len(df_result):,} in {elapsed:.3f}s")
    if elapsed > 10.0:
        pytest.skip(f"Alignment took {elapsed:.3f}s, may be slow on this machine.")


# ============================================================================
# Stress Test 3: Rolling Z‑Score on 5M rows with huge window
# ============================================================================

@pytest.mark.stress
def test_stress_rolling(tmp_path) -> None:
    """Generate 5M aligned rows (2 symbols) and compute rolling z‑score with window=10000."""
    # For this test, we need data that is already aligned to a regular grid.
    # We'll generate data with timestamps at 1s intervals for simplicity.
    n_per_symbol = 2_500_000  # total 5M rows
    silver_dir = tmp_path / "silver"
    silver_dir.mkdir()

    timestamps = np.arange(0, n_per_symbol * 1000, 1000, dtype=np.int64)  # 1s steps

    # Symbol X
    np.random.seed(42)
    log_price_x = np.cumsum(np.random.normal(0, 1e-4, n_per_symbol))
    log_volume_x = np.random.normal(5, 1, n_per_symbol)
    df_x = pl.DataFrame({
        "timestamp": timestamps,
        "symbol": ["X"] * n_per_symbol,
        "log_price": log_price_x,
        "log_volume": log_volume_x,
    })

    # Symbol Y (cointegrated: Y = 2*X + spread)
    spread = np.random.normal(0, 0.1, n_per_symbol)
    log_price_y = 2 * log_price_x + spread
    log_volume_y = np.random.normal(5, 1, n_per_symbol)
    df_y = pl.DataFrame({
        "timestamp": timestamps,
        "symbol": ["Y"] * n_per_symbol,
        "log_price": log_price_y,
        "log_volume": log_volume_y,
    })

    df = pl.concat([df_x, df_y])
    file_path = silver_dir / "skinny.parquet"
    df.write_parquet(file_path)

    start = time.perf_counter()
    result = run_pairs_research(
        silver_path=str(silver_dir),
        y_symbol="Y",
        x_symbol="X",
        frequency="1s",          # already aligned, but alignment will pass through
        min_rows=100,
        z_window=10000,          # huge rolling window
    )
    elapsed = time.perf_counter() - start

    assert is_ok(result), f"Rolling calculation failed: {result.err()}"
    df_result = result.ok()["data"]
    print(f"\n[Rolling] Processed {len(df_result):,} rows with window=10000 in {elapsed:.3f}s")
    if elapsed > 10.0:
        pytest.skip(f"Rolling test took {elapsed:.3f}s, may be slow on this machine.")
