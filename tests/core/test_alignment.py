"""
tests/core/test_alignment.py

Unit tests for the UniverseAligner class.
Tests cover factory, alignment correctness, forward-fill physics, and error handling.
"""

import polars as pl

from dacos.core.alignment import UniverseAligner, create_universe_aligner
from dacos.utils import is_err, is_ok

# ============================================================================
# Test 1: Factory and Initialization
# ============================================================================

def test_factory_initialization() -> None:
    """Ensure create_universe_aligner returns a properly configured instance."""
    frequency = "5m"
    aligner = create_universe_aligner(frequency)

    assert isinstance(aligner, UniverseAligner)
    assert aligner.frequency == frequency


# ============================================================================
# Test 2: Asynchronous Alignment (grid generation)
# ============================================================================

def test_asynchronous_alignment() -> None:
    """
    Two symbols with irregular timestamps should be expanded to a regular grid.
    - Minute 0: both A and B present.
    - Minute 1: only A present.
    - Minute 2: only B present.
    After alignment with 1m frequency, we should have 3 minutes × 2 symbols = 6 rows.
    """
    # Create data with timestamps in milliseconds (1 minute = 60,000 ms)
    data = pl.DataFrame({
        "timestamp": [0, 0, 60_000, 120_000],
        "symbol": ["A", "B", "A", "B"],
        "log_price": [1.0, 2.0, 1.5, 2.5],
        "log_volume": [10.0, 20.0, 15.0, 25.0],
    })

    aligner = UniverseAligner(frequency="1m")
    result = aligner.align(data)

    assert is_ok(result)
    df = result.ok()
    assert isinstance(df, pl.DataFrame)

    # Should have 6 rows: minutes 0,1,2 for both A and B
    assert len(df) == 6

    # Check that all expected timestamps are present for each symbol
    timestamps = df.group_by("symbol").agg(pl.col("timestamp").sort())
    timestamps_dict = {row["symbol"]: row["timestamp"] for row in timestamps.to_dicts()}

    expected_timestamps = pl.datetime_range(
        start=pl.datetime(1970, 1, 1),  # zero in milliseconds corresponds to 1970-01-01
        end=pl.datetime(1970, 1, 1) + pl.duration(minutes=2),
        interval="1m",
        eager=True,
    ).cast(pl.Datetime("ms")).to_list()

    for sym in ["A", "B"]:
        assert timestamps_dict[sym] == expected_timestamps


# ============================================================================
# Test 3: Forward-Fill Physics (no look-ahead bias)
# ============================================================================

def test_forward_fill_physics() -> None:
    """
    After upsample, nulls should be filled with the last observed value per symbol.
    - Minute 0: A=1.0, B=2.0
    - Minute 1: only A=1.5 (B should carry forward 2.0)
    - Minute 2: only B=2.5 (A should carry forward 1.5)
    """
    data = pl.DataFrame({
        "timestamp": [0, 0, 60_000, 120_000],
        "symbol": ["A", "B", "A", "B"],
        "log_price": [1.0, 2.0, 1.5, 2.5],
        "log_volume": [10.0, 20.0, 15.0, 25.0],
    })

    aligner = UniverseAligner(frequency="1m")
    result = aligner.align(data)
    df = result.ok()

    # Check B's price at minute 1 (60,000 ms) – should be 2.0 (carried from minute 0)
    b_min1 = df.filter((pl.col("symbol") == "B") & (pl.col("timestamp") == 60_000))
    assert len(b_min1) == 1
    assert b_min1["log_price"][0] == 2.0

    # Check A's price at minute 2 (120,000 ms) – should be 1.5 (carried from minute 1)
    a_min2 = df.filter((pl.col("symbol") == "A") & (pl.col("timestamp") == 120_000))
    assert len(a_min2) == 1
    assert a_min2["log_price"][0] == 1.5

    # Also ensure that volumes follow the same forward-fill pattern
    b_vol_min1 = df.filter((pl.col("symbol") == "B") & (pl.col("timestamp") == 60_000))["log_volume"][0]
    assert b_vol_min1 == 20.0

    a_vol_min2 = df.filter((pl.col("symbol") == "A") & (pl.col("timestamp") == 120_000))["log_volume"][0]
    assert a_vol_min2 == 15.0


# ============================================================================
# Test 4: Error Handling – Missing timestamp column
# ============================================================================

def test_missing_timestamp_column() -> None:
    """If the input DataFrame lacks a 'timestamp' column, align should return Err."""
    data = pl.DataFrame({
        "time": [0, 60_000, 120_000],  # wrong column name
        "symbol": ["A", "B", "A"],
        "log_price": [1.0, 2.0, 1.5],
        "log_volume": [10.0, 20.0, 15.0],
    })

    aligner = UniverseAligner(frequency="1m")
    result = aligner.align(data)

    assert is_err(result)
    # Optionally, check that the error message mentions the missing column
    assert "timestamp" in str(result.err()).lower()


# ============================================================================
# Additional test: Empty DataFrame
# ============================================================================

def test_empty_dataframe() -> None:
    """Aligning an empty DataFrame should return an empty DataFrame (still Ok)."""
    data = pl.DataFrame({
        "timestamp": [],
        "symbol": [],
        "log_price": [],
        "log_volume": [],
    }, schema={
        "timestamp": pl.Int64,
        "symbol": pl.Utf8,
        "log_price": pl.Float64,
        "log_volume": pl.Float64,
    })

    aligner = UniverseAligner(frequency="1m")
    result = aligner.align(data)

    assert is_ok(result)
    df = result.ok()
    assert len(df) == 0
