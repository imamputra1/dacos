"""
tests/paradigms/test_engines.py

Unit tests for StatArbEngine.
Tests cover factory initialization, perfect cointegration, rolling window mechanics,
and error handling for missing symbols.
"""

import polars as pl
import pytest

from dacos import StatArbEngine, create_stat_arb_engine, is_err, is_ok

# ============================================================================
# Test 1: Factory and Initialization
# ============================================================================

def test_factory_initialization() -> None:
    """Ensure create_stat_arb_engine returns a properly configured instance."""
    window = 50
    engine = create_stat_arb_engine(zscore_window=window)

    assert isinstance(engine, StatArbEngine)
    assert engine.window == window


# ============================================================================
# Test 2: Perfect Cointegration (Y = 2X)
# ============================================================================

def test_perfect_cointegration() -> None:
    """With Y = 2X, the spread should be zero."""
    # Create long-format data
    timestamps = [1, 2, 3, 4, 5]
    data = pl.DataFrame({
        "timestamp": timestamps * 2,
        "symbol": ["X"] * 5 + ["Y"] * 5,
        "log_price": [1.0, 2.0, 3.0, 4.0, 5.0,  # X
                      2.0, 4.0, 6.0, 8.0, 10.0],  # Y
    })

    engine = StatArbEngine(zscore_window=3)
    result = engine.run_engine(data, y_symbol="Y", x_symbol="X")

    assert is_ok(result)
    df = result.ok()

    # Check that spread is near zero for all rows
    spread = df["spread"].to_numpy()
    assert spread == pytest.approx(0.0, abs=1e-10)


# ============================================================================
# Test 3: Rolling Window Mechanics (no look-ahead)
# ============================================================================

def test_rolling_window_mechanics() -> None:
    """With window=3, first two rows should have nulls for rolling stats."""
    # Create data where spread is known (e.g., random)
    timestamps = [1, 2, 3, 4, 5]
    data = pl.DataFrame({
        "timestamp": timestamps * 2,
        "symbol": ["X"] * 5 + ["Y"] * 5,
        "log_price": [1.0, 2.0, 3.0, 4.0, 5.0,  # X
                      2.1, 4.1, 6.1, 8.1, 10.1],  # Y (slight offset)
    })

    engine = StatArbEngine(zscore_window=3)
    result = engine.run_engine(data, y_symbol="Y", x_symbol="X")
    assert is_ok(result)
    df = result.ok()

    # First two rows: spread_mean, spread_std, z_score should be null
    assert df["spread_mean"][:2].null_count() == 2
    assert df["spread_std"][:2].null_count() == 2
    assert df["z_score"][:2].null_count() == 2

    # Rows from index 2 onward should have non-null values
    assert df["spread_mean"][2:].null_count() == 0
    assert df["spread_std"][2:].null_count() == 0
    assert df["z_score"][2:].null_count() == 0


# ============================================================================
# Test 4: Missing Symbol Trap (error handling)
# ============================================================================

def test_missing_symbol_trap() -> None:
    """Requesting a symbol not present in data should return Err."""
    data = pl.DataFrame({
        "timestamp": [1, 2, 3, 4, 5] * 2,
        "symbol": ["BTC"] * 5 + ["ETH"] * 5,
        "log_price": [10.0, 11.0, 12.0, 13.0, 14.0,  # BTC
                      20.0, 21.0, 22.0, 23.0, 24.0],  # ETH
    })

    engine = StatArbEngine(zscore_window=3)
    result = engine.run_engine(data, y_symbol="SOL", x_symbol="BTC")

    assert is_err(result)
    # Optionally check error message contains the missing symbol
    error_msg = str(result.err()).lower()
    assert "sol" in error_msg


# ============================================================================
# Additional test: Empty input
# ============================================================================

def test_empty_dataframe() -> None:
    """Empty input should return Err."""
    data = pl.DataFrame(schema={"timestamp": pl.Int64, "symbol": pl.Utf8, "log_price": pl.Float64})
    engine = StatArbEngine(zscore_window=3)
    result = engine.run_engine(data, y_symbol="Y", x_symbol="X")
    assert is_err(result)
