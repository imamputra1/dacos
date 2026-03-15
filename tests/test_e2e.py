"""
tests/test_e2e.py

End-to-end tests for the main `run_pairs_research` function.
Verifies the Railway Oriented Programming pipeline behavior under various scenarios.
"""

import numpy as np
import polars as pl
import pytest

from dacos.research import run_pairs_research
from dacos.utils import is_err, is_ok

# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def healthy_pairs_data(tmp_path) -> str:
    """
    Create a healthy pairs dataset with sufficient time range (6000 seconds)
    to yield ~100 rows per symbol after 1m upsample.
    Timestamps are in milliseconds.
    """
    n_seconds = 6000                     # 6000 detik = 100 menit
    timestamps = [i * 1000 for i in range(n_seconds)]   # dalam milidetik

    np.random.seed(42)
    x = np.cumsum(np.random.normal(0, 1, n_seconds))
    spread = np.zeros(n_seconds)
    spread[0] = np.random.normal()
    for i in range(1, n_seconds):
        spread[i] = 0.3 * spread[i-1] + np.random.normal(0, 0.5)
    y = 2 * x + spread

    log_volume_x = np.random.normal(5, 1, n_seconds)
    log_volume_y = np.random.normal(5, 1, n_seconds)

    data = pl.DataFrame({
        "timestamp": timestamps * 2,
        "symbol": ["X"] * n_seconds + ["Y"] * n_seconds,
        "log_price": list(x) + list(y),
        "log_volume": list(log_volume_x) + list(log_volume_y),
    })

    silver_dir = tmp_path / "silver"
    silver_dir.mkdir()
    file_path = silver_dir / "skinny.parquet"
    data.write_parquet(file_path)
    return str(silver_dir)


# ============================================================================
# E2E Test 1: Golden Path
# ============================================================================

def test_golden_path(healthy_pairs_data: str) -> None:
    """All parameters correct, should return Ok with data and metrics."""
    result = run_pairs_research(
        silver_path=healthy_pairs_data,
        y_symbol="Y",
        x_symbol="X",
        frequency="1m",
        min_rows=100,
        z_window=50,
    )

    assert is_ok(result)
    output = result.ok()
    assert "data" in output
    assert "metrics" in output

    df = output["data"]
    assert isinstance(df, pl.DataFrame)
    assert "z_score" in df.columns

    metrics = output["metrics"]
    assert metrics["hurst"] is not None
    assert metrics["adf_pvalue"] is not None
    assert metrics["halflife"] is not None
    assert isinstance(metrics["hurst"], float)
    assert isinstance(metrics["adf_pvalue"], float)
    assert isinstance(metrics["halflife"], float) or metrics["halflife"] is None


# ============================================================================
# E2E Test 2: Blind User (bad inputs)
# ============================================================================

def test_blind_user_bad_path(healthy_pairs_data: str) -> None:
    """Non-existent silver path should return Err at ingestion stage."""
    result = run_pairs_research(
        silver_path="/definitely/does/not/exist",
        y_symbol="Y",
        x_symbol="X",
        frequency="1m",
        min_rows=100,
        z_window=50,
    )
    assert is_err(result)
    assert "No such file" in str(result.err()) or "exist" in str(result.err())


def test_blind_user_missing_symbol(healthy_pairs_data: str) -> None:
    """
    Requesting a symbol not present in data should eventually fail at validation
    (ingestion returns empty DataFrame, alignment passes, validation rejects).
    """
    result = run_pairs_research(
        silver_path=healthy_pairs_data,
        y_symbol="Z",  # not in data
        x_symbol="X",
        frequency="1m",
        min_rows=100,
        z_window=50,
    )
    assert is_err(result)
    error_msg = str(result.err()).lower()
    assert "not found" in error_msg or "symbol" in error_msg


# ============================================================================
# E2E Test 3: Dead End (stopped at validation)
# ============================================================================

def test_dead_end_validation(healthy_pairs_data: str) -> None:
    """Data too short (min_rows=1000) should fail at validation stage."""
    result = run_pairs_research(
        silver_path=healthy_pairs_data,
        y_symbol="Y",
        x_symbol="X",
        frequency="1m",
        min_rows=1000,  # much larger than the 500 rows available
        z_window=50,
    )
    assert is_err(result)
    error_msg = str(result.err()).lower()
    assert "short" in error_msg or "rows" in error_msg
