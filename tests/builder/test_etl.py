"""
tests/builder/test_etl.py

Unit tests for the ETL pipeline builder (SkinnyLakeBuilder).
Uses pytest fixtures and temporary directories to mock file I/O.
"""

import math
from pathlib import Path

import polars as pl
import pytest

from dacos.builder import SkinnyLakeBuilder, create_skinny_builder
from dacos.utils import is_err, is_ok


@pytest.fixture
def sample_raw_data() -> pl.DataFrame:
    """Create a sample DataFrame with raw columns including extra ones."""
    return pl.DataFrame({
        "timestamp": [1000, 2000, 3000, 4000],
        "symbol": ["BTCUSDT", "BTCUSDT", "ETHUSDT", "ETHUSDT"],
        "open": [90.0, 95.0, 10.0, 11.0],
        "high": [95.0, 98.0, 11.0, 12.0],
        "low": [89.0, 94.0, 9.5, 10.5],
        "close": [94.0, 97.0, 10.5, 11.5],
        "volume": [1000.0, 1100.0, 5000.0, 5200.0],
        "year": [2024, 2024, 2024, 2024],
        "month": ["01", "01", "01", "01"],
    })


# ============================================================================
# Test 1: Factory and Initialization
# ============================================================================

def test_factory_initialization() -> None:
    """Ensure the factory creates a builder instance with correct paths."""
    raw = "some/raw/path"
    silver = "some/silver/path"
    builder = create_skinny_builder(raw, silver)

    assert isinstance(builder, SkinnyLakeBuilder)
    assert builder.raw_path == Path(raw)
    assert builder.silver_path == Path(silver)


# ============================================================================
# Test 2: Extraction Schema (only 4 columns)
# ============================================================================

def test_extraction_schema(tmp_path: Path, sample_raw_data: pl.DataFrame) -> None:
    """_extract() must return a LazyFrame with only timestamp, symbol, close, volume."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    file_path = raw_dir / "data.parquet"
    sample_raw_data.write_parquet(file_path)

    builder = SkinnyLakeBuilder(raw_dir, tmp_path / "silver")
    lazy = builder._extract()

    assert isinstance(lazy, pl.LazyFrame)
    schema_names = lazy.collect_schema().names()
    assert set(schema_names) == {"timestamp", "symbol", "close", "volume"}


# ============================================================================
# Test 3: Anomaly Filtering (drop null close, zero/negative volume)
# ============================================================================

def test_anomaly_filtering(tmp_path: Path) -> None:
    """_transform() must drop rows with null close or volume <= 0."""
    data = pl.DataFrame({
        "timestamp": [1, 2, 3, 4],
        "symbol": ["A", "B", "C", "D"],
        "close": [100.0, None, 200.0, 300.0],
        "volume": [1000.0, 1000.0, 0.0, -5.0],
    })
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    file_path = raw_dir / "data.parquet"
    data.write_parquet(file_path)

    builder = SkinnyLakeBuilder(raw_dir, tmp_path / "silver")
    raw = builder._extract()
    transformed = builder._transform(raw)
    result = transformed.collect()

    assert len(result) == 1
    # The only remaining row should be the first one (timestamp=1)
    assert result["timestamp"][0] == 1
    assert result["symbol"][0] == "A"


# ============================================================================
# Test 4: Log Transformations and Sorting
# ============================================================================

def test_log_and_sorting(tmp_path: Path) -> None:
    """_transform() must compute log_price, log_volume, sort, and drop original columns."""
    # Deliberately unsorted
    data = pl.DataFrame({
        "timestamp": [3000, 1000, 2000],
        "symbol": ["B", "A", "A"],
        "close": [150.0, 100.0, 100.0],
        "volume": [2000.0, 1000.0, 1000.0],
    })
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    file_path = raw_dir / "data.parquet"
    data.write_parquet(file_path)

    builder = SkinnyLakeBuilder(raw_dir, tmp_path / "silver")
    raw = builder._extract()
    transformed = builder._transform(raw)
    result = transformed.collect()

    # Check schema: only timestamp, symbol, log_price, log_volume
    assert set(result.columns) == {"timestamp", "symbol", "log_price", "log_volume"}

    # Check log calculations
    expected_log_price = math.log(100.0)  # for A rows
    expected_log_volume = math.log(1000.0)
    # For symbol A (first two rows)
    assert result.filter(pl.col("symbol") == "A")["log_price"].to_list() == pytest.approx([expected_log_price] * 2)
    assert result.filter(pl.col("symbol") == "A")["log_volume"].to_list() == pytest.approx([expected_log_volume] * 2)
    # For symbol B
    assert result.filter(pl.col("symbol") == "B")["log_price"][0] == pytest.approx(math.log(150.0))
    assert result.filter(pl.col("symbol") == "B")["log_volume"][0] == pytest.approx(math.log(2000.0))

    # Check sorting: symbol A then B; within A, timestamps ascending
    symbols = result["symbol"].to_list()
    timestamps = result["timestamp"].to_list()
    assert symbols == ["A", "A", "B"]
    assert timestamps == [1000, 2000, 3000]


# ============================================================================
# Test 5: Monadic Error Handling (invalid paths)
# ============================================================================

def test_monad_error() -> None:
    """Pipeline must return Err when raw path does not exist."""
    builder = SkinnyLakeBuilder("/nonexistent/raw/path", "/nonexistent/silver/path")
    result = builder.execute_pipeline()

    assert is_err(result)
    # Also check that it didn't crash (i.e., we got an Err object)
    # Optionally check error type
    assert isinstance(result.err(), Exception)


# ============================================================================
# Test 6: End-to-End Success (with real temp files)
# ============================================================================

def test_end_to_end_success(tmp_path: Path, sample_raw_data: pl.DataFrame) -> None:
    """Pipeline should successfully write a Parquet file and return Ok."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    raw_file = raw_dir / "data.parquet"
    sample_raw_data.write_parquet(raw_file)

    silver_dir = tmp_path / "silver"
    builder = SkinnyLakeBuilder(raw_dir, silver_dir)

    result = builder.execute_pipeline()

    assert is_ok(result)
    expected_file = silver_dir / "skinny.parquet"
    assert expected_file.exists()

    # Optionally, verify the content of the output file
    output_df = pl.read_parquet(expected_file)
    assert set(output_df.columns) == {"timestamp", "symbol", "log_price", "log_volume"}
    # Ensure filtering and logs were applied correctly
    # The sample data has no nulls or zero volumes, so all rows should be present
    assert len(output_df) == len(sample_raw_data)
