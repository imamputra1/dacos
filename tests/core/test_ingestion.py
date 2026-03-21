from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
from dacos.contracts import SILVER_SCHEMA
from dacos.core.ingestion import ingest_silver_data, validate_silver_schema


@pytest.fixture
def valid_silver_dataframe() -> pl.DataFrame:
    """Creates a dummy DataFrame conforming exactly to SILVER_SCHEMA."""
    return pl.DataFrame(
        {
            "timestamp": [1704067200000, 1704067260000, 1704067320000],  # ms timestamps
            "symbol": ["BTC-USDT", "ETH-USDT", "BTC-USDT"],
            "open": [42000.0, 2200.0, 42050.0],
            "high": [42100.0, 2210.0, 42150.0],
            "low": [41900.0, 2190.0, 42000.0],
            "close": [42050.0, 2205.0, 42100.0],
            "volume": [1.5, 10.0, 2.0],
        },
        schema=SILVER_SCHEMA,
    )


@pytest.fixture
def invalid_silver_dataframe_wrong_dtype() -> pl.DataFrame:
    """Creates a DataFrame with a wrong data type (volume as Int64 instead of Float64)."""
    schema = SILVER_SCHEMA.copy()
    schema["volume"] = pl.Int64
    return pl.DataFrame(
        {
            "timestamp": [1704067200000],
            "symbol": ["BTC-USDT"],
            "open": [42000.0],
            "high": [42100.0],
            "low": [41900.0],
            "close": [42050.0],
            "volume": [1],  # Int instead of Float
        },
        schema=schema,
    )


@pytest.fixture
def invalid_silver_dataframe_missing_column() -> pl.DataFrame:
    """Creates a DataFrame missing the 'volume' column."""
    schema = {k: v for k, v in SILVER_SCHEMA.items() if k != "volume"}
    return pl.DataFrame(
        {
            "timestamp": [1704067200000],
            "symbol": ["BTC-USDT"],
            "open": [42000.0],
            "high": [42100.0],
            "low": [41900.0],
            "close": [42050.0],
        },
        schema=schema,
    )


@pytest.fixture
def valid_silver_file(tmp_path: Path, valid_silver_dataframe: pl.DataFrame) -> Path:
    """Writes valid silver dataframe to a temporary Parquet file."""
    file_path = tmp_path / "valid_silver.parquet"
    valid_silver_dataframe.write_parquet(file_path)
    return file_path


@pytest.fixture
def invalid_dtype_file(tmp_path: Path, invalid_silver_dataframe_wrong_dtype: pl.DataFrame) -> Path:
    """Writes invalid dtype dataframe to a temporary Parquet file."""
    file_path = tmp_path / "invalid_dtype.parquet"
    invalid_silver_dataframe_wrong_dtype.write_parquet(file_path)
    return file_path


@pytest.fixture
def missing_column_file(tmp_path: Path, invalid_silver_dataframe_missing_column: pl.DataFrame) -> Path:
    """Writes missing column dataframe to a temporary Parquet file."""
    file_path = tmp_path / "missing_column.parquet"
    invalid_silver_dataframe_missing_column.write_parquet(file_path)
    return file_path


def test_validate_silver_schema_returns_ok_for_valid_file(valid_silver_file: Path) -> None:
    """Tests schema validation passes for a file strictly following SILVER_SCHEMA."""
    result = validate_silver_schema(valid_silver_file)

    assert result.is_ok()
    assert result.unwrap() is True


def test_validate_silver_schema_returns_err_for_missing_column(missing_column_file: Path) -> None:
    """Tests schema validation fails and returns Err when a required column is missing."""
    result = validate_silver_schema(missing_column_file)

    assert result.is_err()
    error = result.unwrap_err()
    assert isinstance(error, TypeError)
    assert "Missing required column" in str(error)


def test_validate_silver_schema_returns_err_for_wrong_dtype(invalid_dtype_file: Path) -> None:
    """Tests schema validation fails and returns Err when a column has the wrong datatype."""
    result = validate_silver_schema(invalid_dtype_file)

    assert result.is_err()
    error = result.unwrap_err()
    assert isinstance(error, TypeError)
    assert "Schema violation for 'volume'" in str(error)


def test_ingest_silver_data_returns_err_when_file_not_found(tmp_path: Path) -> None:
    """Tests ingestion guard clause for non-existent file."""
    missing_file = tmp_path / "ghost.parquet"

    result = ingest_silver_data(missing_file, symbols=["BTC-USDT"])

    assert result.is_err()
    assert isinstance(result.unwrap_err(), FileNotFoundError)


def test_ingest_silver_data_blocks_on_schema_violation(invalid_dtype_file: Path) -> None:
    """Tests ingestion is completely blocked if the schema is violated."""
    result = ingest_silver_data(invalid_dtype_file, symbols=["BTC-USDT"])

    assert result.is_err()
    assert isinstance(result.unwrap_err(), TypeError)


def test_ingest_silver_data_applies_symbol_and_time_filters_correctly(valid_silver_file: Path) -> None:
    """Tests ingestion successfully filters by symbol and time range via predicate pushdown."""
    start_time = 1704067200000
    end_time = 1704067260000

    result = ingest_silver_data(
        valid_silver_file,
        symbols=["BTC-USDT"],
        start_time=start_time,
        end_time=end_time,
    )

    assert result.is_ok()

    lazy_frame = result.unwrap()
    collected_df = lazy_frame.collect()

    assert collected_df.height == 1
    assert collected_df.select(pl.col("symbol").first()).item() == "BTC-USDT"

    # BEDAH: Cast ke Int64 secara internal di Polars untuk menghindari jebakan timezone lokal (WIB) Python
    actual_timestamp_ms = collected_df.select(pl.col("timestamp").cast(pl.Int64).first()).item()
    assert actual_timestamp_ms == start_time
