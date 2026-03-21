from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from dacos.builder.etl import (
    execute_etl_pipeline,
    extract_raw_parquet,
    transform_to_silver_format,
    validate_source_directory,
    write_silver_parquet,
)
from dacos.contracts import RAW_SCHEMA, SILVER_SCHEMA


@pytest.fixture
def dummy_raw_dataframe() -> pl.DataFrame:
    """
    Creates a dummy DataFrame conforming to RAW_SCHEMA.

    Returns:
        pl.DataFrame: A Polars DataFrame with raw market data.
    """
    return pl.DataFrame(
        {
            "timestamp": [1000, 2000, 3000, 4000],
            "open": [100.0, 101.0, 102.0, 103.0],
            "high": [105.0, 106.0, 107.0, 108.0],
            "low": [95.0, 96.0, 97.0, 98.0],
            "close": [101.0, None, 103.0, 104.0],
            "volume": [10.0, 20.0, 0.0, 30.0],
            "symbol": ["BTC-USDT", "BTC-USDT", "ETH-USDT", "ETH-USDT"],
            "interval": ["1m", "1m", "1m", "1m"],
            "year": [2024, 2024, 2024, 2024],
            "month": ["01", "01", "01", "01"],
        },
        schema=RAW_SCHEMA,
    )


@pytest.fixture
def raw_directory_with_data(tmp_path: Path, dummy_raw_dataframe: pl.DataFrame) -> Path:
    """
    Creates a temporary directory with Hive-partitioned Parquet files.

    Args:
        tmp_path: Pytest built-in temporary path fixture.
        dummy_raw_dataframe: Fixture providing raw DataFrame.

    Returns:
        Path: The root directory containing the partitioned Parquet files.
    """
    partition_path = tmp_path / "raw" / "year=2024" / "month=01"
    partition_path.mkdir(parents=True, exist_ok=True)

    file_path = partition_path / "data.parquet"
    dummy_raw_dataframe.write_parquet(file_path)

    return tmp_path / "raw"


@pytest.fixture
def empty_silver_directory(tmp_path: Path) -> Path:
    """
    Creates an empty temporary directory for Silver Lake output.

    Args:
        tmp_path: Pytest built-in temporary path fixture.

    Returns:
        Path: The directory path for silver data.
    """
    silver_path = tmp_path / "silver"
    silver_path.mkdir(parents=True, exist_ok=True)
    return silver_path


def test_validate_source_directory_returns_ok_when_directory_exists(tmp_path: Path) -> None:
    """
    Tests that validate_source_directory returns an Ok result with the resolved path.
    """
    result = validate_source_directory(tmp_path)

    assert result.is_ok()
    assert result.unwrap() == tmp_path.resolve()


def test_validate_source_directory_returns_err_when_directory_is_missing(tmp_path: Path) -> None:
    """
    Tests that validate_source_directory returns an Err when path does not exist.
    """
    missing_path = tmp_path / "nonexistent"

    result = validate_source_directory(missing_path)

    assert result.is_err()
    assert isinstance(result.unwrap_err(), FileNotFoundError)


def test_validate_source_directory_returns_err_when_path_is_file(tmp_path: Path) -> None:
    """
    Tests that validate_source_directory returns an Err when path points to a file.
    """
    file_path = tmp_path / "file.txt"
    file_path.touch()

    result = validate_source_directory(file_path)

    assert result.is_err()
    assert isinstance(result.unwrap_err(), FileNotFoundError)


def test_extract_raw_parquet_returns_ok_with_lazyframe(raw_directory_with_data: Path) -> None:
    """
    Tests that extract_raw_parquet successfully creates a LazyFrame from valid directory.
    """
    result = extract_raw_parquet(raw_directory_with_data)

    assert result.is_ok()
    assert isinstance(result.unwrap(), pl.LazyFrame)


def test_extract_raw_parquet_returns_err_on_invalid_read(tmp_path: Path) -> None:
    """
    Tests that extract_raw_parquet returns an Err when directory is empty or invalid.
    """
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    result = extract_raw_parquet(empty_dir)

    assert result.is_err()
    assert isinstance(result.unwrap_err(), Exception)


def test_transform_to_silver_format_applies_correct_filters_and_schema(dummy_raw_dataframe: pl.DataFrame) -> None:
    """
    Tests that transform_to_silver_format filters nulls, zeros, casts timestamps, and selects correct columns.
    """
    raw_lazy = dummy_raw_dataframe.lazy()

    result = transform_to_silver_format(raw_lazy)

    assert result.is_ok()

    transformed_df = result.unwrap().collect()

    assert transformed_df.height == 2
    assert transformed_df.columns == list(SILVER_SCHEMA.keys())
    assert transformed_df.schema["timestamp"] == pl.Datetime("ms")
    assert transformed_df.select(pl.col("volume").min()).item() > 0.0


def test_write_silver_parquet_creates_file_successfully(
    dummy_raw_dataframe: pl.DataFrame, empty_silver_directory: Path
) -> None:
    """
    Tests that write_silver_parquet materializes the LazyFrame to a Parquet file.
    """
    silver_lazy = transform_to_silver_format(dummy_raw_dataframe.lazy()).unwrap()

    result = write_silver_parquet(silver_lazy, empty_silver_directory)

    assert result.is_ok()

    output_path = result.unwrap()
    assert output_path.exists()
    assert output_path.is_file()
    assert output_path.name == "silver_master.parquet"

    written_df = pl.read_parquet(output_path)
    assert written_df.height == 2


def test_execute_etl_pipeline_succeeds_end_to_end(raw_directory_with_data: Path, empty_silver_directory: Path) -> None:
    """
    Tests the complete execution of the ETL pipeline producing a success message and valid file.
    """
    result = execute_etl_pipeline(raw_directory_with_data, empty_silver_directory)

    assert result.is_ok()
    assert "successfully written" in result.unwrap()

    output_file = empty_silver_directory / "silver_master.parquet"
    assert output_file.exists()


def test_execute_etl_pipeline_fails_at_validation_for_missing_source(
    tmp_path: Path, empty_silver_directory: Path
) -> None:
    """
    Tests that execute_etl_pipeline short-circuits and returns Err if raw path is missing.
    """
    missing_source = tmp_path / "missing_source"

    result = execute_etl_pipeline(missing_source, empty_silver_directory)

    assert result.is_err()
    assert isinstance(result.unwrap_err(), FileNotFoundError)
