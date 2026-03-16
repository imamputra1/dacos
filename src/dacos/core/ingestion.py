from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

from dacos.contracts import SILVER_SCHEMA
from dacos.protocols import PathLike, Symbol, Timestamp
from dacos.utils import Err, Ok, Result

if TYPE_CHECKING:
    from dacos.protocols import LazyFrame

logger = logging.getLogger(__name__)


def validate_silver_schema(file_path: Path) -> Result[bool, TypeError]:
    """
    Validates the schema of a Parquet file against the strictly defined SILVER_SCHEMA.
    Blocks execution if the file schema does not exactly match.

    Args:
        file_path: Resolved path to the Silver Lake parquet file.

    Returns:
        Ok(True) if schema matches exactly, Err(TypeError) with mismatch details otherwise.
    """
    try:
        actual_schema = pl.read_parquet_schema(file_path)
    except Exception as exception:
        return Err(TypeError(f"Failed to read Parquet schema from {file_path}: {exception}"))

    for column_name, expected_dtype in SILVER_SCHEMA.items():
        if column_name not in actual_schema:
            return Err(TypeError(f"Schema violation: Missing required column '{column_name}'."))

        actual_dtype = actual_schema[column_name]
        if actual_dtype != expected_dtype:
            return Err(
                TypeError(
                    f"Schema violation for '{column_name}': "
                    f"Expected {expected_dtype}, got {actual_dtype}."
                )
            )

    return Ok(True)


def ingest_silver_data(
    source_path: PathLike,
    symbols: list[Symbol],
    *,
    start_time: Timestamp | None = None,
    end_time: Timestamp | None = None,
) -> Result[LazyFrame, Exception]:
    """
    Lazily ingests Silver Lake data with strict schema validation and predicate pushdown.

    Args:
        source_path: Path to the Silver Lake Parquet file.
        symbols: List of ticker symbols to filter.
        start_time: Optional start timestamp in Unix milliseconds.
        end_time: Optional end timestamp in Unix milliseconds.

    Returns:
        Ok(LazyFrame) with applied filters, or Err(Exception) on failure or schema violation.
    """
    resolved_path = Path(source_path).resolve()

    if not resolved_path.exists():
        return Err(FileNotFoundError(f"Silver data file not found: {resolved_path}"))

    if not resolved_path.is_file():
        return Err(FileNotFoundError(f"Target path is not a file: {resolved_path}"))

    schema_validation_result = validate_silver_schema(resolved_path)
    if schema_validation_result.is_err():
        error_schema = schema_validation_result.unwrap_err()
        logger.error(f"Ingestion blocked due to schema violation: {error_schema}")
        return Err(error_schema)

    try:
        lazy_dataframe = pl.scan_parquet(
            source=resolved_path,
        )

        if symbols:
            lazy_dataframe = lazy_dataframe.filter(pl.col("symbol").is_in(symbols))

        if start_time is not None:
            start_expr = pl.lit(start_time).cast(pl.Datetime("ms"))
            lazy_dataframe = lazy_dataframe.filter(pl.col("timestamp") >= start_expr)

        if end_time is not None:
            end_expr = pl.lit(end_time).cast(pl.Datetime("ms"))
            lazy_dataframe = lazy_dataframe.filter(pl.col("timestamp") <= end_expr)

        return Ok(lazy_dataframe)

    except Exception as exception:
        logger.error(f"Ingestion failed during query construction: {exception}")
        return Err(exception)


__all__ = [
    "validate_silver_schema",
    "ingest_silver_data",
]
