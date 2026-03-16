from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

from dacos.contracts import RAW_SCHEMA, SILVER_SCHEMA
from dacos.protocols import PathLike
from dacos.utils import Err, Ok, Result

if TYPE_CHECKING:
    from dacos.protocols import LazyFrame

logger = logging.getLogger(__name__)


def validate_source_directory(source_path: PathLike) -> Result[Path, FileNotFoundError]:
    """
    Validates the existence of the source directory.

    Args:
        source_path: The filesystem path to the raw data directory.

    Returns:
        Ok(Path) if the directory exists, Err(FileNotFoundError) otherwise.
    """
    resolved_path = Path(source_path).resolve()

    if not resolved_path.exists():
        return Err(FileNotFoundError(f"Directory not found: {resolved_path}"))

    if not resolved_path.is_dir():
        return Err(FileNotFoundError(f"Path is not a directory: {resolved_path}"))

    return Ok(resolved_path)


def extract_raw_parquet(raw_directory: Path) -> Result[LazyFrame, Exception]:
    """
    Extracts raw data lazily from a Hive-partitioned Parquet directory.

    Args:
        raw_directory: The validated path to the raw Parquet directory.

    Returns:
        Ok(LazyFrame) containing the un-evaluated query plan, or Err(Exception) on read failure.
    """
    try:
        # Guard clause: Prevent Polars lazy evaluation on empty directories
        if not any(raw_directory.rglob("*.parquet")):
            return Err(FileNotFoundError(f"No parquet files found in {raw_directory}"))

        dataframe_lazy = pl.scan_parquet(
            source=str(raw_directory),
            schema=RAW_SCHEMA,
            hive_partitioning=True,
        )
        return Ok(dataframe_lazy)
    except Exception as exception:
        return Err(exception)


def transform_to_silver_format(raw_data: LazyFrame) -> Result[LazyFrame, Exception]:
    """
    Transforms raw data to comply with the SILVER_SCHEMA definition.

    Args:
        raw_data: The extracted LazyFrame with RAW_SCHEMA.

    Returns:
        Ok(LazyFrame) containing the transformation plan, or Err(Exception) on failure.
    """
    try:
        valid_price_volume = raw_data.filter(
            pl.col("close").is_not_null() & (pl.col("volume") > 0.0)
        )

        timestamp_casted = valid_price_volume.with_columns(
            pl.col("timestamp").cast(pl.Datetime("ms")).alias("timestamp")
        )

        silver_columns_target = list(SILVER_SCHEMA.keys())
        # Unpacking target columns directly into multiple string arguments to avoid GenericAlias error
        data_selected = timestamp_casted.select(*silver_columns_target)

        # Using multiple strings directly instead of a list inside sort
        data_sorted = data_selected.sort("symbol", "timestamp")

        return Ok(data_sorted)
    except Exception as exception:
        return Err(exception)


def write_silver_parquet(silver_data: LazyFrame, destination_path: PathLike) -> Result[Path, Exception]:
    """
    Executes the lazy frame plan and writes the output to a Parquet file.

    Args:
        silver_data: The transformed LazyFrame.
        destination_path: The filesystem path where the output directory should reside.

    Returns:
        Ok(Path) pointing to the written file, or Err(Exception) on write failure.
    """
    try:
        destination_directory = Path(destination_path).resolve()
        destination_directory.mkdir(parents=True, exist_ok=True)

        output_file_path = destination_directory / "silver_master.parquet"

        silver_data.sink_parquet(
            path=output_file_path,
            compression="zstd",
            compression_level=22,
        )

        return Ok(output_file_path)
    except Exception as exception:
        return Err(exception)


def execute_etl_pipeline(raw_path: PathLike, silver_path: PathLike) -> Result[str, Exception]:
    """
    Executes the complete ETL pipeline from raw Parquet files to the Silver Lake.

    Args:
        raw_path: The filesystem path to the raw data directory.
        silver_path: The filesystem path to the destination Silver Lake directory.

    Returns:
        Ok(str) with a success message, or Err(Exception) if any pipeline step fails.
    """
    validation_result = validate_source_directory(raw_path)
    if validation_result.is_err():
        error_validation = validation_result.unwrap_err()
        logger.error(f"Pipeline failed at validation: {error_validation}")
        return Err(error_validation)

    validated_raw_path = validation_result.unwrap()

    extraction_result = extract_raw_parquet(validated_raw_path)
    if extraction_result.is_err():
        error_extraction = extraction_result.unwrap_err()
        logger.error(f"Pipeline failed at extraction: {error_extraction}")
        return Err(error_extraction)

    raw_lazy_frame = extraction_result.unwrap()

    transformation_result = transform_to_silver_format(raw_lazy_frame)
    if transformation_result.is_err():
        error_transformation = transformation_result.unwrap_err()
        logger.error(f"Pipeline failed at transformation: {error_transformation}")
        return Err(error_transformation)

    silver_lazy_frame = transformation_result.unwrap()

    write_result = write_silver_parquet(silver_lazy_frame, silver_path)
    if write_result.is_err():
        error_write = write_result.unwrap_err()
        logger.error(f"Pipeline failed at write: {error_write}")
        return Err(error_write)

    output_path = write_result.unwrap()
    success_message = f"Silver table successfully written to {output_path}"
    logger.info(success_message)

    return Ok(success_message)


__all__ = [
    "validate_source_directory",
    "extract_raw_parquet",
    "transform_to_silver_format",
    "write_silver_parquet",
    "execute_etl_pipeline",
]
