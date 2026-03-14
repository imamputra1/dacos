"""
builder/etl.py

Industrial-grade ETL pipeline for building skinny tables from raw Parquet data.
Uses Polars lazy evaluation to avoid loading data into RAM.
Implements Result monad for explicit error handling.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

from dacos.protocols import PathLike
from dacos.utils import Err, Ok, Result

if TYPE_CHECKING:
    from dacos.protocols import LazyFrame

logger = logging.getLogger(__name__)


class SkinnyLakeBuilder:
    """
    Builder class for constructing skinny tables (timestamp, symbol, log_price, log_volume)
    from raw Parquet files.
    """

    def __init__(self, raw_path: PathLike, silver_path: PathLike) -> None:
        """
        Initialize the builder with source and destination paths.

        Args:
            raw_path: Directory containing raw Parquet files (may include subdirectories).
            silver_path: Directory where the skinny table will be written.
        """
        self.raw_path = Path(raw_path)
        self.silver_path = Path(silver_path)

    def _extract(self) -> LazyFrame:
        """
        Extract raw data lazily.

        Returns:
            LazyFrame with columns: timestamp, symbol, close, volume.
        """
        # Scan all Parquet files recursively
        pattern = str(self.raw_path / "**" / "*.parquet")
        lazy_df = pl.scan_parquet(pattern)

        # Select only required columns
        return lazy_df.select(["timestamp", "symbol", "close", "volume"])

    def _transform(self, data: LazyFrame) -> LazyFrame:
        """
        Transform raw data: filter, compute log prices/volumes, sort, and select final columns.

        Args:
            data: LazyFrame with raw columns.

        Returns:
            LazyFrame with columns: timestamp, symbol, log_price, log_volume.
        """
        # Guard: filter out rows with null close or zero/negative volume
        data = data.filter(
            pl.col("close").is_not_null() & (pl.col("volume") > 0)
        )

        # Add log-transformed columns
        data = data.with_columns(
            pl.col("close").log().alias("log_price"),
            pl.col("volume").log().alias("log_volume"),
        )

        # Sort by symbol and timestamp
        data = data.sort(["symbol", "timestamp"])

        # Select final columns
        return data.select(["timestamp", "symbol", "log_price", "log_volume"])

    def execute_pipeline(self) -> Result[str, Exception]:
        """
        Execute the full ETL pipeline: extract, transform, and sink to Parquet.

        Returns:
            Ok(success_message) if successful, otherwise Err(exception).
        """
        try:
            # Ensure destination directory exists
            self.silver_path.mkdir(parents=True, exist_ok=True)

            # Extract
            raw_data = self._extract()

            # Transform
            clean_data = self._transform(raw_data)

            # Define output file path
            output_file = self.silver_path / "skinny.parquet"

            # Sink to Parquet (lazy write, no collect)
            clean_data.sink_parquet(
                output_file,
                compression="zstd",
                compression_level=22,
            )

            msg = f"✅ Skinny table successfully written to {output_file}"
            logger.info(msg)
            return Ok(msg)

        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}")
            return Err(e)


def create_skinny_builder(
    raw_path: PathLike,
    silver_path: PathLike,
) -> SkinnyLakeBuilder:
    """
    Factory function to create a SkinnyLakeBuilder instance.

    Args:
        raw_path: Directory containing raw Parquet files.
        silver_path: Directory for the skinny table.

    Returns:
        Configured SkinnyLakeBuilder.
    """
    return SkinnyLakeBuilder(raw_path, silver_path)


# Alias for backward compatibility
build_skinny_lake = create_skinny_builder
