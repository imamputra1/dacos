"""
SILVER LAKE BUILDER (THE FACTORY LOGIC)
Location: src/dacos/builder/etl.py
Paradigm: Pure Functions, Monadic Result, Guard Clauses, Strict Schema Enforcement.
"""

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


class SkinnyLakeBuilder:
    """
    Builder class for constructing skinny tables following SILVER_SCHEMA
    from raw Parquet files with Hive partitioning.
    """

    def __init__(self, raw_path: PathLike, silver_path: PathLike) -> None:
        self.raw_path = Path(raw_path)
        self.silver_path = Path(silver_path)

    def _validate_path(self) -> Result[bool, Exception]:
        """Guard clause: Ensure raw path exists."""
        if not self.raw_path.exists():
            return Err(FileNotFoundError(f"Raw path does not exist: {self.raw_path}"))
        return Ok(True)

    def _extract(self) -> LazyFrame:
        """
        Extract raw data lazily from Hive-partitioned directory.
        Raises exception on failure (caught in execute_pipeline).
        """
        return pl.scan_parquet(
            str(self.raw_path),
            schema=RAW_SCHEMA,
            hive_partitioning=True,
            cast_options=pl.ScanCastOptions(integer_cast="upcast"),
        )

    def _transform(self, data: LazyFrame) -> LazyFrame:
        """
        Transform raw data to SILVER_SCHEMA.
        """
        # Filter invalid rows
        data = data.filter(
            pl.col("close").is_not_null() & (pl.col("volume") > 0)
        )

        # Cast timestamp to Datetime('ms')
        data = data.with_columns(
            pl.col("timestamp").cast(pl.Datetime("ms")).alias("timestamp")
        )

        # Select only columns in SILVER_SCHEMA
        silver_columns = list(SILVER_SCHEMA.keys())
        data = data.select(silver_columns)

        # Sort for deterministic order
        data = data.sort(["symbol", "timestamp"])

        return data

    def execute_pipeline(self) -> Result[str, Exception]:
        """
        Execute the full ETL pipeline.
        Returns Ok(success_message) or Err(exception).
        """
        # 1. Validate path
        path_check = self._validate_path()
        if path_check.is_err():
            return Err(path_check.unwrap_err())

        try:
            # 2. Extract
            raw_data = self._extract()

            # 3. Transform
            silver_data = self._transform(raw_data)

            # 4. Write
            self.silver_path.mkdir(parents=True, exist_ok=True)
            output_file = self.silver_path / "silver_master.parquet"
            silver_data.sink_parquet(
                output_file,
                compression="zstd",
                compression_level=22,
            )
            msg = f"✅ Silver table successfully written to {output_file}"
            logger.info(msg)
            return Ok(msg)

        except Exception as e:
            logger.error(f"❌ Pipeline failed during write: {e}")
            return Err(e)


def create_skinny_builder(
    raw_path: PathLike,
    silver_path: PathLike,
) -> SkinnyLakeBuilder:
    return SkinnyLakeBuilder(raw_path, silver_path)


build_skinny_lake = create_skinny_builder
