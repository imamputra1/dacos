"""
Module for loading specific symbols from the silver lake into memory.
Uses Polars lazy evaluation to filter before collecting, ensuring memory safety.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

from dacos.utils import Err, Ok, Result

if TYPE_CHECKING:
    from dacos.protocols import PathLike

logger = logging.getLogger(__name__)


class UniverseIngestor:
    """
    Ingests data from silver lake (skinny table) for a given set of symbols.
    """

    def __init__(self, silver_path: PathLike) -> None:
        """
        Initialize the ingestor with the path to the silver lake.

        Args:
            silver_path: Path to the directory containing skinny Parquet files.
        """
        self.silver_path = Path(silver_path)

    def load_universe(self, symbols: list[str]) -> Result[pl.DataFrame, Exception]:
        """
        Load data for specified symbols into memory as a Polars DataFrame.

        This method performs a lazy scan, filters by the given symbols,
        and finally collects the result into a DataFrame. If the silver path
        contains multiple Parquet files (e.g., partitioned), it will read all.

        Args:
            symbols: List of symbol strings to load (e.g., ["BTCUSDT", "ETHUSDT"]).

        Returns:
            Ok(DataFrame) with columns: timestamp, symbol, log_price, log_volume
            (depending on the schema of the skinny table).
            Err(exception) if any error occurs.
        """
        try:
            if not self.silver_path.exists():
                return Err(FileNotFoundError(f"Silver path does not exist: {self.silver_path}"))

            # Skema lengkap skinny table untuk memastikan tipe konsisten
            # Timestamp dipaksa sebagai Datetime("ms") agar kompatibel dengan alignment
            skinny_schema = {
                "timestamp": pl.Datetime("ms"),
                "symbol": pl.Utf8,
                "log_price": pl.Float64,
                "log_volume": pl.Float64,
            }

            # Tentukan sumber data berdasarkan jenis path
            if self.silver_path.is_dir():
                # Direktori: scan semua file Parquet secara rekursif dengan dukungan Hive partitioning
                lazy_df: pl.LazyFrame = pl.scan_parquet(
                    str(self.silver_path / "**" / "*.parquet"),
                    hive_partitioning=True,
                    schema=skinny_schema,  # Gunakan skema lengkap
                )
            elif self.silver_path.is_file() and self.silver_path.suffix.lower() == ".parquet":
                # File tunggal .parquet
                lazy_df = pl.scan_parquet(
                    self.silver_path,
                    schema=skinny_schema,
                )
            else:
                return Err(ValueError(
                    f"Silver path must be a directory or a .parquet file, got: {self.silver_path}"
                ))

            # Filter berdasarkan simbol
            filtered = lazy_df.filter(pl.col("symbol").is_in(symbols))

            # Kumpulkan ke DataFrame
            df = filtered.collect()

            logger.info(f"Loaded {len(df)} rows for symbols {symbols}")
            return Ok(df)

        except Exception as e:
            logger.error(f"Failed to load universe: {e}", exc_info=True)
            return Err(e)


def create_universe_ingestor(silver_path: PathLike) -> UniverseIngestor:
    """
    Factory function to create a UniverseIngestor instance.

    Args:
        silver_path: Path to the silver lake directory.

    Returns:
        Configured UniverseIngestor.
    """
    return UniverseIngestor(silver_path)
