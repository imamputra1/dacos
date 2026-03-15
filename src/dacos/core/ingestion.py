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
            pattern = str(self.silver_path / "**" / "*.parquet")
            lazy_df: pl.LazyFrame = pl.scan_parquet(pattern)
            filtered = lazy_df.filter(pl.col("symbol").is_in(symbols))
            df = filtered.collect()
            logger.info(f"Loaded {len(df)} rows for symbols {symbols}")
            return Ok(df)

        except Exception as e:
            logger.error(f"Failed to load universe: {e}")
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
