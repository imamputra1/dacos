"""
core/alignment.py

UniverseAligner: Master clock for time series alignment using Polars upsample.
Transforms irregularly spaced data into a regular frequency grid with forward fill.
Ensures all symbols are aligned to the same global time grid.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import polars as pl

from dacos.utils import Err, Ok, Result

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class UniverseAligner:
    """
    Aligns multi-symbol time series to a regular frequency grid.

    Uses Polars' native upsample with forward fill, operating directly on
    long-format data (timestamp, symbol, log_price, log_volume). No joins,
    no wide-format pivoting – purely columnar and parallelised.
    """

    def __init__(self, frequency: str = "1m") -> None:
        """
        Initialize the aligner with a target frequency.

        Args:
            frequency: Target frequency string (e.g., "1m", "5s", "1h").
                       Must be a valid Polars duration string.
        """
        self.frequency = frequency
        logger.info(f"UniverseAligner initialized with frequency={frequency}")

    def align(self, data: pl.DataFrame) -> Result[pl.DataFrame, Exception]:
        """
        Align the input DataFrame to a regular time grid.

        Steps:
        1. Cast timestamp to Datetime (ms precision) – required for upsample.
        2. Sort by timestamp.
        3. Determine global min and max timestamps.
        4. Add dummy rows at global boundaries for each symbol to ensure identical grid range.
        5. Remove duplicate (timestamp, symbol) pairs (keep original data).
        6. Upsample to the target frequency, grouping by symbol.
        7. Forward-fill log_price and log_volume within each symbol group.
        8. Drop rows that remain null (e.g., leading gaps).

        Args:
            data: Polars DataFrame with at least columns:
                  timestamp (int/float), symbol (str), log_price (float), log_volume (float).

        Returns:
            Ok(aligned DataFrame) with same columns, now at regular frequency,
            or Err(exception) if any step fails.
        """
        try:
            if data.is_empty():
                logger.info("Input DataFrame is empty, returning empty result.")
                return Ok(data)

            df = data.with_columns(
                pl.col("timestamp").cast(pl.Datetime("ms")).alias("timestamp")
            )

            df = df.sort("timestamp")

            min_ts = df["timestamp"].min()
            max_ts = df["timestamp"].max()

            symbols = df["symbol"].unique().to_list()

            dummy_data = []
            for sym in symbols:
                dummy_data.append({"timestamp": min_ts, "symbol": sym, "log_price": None, "log_volume": None})
                dummy_data.append({"timestamp": max_ts, "symbol": sym, "log_price": None, "log_volume": None})

            dummy_df = pl.DataFrame(dummy_data, schema=df.schema)

            df = pl.concat([df, dummy_df]).unique(subset=["timestamp", "symbol"], keep="first")

            df = df.sort("timestamp")

            df = df.upsample(
                time_column="timestamp",
                every=self.frequency,
                group_by="symbol",
                maintain_order=True,
            )

            df = df.with_columns(
                pl.col("log_price", "log_volume").forward_fill().over("symbol")
            )

            df = df.drop_nulls()

            logger.info(f"Alignment successful: {len(df)} rows after alignment")
            return Ok(df)

        except Exception as e:
            logger.error(f"Alignment failed: {e}", exc_info=True)
            return Err(e)


def create_universe_aligner(frequency: str = "1m") -> UniverseAligner:
    """
    Factory function to create a UniverseAligner instance.

    Args:
        frequency: Target frequency string (e.g., "1m", "5s", "1h").

    Returns:
        Configured UniverseAligner.
    """
    return UniverseAligner(frequency=frequency)
