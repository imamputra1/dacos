"""
core/validation.py

UniverseValidator: Quality control for aligned data.
Ensures sufficient length, no excessive nulls, and no stagnant coins.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import polars as pl

from dacos.utils import Err, Ok, Result

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class UniverseValidator:
    """
    Validates that the input DataFrame meets minimum quality standards:
    - At least `min_rows` total rows after dropping nulls.
    - No coins with zero variance (stagnant prices).
    """

    def __init__(self, min_rows: int = 100) -> None:
        """
        Initialize the validator with a minimum row threshold.

        Args:
            min_rows: Minimum number of rows required for analysis.
        """
        self.min_rows = min_rows
        logger.info(f"UniverseValidator initialized with min_rows={min_rows}")

    def validate(self, data: pl.DataFrame) -> Result[pl.DataFrame, Exception]:
        """
        Validate the input DataFrame.

        Steps:
        1. Check if total rows >= min_rows.
        2. Drop nulls; if resulting rows < min_rows, reject.
        3. Check for stagnant coins (standard deviation of log_price == 0 per symbol).

        Args:
            data: Polars DataFrame with at least columns:
                  symbol (str), log_price (float).

        Returns:
            Ok(original data) if all checks pass,
            Err(exception) with descriptive message otherwise.
        """
        try:
            if len(data) < self.min_rows:
                msg = f"Data too short: {len(data)} rows < {self.min_rows}"
                logger.warning(msg)
                return Err(ValueError(msg))

            data_clean = data.drop_nulls()
            if len(data_clean) < self.min_rows:
                msg = f"After dropping nulls: {len(data_clean)} rows < {self.min_rows}"
                logger.warning(msg)
                return Err(ValueError(msg))

            std_per_symbol = data_clean.group_by("symbol").agg(
                pl.col("log_price").std().alias("std_dev")
            )

            dead_symbols = std_per_symbol.filter(
                (pl.col("std_dev") == 0) | pl.col("std_dev").is_null()
            )["symbol"].to_list()

            if dead_symbols:
                msg = f"Stagnant coins detected (std_dev = 0): {dead_symbols}"
                logger.warning(msg)
                return Err(ValueError(msg))

            logger.info(f"Validation passed: {len(data)} rows, no stagnant coins")
            return Ok(data)

        except Exception as e:
            logger.error(f"Validation failed with exception: {e}", exc_info=True)
            return Err(e)


def create_universe_validator(min_rows: int = 100) -> UniverseValidator:
    """
    Factory function to create a UniverseValidator instance.

    Args:
        min_rows: Minimum number of rows required for analysis.

    Returns:
        Configured UniverseValidator.
    """
    return UniverseValidator(min_rows=min_rows)
