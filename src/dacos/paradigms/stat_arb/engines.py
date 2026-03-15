"""
StatArbEngine: Computes hedge ratio, spread, and rolling z-score for pairs trading.
"""

import numpy as np
import polars as pl
import statsmodels.api as sm

from dacos.utils import Err, Ok, Result


class StatArbEngine:
    """
    Engine for statistical arbitrage pair trading.
    Calculates static hedge ratio via OLS, then computes spread and rolling z-score.
    """

    def __init__(self, zscore_window: int = 100) -> None:
        """
        Initialize the engine with rolling window size for z-score.

        Args:
            zscore_window: Number of periods for rolling mean and standard deviation.
        """
        self.window = zscore_window

    def _calculate_hedge_ratio(self, y: np.ndarray, x: np.ndarray) -> float:
        """
        Compute hedge ratio (beta) via OLS: y = alpha + beta * x + epsilon.

        Args:
            y: Dependent variable array (1D).
            x: Independent variable array (1D).

        Returns:
            Beta coefficient.
        """
        X = sm.add_constant(x)
        model = sm.OLS(y, X).fit()
        return float(model.params[1])

    def run_engine(
        self,
        data: pl.DataFrame,
        y_symbol: str,
        x_symbol: str,
    ) -> Result[pl.DataFrame, Exception]:
        """
        Execute the pairs trading engine.

        Steps:
        1. Pivot data to wide format: one column per symbol.
        2. Extract numpy arrays for y and x.
        3. Compute static hedge ratio via OLS.
        4. Compute spread = y - beta * x.
        5. Compute rolling mean and std of spread.
        6. Compute z-score = (spread - rolling_mean) / rolling_std.

        Args:
            data: Long-format DataFrame with columns 'timestamp', 'symbol', 'log_price'.
            y_symbol: Symbol of the dependent (Y) asset.
            x_symbol: Symbol of the independent (X) asset.

        Returns:
            Ok(DataFrame) with additional columns: spread, spread_mean, spread_std, z_score.
            Err(exception) if any step fails.
        """
        try:
            if data.is_empty():
                return Err(ValueError("Input data is empty"))

            data_wide = data.select(["timestamp", "symbol", "log_price"]).pivot(
                values="log_price",
                index="timestamp",
                on="symbol",
            )

            if data_wide.is_empty():
                return Err(ValueError("Pivoted data is empty"))

            if y_symbol not in data_wide.columns:
                return Err(ValueError(f"Symbol {y_symbol} not found after pivot"))
            if x_symbol not in data_wide.columns:
                return Err(ValueError(f"Symbol {x_symbol} not found after pivot"))

            data_wide = data_wide.drop_nulls()

            if data_wide.is_empty():
                return Err(ValueError("No complete rows after dropping nulls"))

            arr_y = data_wide.get_column(y_symbol).to_numpy()
            arr_x = data_wide.get_column(x_symbol).to_numpy()

            beta = self._calculate_hedge_ratio(arr_y, arr_x)

            data_wide = data_wide.with_columns(
                (pl.col(y_symbol) - beta * pl.col(x_symbol)).alias("spread")
            )

            data_wide = data_wide.with_columns([
                pl.col("spread")
                .rolling_mean(window_size=self.window, min_samples=self.window)
                .alias("spread_mean"),
                pl.col("spread")
                .rolling_std(window_size=self.window, min_samples=self.window)
                .alias("spread_std"),
            ])

            data_wide = data_wide.with_columns([
                ((pl.col("spread") - pl.col("spread_mean")) / pl.col("spread_std")).alias("z_score")
            ])

            return Ok(data_wide)

        except Exception as e:
            return Err(e)


def create_stat_arb_engine(zscore_window: int = 100) -> StatArbEngine:
    """
    Factory function to create a StatArbEngine instance.

    Args:
        zscore_window: Rolling window size for z-score calculation.

    Returns:
        Configured StatArbEngine.
    """
    return StatArbEngine(zscore_window=zscore_window)
