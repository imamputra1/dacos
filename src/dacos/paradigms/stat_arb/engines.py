from __future__ import annotations

import numpy as np
import polars as pl

from dacos.core import compute_pca_safe
from dacos.protocols import DataFrame
from dacos.utils import Err, Ok, Result


def compute_pairs_zscore(
    aligned_data: DataFrame,
    target_column: str,
    anchor_column: str,
    hedge_ratio_beta: float,
    z_score_rolling_window: int,
) -> Result[DataFrame, ValueError]:
    """
    Computes the Z-Score of the log spread between two cointegrated assets.
    """
    if len(aligned_data) == 0:
        return Err(ValueError("Empty DataFrame provided."))

    if target_column not in aligned_data.columns:
        return Err(ValueError(f"Missing required column for target asset: {target_column}"))

    if anchor_column not in aligned_data.columns:
        return Err(ValueError(f"Missing required column for anchor asset: {anchor_column}"))

    if z_score_rolling_window < 2:
        return Err(ValueError(f"Z-Score rolling window must be at least 2, got {z_score_rolling_window}"))

    try:
        expr_log_target = pl.col(target_column).log()
        expr_log_anchor = pl.col(anchor_column).log()
        expr_log_spread = expr_log_target - (hedge_ratio_beta * expr_log_anchor)

        data_with_spread = aligned_data.with_columns(
            expr_log_spread.alias("log_spread")
        )

        expr_mean = pl.col("log_spread").rolling_mean(window_size=z_score_rolling_window)
        expr_std = pl.col("log_spread").rolling_std(window_size=z_score_rolling_window)

        data_with_stats = data_with_spread.with_columns([
            expr_mean.alias("spread_mean"),
            expr_std.alias("spread_std")
        ])

        expr_z_score = pl.when(pl.col("spread_std") <= 1e-8).then(0.0).otherwise(
            (pl.col("log_spread") - pl.col("spread_mean")) / pl.col("spread_std")
        ).fill_nan(0.0)

        data_final = data_with_stats.with_columns(
            expr_z_score.alias("z_score")
        )

        return Ok(data_final)

    except Exception as computation_error:
        return Err(ValueError(f"Pairs Z-Score computation failed: {computation_error}"))


def compute_basket_zscore(
    aligned_data: DataFrame,
    target_column: str,
    basket_columns: list[str],
    z_score_rolling_window: int,
) -> Result[DataFrame, ValueError]:
    """
    Computes the Z-Score of a Target Asset against a Synthetic PCA Basket.
    """
    if len(aligned_data) == 0:
        return Err(ValueError("Empty DataFrame provided."))

    if target_column not in aligned_data.columns:
        return Err(ValueError(f"Missing required target column: {target_column}"))

    for col in basket_columns:
        if col not in aligned_data.columns:
            return Err(ValueError(f"Missing required basket column: {col}"))

    if len(basket_columns) < 2:
        return Err(ValueError("PCA Basket Engine requires at least 2 anchor assets."))

    if z_score_rolling_window < 2:
        return Err(ValueError(f"Z-Score rolling window must be at least 2, got {z_score_rolling_window}"))

    try:
        all_assets = [target_column] + basket_columns
        returns_expressions = [
            pl.col(asset).log().diff().alias(f"{asset}_ret") for asset in all_assets
        ]
        data_with_returns = aligned_data.with_columns(returns_expressions)

        basket_return_columns = [f"{asset}_ret" for asset in basket_columns]
        matrix_for_pca = data_with_returns.select(basket_return_columns).drop_nulls().to_numpy()

        pca_result = compute_pca_safe(matrix_for_pca)
        if pca_result.is_err():
            return Err(ValueError(f"PCA computation kernel panic: {pca_result.unwrap_err()}"))

        _, eigenvectors = pca_result.unwrap()

        pc1_raw_weights = eigenvectors[:, 0]
        if np.sum(pc1_raw_weights) < 0:
            pc1_raw_weights = -pc1_raw_weights

        pc1_weights = pc1_raw_weights / np.sum(np.abs(pc1_raw_weights))

        synthetic_anchor_expression = pl.sum_horizontal([
            pl.col(col_name) * float(weight)
            for col_name, weight in zip(basket_return_columns, pc1_weights)
        ])

        target_return_col = f"{target_column}_ret"
        spread_expression = pl.col(target_return_col) - synthetic_anchor_expression

        data_with_spread = data_with_returns.with_columns([
            synthetic_anchor_expression.alias("synthetic_anchor_return"),
            spread_expression.alias("basket_spread")
        ])

        expr_mean = pl.col("basket_spread").rolling_mean(window_size=z_score_rolling_window)
        expr_std = pl.col("basket_spread").rolling_std(window_size=z_score_rolling_window)

        data_with_stats = data_with_spread.with_columns([
            expr_mean.alias("spread_mean"),
            expr_std.alias("spread_std")
        ])

        expr_z_score = pl.when(pl.col("spread_std") <= 1e-8).then(0.0).otherwise(
            (pl.col("basket_spread") - pl.col("spread_mean")) / pl.col("spread_std")
        ).fill_nan(0.0)

        data_final = data_with_stats.with_columns(
            expr_z_score.alias("z_score")
        )

        return Ok(data_final)

    except Exception as computation_error:
        return Err(ValueError(f"Basket Z-Score computation failed: {computation_error}"))


__all__ = [
    "compute_pairs_zscore",
    "compute_basket_zscore",
]
