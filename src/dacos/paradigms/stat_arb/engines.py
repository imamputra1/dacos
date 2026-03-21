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
    if len(aligned_data) == 0:
        return Err(ValueError("Empty DataFrame provided."))
    if target_column not in aligned_data.columns:
        return Err(ValueError(f"Missing: {target_column}"))
    if anchor_column not in aligned_data.columns:
        return Err(ValueError(f"Missing: {anchor_column}"))
    if z_score_rolling_window < 2:
        return Err(ValueError("Window must be >= 2"))

    try:
        expr_spread = pl.col(target_column).log() - (hedge_ratio_beta * pl.col(anchor_column).log())
        data_with_spread = aligned_data.with_columns(expr_spread.alias("spread"))

        expr_mean = pl.col("spread").rolling_mean(window_size=z_score_rolling_window)
        expr_std = pl.col("spread").rolling_std(window_size=z_score_rolling_window)
        data_with_stats = data_with_spread.with_columns([expr_mean.alias("spread_mean"), expr_std.alias("spread_std")])

        expr_z_score = (
            pl.when(pl.col("spread_std") < 1e-8)
            .then(0.0)
            .otherwise((pl.col("spread") - pl.col("spread_mean")) / pl.col("spread_std"))
            .fill_nan(0.0)
        )

        return Ok(data_with_stats.with_columns(expr_z_score.alias("z_score")))
    except Exception as e:
        return Err(ValueError(f"Pairs Z-Score failed: {e}"))


def compute_basket_zscore(
    aligned_data: DataFrame,
    target_column: str,
    basket_columns: list[str],
    z_score_rolling_window: int,
) -> Result[DataFrame, ValueError]:
    if len(aligned_data) == 0:
        return Err(ValueError("Empty DataFrame."))
    if len(basket_columns) < 2:
        return Err(ValueError("PCA needs >= 2 assets."))

    try:
        all_assets = [target_column] + basket_columns
        returns_expr = [pl.col(asset).log().diff().alias(f"{asset}_ret") for asset in all_assets]
        data_with_returns = aligned_data.with_columns(returns_expr)

        basket_ret_cols = [f"{asset}_ret" for asset in basket_columns]
        matrix_pca = data_with_returns.select(basket_ret_cols).drop_nulls().to_numpy()

        pca_result = compute_pca_safe(matrix_pca)
        if pca_result.is_err():
            return Err(ValueError(f"PCA failed: {pca_result.unwrap_err()}"))
        _, eigenvectors = pca_result.unwrap()

        pc1_raw = eigenvectors[:, 0]
        if np.sum(pc1_raw) < 0:
            pc1_raw = -pc1_raw
        pc1_weights = pc1_raw / np.sum(np.abs(pc1_raw))

        synthetic_anchor = pl.sum_horizontal(
            [pl.col(c) * float(w) for c, w in zip(basket_ret_cols, pc1_weights, strict=False)]
        )
        expr_spread = pl.col(f"{target_column}_ret") - synthetic_anchor

        data_with_spread = data_with_returns.with_columns(
            [synthetic_anchor.alias("synthetic_anchor_return"), expr_spread.alias("spread")]
        )

        expr_mean = pl.col("spread").rolling_mean(window_size=z_score_rolling_window)
        expr_std = pl.col("spread").rolling_std(window_size=z_score_rolling_window)
        data_with_stats = data_with_spread.with_columns([expr_mean.alias("spread_mean"), expr_std.alias("spread_std")])

        expr_z_score = (
            pl.when(pl.col("spread_std") < 1e-8)
            .then(0.0)
            .otherwise((pl.col("spread") - pl.col("spread_mean")) / pl.col("spread_std"))
            .fill_nan(0.0)
        )

        return Ok(data_with_stats.with_columns(expr_z_score.alias("z_score")))
    except Exception as e:
        return Err(ValueError(f"Basket Z-Score failed: {e}"))


__all__ = ["compute_pairs_zscore", "compute_basket_zscore"]
