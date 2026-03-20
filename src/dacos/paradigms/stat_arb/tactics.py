from __future__ import annotations

import math
from typing import Any

import polars as pl

from dacos.protocols import DataFrame
from dacos.utils import Err, Ok, Result


def apply_mean_reversion_tactics_strict(
    data: DataFrame | dict[str, Any],
    symbol: str,
    entry_z_threshold: float = 2.0,
    exit_z_threshold: float = 0.0,
    allow_short: bool = True,
) -> Result[DataFrame | dict[str, Any], ValueError]:
    """
    Converts continuous Z-Scores into discrete absolute trading actions (BUY, SELL, EXIT, NEUTRAL).
    Supports both Vectorized Polars DataFrames (Research) and Dicts (Live Tick-by-Tick).

    Args:
        data: Polars DataFrame or Dictionary containing at least 'timestamp' and 'z_score'.
        symbol: The trading pair or synthetic basket identifier.
        entry_z_threshold: Absolute Z-Score required to trigger an entry (Long/Short).
        exit_z_threshold: Absolute Z-Score required to trigger a mean-reversion exit.
        allow_short: If False (Spot mode), all SELL (-1) signals are overridden to NEUTRAL.

    Returns:
        Ok(DataFrame | Dict) with standard schema: [timestamp, symbol, action, strength, z_score].
    """
    # ============================================================================
    # MODE 1: VECTORIZED RESEARCH (POLARS DATAFRAME)
    # ============================================================================
    if isinstance(data, pl.DataFrame):
        if len(data) == 0:
            return Err(ValueError("Tactics Engine received an empty DataFrame."))

        if "timestamp" not in data.columns:
            return Err(ValueError("Missing required column: 'timestamp'."))

        if "z_score" not in data.columns:
            return Err(ValueError("Missing required column: 'z_score'."))

        try:
            expr_raw_action = (
                pl.when(pl.col("z_score") > entry_z_threshold).then(-1)
                .when(pl.col("z_score") < -entry_z_threshold).then(1)
                .when(pl.col("z_score").abs() <= exit_z_threshold).then(0)
                .otherwise(None)
            )

            if not allow_short:
                expr_raw_action = pl.when(expr_raw_action == -1).then(None).otherwise(expr_raw_action)

            expr_action_string = (
                pl.when(expr_raw_action == 1).then(pl.lit("BUY"))
                .when(expr_raw_action == -1).then(pl.lit("SELL"))
                .when(expr_raw_action == 0).then(pl.lit("EXIT"))
                .otherwise(pl.lit("NEUTRAL"))
            )

            df_final = data.with_columns([
                pl.lit(symbol).alias("symbol"),
                expr_action_string.alias("action"),
                pl.col("z_score").abs().alias("strength")
            ]).select(["timestamp", "symbol", "action", "strength", "z_score"])

            return Ok(df_final)

        except Exception as e:
            return Err(ValueError(f"Vectorized tactics evaluation failed: {e}"))

    # ============================================================================
    # MODE 2: LIVE TICK-BY-TICK (DICTIONARY / JSON)
    # ============================================================================
    elif isinstance(data, dict):
        if "timestamp" not in data:
            return Err(ValueError("Missing required key in tick data: 'timestamp'."))

        if "z_score" not in data:
            return Err(ValueError("Missing required key in tick data: 'z_score'."))

        try:
            z = data["z_score"]
            if z is None or math.isnan(z):
                raw_action = None
            elif z > entry_z_threshold:
                raw_action = -1
            elif z < -entry_z_threshold:
                raw_action = 1
            elif abs(z) <= exit_z_threshold:
                raw_action = 0
            else:
                raw_action = None

            if not allow_short and raw_action == -1:
                raw_action = None

            if raw_action == 1:
                action_str = "BUY"
            elif raw_action == -1:
                action_str = "SELL"
            elif raw_action == 0:
                action_str = "EXIT"
            else:
                action_str = "NEUTRAL"

            tick_final = {
                "timestamp": data["timestamp"],
                "symbol": symbol,
                "action": action_str,
                "strength": abs(z) if z is not None and not math.isnan(z) else 0.0,
                "z_score": z
            }

            return Ok(tick_final)

        except Exception as e:
            return Err(ValueError(f"Live tick tactics evaluation failed: {e}"))

    else:
        return Err(ValueError(f"Unsupported data type for tactics engine: {type(data)}. Expected Polars DataFrame or Dict."))


__all__ = [
    "apply_mean_reversion_tactics_strict",
]
