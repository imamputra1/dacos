from __future__ import annotations

import math
from typing import Any

import polars as pl

from dacos.config import StatArbConfig
from dacos.protocols import DataFrame
from dacos.utils import Err, Ok, Result


def apply_mean_reversion_tactics_strict(
    data: DataFrame | dict[str, Any],
    symbol: str,
    config: StatArbConfig | None = None,
) -> Result[DataFrame | dict[str, Any], ValueError]:
    config = config or StatArbConfig()
    # 1. VECTORIZED RESEARCH
    if isinstance(data, pl.DataFrame):
        if len(data) == 0:
            return Err(ValueError("Empty DataFrame."))
        for col in ["timestamp", "z_score", "spread"]:
            if col not in data.columns:
                return Err(ValueError(f"Missing '{col}'"))

        try:
            expr_raw = (
                pl.when(pl.col("z_score") > config.entry_z)
                .then(-1)
                .when(pl.col("z_score") < -config.entry_z)
                .then(1)
                .when(pl.col("z_score").abs() <= config.exit_z)
                .then(0)
                .otherwise(None)
            )

            if not config.allow_short:
                expr_raw = pl.when(expr_raw == -1).then(None).otherwise(expr_raw)

            expr_action = (
                pl.when(expr_raw == 1)
                .then(pl.lit("BUY"))
                .when(expr_raw == -1)
                .then(pl.lit("SELL"))
                .when(expr_raw == 0)
                .then(pl.lit("EXIT"))
                .otherwise(pl.lit("NEUTRAL"))
            )
            df_final = data.with_columns(
                [
                    pl.lit(symbol).alias("symbol"),
                    expr_action.alias("action"),
                    expr_raw.fill_null(0).cast(pl.Int8).alias("position"),
                    pl.col("z_score").abs().alias("strength"),
                ]
            ).select(["timestamp", "symbol", "action", "position", "strength", "z_score", "spread"])

            return Ok(df_final)
        except Exception as e:
            return Err(ValueError(f"Vectorized failure: {e}"))

    # 2. LIVE TICK
    elif isinstance(data, dict):
        for key in ["timestamp", "z_score", "spread"]:
            if key not in data:
                return Err(ValueError(f"Missing '{key}'"))
        try:
            z = data["z_score"]
            if z is None or math.isnan(z):
                raw = None
            elif z > config.entry_z:
                raw = -1
            elif z < -config.entry_z:
                raw = 1
            elif abs(z) <= config.exit_z:
                raw = 0
            else:
                raw = None

            if not config.allow_short and raw == -1:
                raw = None

            if raw == 1:
                act = "BUY"
            elif raw == -1:
                act = "SELL"
            elif raw == 0:
                act = "EXIT"
            else:
                act = "NEUTRAL"

            return Ok(
                {
                    "timestamp": data["timestamp"],
                    "symbol": symbol,
                    "action": act,
                    "position": int(raw) if raw is not None else 0,
                    "strength": abs(z) if z is not None and not math.isnan(z) else 0.0,
                    "z_score": z,
                    "spread": data["spread"],
                }
            )
        except Exception as e:
            return Err(ValueError(f"Live tick failure: {e}"))
    else:
        return Err(ValueError("Unsupported data type."))


__all__ = ["apply_mean_reversion_tactics_strict"]
