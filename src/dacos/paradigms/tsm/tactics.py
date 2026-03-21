from __future__ import annotations

import math
from typing import Any

import polars as pl

from dacos.config import TSMConfig  # IMPORT CONFIG
from dacos.protocols import DataFrame
from dacos.utils import Err, Ok, Result


def apply_momentum_tactics_strict(
    data: DataFrame | dict[str, Any],
    target_symbol: str,
    config: TSMConfig | None = None,
) -> Result[DataFrame | dict[str, Any], ValueError]:
    config = config or TSMConfig()

    # 1. VECTORIZED RESEARCH
    if isinstance(data, pl.DataFrame):
        if len(data) == 0:
            return Err(ValueError("Empty DataFrame."))
        for col in ["timestamp", "close", "upper_band", "lower_band", "atr"]:
            if col not in data.columns:
                return Err(ValueError(f"Missing '{col}'"))

        try:
            expr_mid = (pl.col("upper_band") + pl.col("lower_band")) / 2.0
            expr_raw = (
                pl.when(pl.col("close") > pl.col("upper_band"))
                .then(1)
                .when(pl.col("close") < pl.col("lower_band"))
                .then(-1)
                .when(pl.col("close") == expr_mid)
                .then(0)
                .otherwise(None)
            )

            if not config.allow_short:
                expr_raw = pl.when(expr_raw == -1).then(None).otherwise(expr_raw)

            expr_rel_vol = pl.col("atr") / pl.col("close")
            expr_strength = (
                pl.when(pl.col("atr") <= 1e-8).then(0.0).otherwise(config.target_risk_pct / expr_rel_vol).fill_nan(0.0)
            )

            expr_action = (
                pl.when(expr_raw == 1)
                .then(pl.lit("BUY"))
                .when(expr_raw == -1)
                .then(pl.lit("SELL"))
                .when(expr_raw == 0)
                .then(pl.lit("EXIT"))
                .otherwise(pl.lit("NEUTRAL"))
            )

            # SURGERY: Inject 'position' as Int8, Remove 'close' to strictly match TSM_SIGNAL_SCHEMA
            df_final = data.with_columns(
                [
                    pl.lit(target_symbol).alias("symbol"),
                    expr_action.alias("action"),
                    expr_raw.fill_null(0).cast(pl.Int8).alias("position"),
                    expr_strength.alias("strength"),
                ]
            ).select(["timestamp", "symbol", "action", "position", "strength", "atr"])

            return Ok(df_final)
        except Exception as e:
            return Err(ValueError(f"Vectorized failure: {e}"))

    # 2. LIVE TICK
    elif isinstance(data, dict):
        for key in ["timestamp", "close", "upper_band", "lower_band", "atr"]:
            if key not in data:
                return Err(ValueError(f"Missing '{key}'"))
        try:
            close_val = float(data["close"])
            upper_val = float(data["upper_band"])
            lower_val = float(data["lower_band"])
            atr = float(data["atr"])

            if math.isnan(close_val) or math.isnan(upper_val) or math.isnan(lower_val):
                return Ok(
                    {
                        "timestamp": data["timestamp"],
                        "symbol": target_symbol,
                        "action": "NEUTRAL",
                        "position": 0,
                        "strength": 0.0,
                        "atr": atr,
                    }
                )

            mid = (upper_val + lower_val) / 2.0
            if close_val > upper_val:
                raw = 1
            elif close_val < lower_val:
                raw = -1
            elif math.isclose(close_val, mid, rel_tol=1e-6):
                raw = 0
            else:
                raw = None

            if not config.allow_short and raw == -1:
                raw = None

            strength = 0.0 if (atr <= 1e-8 or math.isnan(atr)) else (config.target_risk_pct / (atr / close_val))

            if raw == 1:
                act = "BUY"
            elif raw == -1:
                act = "SELL"
            elif raw == 0:
                act = "EXIT"
            else:
                act = "NEUTRAL"

            # SURGERY: Add 'position', remove 'close'
            return Ok(
                {
                    "timestamp": data["timestamp"],
                    "symbol": target_symbol,
                    "action": act,
                    "position": int(raw) if raw is not None else 0,
                    "strength": strength,
                    "atr": atr,
                }
            )
        except Exception as e:
            return Err(ValueError(f"Live tick failure: {e}"))
    else:
        return Err(ValueError("Unsupported data type."))


__all__ = ["apply_momentum_tactics_strict"]
