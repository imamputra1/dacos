from __future__ import annotations

import math
from typing import Any

import polars as pl

from dacos.protocols import DataFrame
from dacos.utils import Err, Ok, Result


def apply_momentum_tactics_strict(
    data: DataFrame | dict[str, Any],
    target_symbol: str,
    target_risk_pct: float = 0.01,
    allow_short: bool = True,
) -> Result[DataFrame | dict[str, Any], ValueError]:
    """
    Evaluates TSM indicators to generate discrete trading actions (BUY, SELL, EXIT, NEUTRAL)
    along with Risk Parity-adjusted position sizing (strength).
    
    Args:
        data: Polars DataFrame (Research) or Dictionary (Live Tick) with indicator columns.
        target_symbol: Identifier for the traded asset.
        target_risk_pct: The percentage of capital to risk per trade (e.g., 0.01 for 1%).
        allow_short: If False, blocks all SELL (-1) signals (Spot Market Armor).
        
    Returns:
        Ok(DataFrame | Dict) containing strictly [timestamp, symbol, action, strength, close, atr].
    """

    # ============================================================================
    # MODE 1: VECTORIZED RESEARCH (POLARS DATAFRAME)
    # ============================================================================
    if isinstance(data, pl.DataFrame):
        if len(data) == 0:
            return Err(ValueError("TSM Tactics received an empty DataFrame."))

        required_columns = ["timestamp", "close", "upper_band", "lower_band", "atr"]
        for col in required_columns:
            if col not in data.columns:
                return Err(ValueError(f"Missing required column for TSM Tactics: '{col}'."))

        try:
            # 1. The Directional Matrix (Breakout Logic)
            expr_midline = (pl.col("upper_band") + pl.col("lower_band")) / 2.0

            expr_raw_position = (
                pl.when(pl.col("close") > pl.col("upper_band")).then(1)
                .when(pl.col("close") < pl.col("lower_band")).then(-1)
                # Stateless midline touch exit (exact or crossed, mapped strictly to Neutral otherwise)
                .when(pl.col("close") == expr_midline).then(0)
                .otherwise(None)
            )

            # 2. The Spot Armor
            if not allow_short:
                expr_raw_position = pl.when(expr_raw_position == -1).then(None).otherwise(expr_raw_position)

            # 3. Risk Parity Engine (Position Sizing with Flatline Protection)
            expr_relative_volatility = pl.col("atr") / pl.col("close")
            expr_strength = pl.when(pl.col("atr") <= 1e-8).then(0.0).otherwise(
                target_risk_pct / expr_relative_volatility
            ).fill_nan(0.0)

            # 4. Schema Alignment (Orca Bot Dialect)
            expr_action_string = (
                pl.when(expr_raw_position == 1).then(pl.lit("BUY"))
                .when(expr_raw_position == -1).then(pl.lit("SELL"))
                .when(expr_raw_position == 0).then(pl.lit("EXIT"))
                .otherwise(pl.lit("NEUTRAL"))
            )

            # 5. Selection and Output
            df_final = data.with_columns([
                pl.lit(target_symbol).alias("symbol"),
                expr_action_string.alias("action"),
                expr_strength.alias("strength")
            ]).select([
                "timestamp", "symbol", "action", "strength", "close", "atr"
            ])

            return Ok(df_final)

        except Exception as e:
            return Err(ValueError(f"Vectorized TSM tactics evaluation failed: {e}"))

    # ============================================================================
    # MODE 2: LIVE TICK-BY-TICK (DICTIONARY)
    # ============================================================================
    elif isinstance(data, dict):
        required_keys = ["timestamp", "close", "upper_band", "lower_band", "atr"]
        for key in required_keys:
            if key not in data:
                return Err(ValueError(f"Missing required key in tick data: '{key}'."))

        try:
            close_px = float(data["close"])
            upper_band = float(data["upper_band"])
            lower_band = float(data["lower_band"])
            atr = float(data["atr"])

            # Guard against propagating NaNs from the warmup phase
            if math.isnan(close_px) or math.isnan(upper_band) or math.isnan(lower_band):
                return Ok({
                    "timestamp": data["timestamp"],
                    "symbol": target_symbol,
                    "action": "NEUTRAL",
                    "strength": 0.0,
                    "close": close_px,
                    "atr": atr
                })

            # 1. The Directional Matrix
            midline = (upper_band + lower_band) / 2.0

            if close_px > upper_band:
                raw_position = 1
            elif close_px < lower_band:
                raw_position = -1
            elif math.isclose(close_px, midline, rel_tol=1e-6):
                raw_position = 0
            else:
                raw_position = None

            # 2. The Spot Armor
            if not allow_short and raw_position == -1:
                raw_position = None

            # 3. Risk Parity Engine
            if atr <= 1e-8 or math.isnan(atr):
                strength = 0.0
            else:
                relative_volatility = atr / close_px
                strength = target_risk_pct / relative_volatility

            # 4. Schema Alignment
            if raw_position == 1:
                action_str = "BUY"
            elif raw_position == -1:
                action_str = "SELL"
            elif raw_position == 0:
                action_str = "EXIT"
            else:
                action_str = "NEUTRAL"

            # 5. Output
            tick_final = {
                "timestamp": data["timestamp"],
                "symbol": target_symbol,
                "action": action_str,
                "strength": strength,
                "close": close_px,
                "atr": atr
            }

            return Ok(tick_final)

        except Exception as e:
            return Err(ValueError(f"Live tick TSM tactics evaluation failed: {e}"))

    else:
        return Err(ValueError(f"Unsupported data type for TSM tactics: {type(data)}. Expected DataFrame or Dict."))


__all__ = [
    "apply_momentum_tactics_strict",
]
