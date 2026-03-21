from __future__ import annotations

import numpy as np
import polars as pl

from dacos.laws.volatility import compute_atr_safe, compute_donchian_channels_safe
from dacos.protocols import DataFrame
from dacos.utils import Err, Ok, Result


def compute_tsm_indicators(
    data: DataFrame | dict[str, np.ndarray],
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    atr_window: int = 14,
    donchian_window: int = 20,
) -> Result[DataFrame | dict[str, float], ValueError]:
    """
    Computes Time Series Momentum indicators (ATR and Donchian Channels).
    Supports Vectorized Research (Polars) and Live Tick-by-Tick buffers (Dict of Numpy Arrays).

    Args:
        data: Polars DataFrame or Dictionary of Numpy arrays containing historical window.
        high_col: Name of the high price column/key.
        low_col: Name of the low price column/key.
        close_col: Name of the close price column/key.
        atr_window: Lookback period for Average True Range.
        donchian_window: Lookback period for Donchian Breakout Channels.

    Returns:
        Ok(DataFrame) with new columns appended, or Ok(Dict) with latest tick values.
    """
    if atr_window < 2 or donchian_window < 2:
        return Err(ValueError("Windows must be at least 2."))

    # ============================================================================
    # MODE 1: VECTORIZED RESEARCH (POLARS DATAFRAME)
    # ============================================================================
    if isinstance(data, pl.DataFrame):
        if len(data) == 0:
            return Err(ValueError("TSM Engine received an empty DataFrame."))

        for col in [high_col, low_col, close_col]:
            if col not in data.columns:
                return Err(ValueError(f"Missing required column: '{col}'."))

        try:
            # 1. Zero-Copy Extraction to Numpy (Hardware Handoff)
            high_np = data.get_column(high_col).cast(pl.Float64).to_numpy(allow_copy=True)
            low_np = data.get_column(low_col).cast(pl.Float64).to_numpy(allow_copy=True)
            close_np = data.get_column(close_col).cast(pl.Float64).to_numpy(allow_copy=True)

            # 2. Numba Execution
            atr_result = compute_atr_safe(high_np, low_np, close_np, atr_window)
            if atr_result.is_err():
                return Err(ValueError(f"ATR Kernel Panic: {atr_result.unwrap_err()}"))

            dc_result = compute_donchian_channels_safe(high_np, low_np, donchian_window)
            if dc_result.is_err():
                return Err(ValueError(f"Donchian Kernel Panic: {dc_result.unwrap_err()}"))

            atr_array = atr_result.unwrap()
            upper_array, lower_array = dc_result.unwrap()

            # 3. Re-attachment & NaN Propagation Control (Schema Preservation)
            df_final = data.with_columns(
                [
                    pl.Series("atr", atr_array).fill_nan(0.0),
                    pl.Series("upper_band", upper_array),
                    pl.Series("lower_band", lower_array),
                ]
            )

            return Ok(df_final)

        except Exception as e:
            return Err(ValueError(f"Vectorized TSM computation failed: {e}"))

    # ============================================================================
    # MODE 2: LIVE TICK-BY-TICK (DICTIONARY BUFFER)
    # ============================================================================
    elif isinstance(data, dict):
        for col in [high_col, low_col, close_col]:
            if col not in data:
                return Err(ValueError(f"Missing required key in live buffer: '{col}'."))

        try:
            # 1. Type Enforcement
            high_np = np.asarray(data[high_col], dtype=np.float64)
            low_np = np.asarray(data[low_col], dtype=np.float64)
            close_np = np.asarray(data[close_col], dtype=np.float64)

            max_window = max(atr_window, donchian_window)
            if len(high_np) < max_window:
                return Err(ValueError(f"Insufficient live buffer. Need at least {max_window} ticks."))

            # 2. Numba Execution
            atr_result = compute_atr_safe(high_np, low_np, close_np, atr_window)
            if atr_result.is_err():
                return Err(ValueError(f"Live ATR Kernel Panic: {atr_result.unwrap_err()}"))

            dc_result = compute_donchian_channels_safe(high_np, low_np, donchian_window)
            if dc_result.is_err():
                return Err(ValueError(f"Live Donchian Kernel Panic: {dc_result.unwrap_err()}"))

            atr_array = atr_result.unwrap()
            upper_array, lower_array = dc_result.unwrap()

            # 3. Output Extraction (Latest Tick Only)
            latest_atr = float(atr_array[-1])

            tick_final = {
                "atr": 0.0 if np.isnan(latest_atr) else latest_atr,
                "upper_band": float(upper_array[-1]),
                "lower_band": float(lower_array[-1]),
            }

            return Ok(tick_final)

        except Exception as e:
            return Err(ValueError(f"Live buffer TSM computation failed: {e}"))

    else:
        return Err(ValueError(f"Unsupported data type: {type(data)}. Expected DataFrame or Dict."))


__all__ = [
    "compute_tsm_indicators",
]
