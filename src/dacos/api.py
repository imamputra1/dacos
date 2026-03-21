"""
DACOS PUBLIC API (THE CONDUCTOR)
Location: src/dacos/api.py
Paradigm: Separation of Concerns, Railway Oriented Programming (ROP), 100% Monadic Boundary.
"""

from __future__ import annotations

import logging
from typing import Any, cast

import numpy as np
import polars as pl

from dacos.config import StatArbConfig, TSMConfig
from dacos.paradigms import (
    apply_mean_reversion_tactics_strict,
    apply_momentum_tactics_strict,
    compute_pairs_zscore,
    compute_tsm_indicators,
)
from dacos.protocols import DataFrame
from dacos.utils import Err, Ok, Result

logger = logging.getLogger(__name__)

# ============================================================================
# PILAR 1: THE VECTORIZED RAILWAY (RESEARCH MODE)
# ============================================================================

def run_stat_arb_research(
    aligned_data: DataFrame,
    target_symbol: str,
    anchor_symbol: str,
    hedge_ratio_beta: float,
    config: StatArbConfig | None = None,
) -> Result[DataFrame, ValueError]:
    config = config or StatArbConfig()
    prefix = f"[STAT-ARB | RESEARCH | {target_symbol}]"

    if not isinstance(aligned_data, pl.DataFrame) or aligned_data.is_empty():
        return Err(ValueError(f"{prefix} API Rejected: Empty or invalid DataFrame provided."))

    try:
        engine_step = compute_pairs_zscore(
            aligned_data=aligned_data,
            target_column=target_symbol,
            anchor_column=anchor_symbol,
            hedge_ratio_beta=hedge_ratio_beta,
            z_score_rolling_window=config.z_window
        )
        if engine_step.is_err():
            return Err(engine_step.unwrap_err())  # Propagate explicitly

        tactics_step = apply_mean_reversion_tactics_strict(
            data=engine_step.unwrap(),
            symbol=target_symbol,
            config=config
        )
        if tactics_step.is_err():
            return Err(tactics_step.unwrap_err())

        # FIX MYPY: Beri tahu MyPy secara eksplisit bahwa hasilnya adalah DataFrame
        final_df = cast(pl.DataFrame, tactics_step.unwrap())
        return Ok(final_df)

    except Exception as e:
        logger.exception(f"{prefix} UNHANDLED PANIC.")
        return Err(ValueError(f"StatArb Pipeline Panic: {str(e)}"))


def run_tsm_research(
    silver_data: DataFrame,
    target_symbol: str,
    config: TSMConfig | None = None,
) -> Result[DataFrame, ValueError]:
    config = config or TSMConfig()
    prefix = f"[TSM | RESEARCH | {target_symbol}]"

    if not isinstance(silver_data, pl.DataFrame) or silver_data.is_empty():
        return Err(ValueError(f"{prefix} API Rejected: Empty or invalid DataFrame provided."))

    try:
        engine_step = compute_tsm_indicators(
            data=silver_data,
            atr_window=config.atr_window,
            donchian_window=config.donchian_window
        )
        if engine_step.is_err():
            return Err(engine_step.unwrap_err())

        tactics_step = apply_momentum_tactics_strict(
            data=engine_step.unwrap(),
            target_symbol=target_symbol,
            config=config
        )
        if tactics_step.is_err():
            return Err(tactics_step.unwrap_err())

        # FIX MYPY: Beri tahu MyPy secara eksplisit bahwa hasilnya adalah DataFrame
        final_df = cast(pl.DataFrame, tactics_step.unwrap())
        return Ok(final_df)

    except Exception as e:
        logger.exception(f"{prefix} UNHANDLED PANIC.")
        return Err(ValueError(f"TSM Pipeline Panic: {str(e)}"))


# ============================================================================
# PILAR 2: THE LIVE TICK PIPELINE (EXECUTION MODE)
# ============================================================================

def evaluate_stat_arb_live(
    live_buffer: DataFrame | dict[str, Any],
    target_symbol: str,
    anchor_symbol: str,
    hedge_ratio_beta: float,
    config: StatArbConfig | None = None,
) -> Result[dict[str, Any], ValueError]:
    config = config or StatArbConfig()

    try:
        df_buffer = pl.DataFrame(live_buffer) if isinstance(live_buffer, dict) else live_buffer

        engine_step = compute_pairs_zscore(
            aligned_data=df_buffer,
            target_column=target_symbol,
            anchor_column=anchor_symbol,
            hedge_ratio_beta=hedge_ratio_beta,
            z_score_rolling_window=config.z_window
        )
        if engine_step.is_err():
            return Err(engine_step.unwrap_err())

        latest_tick = engine_step.unwrap().row(-1, named=True)
        tactics_res = apply_mean_reversion_tactics_strict(
            data=latest_tick,
            symbol=target_symbol,
            config=config
        )
        if tactics_res.is_err():
            return Err(tactics_res.unwrap_err())

        # FIX MYPY: Casting eksplisit ke Dict
        final_dict = cast(dict[str, Any], tactics_res.unwrap())
        return Ok(final_dict)

    except Exception as e:
        return Err(ValueError(f"StatArb Live Panic: {str(e)}"))


def evaluate_tsm_live(
    live_buffer: DataFrame | dict[str, Any],
    target_symbol: str,
    config: TSMConfig | None = None,
) -> Result[dict[str, Any], ValueError]:
    config = config or TSMConfig()

    try:
        engine_step = compute_tsm_indicators(
            data=live_buffer,
            atr_window=config.atr_window,
            donchian_window=config.donchian_window
        )
        if engine_step.is_err():
            return Err(engine_step.unwrap_err())

        engine_data = engine_step.unwrap()

        # FIX MYPY: Evaluasi `engine_data` secara eksplisit agar MyPy paham itu Dict atau DataFrame
        if isinstance(engine_data, dict):
            # Casting aman karena jika masuk sini, live_buffer pasti dict yang punya list/ndarray
            ts_list = cast(Any, live_buffer)["timestamp"]
            cl_list = cast(Any, live_buffer)["close"]

            ts = ts_list[-1] if isinstance(ts_list, list | np.ndarray) else ts_list
            cl = cl_list[-1] if isinstance(cl_list, list | np.ndarray) else cl_list

            engine_data["timestamp"] = ts
            engine_data["close"] = cl
            latest_tick = engine_data
        elif isinstance(engine_data, pl.DataFrame):
            latest_tick = engine_data.row(-1, named=True)
        else:
            return Err(ValueError("Engine returned invalid data type."))

        tactics_res = apply_momentum_tactics_strict(
            data=latest_tick,
            target_symbol=target_symbol,
            config=config
        )

        if tactics_res.is_err():
            return Err(tactics_res.unwrap_err())

        # FIX MYPY: Casting eksplisit ke Dict
        final_dict = cast(dict[str, Any], tactics_res.unwrap())
        return Ok(final_dict)

    except Exception as e:
        return Err(ValueError(f"TSM Live Panic: {str(e)}"))

__all__ = [
    "run_stat_arb_research",
    "run_tsm_research",
    "evaluate_stat_arb_live",
    "evaluate_tsm_live",
]
