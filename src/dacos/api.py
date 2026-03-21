"""
This module exposes the main 4 endpoints for the Dacos library.
No mathematical business logic is allowed here.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import polars as pl

from dacos.config import StatArbConfig, TSMConfig

# Import Engines
from dacos.paradigms import compute_pairs_zscore

# Import Tactics
from dacos.paradigms.stat_arb.tactics import apply_mean_reversion_tactics_strict
from dacos.paradigms.tsm.engine import compute_tsm_indicators
from dacos.paradigms.tsm.tactics import apply_momentum_tactics_strict
from dacos.protocols import DataFrame
from dacos.utils import Err, Ok, Result

# Asumsi fungsi ingestion tersedia (sesuaikan dengan struktur file Anda)
# from dacos.core.data import ingest_silver_data

# ============================================================================
# EXECUTIVE LOGGER CONFIGURATION
# ============================================================================
logger = logging.getLogger(__name__)


# ============================================================================
# PILAR 2: THE VECTORIZED RAILWAY (RESEARCH MODE)
# ============================================================================

def run_stat_arb_research(
    aligned_data: DataFrame, # Data yang sudah melalui ingest_silver_data & sejajar
    target_symbol: str,
    anchor_symbol: str,
    hedge_ratio_beta: float,
    config: StatArbConfig | None = None,
) -> Result[DataFrame, ValueError]:
    """
    (Mode 1) Executes a full vectorized backtest pipeline for Statistical Arbitrage.
    Args:
        aligned_data: Polars DataFrame containing aligned 'target' and 'anchor' price columns.
        target_symbol: The coin being traded (Y).
        anchor_symbol: The hedge coin (X).
        hedge_ratio_beta: The cointegration multiplier binding X to Y.
        config: Immutable StatArb configuration. Defaults to StatArbConfig().

    Returns:
        Ok(DataFrame) matching STAT_ARB_SIGNAL_SCHEMA, or Err(ValueError).
    """
    config = config or StatArbConfig()

    try:
        logger.info(f"START [StatArb Research]: Evaluating {target_symbol} vs {anchor_symbol}")

        # 1. Stasiun Mesin (Engines)
        logger.info("--> Entering Engine Station: Computing Z-Scores...")
        engine_res = compute_pairs_zscore(
            aligned_data=aligned_data,
            target_column=target_symbol,
            anchor_column=anchor_symbol,
            hedge_ratio_beta=hedge_ratio_beta,
            z_score_rolling_window=config.z_window
        )
        if engine_res.is_err():
            logger.error(f"Engine Failure: {engine_res.unwrap_err()}")
            return engine_res

        # 2. Stasiun Taktik & Bea Cukai (Tactics & Schema Enforcement)
        logger.info("--> Entering Tactics Station: Translating continuous metrics to discrete signals...")
        tactics_res = apply_mean_reversion_tactics_strict(
            data=engine_res.unwrap(),
            symbol=target_symbol,
            config=config
        )
        if tactics_res.is_err():
            logger.error(f"Tactics Failure: {tactics_res.unwrap_err()}")
            return tactics_res

        logger.info(f"SUCCESS [StatArb Research]: Pipeline completed for {target_symbol}.")
        return Ok(tactics_res.unwrap())

    except Exception as e:
        logger.error(f"PANIC [StatArb Research]: Unhandled exception breached the pipeline: {e}")
        return Err(ValueError(f"StatArb Pipeline Panic: {e}"))


def run_tsm_research(
    silver_data: DataFrame, # Data yang sudah melalui ingest_silver_data
    target_symbol: str,
    config: TSMConfig | None = None,
) -> Result[DataFrame, ValueError]:
    """
    (Mode 1) Executes a full vectorized backtest pipeline for Time Series Momentum (CTA).

    Args:
        silver_data: Polars DataFrame containing standard OHLCV columns.
        target_symbol: The asset being evaluated.
        config: Immutable TSM configuration. Defaults to TSMConfig().

    Returns:
        Ok(DataFrame) matching TSM_SIGNAL_SCHEMA, or Err(ValueError).
    """
    config = config or TSMConfig()

    try:
        logger.info(f"START [TSM Research]: Evaluating {target_symbol}")

        # 1. Stasiun Mesin (Engines)
        logger.info("--> Entering Engine Station: Computing Donchian Channels & ATR...")
        engine_res = compute_tsm_indicators(
            data=silver_data,
            atr_window=config.atr_window,
            donchian_window=config.donchian_window
        )
        if engine_res.is_err():
            logger.error(f"Engine Failure: {engine_res.unwrap_err()}")
            return engine_res

        # 2. Stasiun Taktik & Bea Cukai (Tactics & Schema Enforcement)
        logger.info("--> Entering Tactics Station: Applying Breakout & Risk Parity sizing...")
        tactics_res = apply_momentum_tactics_strict(
            data=engine_res.unwrap(),
            target_symbol=target_symbol,
            config=config
        )
        if tactics_res.is_err():
            logger.error(f"Tactics Failure: {tactics_res.unwrap_err()}")
            return tactics_res

        logger.info(f"SUCCESS [TSM Research]: Pipeline completed for {target_symbol}.")
        return Ok(tactics_res.unwrap())

    except Exception as e:
        logger.error(f"PANIC [TSM Research]: Unhandled exception breached the pipeline: {e}")
        return Err(ValueError(f"TSM Pipeline Panic: {e}"))


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
    """
    (Mode 2) High-speed evaluation of a single live tick/buffer for Statistical Arbitrage.
    Bypasses Ingestion and Cointegration Testing.

    Args:
        live_buffer: Small DataFrame or Dict containing the recent lookback window.
        target_symbol: The coin being traded (Y).
        anchor_symbol: The hedge coin (X).
        hedge_ratio_beta: The pre-calculated beta from the research phase.
        config: Immutable StatArb configuration.

    Returns:
        Ok(Dict) containing exactly one actionable signal matching STAT_ARB_SIGNAL_SCHEMA.
    """


    config = config or StatArbConfig()

    try:
        # SURGERY: Mesin StatArb Polars mewajibkan DataFrame. Casting dict (20 rows) membutuhkan < 1ms.
        df_buffer = pl.DataFrame(live_buffer) if isinstance(live_buffer, dict) else live_buffer

        engine_step = compute_pairs_zscore(
            aligned_data=df_buffer,
            target_column=target_symbol,
            anchor_column=anchor_symbol,
            hedge_ratio_beta=hedge_ratio_beta,
            z_score_rolling_window=config.z_window
        )
        if engine_step.is_err():
            return engine_step

        # Ambil baris terujung (Latest Tick) dan suapkan ke Tactics Mode 2 (Dict)
        latest_tick = engine_step.unwrap().row(-1, named=True)
        return apply_mean_reversion_tactics_strict(
            data=latest_tick,
            symbol=target_symbol,
            config=config
        )

    except Exception as e:
        logger.error(f"[STAT-ARB | LIVE | {target_symbol}] Panic: {e}")
        return Err(ValueError(f"StatArb Live Panic: {str(e)}"))


def evaluate_tsm_live(
    live_buffer: DataFrame | dict[str, Any],
    target_symbol: str,
    config: TSMConfig | None = None,
) -> Result[dict[str, Any], ValueError]:
    """(Mode 2) High-speed evaluation of a single live tick/buffer for Time Series Momentum."""
    config = config or TSMConfig()

    try:
        engine_step = compute_tsm_indicators(
            data=live_buffer,
            atr_window=config.atr_window,
            donchian_window=config.donchian_window
        )
        if engine_step.is_err():
            return engine_step

        engine_data = engine_step.unwrap()

        # SURGERY: Rekonstruksi Dict jika data dari Numba (yang tidak membawa timestamp)
        if isinstance(live_buffer, dict):
            # Ambil elemen terakhir dari array timestamp/close
            ts = live_buffer["timestamp"][-1] if isinstance(live_buffer["timestamp"], (list, np.ndarray)) else live_buffer["timestamp"]
            cl = live_buffer["close"][-1] if isinstance(live_buffer["close"], (list, np.ndarray)) else live_buffer["close"]

            engine_data["timestamp"] = ts
            engine_data["close"] = cl
            latest_tick = engine_data
        else:
            latest_tick = engine_data.row(-1, named=True)

        return apply_momentum_tactics_strict(
            data=latest_tick,
            target_symbol=target_symbol,
            config=config
        )

    except Exception as e:
        logger.error(f"[TSM | LIVE | {target_symbol}] Panic: {e}")
        return Err(ValueError(f"TSM Live Panic: {str(e)}"))


__all__ = [
    "run_stat_arb_research",
    "run_tsm_research",
    "evaluate_stat_arb_live",
    "evaluate_tsm_live",
]
