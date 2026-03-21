from __future__ import annotations

import numpy as np
import polars as pl
from dacos.config import StatArbConfig
from dacos.paradigms.stat_arb.tactics import apply_mean_reversion_tactics_strict

# ============================================================================
# KUADRAN 1: THE CORE LOGIC (Akurasi Threshold)
# ============================================================================


def test_long_entry_signal() -> None:
    """1.1: Z-Score < -Entry -> BUY"""
    config = StatArbConfig(entry_z=2.0, exit_z=0.5)

    # Test Vectorized (Polars)
    df = pl.DataFrame({"timestamp": [1], "z_score": [-2.5], "spread": [0.1]})
    res_df = apply_mean_reversion_tactics_strict(df, "BTC", config=config).unwrap()
    assert res_df["action"][0] == "BUY"
    assert res_df["position"][0] == 1

    # Test Live Tick (Dict)
    tick = {"timestamp": 1, "z_score": -2.5, "spread": 0.1}
    res_tick = apply_mean_reversion_tactics_strict(tick, "BTC", config=config).unwrap()
    assert res_tick["action"] == "BUY"
    assert res_tick["position"] == 1


def test_short_entry_signal() -> None:
    """1.2: Z-Score > Entry -> SELL"""
    config = StatArbConfig(entry_z=2.0, exit_z=0.5)

    df = pl.DataFrame({"timestamp": [1], "z_score": [3.0], "spread": [0.1]})
    res_df = apply_mean_reversion_tactics_strict(df, "BTC", config=config).unwrap()
    assert res_df["action"][0] == "SELL"
    assert res_df["position"][0] == -1

    tick = {"timestamp": 1, "z_score": 3.0, "spread": 0.1}
    res_tick = apply_mean_reversion_tactics_strict(tick, "BTC", config=config).unwrap()
    assert res_tick["action"] == "SELL"
    assert res_tick["position"] == -1


def test_exit_signal_inside_band() -> None:
    """1.3: |Z-Score| <= Exit -> EXIT"""
    config = StatArbConfig(entry_z=2.0, exit_z=0.5)

    df = pl.DataFrame({"timestamp": [1, 2], "z_score": [0.2, -0.5], "spread": [0.1, 0.1]})
    res_df = apply_mean_reversion_tactics_strict(df, "BTC", config=config).unwrap()
    assert res_df["action"][0] == "EXIT"
    assert res_df["action"][1] == "EXIT"
    assert res_df["position"][0] == 0

    tick = {"timestamp": 1, "z_score": 0.2, "spread": 0.1}
    res_tick = apply_mean_reversion_tactics_strict(tick, "BTC", config=config).unwrap()
    assert res_tick["action"] == "EXIT"
    assert res_tick["position"] == 0


def test_neutral_zone_ignore() -> None:
    """1.4: Exit < |Z-Score| <= Entry -> NEUTRAL (Area abu-abu)"""
    config = StatArbConfig(entry_z=2.0, exit_z=0.5)

    df = pl.DataFrame({"timestamp": [1], "z_score": [1.5], "spread": [0.1]})
    res_df = apply_mean_reversion_tactics_strict(df, "BTC", config=config).unwrap()
    assert res_df["action"][0] == "NEUTRAL"
    assert res_df["position"][0] == 0

    tick = {"timestamp": 1, "z_score": -1.5, "spread": 0.1}
    res_tick = apply_mean_reversion_tactics_strict(tick, "BTC", config=config).unwrap()
    assert res_tick["action"] == "NEUTRAL"
    assert res_tick["position"] == 0


# ============================================================================
# KUADRAN 2: THE SPOT ARMOR (Proteksi Short-Selling)
# ============================================================================


def test_spot_market_blocks_short() -> None:
    """2.1: allow_short=False MUST override SELL to NEUTRAL."""
    config_spot = StatArbConfig(entry_z=2.0, exit_z=0.5, allow_short=False)

    # DF
    df = pl.DataFrame({"timestamp": [1], "z_score": [3.0], "spread": [0.1]})
    res_df = apply_mean_reversion_tactics_strict(df, "BTC", config=config_spot).unwrap()
    assert res_df["action"][0] == "NEUTRAL"
    assert res_df["position"][0] == 0

    # Dict
    tick = {"timestamp": 1, "z_score": 3.0, "spread": 0.1}
    res_tick = apply_mean_reversion_tactics_strict(tick, "BTC", config=config_spot).unwrap()
    assert res_tick["action"] == "NEUTRAL"
    assert res_tick["position"] == 0


def test_spot_market_allows_long() -> None:
    """2.2: allow_short=False MUST still allow BUY."""
    config_spot = StatArbConfig(entry_z=2.0, exit_z=0.5, allow_short=False)

    tick = {"timestamp": 1, "z_score": -3.0, "spread": 0.1}
    res_tick = apply_mean_reversion_tactics_strict(tick, "BTC", config=config_spot).unwrap()
    assert res_tick["action"] == "BUY"
    assert res_tick["position"] == 1


def test_futures_allows_short() -> None:
    """2.3: allow_short=True permits SELL."""
    config_futures = StatArbConfig(entry_z=2.0, exit_z=0.5, allow_short=True)

    tick = {"timestamp": 1, "z_score": 3.0, "spread": 0.1}
    res_tick = apply_mean_reversion_tactics_strict(tick, "BTC", config=config_futures).unwrap()
    assert res_tick["action"] == "SELL"
    assert res_tick["position"] == -1


# ============================================================================
# KUADRAN 3: MODE 2 - LIVE TICK (Robustness)
# ============================================================================


def test_live_dict_schema_output() -> None:
    """3.1: Output schema must strictly match STAT_ARB_SIGNAL_SCHEMA."""
    tick = {"timestamp": 1690000000, "z_score": 2.5, "spread": 0.05, "extra_noise": "ignore_me"}
    res = apply_mean_reversion_tactics_strict(tick, "ETH").unwrap()

    # SURGERY: Memastikan kolom position dan spread ada
    expected_keys = {"timestamp", "symbol", "action", "position", "strength", "z_score", "spread"}
    assert set(res.keys()) == expected_keys
    assert res["symbol"] == "ETH"
    assert res["strength"] == 2.5
    assert res["position"] == -1


def test_live_dict_missing_keys() -> None:
    """3.2: Missing required keys must return Err."""
    tick_no_z = {"timestamp": 123, "spread": 0.05}
    assert apply_mean_reversion_tactics_strict(tick_no_z, "BTC").is_err()

    tick_no_spread = {"timestamp": 123, "z_score": 2.5}
    assert apply_mean_reversion_tactics_strict(tick_no_spread, "BTC").is_err()


def test_live_dict_type_safety() -> None:
    """3.3: Graceful degradation for None/NaN in live stream (returns Ok(NEUTRAL) instead of crashing)."""
    tick_none = {"timestamp": "now", "z_score": None, "spread": 0.05}
    res_none = apply_mean_reversion_tactics_strict(tick_none, "BTC").unwrap()
    assert res_none["action"] == "NEUTRAL"
    assert res_none["strength"] == 0.0
    assert res_none["position"] == 0

    tick_nan = {"timestamp": "now", "z_score": float("nan"), "spread": 0.05}
    res_nan = apply_mean_reversion_tactics_strict(tick_nan, "BTC").unwrap()
    assert res_nan["action"] == "NEUTRAL"
    assert res_nan["strength"] == 0.0
    assert res_nan["position"] == 0


# ============================================================================
# KUADRAN 4: MODE 1 - VECTORIZED (Polars Integrity)
# ============================================================================


def test_polars_empty_dataframe() -> None:
    """4.1: Empty dataframe must return Err."""
    df = pl.DataFrame()
    assert apply_mean_reversion_tactics_strict(df, "SOL").is_err()


def test_polars_missing_columns() -> None:
    """4.2: Missing required columns must return Err."""
    df_no_z = pl.DataFrame({"timestamp": [1, 2], "spread": [0.1, 0.1]})
    assert apply_mean_reversion_tactics_strict(df_no_z, "SOL").is_err()

    df_no_spread = pl.DataFrame({"timestamp": [1, 2], "z_score": [1.0, 1.0]})
    assert apply_mean_reversion_tactics_strict(df_no_spread, "SOL").is_err()


def test_polars_schema_preservation() -> None:
    """4.3: Mass processing must preserve STAT_ARB_SIGNAL_SCHEMA exactly."""
    np.random.seed(42)
    n_rows = 10_000
    df = pl.DataFrame(
        {
            "timestamp": np.arange(n_rows),
            "z_score": np.random.randn(n_rows) * 3,
            "spread": np.random.randn(n_rows) * 0.1,
        }
    )

    res = apply_mean_reversion_tactics_strict(df, "BASKET_1")
    assert res.is_ok()

    out_df = res.unwrap()
    assert len(out_df) == n_rows

    # SURGERY: Memastikan kolom position dan spread terbawa, dan position bertipe Int8
    expected_cols = {"timestamp", "symbol", "action", "position", "strength", "z_score", "spread"}
    assert set(out_df.columns) == expected_cols
    assert out_df["position"].dtype == pl.Int8

    # Ensure Spot Armor defaults cleanly across mass data
    config_spot = StatArbConfig(allow_short=False)
    res_spot = apply_mean_reversion_tactics_strict(df, "BASKET_1", config=config_spot).unwrap()
    assert "SELL" not in res_spot["action"].to_list()
    assert -1 not in res_spot["position"].to_list()
