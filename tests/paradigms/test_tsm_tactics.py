from __future__ import annotations

import polars as pl
from dacos.config import TSMConfig
from dacos.paradigms.tsm.tactics import apply_momentum_tactics_strict
from numpy.testing import assert_allclose

# ============================================================================
# KUADRAN 1: THE DIRECTIONAL MATRIX (Akurasi Sinyal Breakout)
# ============================================================================


def test_bullish_breakout() -> None:
    """1.1: Close > Upper Band triggers BUY."""
    df = pl.DataFrame({"timestamp": [1], "close": [105.0], "upper_band": [100.0], "lower_band": [80.0], "atr": [2.0]})
    res = apply_momentum_tactics_strict(df, "BTC", config=TSMConfig()).unwrap()
    assert res["action"][0] == "BUY"
    assert res["position"][0] == 1


def test_bearish_breakout() -> None:
    """1.2: Close < Lower Band triggers SELL."""
    df = pl.DataFrame({"timestamp": [1], "close": [75.0], "upper_band": [100.0], "lower_band": [80.0], "atr": [2.0]})
    res = apply_momentum_tactics_strict(df, "BTC", config=TSMConfig()).unwrap()
    assert res["action"][0] == "SELL"
    assert res["position"][0] == -1


def test_inside_channel_neutral_and_exit() -> None:
    """1.3: Inside channel logic. Midline touch triggers EXIT, otherwise NEUTRAL."""
    config = TSMConfig()
    # Skenario 1: Sentuh persis garis tengah (Midline) -> EXIT
    df_exit = pl.DataFrame(
        {"timestamp": [1], "close": [90.0], "upper_band": [100.0], "lower_band": [80.0], "atr": [2.0]}
    )
    res_exit = apply_momentum_tactics_strict(df_exit, "BTC", config=config).unwrap()
    assert res_exit["action"][0] == "EXIT"
    assert res_exit["position"][0] == 0

    # Skenario 2: Melayang di dalam channel (Area Abu-abu) -> NEUTRAL
    df_neutral = pl.DataFrame(
        {"timestamp": [1], "close": [95.0], "upper_band": [100.0], "lower_band": [80.0], "atr": [2.0]}
    )
    res_neutral = apply_momentum_tactics_strict(df_neutral, "BTC", config=config).unwrap()
    assert res_neutral["action"][0] == "NEUTRAL"
    assert res_neutral["position"][0] == 0


def test_exact_touch_no_breakout() -> None:
    """1.4: Exact touch on upper/lower band is NOT a breakout -> NEUTRAL."""
    df = pl.DataFrame({"timestamp": [1], "close": [100.0], "upper_band": [100.0], "lower_band": [80.0], "atr": [2.0]})
    res = apply_momentum_tactics_strict(df, "BTC", config=TSMConfig()).unwrap()
    assert res["action"][0] == "NEUTRAL"
    assert res["position"][0] == 0


# ============================================================================
# KUADRAN 2: THE RISK PARITY ENGINE (Akurasi Sizing & Proteksi)
# ============================================================================


def test_normal_volatility_sizing() -> None:
    """2.1: Normal volatility (2%). Strength should be exactly 0.5."""
    tick = {"timestamp": 1, "close": 100.0, "upper_band": 105.0, "lower_band": 95.0, "atr": 2.0}
    res = apply_momentum_tactics_strict(tick, "BTC", config=TSMConfig(target_risk_pct=0.01)).unwrap()
    assert_allclose(res["strength"], 0.5)


def test_extreme_volatility_sizing() -> None:
    """2.2: Extreme volatility (10%). Strength shrinks dramatically to 0.1."""
    tick = {"timestamp": 1, "close": 100.0, "upper_band": 110.0, "lower_band": 90.0, "atr": 10.0}
    res = apply_momentum_tactics_strict(tick, "BTC", config=TSMConfig(target_risk_pct=0.01)).unwrap()
    assert_allclose(res["strength"], 0.1)


def test_absolute_flatline_sizing() -> None:
    """2.3: CRITICAL! Zero volatility (ATR=0) must yield 0.0 strength without Division by Zero crash."""
    tick = {"timestamp": 1, "close": 100.0, "upper_band": 100.0, "lower_band": 100.0, "atr": 0.0}
    res = apply_momentum_tactics_strict(tick, "BTC", config=TSMConfig(target_risk_pct=0.01)).unwrap()
    assert_allclose(res["strength"], 0.0)


# ============================================================================
# KUADRAN 3: THE SPOT ARMOR (Proteksi Tipe Bursa)
# ============================================================================


def test_spot_blocks_bearish_breakout() -> None:
    """3.1: allow_short=False MUST override bearish breakouts to NEUTRAL."""
    df = pl.DataFrame({"timestamp": [1], "close": [75.0], "upper_band": [100.0], "lower_band": [80.0], "atr": [2.0]})
    res = apply_momentum_tactics_strict(df, "BTC", config=TSMConfig(allow_short=False)).unwrap()
    assert res["action"][0] == "NEUTRAL"
    assert res["position"][0] == 0


def test_spot_allows_bullish_breakout() -> None:
    """3.2: allow_short=False permits bullish breakouts (BUY)."""
    df = pl.DataFrame({"timestamp": [1], "close": [105.0], "upper_band": [100.0], "lower_band": [80.0], "atr": [2.0]})
    res = apply_momentum_tactics_strict(df, "BTC", config=TSMConfig(allow_short=False)).unwrap()
    assert res["action"][0] == "BUY"
    assert res["position"][0] == 1


def test_futures_allows_both_sides() -> None:
    """3.3: allow_short=True permits bearish breakouts (SELL)."""
    df = pl.DataFrame({"timestamp": [1], "close": [75.0], "upper_band": [100.0], "lower_band": [80.0], "atr": [2.0]})
    res = apply_momentum_tactics_strict(df, "BTC", config=TSMConfig(allow_short=True)).unwrap()
    assert res["action"][0] == "SELL"
    assert res["position"][0] == -1


# ============================================================================
# KUADRAN 4: DUAL-MODE & SCHEMA CONSERVATION
# ============================================================================


def test_dual_mode_symmetry_tsm() -> None:
    """4.1: Proves Polars mode and Dict mode output perfectly identical signals and sizing."""
    config = TSMConfig(target_risk_pct=0.02)
    df = pl.DataFrame({"timestamp": [1], "close": [105.0], "upper_band": [100.0], "lower_band": [80.0], "atr": [2.0]})
    tick = {"timestamp": 1, "close": 105.0, "upper_band": 100.0, "lower_band": 80.0, "atr": 2.0}

    res_df = apply_momentum_tactics_strict(df, "SOL", config=config).unwrap()
    res_tick = apply_momentum_tactics_strict(tick, "SOL", config=config).unwrap()

    assert res_df["action"][0] == res_tick["action"]
    assert res_df["position"][0] == res_tick["position"]
    assert_allclose(res_df["strength"][0], res_tick["strength"])


def test_schema_output_trimming() -> None:
    """4.2: Output must be rigorously trimmed down to TSM_SIGNAL_SCHEMA."""
    data = {
        "timestamp": [1],
        "close": [105.0],
        "upper_band": [100.0],
        "lower_band": [80.0],
        "atr": [2.0],
        "noise_volume": [5000],
        "noise_vwap": [102.0],
        "noise_ma": [101.0],
    }
    df = pl.DataFrame(data)
    res = apply_momentum_tactics_strict(df, "SOL", config=TSMConfig()).unwrap()

    # SURGERY: 'close' harus hilang, 'position' harus ada
    expected_columns = {"timestamp", "symbol", "action", "position", "strength", "atr"}
    assert set(res.columns) == expected_columns
    assert "noise_volume" not in res.columns
    assert "close" not in res.columns
    assert res["position"].dtype == pl.Int8  # Tipe wajib dari Orca


def test_missing_required_keys() -> None:
    """4.3: Missing indicator columns trigger the Monadic Error Shield."""
    tick_incomplete = {"timestamp": 1, "close": 105.0}  # Missing bands and ATR
    res_tick = apply_momentum_tactics_strict(tick_incomplete, "SOL")
    assert res_tick.is_err()
    assert "Missing" in str(res_tick.unwrap_err())

    df_incomplete = pl.DataFrame({"timestamp": [1], "close": [105.0]})
    res_df = apply_momentum_tactics_strict(df_incomplete, "SOL")
    assert res_df.is_err()
    assert "Missing" in str(res_df.unwrap_err())
