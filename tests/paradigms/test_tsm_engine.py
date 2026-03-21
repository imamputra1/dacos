from __future__ import annotations

import numpy as np
import polars as pl
from numpy.testing import assert_allclose

from dacos.paradigms import compute_tsm_indicators

# ============================================================================
# KUADRAN 1: THE PHYSICS VALIDATION (Akurasi Matematis)
# ============================================================================

def test_donchian_uptrend_accuracy() -> None:
    """1.1: Upper band correctly tracks Higher Highs in an uptrend."""
    n = 100
    window = 20
    # Harga naik konstan +1 setiap baris
    highs = np.arange(10, 10 + n, dtype=float)
    df = pl.DataFrame({
        "high": highs,
        "low": highs - 2,
        "close": highs - 1
    })

    res = compute_tsm_indicators(df, donchian_window=window).unwrap()
    upper_band = res["upper_band"].to_numpy()

    # Setelah warmup, upper band di index i harus sama dengan high[i]
    assert_allclose(upper_band[window:], highs[window:])


def test_donchian_downtrend_accuracy() -> None:
    """1.2: Lower band correctly tracks Lower Lows in a downtrend."""
    n = 100
    window = 20
    # Harga turun konstan -1 setiap baris
    lows = np.arange(100, 100 - n, -1, dtype=float)
    df = pl.DataFrame({
        "high": lows + 2,
        "low": lows,
        "close": lows + 1
    })

    res = compute_tsm_indicators(df, donchian_window=window).unwrap()
    lower_band = res["lower_band"].to_numpy()

    # Setelah warmup, lower band di index i harus sama dengan low[i]
    assert_allclose(lower_band[window:], lows[window:])


def test_atr_constant_volatility() -> None:
    """1.3: ATR exactly matches a constant True Range without drift."""
    n = 100
    window = 14
    constant_tr = 5.0

    df = pl.DataFrame({
        "high": np.full(n, 105.0),
        "low": np.full(n, 100.0),
        "close": np.full(n, 102.5)
    })

    res = compute_tsm_indicators(df, atr_window=window).unwrap()
    atr = res["atr"].to_numpy()

    # ATR harus stabil di 5.0 setelah periode warmup
    assert_allclose(atr[window:], constant_tr)


def test_warmup_nan_preservation() -> None:
    """1.4: Ensures NaNs are preserved during warmup (No data dropping)."""
    df = pl.DataFrame({
        "high": np.random.rand(100) + 10,
        "low": np.random.rand(100) + 5,
        "close": np.random.rand(100) + 7
    })

    atr_win = 14
    dc_win = 20
    res = compute_tsm_indicators(df, atr_window=atr_win, donchian_window=dc_win).unwrap()

    # Donchian: 19 baris pertama (index 0 to 18) harus Null/NaN
    assert res["upper_band"][:dc_win - 1].is_nan().all()
    assert res["lower_band"][:dc_win - 1].is_nan().all()
    assert not np.isnan(res["upper_band"][dc_win - 1])

    # ATR: 13 baris pertama harus berisi 0.0 (karena fill_nan(0.0) di engine)
    # atau jika kita mengecek original logic, kita pastikan struktur ukurannya tetap utuh
    assert len(res) == 100


# ============================================================================
# KUADRAN 2: EXTREME MARKET ANOMALIES (Kekebalan Mesin)
# ============================================================================

def test_the_absolute_flatline() -> None:
    """2.1: CRITICAL! Machine must survive a completely frozen/flatline market."""
    n = 50
    df = pl.DataFrame({
        "high": np.full(n, 100.0),
        "low": np.full(n, 100.0),
        "close": np.full(n, 100.0)
    })

    res = compute_tsm_indicators(df, atr_window=14, donchian_window=20).unwrap()

    # Bands harus menyempit ke harga close (100.0)
    assert_allclose(res["upper_band"].to_numpy()[20:], 100.0)
    assert_allclose(res["lower_band"].to_numpy()[20:], 100.0)

    # ATR mutlak harus 0.0 karena tidak ada volatilitas sama sekali
    assert_allclose(res["atr"].to_numpy()[14:], 0.0)


def test_overnight_gap_up() -> None:
    """2.2: True Range must account for massive gaps between previous close and current open/high."""
    high = np.full(50, 105.0)
    low = np.full(50, 95.0)
    close = np.full(50, 100.0)

    # Massive Gap Up di hari ke-30
    high[30] = 160.0
    low[30] = 150.0
    close[30] = 155.0

    df = pl.DataFrame({"high": high, "low": low, "close": close})
    res = compute_tsm_indicators(df, atr_window=14).unwrap()

    atr = res["atr"].to_numpy()
    # TR pada gap day: max(160-150, 160-100, 150-100) = 60.
    # ATR harus melonjak signifikan merespons ini, bukan hanya merespons selisih H-L (10)
    assert atr[30] > atr[29] + 2.0  # Ekspektasi lonjakan tajam


def test_flash_crash_recovery() -> None:
    """2.3: Lower band pins to the crash value and drops it exactly after the window passes."""
    n = 100
    window = 20
    low = np.full(n, 100.0)

    # Flash Crash di Index 50
    low[50] = 10.0

    df = pl.DataFrame({
        "high": np.full(n, 105.0),
        "low": low,
        "close": np.full(n, 102.0)
    })

    res = compute_tsm_indicators(df, donchian_window=window).unwrap()
    lower_band = res["lower_band"].to_numpy()

    # Baris 49 normal
    assert lower_band[49] == 100.0
    # Baris 50 menancap di 10.0 dan bertahan selama window (50 hingga 69)
    assert_allclose(lower_band[50:50 + window], 10.0)
    # Baris 70 (tepat setelah window flash crash terlewati) harus pulih ke 100.0
    assert lower_band[50 + window] == 100.0


# ============================================================================
# KUADRAN 3: THE IRON GUARDS (Integritas Skema & Tipe)
# ============================================================================

def test_missing_ohlc_columns() -> None:
    """3.1: Rejects DataFrames without proper OHLC columns."""
    df = pl.DataFrame({"timestamp": [1, 2], "close": [10, 11]})
    res = compute_tsm_indicators(df)
    assert res.is_err()
    assert "Missing required column" in str(res.unwrap_err())


def test_invalid_window_parameters() -> None:
    """3.2: Rejects mathematically impossible windows."""
    df = pl.DataFrame({"high": [1], "low": [1], "close": [1]})
    res = compute_tsm_indicators(df, atr_window=1, donchian_window=20)
    assert res.is_err()
    assert "at least 2" in str(res.unwrap_err())


def test_dict_live_window_too_small() -> None:
    """3.3: Rejects Live Dict buffers that are smaller than the required window."""
    # Engine secara ketat me-return Err jika panjang array < max_window
    tick_dict = {
        "high": np.array([1, 2, 3]),
        "low": np.array([1, 2, 3]),
        "close": np.array([1, 2, 3])
    }
    res = compute_tsm_indicators(tick_dict, donchian_window=20)
    assert res.is_err()
    assert "Insufficient live buffer" in str(res.unwrap_err())


# ============================================================================
# KUADRAN 4: DUAL-MODE SYMMETRY (Polars vs Numba Dict)
# ============================================================================

def test_dual_mode_identical_output() -> None:
    """4.1: Proves that Vectorized Mass Processing yields exact same values as Live Tick Buffer."""
    np.random.seed(42)
    n_rows = 1000
    highs = np.random.rand(n_rows) * 10 + 100
    lows = highs - np.random.rand(n_rows) * 5
    closes = lows + np.random.rand(n_rows) * 2

    # MODE 1: Vectorized DF
    df = pl.DataFrame({"high": highs, "low": lows, "close": closes})
    res_df = compute_tsm_indicators(df, atr_window=14, donchian_window=20).unwrap()

    # Ambil nilai baris paling ujung (terakhir) dari kalkulasi Polars
    expected_atr = res_df["atr"][-1]
    expected_upper = res_df["upper_band"][-1]
    expected_lower = res_df["lower_band"][-1]

    # MODE 2: Live Tick Buffer (Suapkan seluruh history agar ATR konvergen sempurna)
    live_dict = {
        "high": highs,
        "low": lows,
        "close": closes
    }
    res_dict = compute_tsm_indicators(live_dict, atr_window=14, donchian_window=20).unwrap()

    # HARUS IDENTIK 100%
    assert_allclose(res_dict["atr"], expected_atr, rtol=1e-8)
    assert_allclose(res_dict["upper_band"], expected_upper, rtol=1e-8)
    assert_allclose(res_dict["lower_band"], expected_lower, rtol=1e-8)


def test_mode_1_schema_conservation() -> None:
    """4.2: Vectorized mode must append exactly 3 columns and preserve all original columns."""
    df = pl.DataFrame({
        "timestamp": np.arange(100),
        "open": np.random.rand(100),
        "high": np.random.rand(100) + 10,
        "low": np.random.rand(100) + 5,
        "close": np.random.rand(100) + 7,
        "volume": np.random.rand(100) * 1000,
        "vwap": np.random.rand(100) + 8,
    })

    original_cols = set(df.columns)
    res_df = compute_tsm_indicators(df).unwrap()

    new_cols = set(res_df.columns)

    # Harus ada 3 tambahan kolom spesifik
    expected_additions = {"atr", "upper_band", "lower_band"}

    assert original_cols.issubset(new_cols)
    assert new_cols - original_cols == expected_additions
