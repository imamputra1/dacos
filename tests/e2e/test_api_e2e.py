"""
END-TO-END (E2E) API TESTS
Location: tests/e2e/test_api_e2e.py
Paradigm: Black Box Testing, Determinism, Monadic Resilience, Sub-millisecond Benchmarking.

Membuktikan bahwa dacos/api.py (The Conductor) mampu menangani aliran data
dari ujung ke ujung tanpa membocorkan Exception, mematuhi kontrak skema,
dan beroperasi dengan latensi tingkat HFT (High-Frequency Trading).
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta

import numpy as np
import polars as pl
from polars.testing import assert_frame_equal

from dacos.api import (
    evaluate_stat_arb_live,
    evaluate_tsm_live,
    run_stat_arb_research,
    run_tsm_research,
)
from dacos.config import StatArbConfig, TSMConfig

# ============================================================================
# HELPER: MOCK DATA GENERATORS (The Fuel)
# ============================================================================


def generate_silver_tsm_data(n_rows: int = 100) -> pl.DataFrame:
    """Menghasilkan mock data Silver Lake dengan skema waktu presisi milidetik."""
    np.random.seed(42)
    base_time = datetime(2023, 1, 1)
    timestamps = [base_time + timedelta(minutes=i) for i in range(n_rows)]

    highs = np.random.rand(n_rows) * 10 + 100
    lows = highs - np.random.rand(n_rows) * 5
    closes = lows + np.random.rand(n_rows) * 2

    return pl.DataFrame(
        {
            "timestamp": timestamps,
            "open": closes * 0.99,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": np.random.rand(n_rows) * 1000,
        }
    ).with_columns(pl.col("timestamp").cast(pl.Datetime("ms")))


def generate_aligned_statarb_data(n_rows: int = 100) -> pl.DataFrame:
    """Menghasilkan mock data pasangan koin yang sudah disejajarkan."""
    np.random.seed(42)
    base_time = datetime(2023, 1, 1)
    timestamps = [base_time + timedelta(minutes=i) for i in range(n_rows)]

    return pl.DataFrame(
        {
            "timestamp": timestamps,
            "target_coin": np.random.rand(n_rows) * 10 + 100,
            "anchor_coin": np.random.rand(n_rows) * 5 + 50,
        }
    ).with_columns(pl.col("timestamp").cast(pl.Datetime("ms")))


# ============================================================================
# 📦 KRITERIA 1: INTEGRITAS KONTRAK (The Black Box Schema)
# ============================================================================


def test_e2e_tsm_schema_integrity() -> None:
    """1.1 Zero-Leakage & Casting Validation untuk TSM."""
    df = generate_silver_tsm_data(100)
    res = run_tsm_research(df, "BTC-USDT")

    assert res.is_ok()
    out_df = res.unwrap()

    # Validasi Skema Presisi (Zero Leakage)
    expected_columns = {"timestamp", "symbol", "action", "position", "strength", "atr"}
    assert set(out_df.columns) == expected_columns

    # Validasi Tipe Data Paksa (Casting)
    assert out_df["timestamp"].dtype == pl.Datetime("ms")
    assert out_df["position"].dtype == pl.Int8
    assert out_df["symbol"][0] == "BTC-USDT"


def test_e2e_statarb_schema_integrity() -> None:
    """1.2 Zero-Leakage & Casting Validation untuk StatArb."""
    df = generate_aligned_statarb_data(100)
    res = run_stat_arb_research(df, "target_coin", "anchor_coin", hedge_ratio_beta=1.0)

    assert res.is_ok()
    out_df = res.unwrap()

    expected_columns = {"timestamp", "symbol", "action", "position", "strength", "z_score", "spread"}
    assert set(out_df.columns) == expected_columns
    assert out_df["position"].dtype == pl.Int8


# ============================================================================
# 🔄 KRITERIA 2: DETERMINISME (Idempotency Metrics)
# ============================================================================


def test_e2e_idempotency_100_percent() -> None:
    """2.1 & 2.2 Replikasi identik dan State Isolation (Statelessness)."""
    df = generate_silver_tsm_data(200)

    # Eksekusi pertama
    res_1 = run_tsm_research(df, "ETH-USDT").unwrap()

    # Eksekusi berulang (Mensimulasikan panggilan berulang dari orca)
    for _ in range(5):
        res_n = run_tsm_research(df, "ETH-USDT").unwrap()
        # Harus 100% identik, tidak ada floating-point drift atau sisa state memory
        assert_frame_equal(res_1, res_n)


# ============================================================================
# 🛡️ KRITERIA 3: KETAHANAN MONADIK (The Exception Shield)
# ============================================================================


def test_e2e_exception_shield_corrupt_data() -> None:
    """3.1 Injeksi Data Cacat. API tidak boleh melempar Exception merah, wajib Err()."""
    bad_df = pl.DataFrame({"timestamp": [1, 2, 3], "wrong_column": [10, 20, 30]})

    res = run_tsm_research(bad_df, "SOL-USDT")
    assert res.is_err()
    assert "Missing" in str(res.unwrap_err())  # Ditangkap secara elegan oleh Guard Clause


def test_e2e_exception_shield_empty_data() -> None:
    """3.2 Validasi set kosong (Misal: bursa maintenance/libur)."""
    empty_df = pl.DataFrame()

    res = run_stat_arb_research(empty_df, "A", "B", 1.0)
    assert res.is_err()
    assert "Empty" in str(res.unwrap_err())


def test_e2e_flatline_chain_protection() -> None:
    """3.3 Proteksi pembagian nol berantai hingga ke output akhir."""
    # Data stagnan, volatilitas mati total
    df = pl.DataFrame(
        {
            "timestamp": [datetime(2023, 1, 1)] * 50,
            "open": [100.0] * 50,
            "high": [100.0] * 50,
            "low": [100.0] * 50,
            "close": [100.0] * 50,
            "volume": [0.0] * 50,
        }
    ).with_columns(pl.col("timestamp").cast(pl.Datetime("ms")))

    res = run_tsm_research(df, "FLAT-USDT").unwrap()

    # Engine harus selamat. Midline touch = EXIT. Sizing flatline = 0.0
    last_row = res.row(-1, named=True)
    assert last_row["strength"] == 0.0
    assert last_row["action"] == "EXIT"  # SURGERY: Diubah dari NEUTRAL ke EXIT
    assert last_row["position"] == 0


# ============================================================================
# 🪞 KRITERIA 4: SIMETRI DUAL-MODE (Research vs Live Parity)
# ============================================================================


def test_e2e_dual_mode_symmetry_tick_to_vector() -> None:
    """4.1 Membuktikan output Live identik dengan baris terakhir output Research."""
    df = generate_silver_tsm_data(100)
    config = TSMConfig(atr_window=14, donchian_window=20)

    # Mode 1: Vectorized Research (Landasan pacu 100 baris)
    res_research = run_tsm_research(df, "DOGE-USDT", config).unwrap()
    last_research_row = res_research.row(-1, named=True)

    # Mode 2: Live Tick
    # SURGERY: Jangan gunakan .tail(30)!
    # Suapkan SELURUH 100 baris (df.to_dict) agar efek Wilder's Smoothing pada ATR
    # memiliki titik awal (warmup) yang 100% identik dengan Mode Research.
    live_buffer = df.to_dict(as_series=False)

    res_live = evaluate_tsm_live(live_buffer, "DOGE-USDT", config).unwrap()

    assert last_research_row["action"] == res_live["action"]
    assert last_research_row["position"] == res_live["position"]
    # Perbandingan floating point toleransi ketat (1e-8) sekarang akan lulus!
    assert np.isclose(last_research_row["strength"], res_live["strength"], atol=1e-8)


# ============================================================================
# ⏱️ KRITERIA 5: LATENSI EKSEKUSI (Live Mode Benchmark)
# ============================================================================
def test_e2e_hft_latency_benchmark() -> None:
    """5.1 Waktu evaluasi Live Tick-by-Tick wajib di bawah 5 milidetik."""
    df = generate_aligned_statarb_data(50)
    live_buffer = df.to_dict(as_series=False)
    config = StatArbConfig(z_window=20)

    # WARMUP RUN: Memaksa Numba untuk melakukan JIT Compilation
    # SURGERY: Ubah "TGT", "ANC" menjadi "target_coin", "anchor_coin" sesuai mock data
    _ = evaluate_stat_arb_live(live_buffer, "target_coin", "anchor_coin", 1.0, config)

    latencies = []
    iterations = 100

    for _ in range(iterations):
        start_time = time.perf_counter()
        # SURGERY: Ubah "TGT", "ANC" menjadi "target_coin", "anchor_coin"
        res = evaluate_stat_arb_live(live_buffer, "target_coin", "anchor_coin", 1.0, config)
        end_time = time.perf_counter()

        assert res.is_ok()
        latencies.append((end_time - start_time) * 1000)  # Konversi ke milidetik

    avg_latency = sum(latencies) / iterations
    max_latency = max(latencies)

    # Syarat mutlak kelulusan E2E HFT
    assert avg_latency < 5.0, f"Average latency too high: {avg_latency:.2f} ms"
    assert max_latency < 10.0, f"Max latency spike detected: {max_latency:.2f} ms"
