"""
STRESS & PERFORMANCE TESTS
Location: tests/e2e/test_stress.py
Paradigm: Hardware Limits, Memory Management, Numba JIT Stability, O(1) Validation.

PERINGATAN: Tes ini dirancang untuk menyiksa CPU dan RAM.
Kipas komputer Anda mungkin akan berputar kencang saat tes ini berjalan.
"""

from __future__ import annotations

import time
import tracemalloc

import numpy as np
import polars as pl
import pytest
from dacos.api import evaluate_tsm_live, run_tsm_research
from dacos.config import TSMConfig
from dacos.paradigms import compute_basket_zscore

# ============================================================================
# KUADRAN 1: THE MEMORY CRUNCH (10 JUTA BARIS)
# ============================================================================


# Menggunakan marker khusus agar tes berat ini bisa di-skip jika hanya ingin tes logika cepat
@pytest.mark.performance
def test_stress_memory_crunch_10m_rows() -> None:
    """
    Skenario: Backtest Vectorized 1 koin (TSM) selama 10 Juta Baris.
    Target: < 3.0 detik. Tidak boleh terjadi MemoryError.
    """
    n_rows = 10_000_000

    # 1. Menghasilkan 10 Juta Baris Data Sintetis secara efisien di memori
    np.random.seed(42)
    closes = np.random.rand(n_rows) * 100 + 1000

    df_massive = pl.DataFrame(
        {
            "timestamp": np.arange(n_rows),  # Angka int lebih ringan di RAM untuk inisiasi
            "open": closes * 0.99,
            "high": closes + 5,
            "low": closes - 5,
            "close": closes,
            "volume": np.ones(n_rows) * 100,
        }
    ).with_columns(pl.col("timestamp").cast(pl.Datetime("ms")))

    config = TSMConfig(atr_window=14, donchian_window=20)

    # 2. Mulai Stopwatch & Perekam Memori
    tracemalloc.start()
    start_time = time.perf_counter()

    # EKSEKUSI
    res = run_tsm_research(df_massive, "BTC-USDT", config)

    end_time = time.perf_counter()
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert res.is_ok(), f"Pipeline gagal di beban tinggi: {res.unwrap_err()}"
    out_df = res.unwrap()
    assert len(out_df) == n_rows

    # 3. Metrik Kelulusan
    execution_time = end_time - start_time
    peak_mb = peak_mem / 10**6

    print(f"\n[STRESS Q1] 10M Rows executed in: {execution_time:.3f} seconds.")
    print(f"[STRESS Q1] Peak Memory Usage: {peak_mb:.2f} MB.")

    # Toleransi 3.5 detik (memberi ruang jika dijalankan di CI/CD gratisan seperti GitHub Actions)
    assert execution_time < 3.5, f"Vectorized engine too slow: {execution_time:.2f}s (Limit: 3.5s)"


# ============================================================================
# KUADRAN 2: THE MULTI-ASSET BASKET CRUNCH (PCA STRESS)
# ============================================================================


@pytest.mark.performance
def test_stress_pca_basket_50_coins() -> None:
    """
    Skenario: PCA Eigen-Decomposition pada 50 Koin sekaligus (100.000 baris).
    Target: Matriks Covariance & SVD Numba tidak crash. < 10.0 detik.
    """
    n_rows = 100_000
    n_coins = 50
    np.random.seed(42)

    # Menyiapkan 50 kolom koin basket
    data = {"target": np.random.rand(n_rows) + 10}
    basket_columns = []

    for i in range(n_coins):
        col_name = f"coin_{i}"
        data[col_name] = np.random.rand(n_rows) + 10
        basket_columns.append(col_name)

    df_pca = pl.DataFrame(data)

    start_time = time.perf_counter()

    # EKSEKUSI (Langsung menghajar PCA Engine)
    res = compute_basket_zscore(df_pca, "target", basket_columns, 20)

    end_time = time.perf_counter()

    assert res.is_ok(), f"PCA Math Engine crashed: {res.unwrap_err()}"

    execution_time = end_time - start_time
    print(f"\n[STRESS Q2] PCA (50 coins x 100k rows) executed in: {execution_time:.3f} seconds.")

    assert execution_time < 10.0, f"PCA matrix decomposition too slow: {execution_time:.2f}s"


# ============================================================================
# KUADRAN 3: THE LIVE TICK BARRAGE (10.000 REQUEST/DETIK)
# ============================================================================


@pytest.mark.performance
def test_stress_live_tick_barrage() -> None:
    """
    Skenario: Simulasi Badai Volatilitas (10.000 live tick berturut-turut).
    Target: Degradasi performa tidak terjadi (O(1)). Numba JIT stabil.
    """
    # Mock data Live Buffer berukuran 30 baris (Sesuai memori bot)
    live_buffer = {
        "timestamp": np.arange(30),
        "open": np.random.rand(30) + 100,
        "high": np.random.rand(30) + 105,
        "low": np.random.rand(30) + 95,
        "close": np.random.rand(30) + 102,
        "volume": np.random.rand(30) * 1000,
    }

    config = TSMConfig()
    iterations = 10_000

    # Pemanasan JIT (Warmup) agar kompilasi awal tidak merusak metrik
    _ = evaluate_tsm_live(live_buffer, "SOL", config)

    latencies = []

    for _ in range(iterations):
        start_time = time.perf_counter()
        res = evaluate_tsm_live(live_buffer, "SOL", config)
        end_time = time.perf_counter()

        assert res.is_ok()
        latencies.append((end_time - start_time) * 1000)  # dalam milidetik

    # Membagi sesi menjadi dua untuk membandingkan degradasi kecepatan
    first_1000_avg = sum(latencies[:1000]) / 1000
    last_1000_avg = sum(latencies[-1000:]) / 1000

    print("\n[STRESS Q3] 10,000 Live Requests Barrage completed.")
    print(f"[STRESS Q3] Avg Latency (First 1K): {first_1000_avg:.4f} ms")
    print(f"[STRESS Q3] Avg Latency (Last 1K):  {last_1000_avg:.4f} ms")

    # Syarat mutlak: Iterasi akhir tidak boleh lebih lambat 2x lipat dari iterasi awal (Indikasi Memory Leak)
    assert last_1000_avg < (first_1000_avg * 2.0), "Terdeteksi degradasi performa (kemungkinan Memory Leak Numba/Dict)"
    # Rata-rata keseluruhan harus tetap di bawah 2 ms
    assert (sum(latencies) / iterations) < 2.0, "Sistem gagal mempertahankan latensi HFT"
