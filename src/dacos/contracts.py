"""Single Source of Truth untuk skema data. Dirancang agar 100% kompatibel dengan OMS Orca."""

import polars as pl

# ==========================================
# 1. KONTRAK DATA MENTAH (RAW LAKE)
# Sesuai dengan struktur Parquet asli yang memiliki partisi Hive
# ==========================================
RAW_SCHEMA = {
    "timestamp": pl.Int16,
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Float64,
    "symbol": pl.String,
    "interval": pl.String,
    "year": pl.Int64,
    "month": pl.String
}
"""
Skema untuk data mentah di raw lake.
- timestamp: dalam milidetik (Unix time)
- open, high, low, close, volume: harga dan volume
- symbol: pasangan perdagangan (contoh: "BTC-USDT")
- interval: resolusi waktu (contoh: "1m")
- year, month: partisi Hive (tahun dan bulan)
"""

# ==========================================
# 2. KONTRAK DATA UNIVERSAL (SILVER LAKE)
# Input murni untuk Dacos (Waktu hidup, tanpa metadata Hive)
# ==========================================
SILVER_SCHEMA = {
    "timestamp": pl.Datetime("ms"),
    "symbol": pl.String,
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Float64
}
"""
Skema untuk data silver (skinny table) setelah proses ETL.
- timestamp: sudah dikonversi ke Datetime dengan presisi milidetik
- symbol: tetap string
- open, high, low, close, volume: harga dan volume dalam bentuk asli (belum di-log)
"""

# ==========================================
# 3. KONTRAK DATA EKSEKUSI (SIGNAL / BATCH OUTPUT)
# Dirancang KHUSUS untuk lolos dari 'SignalValidator' di orca/filters.py
# ==========================================
SIGNAL_SCHEMA = {
    "timestamp": pl.Datetime("ms"),
    "symbol": pl.String,
    "action": pl.String,
    "position": pl.Int8,
    "strength": pl.Float64,
    "z_score": pl.Float64,
    "spread": pl.Float64
}
"""
Skema untuk sinyal trading yang dihasilkan oleh engine.
- timestamp: waktu sinyal
- symbol: pasangan perdagangan
- action: aksi yang direkomendasikan (string sesuai definisi Orca)
- position: posisi numerik (-1, 0, 1)
- strength: kekuatan sinyal (biasanya |z-score|)
- z_score: nilai z-score mentah
- spread: nilai spread mentah
"""

__all__ = [
    "RAW_SCHEMA",
    "SILVER_SCHEMA",
    "SIGNAL_SCHEMA",
]
