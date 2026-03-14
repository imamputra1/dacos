"""
dacos.protocols - Type hints, protocols, and interfaces for the entire library.

This module serves as the central "dictionary" for all data types and contracts.
It enables IDE autocompletion, prevents circular imports, and defines the
shape of data flowing through the system.

RULES:
1. NO execution logic (no calculations, I/O, or data manipulation).
2. NO heavy imports outside TYPE_CHECKING.
3. All type aliases and protocols must be defined here.
4. Use `from __future__ import annotations` for lazy evaluation.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

# ====================================================================
# DIMENSI BAYANGAN: Impor berat hanya untuk type checking
# ====================================================================
if TYPE_CHECKING:
    import polars as pl

    type LazyFrame = pl.LazyFrame
    type DataFrame = pl.DataFrame
else:
    # Runtime placeholders (mencegah NameError saat digunakan dalam anotasi)
    LazyFrame = Any
    DataFrame = Any


# ====================================================================
# TYPE ALIASES: Memperpendek notasi tipe di seluruh kode
# ====================================================================

type Symbol = str
"""Ticker symbol, e.g., 'BTCUSDT'."""

type Interval = str
"""Kline interval, e.g., '1m', '1h', '1d'."""

type Timestamp = int
"""Unix timestamp in seconds or milliseconds (consistency required)."""

type Price = float
"""Asset price."""

type Volume = float
"""Trading volume."""

# ====================================================================
# FUNCTION SIGNATURE ALIASES
# ====================================================================

type TransformFunc = Callable[[LazyFrame], LazyFrame]
"""Fungsi yang menerima LazyFrame dan mengembalikan LazyFrame."""

type SignalFunc = Callable[[LazyFrame], LazyFrame]
"""Fungsi yang menambahkan kolom sinyal (biasanya 'signal') ke LazyFrame."""


# ====================================================================
# PROTOCOLS (INTERFACES): Kontrak untuk komponen dacos
# ====================================================================

@runtime_checkable
class DataTransformer(Protocol):
    """
    Kontrak untuk komponen yang melakukan transformasi data.
    Siapa pun yang mengimplementasikan protocol ini WAJIB memiliki
    method `transform` dengan signature berikut.
    """

    def transform(self, data: LazyFrame) -> LazyFrame:
        """
        Terima LazyFrame, kembalikan LazyFrame yang telah ditransformasi.
        Transformasi bisa berupa filtering, penambahan kolom, agregasi, dll.
        """
        ...


@runtime_checkable
class IngestionProtocol(Protocol):
    """
    Kontrak untuk pembaca data dari sumber (raw lake atau silver lake).
    """

    def read(
        self,
        symbols: list[Symbol],
        interval: Interval,
        *,
        start_time: Timestamp | None = None,
        end_time: Timestamp | None = None,
    ) -> LazyFrame:
        """
        Baca data untuk daftar simbol dan interval tertentu.
        Mengembalikan LazyFrame dengan kolom standar: timestamp, symbol,
        open, high, low, close, volume.
        """
        ...


@runtime_checkable
class AlignmentProtocol(Protocol):
    """
    Kontrak untuk penyelarasan timestamp multi-asset (as-of join).
    """

    def align(
        self,
        data: LazyFrame,
        symbols: list[Symbol],
        frequency: str = "1s",
        *,
        how: str = "forward",
    ) -> LazyFrame:
        """
        Lakukan as-of join untuk menyelaraskan data beberapa simbol
        ke grid waktu yang sama.
        """
        ...


@runtime_checkable
class StatisticalTestProtocol(Protocol):
    """
    Kontrak untuk uji statistik (Hurst, ADF, dll).
    """

    def compute(self, series: LazyFrame, column: str) -> dict[str, float]:
        """
        Hitung statistik dari satu kolom numerik.
        Mengembalikan dictionary berisi hasil uji (misal: {'hurst': 0.35}).
        """
        ...
