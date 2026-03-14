"""
dacos.protocols - Type hints, protocols, and interfaces for the entire library.

This module serves as the central "dictionary" for all data types and contracts.
It enables IDE autocompletion, prevents circular imports, and defines the
shape of data flowing through the system.
"""

from __future__ import annotations

import os
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
    # Runtime placeholders
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

type PathLike = str | os.PathLike
"""Path-like object (string or os.PathLike)."""


# ====================================================================
# FUNCTION SIGNATURE ALIASES
# ====================================================================

type TransformFunc = Callable[[LazyFrame], LazyFrame]
"""Function that takes a LazyFrame and returns a transformed LazyFrame."""

type SignalFunc = Callable[[LazyFrame], LazyFrame]
"""Function that adds a signal column (usually 'signal') to the LazyFrame."""


# ====================================================================
# PROTOCOLS (INTERFACES): Kontrak untuk komponen dacos
# ====================================================================

@runtime_checkable
class DataTransformer(Protocol):
    """Contract for components that perform data transformations."""

    def transform(self, data: LazyFrame) -> LazyFrame:
        """
        Transform a LazyFrame and return a new LazyFrame.
        Transformations can include filtering, column additions, aggregations, etc.
        """
        ...


@runtime_checkable
class IngestionProtocol(Protocol):
    """Contract for data readers from raw or silver lake."""

    def read(
        self,
        symbols: list[Symbol],
        interval: Interval,
        *,
        start_time: Timestamp | None = None,
        end_time: Timestamp | None = None,
    ) -> LazyFrame:
        """
        Read data for given symbols and interval.
        Returns a LazyFrame with standard columns: timestamp, symbol,
        open, high, low, close, volume.
        """
        ...


@runtime_checkable
class AlignmentProtocol(Protocol):
    """Contract for multi-asset timestamp alignment (as-of join)."""

    def align(
        self,
        data: LazyFrame,
        symbols: list[Symbol],
        frequency: str = "1s",
        *,
        how: str = "forward",
    ) -> LazyFrame:
        """
        Perform as-of join to align multiple symbols to the same time grid.
        """
        ...


@runtime_checkable
class StatisticalTestProtocol(Protocol):
    """Contract for statistical tests (Hurst, ADF, etc.)."""

    def compute(self, series: LazyFrame, column: str) -> dict[str, float]:
        """
        Compute statistics from a single numeric column.
        Returns a dictionary of results (e.g., {'hurst': 0.35}).
        """
        ...


# ====================================================================
# EXPORTS
# ====================================================================

__all__ = [
    "Symbol",
    "Interval",
    "Timestamp",
    "Price",
    "Volume",
    "PathLike",
    "LazyFrame",
    "DataFrame",
    "TransformFunc",
    "SignalFunc",
    "DataTransformer",
    "IngestionProtocol",
    "AlignmentProtocol",
    "StatisticalTestProtocol",
]
