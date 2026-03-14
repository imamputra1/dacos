"""
builder/etl.py

ETL pipeline untuk membaca data mentah dari raw lake, memproyeksikan kolom yang diperlukan,
dan menyimpannya sebagai skinny table di silver lake.
Menggunakan Result monad untuk error handling yang eksplisit.
"""
import logging
from pathlib import Path

import polars as pl

from dacos.utils import Err, Ok, Result, safe

logger = logging.getLogger(__name__)


def build_skinny_lake(
    raw_path: str | Path,
    silver_path: str | Path,
) -> Result[None, Exception]:
    """
    Membangun skinny table dengan membaca semua file Parquet dari raw_path,
    memilih kolom 'timestamp', 'symbol', 'close', 'volume', dan menulis hasilnya
    ke silver_path sebagai file Parquet terkompresi (ZSTD).

    Parameters
    ----------
    raw_path : str or Path
        Direktori yang berisi file-file Parquet mentah (bisa dengan subdirektori).
    silver_path : str or Path
        Direktori tujuan untuk menyimpan skinny table. Jika belum ada, akan dibuat.

    Returns
    -------
    Result[None, Exception]
        Ok(None) jika sukses, atau Err(exception) jika gagal.
    """
    raw = Path(raw_path)
    silver = Path(silver_path)

    if not raw.exists():
        return Err(FileNotFoundError(f"Raw path tidak ditemukan: {raw}"))
    if not raw.is_dir():
        return Err(NotADirectoryError(f"Raw path harus berupa direktori: {raw}"))

    try:
        silver.mkdir(parents=True, exist_ok=True)

        pattern = str(raw / "**" / "*.parquet")
        lazy_df: pl.LazyFrame = pl.scan_parquet(pattern)

        lazy_df = lazy_df.select(["timestamp", "symbol", "close", "volume"])

        output_file = silver / "skinny.parquet"
        lazy_df.sink_parquet(
            output_file,
            compression="zstd",
            compression_level=22,
        )

        logger.info(f"✅ Skinny table berhasil ditulis ke {output_file}")
        return Ok(None)

    except Exception as e:
        logger.error(f"❌ Gagal membangun skinny table: {e}")
        return Err(e)


build_skinny_lake_safe = safe(build_skinny_lake)
