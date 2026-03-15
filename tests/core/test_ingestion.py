import polars as pl
import pytest

from dacos.core import UniverseIngestor, create_universe_ingestor
from dacos.utils import is_err, is_ok


@pytest.fixture
def sample_silver_data() -> pl.DataFrame:
    return pl.DataFrame({
        "timestamp": [100, 200, 300, 400, 500, 600],
        "symbol": ["BTC_USDT", "BTC_USDT", "ETH_USDT", "ETH_USDT", "SOL_USDT", "SOL_USDT"],
        "log_price": [4.5, 4.6, 3.2, 3.3, 2.1, 2.2],
        "log_volume": [6.1, 6.2, 5.8, 5.9, 4.5, 4.6],
    })

# ============================================================================
# Test 1: Factory and Initialization
# ============================================================================

def test_factory_initialization() -> None:
    silver = "some/silver/path"
    ingestor = create_universe_ingestor(silver)

    assert isinstance(ingestor, UniverseIngestor)
    assert str(ingestor.silver_path) == silver

# ============================================================================
# Test 2: Precision Ingestion (filter by symbols)
# ============================================================================

def test_precision_ingestion(tmp_path, sample_silver_data: pl.DataFrame) -> None:
    silver_dir = tmp_path / "silver"
    silver_dir.mkdir()
    file_path = silver_dir / "skinny.parquet"
    sample_silver_data.write_parquet(file_path)

    ingestor = UniverseIngestor(silver_dir)
    result = ingestor.load_universe(["BTC_USDT", "ETH_USDT"])

    assert is_ok(result)

    df = result.ok()
    assert isinstance(df, pl.DataFrame)

    symbols = df["symbol"].unique().to_list()
    assert set(symbols) == {"BTC_USDT", "ETH_USDT"}
    assert len(df) == 4

# ============================================================================
# Test 3: Empty Universe (requested symbols not present)
# ============================================================================

def test_empty_universe(tmp_path, sample_silver_data: pl.DataFrame) -> None:
    silver_dir = tmp_path / "silver_dir"
    silver_dir.mkdir()
    file_path = silver_dir /"skinny.parquet"
    sample_silver_data.write_parquet(file_path)

    ingestor = UniverseIngestor(silver_dir)
    result = ingestor.load_universe(["DOGE_USDT"])
    assert is_ok(result)

    df = result.ok()
    assert isinstance(df, pl.DataFrame)
    assert len(df) == 0

# ============================================================================
# Test 4: Monadic Error Handling (invalid path)
# ============================================================================

def test_monadic_error(tmp_path) -> None:
    non_existent_path = tmp_path / "nonexistent" / "silver"
    ingestor = UniverseIngestor(non_existent_path)
    result = ingestor.load_universe(["BTC_USDT"])
    assert is_err(result)
    assert isinstance(result.err(), Exception)
