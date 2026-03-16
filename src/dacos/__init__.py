"""
Arsitektur:
- builder: ETL pipeline untuk menyiapkan data (skinny table)
- core: Pembacaan, penyelarasan, dan validasi data
- laws: Hukum alam statistik (Hurst, ADF, dll.) - coming soon
- paradigms: Keluarga strategi (stat arb, CTA) - coming soon
- utils: Utilitas (Result monad, visualisasi)
- config: Konstanta global
- protocols: Type hints dan kontrak antarmuka
- facade: Pintu masuk utama untuk user notebook (coming soon)
"""

__version__ = "0.1.1"
from dacos import (
    builder,
    config,
    core,
    laws,
    paradigms,
    protocols,
    utils,
)
from dacos.builder import execute_etl_pipeline
from dacos.core import (
    ingest_silver_data,
    synchronize_asset_to_master_grid_strict,
    validate_market_integrity,
    validate_silver_schema,
)
from dacos.laws import (
    calculate_adf_pvalue,
    calculate_halflife,
    calculate_hurst,
)
from dacos.paradigms import (
    StatArbEngine,
    create_stat_arb_engine,
)
from dacos.protocols import DataFrame

# from dacos.research import run_pairs_research
from dacos.utils import (
    Err,
    NoneType,
    Ok,
    Option,
    Result,
    Some,
    is_err,
    is_ok,
    safe,
    safe_async,
)

__all__ = [
    "__version__",
    # "run_pairs_research",
    "builder",
    "config",
    "core",
    "laws",
    "paradigms",
    "protocols",
    "utils",
    "DataFrame",
    "validate_silver_schema",
    "ingest_silver_data",
    "UniverseAligner",
    "create_universe_aligner",
    "validate_market_integrity",
    "calculate_hurst",
    "calculate_adf_pvalue",
    "calculate_halflife",
    "StatArbEngine",
    "create_stat_arb_engine",
    "execute_etl_pipeline",
    "Result",
    "Ok",
    "Err",
    "Option",
    "Some",
    "NoneType",
    "safe",
    "safe_async",
    "is_ok",
    "is_err",
]
