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
    compute_pca_safe,
    ingest_silver_data,
    invert_matrix_safe,
    synchronize_asset_to_master_grid_strict,
    validate_market_integrity,
    validate_silver_schema,
)
from dacos.laws import (
    compute_adf_test_safe,
    compute_atr_safe,
    compute_donchian_channels_safe,
    compute_engle_arch_test_safe,
    compute_garman_klass_safe,
    compute_half_life_safe,
    compute_hurst_exponent_safe,
    compute_yang_zhang_safe,
)
from dacos.paradigms import (
    compute_basket_zscore,
    compute_pairs_zscore,
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
    "invert_matrix_safe",
    "compute_pca_safe",
    "validate_silver_schema",
    "ingest_silver_data",
    "synchronize_asset_to_master_grid_strict",
    "validate_market_integrity",
    "compute_hurst_exponent_safe",
    "compute_adf_test_safe",
    "compute_half_life_safe",
    "compute_engle_arch_test_safe",
    "compute_yang_zhang_safe",
    "compute_garman_klass_safe",
    "compute_donchian_channels_safe",
    "compute_atr_safe",
    "compute_pairs_zscore",
    "compute_basket_zscore",
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
