"""
Architecture:
- builder: ETL pipeline for preparing data (skinny table)
- core: Data reading, alignment, and validation
- laws: Statistical laws of nature (Hurst, ADF, etc.) - coming soon
- paradigms: Strategy families (stat arb, CTA) - coming soon
- utils: Utilities (Result monad, visualization)
- config: Global constants
- protocols: Type hints and interface contracts
- facade: Main entry point for user notebooks (coming soon)
"""

__version__ = "0.1.1"

# ============================================================================
# 1. SUBMODULE EXPORTS
# ============================================================================
from . import api, builder, config, core, laws, paradigms, protocols, utils

# ============================================================================
# 2. PUBLIC API (THE CONDUCTOR)
# ============================================================================
from .api import (
    evaluate_stat_arb_live,
    evaluate_tsm_live,
    run_stat_arb_research,
    run_tsm_research,
)

# ============================================================================
# 3. BUILDER & CORE PIPELINE
# ============================================================================
from .builder import execute_etl_pipeline
from .core import (
    compute_pca_safe,
    ingest_silver_data,
    invert_matrix_safe,
    synchronize_asset_to_master_grid_strict,
    validate_market_integrity,
    validate_silver_schema,
)

# ============================================================================
# 4. LAWS (MATHEMATICAL KERNELS)
# ============================================================================
from .laws import (
    compute_adf_test_safe,
    compute_atr_safe,
    compute_donchian_channels_safe,
    compute_engle_arch_test_safe,
    compute_garman_klass_safe,
    compute_half_life_safe,
    compute_hurst_exponent_safe,
    compute_yang_zhang_safe,
)

# ============================================================================
# 5. PARADIGMS (ENGINES & TACTICS)
# ============================================================================
from .paradigms import (
    apply_mean_reversion_tactics_strict,
    apply_momentum_tactics_strict,
    compute_basket_zscore,
    compute_pairs_zscore,
    compute_tsm_indicators,
)

# ============================================================================
# 6. PROTOCOLS & UTILITIES (MONADS)
# ============================================================================
from .protocols import DataFrame
from .utils import (
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

# ============================================================================
__all__ = [
    # Metadata
    "__version__",

    # Submodules
    "api",
    "builder",
    "config",
    "core",
    "laws",
    "paradigms",
    "protocols",
    "utils",

    # Public API Endpoints
    "run_stat_arb_research",
    "run_tsm_research",
    "evaluate_stat_arb_live",
    "evaluate_tsm_live",

    # Builder & Core
    "execute_etl_pipeline",
    "ingest_silver_data",
    "synchronize_asset_to_master_grid_strict",
    "validate_market_integrity",
    "validate_silver_schema",
    "invert_matrix_safe",
    "compute_pca_safe",

    # Laws
    "compute_hurst_exponent_safe",
    "compute_adf_test_safe",
    "compute_half_life_safe",
    "compute_engle_arch_test_safe",
    "compute_yang_zhang_safe",
    "compute_garman_klass_safe",
    "compute_donchian_channels_safe",
    "compute_atr_safe",

    # Paradigms
    "compute_pairs_zscore",
    "compute_basket_zscore",
    "compute_tsm_indicators",
    "apply_mean_reversion_tactics_strict",
    "apply_momentum_tactics_strict",

    # Protocols & Monads
    "DataFrame",
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
