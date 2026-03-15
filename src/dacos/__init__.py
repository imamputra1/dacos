"""
dacos - Universal Python library for Medium-Frequency Trading alpha research.

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
from dacos.builder import SkinnyLakeBuilder, create_skinny_builder
from dacos.core import (
    UniverseAligner,
    UniverseIngestor,
    UniverseValidator,
    create_universe_aligner,
    create_universe_ingestor,
    create_universe_validator,
)
from dacos.laws import (
    calculate_adf_pvalue,
    calculate_halflife,
    calculate_hurst,
)
from dacos.protocols import DataFrame
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
    "builder",
    "config",
    "core",
    "laws",
    "paradigms",
    "protocols",
    "utils",
    "DataFrame",
    "UniverseIngestor",
    "create_universe_ingestor",
    "UniverseAligner",
    "create_universe_aligner",
    "UniverseValidator",
    "create_universe_validator",
    "calculate_hurst",
    "calculate_adf_pvalue",
    "calculate_halflife",
    "SkinnyLakeBuilder",
    "create_skinny_builder",
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
