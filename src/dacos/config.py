"""
dacos.config - Single Source of Truth for universal constants.

This module defines all magic numbers and thresholds used across the library.
No hardcoded paths or user-specific locations are stored here.
All constants are environment-agnostic and must be used instead of inline literals.
"""

from typing import Final

# ====================================================================
# ZONE 1: ALPHA FILTERING PARAMETERS
# ====================================================================
MAX_HURST_EXPONENT: Final[float] = 0.45
MAX_ADF_PVALUE: Final[float] = 0.05
MAX_HALF_LIFE_MINUTES: Final[float] = 1440

# ====================================================================
# ZONE 2: HARDWARE SAFEGUARDS
# ====================================================================
MAX_CPU_WORKERS: Final[int] = 3

# ====================================================================
# ZONE 3: STANDARD COLUMN NAMES
# ====================================================================
COLUMN_TIMESTAMP: Final[str] = "timestamp"
COLUMN_SYMBOL: Final[str] = "symbol"
