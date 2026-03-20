"""
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

# ====================================================================
# ZONE 4: STRATEGY DEFAULT CONFIGURATIONS (IMMUTABLE)
# ====================================================================
@dataclass(frozen=True)
class StatArbConfig:
    """Default parameters for Statistical Arbitrage / Pairs Trading."""
    z_window: int = 50
    entry_z: float = 2.0
    exit_z: float = 0.5
    allow_short: bool = True

@dataclass(frozen=True)
class TSMConfig:
    """Default parameters for Time Series Momentum (CTA)."""
    donchian_window: int = 20
    atr_window: int = 14
    target_risk_pct: float = 0.01  # 1% risk sizing
    allow_short: bool = True
