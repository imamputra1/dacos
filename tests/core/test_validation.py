import polars as pl

from dacos.core.validation import UniverseValidator, create_universe_validator
from dacos.utils import is_err, is_ok

# ============================================================================
# Test 1: Factory and Initialization
# ============================================================================

def test_factory_initialization() -> None:
    """Ensure create_universe_validator returns a properly configured instance."""
    min_rows = 50
    validator = create_universe_validator(min_rows)

    assert isinstance(validator, UniverseValidator)
    assert validator.min_rows == min_rows


# ============================================================================
# Test 2: Short Data (below min_rows)
# ============================================================================

def test_short_data() -> None:
    """Data with fewer rows than min_rows should be rejected."""
    data = pl.DataFrame({
        "symbol": ["A"] * 99,
        "log_price": [1.0] * 99,
        "log_volume": [10.0] * 99,
    })

    validator = UniverseValidator(min_rows=100)
    result = validator.validate(data)

    assert is_err(result)
    error_msg = str(result.err()).lower()
    assert "short" in error_msg or "rows" in error_msg


# ============================================================================
# Test 3: Null Infestation (too many nulls after dropping)
# ============================================================================

def test_null_infestation() -> None:
    """Data with many nulls should be rejected after drop_nulls."""
    data = pl.DataFrame({
        "symbol": ["A"] * 20,
        "log_price": [1.0] * 5 + [None] * 15,
        "log_volume": [10.0] * 20,
    })

    validator = UniverseValidator(min_rows=10)
    result = validator.validate(data)

    assert is_err(result)
    error_msg = str(result.err()).lower()
    assert "null" in error_msg or "rows" in error_msg


# ============================================================================
# Test 4: Stagnant Coin (zero variance)
# ============================================================================

def test_stagnant_coin() -> None:
    """A coin with constant log_price should be rejected."""
    data = pl.DataFrame({
        "symbol": ["USDT"] * 10,
        "log_price": [1.0] * 10,
        "log_volume": [10.0] * 10,
    })

    validator = UniverseValidator(min_rows=5)
    result = validator.validate(data)

    assert is_err(result)
    error_msg = str(result.err()).lower()
    assert "stagnant" in error_msg or "std_dev" in error_msg


# ============================================================================
# Test 5: Corrupt Schema (missing required column)
# ============================================================================

def test_corrupt_schema_missing_symbol() -> None:
    """Missing 'symbol' column should cause Err."""
    data = pl.DataFrame({
        "not_symbol": ["A", "B"],
        "log_price": [1.0, 2.0],
        "log_volume": [10.0, 20.0],
    })

    validator = UniverseValidator(min_rows=1)
    result = validator.validate(data)

    assert is_err(result)


def test_corrupt_schema_missing_log_price() -> None:
    """Missing 'log_price' column should cause Err."""
    data = pl.DataFrame({
        "symbol": ["A", "B"],
        "not_log_price": [1.0, 2.0],
        "log_volume": [10.0, 20.0],
    })

    validator = UniverseValidator(min_rows=1)
    result = validator.validate(data)

    assert is_err(result)


# ============================================================================
# Test 6: Golden Path (valid data passes through unchanged)
# ============================================================================

def test_golden_path() -> None:
    """Valid data should be returned unchanged as Ok."""
    original_data = pl.DataFrame({
        "symbol": ["A", "A", "B", "B"],
        "log_price": [1.1, 1.2, 2.1, 2.2],
        "log_volume": [10.0, 11.0, 20.0, 21.0],
    })

    validator = UniverseValidator(min_rows=4)
    result = validator.validate(original_data)

    assert is_ok(result)
    returned_data = result.ok()
    assert returned_data.shape == original_data.shape
    assert returned_data.columns == original_data.columns
    assert returned_data.to_dict(as_series=False) == original_data.to_dict(as_series=False)
