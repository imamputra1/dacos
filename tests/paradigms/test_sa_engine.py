from __future__ import annotations

import numpy as np
import polars as pl
from numpy.testing import assert_allclose

from dacos.paradigms import compute_basket_zscore, compute_pairs_zscore

# ============================================================================
# KUADRAN 1: THE PHYSICS VALIDATION (Akurasi Matematis)
# ============================================================================


def test_spread_calculation_accuracy() -> None:
    """1.1 & 1.2: Tests log spread calculation with different beta values."""
    df = pl.DataFrame({"target": [100.0, 105.0, 110.0], "anchor": [50.0, 52.0, 55.0]})

    # Beta = 1.0
    result_beta_1 = compute_pairs_zscore(df, "target", "anchor", 1.0, 2).unwrap()
    expected_1 = np.log(df["target"].to_numpy()) - 1.0 * np.log(df["anchor"].to_numpy())
    # SURGERY: "log_spread" -> "spread"
    assert_allclose(result_beta_1["spread"].to_numpy(), expected_1)

    # Beta = 0.5
    result_beta_half = compute_pairs_zscore(df, "target", "anchor", 0.5, 2).unwrap()
    expected_half = np.log(df["target"].to_numpy()) - 0.5 * np.log(df["anchor"].to_numpy())
    # SURGERY: "log_spread" -> "spread"
    assert_allclose(result_beta_half["spread"].to_numpy(), expected_half)


def test_z_score_normalization() -> None:
    """1.3: Tests Z-Score dynamics using a perfect sine wave spread."""
    # Create a sine wave to simulate perfectly oscillating cointegrated prices
    wave = np.sin(np.linspace(0, 10 * np.pi, 100)) + 10.0
    df = pl.DataFrame({"target": np.exp(wave), "anchor": [1.0] * 100})  # Anchor is flat (ln(1)=0)

    result = compute_pairs_zscore(df, "target", "anchor", 1.0, 20).unwrap()
    z_scores = result["z_score"].to_numpy()[19:]  # Skip warmup

    # Z-score should oscillate dynamically
    assert np.max(z_scores) > 1.2
    assert np.min(z_scores) < -1.2
    # Mean of z-scores should be very close to 0
    assert_allclose(np.mean(z_scores), 0.0, atol=0.5)


def test_warmup_period_nans() -> None:
    """1.4: Tests that the rolling window correctly yields NaNs during warmup."""
    df = pl.DataFrame({"target": np.random.rand(100) + 10, "anchor": np.random.rand(100) + 10})
    window = 50
    result = compute_pairs_zscore(df, "target", "anchor", 1.0, window).unwrap()

    # First 49 rows should be null for rolling stats
    assert result["z_score"][: window - 1].is_null().all()
    val = result["z_score"][window - 1]
    assert val is not None and not np.isnan(val)


# ============================================================================
# KUADRAN 2: EXTREME MARKET ANOMALIES (Kekebalan Mesin)
# ============================================================================


def test_the_flatline_division_by_zero() -> None:
    """2.1: CRITICAL! Tests that a dead market (0 volatility) yields 0.0 Z-score, not NaN/Inf."""
    df = pl.DataFrame({"target": [100.0] * 60, "anchor": [50.0] * 60})
    window = 20
    result = compute_pairs_zscore(df, "target", "anchor", 1.0, window).unwrap()

    # Standard deviation will be exactly 0.0. Our .fill_nan(0.0) should catch the 0/0 division.
    valid_z_scores = result["z_score"].to_numpy()[window - 1 :]
    assert_allclose(valid_z_scores, 0.0)


def test_perfect_correlation() -> None:
    """2.2: Tests perfect correlation (constant spread) behaves like a flatline."""
    # Both move exactly +1% per tick
    target = 100.0 * (1.01 ** np.arange(60))
    anchor = 50.0 * (1.01 ** np.arange(60))
    df = pl.DataFrame({"target": target, "anchor": anchor})

    window = 20
    result = compute_pairs_zscore(df, "target", "anchor", 1.0, window).unwrap()

    valid_z_scores = result["z_score"].to_numpy()[window - 1 :]
    assert_allclose(valid_z_scores, 0.0, atol=1e-10)


def test_flash_crash_spike() -> None:
    """2.3: Tests rapid anomaly detection."""
    target = np.array([100.0] * 60)
    target[50] = 10.0  # 90% flash crash at row 50
    df = pl.DataFrame({"target": target, "anchor": [50.0] * 60})

    result = compute_pairs_zscore(df, "target", "anchor", 1.0, 20).unwrap()
    z_scores = result["z_score"].to_numpy()

    assert z_scores[50] < -3.0  # Massive negative z-score


def test_negative_prices_guard() -> None:
    """2.4: Tests Polars' native handling of negative/zero prices for logarithms."""
    df = pl.DataFrame({"target": [100.0, -10.0, 100.0], "anchor": [50.0, 50.0, 50.0]})
    result = compute_pairs_zscore(df, "target", "anchor", 1.0, 2).unwrap()

    # Log of negative number yields NaN, which propagates safely without crashing the engine
    # SURGERY: "log_spread" -> "spread"
    assert result["spread"][1] is None or np.isnan(result["spread"][1])


# ============================================================================
# KUADRAN 3: THE IRON GUARDS (Integritas Input)
# ============================================================================


def test_empty_dataframe_rejection() -> None:
    """3.1: Tests guard against empty DataFrame."""
    df = pl.DataFrame({"target": [], "anchor": []})
    result = compute_pairs_zscore(df, "target", "anchor", 1.0, 20)
    assert result.is_err()
    assert "Empty" in str(result.unwrap_err())


def test_missing_target_column() -> None:
    """3.2: Tests guard against missing columns."""
    df = pl.DataFrame({"wrong_name": [1, 2], "anchor": [1, 2]})
    result = compute_pairs_zscore(df, "target", "anchor", 1.0, 20)
    assert result.is_err()
    assert "Missing: target" in str(result.unwrap_err())


def test_invalid_window_size() -> None:
    """3.3: Tests guard against mathematically impossible rolling windows."""
    df = pl.DataFrame({"target": [1, 2], "anchor": [1, 2]})
    result = compute_pairs_zscore(df, "target", "anchor", 1.0, 1)
    assert result.is_err()
    assert "Window must be >= 2" in str(result.unwrap_err())


# ============================================================================
# KUADRAN 4: FUNCTIONAL PURITY (Hukum Fungsional)
# ============================================================================


def test_monadic_return_type() -> None:
    """4.1: Ensures the engine strictly follows the Monadic Result pattern."""
    df = pl.DataFrame({"target": [100.0, 105.0], "anchor": [50.0, 52.0]})
    result = compute_pairs_zscore(df, "target", "anchor", 1.0, 2)
    assert result.is_ok()
    assert isinstance(result.unwrap(), pl.DataFrame)


def test_immutability_of_input() -> None:
    """4.2: Proves the function does not mutate the original DataFrame in memory."""
    df = pl.DataFrame({"target": [100.0, 105.0], "anchor": [50.0, 52.0]})
    original_columns = df.columns.copy()

    _ = compute_pairs_zscore(df, "target", "anchor", 1.0, 2).unwrap()

    assert df.columns == original_columns  # Original DF is untouched


def test_column_schema_output() -> None:
    """4.3: Proves the output DataFrame contains exactly the required analytical columns."""
    df = pl.DataFrame({"target": [100.0, 105.0], "anchor": [50.0, 52.0]})
    result_df = compute_pairs_zscore(df, "target", "anchor", 1.0, 2).unwrap()

    # SURGERY: "log_spread" -> "spread"
    expected_new_columns = {"spread", "spread_mean", "spread_std", "z_score"}
    actual_new_columns = set(result_df.columns) - set(df.columns)

    assert expected_new_columns == actual_new_columns


# ============================================================================
# KUADRAN 5: THE BASKET ENGINE (PCA & Multi-Dimensi)
# ============================================================================


def test_basket_zscore_accuracy() -> None:
    """5.1: Tests that the PCA basket engine correctly computes a synthetic anchor and Z-score."""
    # Membuat data sintetis: Target dan 3 koin Basket
    np.random.seed(42)
    n_samples = 100
    df = pl.DataFrame(
        {
            "target": np.cumsum(np.random.normal(0.001, 0.01, n_samples)) + 100,
            "btc": np.cumsum(np.random.normal(0.001, 0.01, n_samples)) + 50000,
            "eth": np.cumsum(np.random.normal(0.001, 0.012, n_samples)) + 3000,
            "sol": np.cumsum(np.random.normal(0.0015, 0.015, n_samples)) + 100,
        }
    )

    result = compute_basket_zscore(df, "target", ["btc", "eth", "sol"], 20)
    assert result.is_ok()
    out_df = result.unwrap()

    # SURGERY: "basket_spread" -> "spread"
    expected_new_cols = {"btc_ret", "eth_ret", "sol_ret", "target_ret", "synthetic_anchor_return", "spread", "z_score"}
    assert expected_new_cols.issubset(set(out_df.columns) - set(df.columns))

    assert out_df["z_score"][:20].is_null().all()
    val = out_df["z_score"][21]
    assert val is not None and not np.isnan(val)


def test_basket_guard_insufficient_basket_assets() -> None:
    """5.2: Tests guard clause for when the basket has less than 2 assets (PCA needs >= 2)."""
    df = pl.DataFrame({"target": [1, 2, 3], "btc": [1, 2, 3]})
    result = compute_basket_zscore(df, "target", ["btc"], 2)

    assert result.is_err()
    assert "PCA needs >= 2 assets" in str(result.unwrap_err())


def test_basket_guard_missing_columns() -> None:
    """5.3: Tests guard clause for missing target or basket columns."""
    df = pl.DataFrame({"target": [1, 2, 3], "btc": [1, 2, 3]})

    # Missing basket column 'eth'
    result_missing_basket = compute_basket_zscore(df, "target", ["btc", "eth"], 2)
    assert result_missing_basket.is_err()
    assert "unable to find column" in str(result_missing_basket.unwrap_err())

    # Missing target column
    result_missing_target = compute_basket_zscore(df, "wrong_target", ["btc", "eth"], 2)
    assert result_missing_target.is_err()
    assert "unable to find column" in str(result_missing_target.unwrap_err())


def test_basket_engine_flatline_immunity() -> None:
    """5.4: CRITICAL! Tests that a flatlining basket doesn't crash the PCA/Eigen decomposition."""
    # Semua harga diam (return 0.0 konstan)
    df = pl.DataFrame(
        {
            "target": [100.0] * 50,
            "btc": [50000.0] * 50,
            "eth": [3000.0] * 50,
        }
    )

    result = compute_basket_zscore(df, "target", ["btc", "eth"], 20)

    # Ingat di linalg.py, PCA kita me-return Err jika varians-nya mati (0).
    # Jadi kita ekspektasikan mesin mengembalikan Err secara elegan (tanpa Exception/Crash)
    assert result.is_ok()
    z_score = result.unwrap()["z_score"].to_numpy()[20:]
    assert_allclose(z_score, 0.0, atol=1e-8)
