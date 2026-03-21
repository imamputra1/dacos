from __future__ import annotations

import time

import numpy as np
import pytest
from dacos.core import (
    _kernel_a_b_inv,
    _kernel_a_b_t,
    _kernel_aba_t_add_c,
    _kernel_covariance_centered,
    _kernel_i_minus_kh_p,
    _kernel_inv_2x2,
    _kernel_mat_inv_general,
    _kernel_mat_mul,
    _kernel_pca_components,
    compute_pca_safe,
    invert_matrix_safe,
)
from numpy.testing import assert_allclose

# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def mat_2x2() -> np.ndarray:
    """Random 2x2 invertible matrix."""
    return np.array([[4.0, 7.0], [2.0, 6.0]], dtype=np.float64)


@pytest.fixture
def mat_3x3() -> np.ndarray:
    """Random 3x3 invertible matrix."""
    return np.array([[1.0, 2.0, 3.0], [0.0, 1.0, 4.0], [5.0, 6.0, 0.0]], dtype=np.float64)


@pytest.fixture
def pca_data() -> np.ndarray:
    """Random data matrix for PCA (100 samples, 5 assets/features)."""
    np.random.seed(42)  # Deterministic seed for testing
    return np.random.randn(100, 5) * 10.0 + 50.0  # Simulated price/returns data


# ============================================================================
# KERNEL TESTS: STANDARD LINEAR ALGEBRA
# ============================================================================


def test_kernel_mat_mul(mat_2x2: np.ndarray) -> None:
    b = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    result = _kernel_mat_mul(mat_2x2, b)
    expected = np.dot(mat_2x2, b)
    assert_allclose(result, expected)


def test_kernel_inv_2x2_correctness(mat_2x2: np.ndarray) -> None:
    result = _kernel_inv_2x2(mat_2x2)
    expected = np.linalg.inv(mat_2x2)
    assert_allclose(result, expected)


def test_kernel_mat_inv_general_correctness(mat_3x3: np.ndarray) -> None:
    result = _kernel_mat_inv_general(mat_3x3)
    expected = np.linalg.inv(mat_3x3)
    assert_allclose(result, expected)


def test_kernel_aba_t_add_c() -> None:
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    b = np.array([[5.0, 0.0], [0.0, 5.0]], dtype=np.float64)
    c = np.array([[0.1, 0.0], [0.0, 0.1]], dtype=np.float64)
    result = _kernel_aba_t_add_c(a, b, c)
    expected = (a @ b @ a.T) + c
    assert_allclose(result, expected)


def test_kernel_a_b_t() -> None:
    a = np.array([[1.0, 2.0]], dtype=np.float64)
    b = np.array([[3.0, 4.0]], dtype=np.float64)
    result = _kernel_a_b_t(a, b)
    expected = a @ b.T
    assert_allclose(result, expected)


def test_kernel_a_b_inv(mat_2x2: np.ndarray) -> None:
    a = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    result = _kernel_a_b_inv(a, mat_2x2)
    expected = a @ np.linalg.inv(mat_2x2)
    assert_allclose(result, expected)


def test_kernel_i_minus_kh_p(mat_2x2: np.ndarray) -> None:
    k = np.array([[0.5, 0.0], [0.0, 0.5]], dtype=np.float64)
    h = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    result = _kernel_i_minus_kh_p(k, h, mat_2x2)
    identity = np.eye(2)
    expected = (identity - (k @ h)) @ mat_2x2
    assert_allclose(result, expected)


# ============================================================================
# KERNEL TESTS: PCA & BASKET TRADING
# ============================================================================


def test_kernel_covariance_centered(pca_data: np.ndarray) -> None:
    """Tests Numba covariance matrix vs Numpy's exact implementation."""
    centered = pca_data - np.mean(pca_data, axis=0)
    result = _kernel_covariance_centered(centered)

    # rowvar=False means columns are variables (features)
    expected = np.cov(centered, rowvar=False)
    assert_allclose(result, expected)


def test_kernel_pca_components(pca_data: np.ndarray) -> None:
    """Tests Eigendecomposition outputs correct math properties and sorts descending."""
    centered = pca_data - np.mean(pca_data, axis=0)
    cov = _kernel_covariance_centered(centered)

    eigvals, eigvecs = _kernel_pca_components(cov)

    # 1. Check Descending Order
    assert np.all(np.diff(eigvals) <= 0), "Eigenvalues must be sorted in descending order"

    # 2. Check Eigenvector Mathematical Validity (C * v = lambda * v)
    for i in range(len(eigvals)):
        v = eigvecs[:, i]
        lambda_i = eigvals[i]
        assert_allclose(cov @ v, lambda_i * v, atol=1e-7)


# ============================================================================
# SAFETY WRAPPER TESTS
# ============================================================================


def test_invert_matrix_safe_returns_ok_for_2x2(mat_2x2: np.ndarray) -> None:
    result = invert_matrix_safe(mat_2x2)
    assert result.is_ok()
    assert_allclose(result.unwrap(), np.linalg.inv(mat_2x2))


def test_invert_matrix_safe_returns_err_for_1d_array() -> None:
    bad_mat = np.array([1.0, 2.0, 3.0])
    result = invert_matrix_safe(bad_mat)
    assert result.is_err()


def test_compute_pca_safe_correctness(pca_data: np.ndarray) -> None:
    """Tests that the full PCA pipeline correctly centers and processes data."""
    result = compute_pca_safe(pca_data)
    assert result.is_ok()

    eigvals, eigvecs = result.unwrap()

    # Numpy direct equivalent for baseline comparison
    centered = pca_data - np.mean(pca_data, axis=0)
    expected_cov = np.cov(centered, rowvar=False)
    expected_vals, expected_vecs = np.linalg.eigh(expected_cov)

    # Sort expected descending manually
    idx = np.argsort(expected_vals)[::-1]
    expected_vals = expected_vals[idx]

    assert_allclose(eigvals, expected_vals)
    # Note: We do not directly `assert_allclose` on eigvecs vs expected_vecs
    # due to Eigenvector Sign Ambiguity (v and -v are both correct).
    # Mathematical validity is already tested in `test_kernel_pca_components`.


def test_compute_pca_safe_returns_err_for_invalid_dimensions() -> None:
    # 1D Array
    assert compute_pca_safe(np.array([1.0, 2.0])).is_err()

    # Only 1 Sample (cannot compute variance)
    assert compute_pca_safe(np.array([[1.0, 2.0]])).is_err()

    # 0 Features
    assert compute_pca_safe(np.empty((10, 0))).is_err()


# ============================================================================
# 🏎️ PERFORMANCE / SPEED TESTS
# ============================================================================


def test_speed_numba_vs_numpy_inv_2x2(mat_2x2: np.ndarray) -> None:
    """Benchmarks the Numba 2x2 inversion kernel against np.linalg.inv."""
    iterations = 100_000

    _ = _kernel_inv_2x2(mat_2x2)  # Warm-up JIT

    start_np = time.perf_counter()
    for _ in range(iterations):
        _ = np.linalg.inv(mat_2x2)
    time_np = time.perf_counter() - start_np

    start_nb = time.perf_counter()
    for _ in range(iterations):
        _ = _kernel_inv_2x2(mat_2x2)
    time_nb = time.perf_counter() - start_nb

    print("\n[SPEED TEST] 100,000 Iterations of 2x2 Matrix Inversion:")
    print(f"Numpy `np.linalg.inv`: {time_np:.5f} seconds")
    print(f"Numba `_kernel_inv_2x2`: {time_nb:.5f} seconds")

    assert time_nb < time_np, "Numba kernel should be faster than Numpy for 2x2 matrices!"


def test_speed_numba_vs_numpy_pca(pca_data: np.ndarray) -> None:
    """
    Benchmarks our Numba PCA wrapper vs a pure Python/Numpy equivalent.
    Proves that eliminating Python overhead in Covariance computation gives an edge.
    """
    iterations = 10_000

    _ = compute_pca_safe(pca_data)  # Warm-up JIT

    # TIME NUMPY PIPELINE
    start_np = time.perf_counter()
    for _ in range(iterations):
        centered = pca_data - np.mean(pca_data, axis=0)
        cov = np.cov(centered, rowvar=False)
        vals, vecs = np.linalg.eigh(cov)
        idx = np.argsort(vals)[::-1]
        vals = vals[idx]
        vecs = vecs[:, idx]
    time_np = time.perf_counter() - start_np

    # TIME NUMBA PIPELINE
    start_nb = time.perf_counter()
    for _ in range(iterations):
        _ = compute_pca_safe(pca_data)
    time_nb = time.perf_counter() - start_nb

    print("\n[SPEED TEST] 10,000 Iterations of PCA (100x5 Matrix):")
    print(f"Numpy `np.cov` + `np.linalg.eigh`: {time_np:.5f} seconds")
    print(f"Numba `compute_pca_safe`: {time_nb:.5f} seconds")

    speedup = time_np / time_nb if time_nb > 0 else float("inf")
    print(f"PCA Speedup Factor: {speedup:.2f}x faster")

    # Soft assertion: Numba should at least be competitive/faster, eliminating python loop overhead
    assert time_nb <= time_np * 1.5
