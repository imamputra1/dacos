"""
LINEAR ALGEBRA MODULE (THE MATH ENGINE)
Location: src/dacos/core/linalg.py
Paradigm: Pure Hardware Kernels (Numba), No-Magic, Explicit Matrix Operations.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from dacos.utils import Err, Ok, Result

# ============================================================================
# ⚙️ LEVEL 1: HARDWARE KERNELS (C-LEVEL EXECUTION via NUMBA)
# Fungsi-fungsi ini dirancang untuk dieksekusi di dalam tight-loop (seperti Kalman)
# tanpa overhead Python atau BLAS yang berlebihan untuk matriks kecil.
# ============================================================================

@njit(cache=True, fastmath=True)
def _kernel_mat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Standard Matrix Multiplication."""
    return np.dot(a, b)


@njit(cache=True, fastmath=True)
def _kernel_inv_2x2(a: np.ndarray) -> np.ndarray:
    """
    Hyper-optimized 2x2 matrix inversion bypassing standard BLAS overhead.
    Sangat krusial untuk spread/hedge ratio pada Pairs Trading (2 variabel state).
    """
    determinant = a[0, 0] * a[1, 1] - a[0, 1] * a[1, 0]
    inv_det = 1.0 / determinant

    out = np.empty((2, 2), dtype=np.float64)
    out[0, 0] = a[1, 1] * inv_det
    out[0, 1] = -a[0, 1] * inv_det
    out[1, 0] = -a[1, 0] * inv_det
    out[1, 1] = a[0, 0] * inv_det

    return out


@njit(cache=True, fastmath=True)
def _kernel_mat_inv_general(a: np.ndarray) -> np.ndarray:
    """General matrix inversion for N > 2."""
    return np.linalg.inv(a)


@njit(cache=True, fastmath=True)
def _kernel_aba_t_add_c(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """
    Computes: (A @ B @ A.T) + C
    Operasi linear murni. Di dunia luar sering dipakai untuk Covariance Prediction (F*P*F.T + Q).
    """
    # A @ B
    step_1 = np.dot(a, b)
    # (A @ B) @ A.T
    step_2 = np.dot(step_1, a.T)
    # + C
    return step_2 + c


@njit(cache=True, fastmath=True)
def _kernel_a_b_t(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Computes: A @ B.T
    Operasi linear murni. Sering dipakai untuk cross-covariance.
    """
    return np.dot(a, b.T)


@njit(cache=True, fastmath=True)
def _kernel_a_b_inv(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Computes: A @ B^-1
    Digunakan untuk komputasi rasio seperti Kalman Gain (P @ H.T @ S^-1).
    """
    # Catatan: Untuk optimasi ekstrim jika B adalah 2x2, kita bisa pakai _kernel_inv_2x2
    if b.shape[0] == 2 and b.shape[1] == 2:
        b_inv = _kernel_inv_2x2(b)
    else:
        b_inv = np.linalg.inv(b)

    return np.dot(a, b_inv)


@njit(cache=True, fastmath=True)
def _kernel_i_minus_kh_p(k: np.ndarray, h: np.ndarray, p: np.ndarray) -> np.ndarray:
    """
    Computes: (I - K @ H) @ P
    Operasi linear murni. Membutuhkan pembuatan Identity Matrix secara instan.
    """
    state_dim = p.shape[0]
    identity = np.eye(state_dim, dtype=np.float64)
    kh = np.dot(k, h)
    i_minus_kh = identity - kh
    return np.dot(i_minus_kh, p)

# ============================================================================
# ⚙️ LEVEL 1.5: PCA HARDWARE KERNELS (BASKET TRADING)
# ============================================================================

@njit(cache=True, fastmath=True)
def _kernel_covariance_centered(x: np.ndarray) -> np.ndarray:
    """
    Computes the covariance matrix for a mean-centered matrix X.
    X shape: (n_samples, n_features/assets).
    Sangat cepat untuk Numba karena hanya operasi dot product dan skalar.
    """
    n_samples = x.shape[0]
    if n_samples <= 1:
        # Menghindari pembagian dengan nol
        return np.zeros((x.shape[1], x.shape[1]), dtype=np.float64)

    return np.dot(x.T, x) / (n_samples - 1.0)


@njit(cache=True, fastmath=True)
def _kernel_pca_components(cov_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes Eigendecomposition for a symmetric covariance matrix.
    Returns:
        - sorted_eigenvalues (1D array, descending)
        - sorted_eigenvectors (2D array, columns are principal components)
    """
    # eigh khusus untuk matriks simetris (Covariance), jauh lebih cepat dari eig biasa
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # argsort secara default adalah ascending. Numba tidak support [::-1] pada argsort secara langsung dengan mudah,
    # jadi kita lakukan pembalikan indeks manual.
    n = eigenvalues.shape[0]
    idx = np.argsort(eigenvalues)

    # Reverse index untuk Descending Order (PC1 di kolom indeks 0)
    desc_idx = np.empty(n, dtype=np.int64)
    for i in range(n):
        desc_idx[i] = idx[n - 1 - i]

    sorted_eigenvalues = eigenvalues[desc_idx]
    sorted_eigenvectors = eigenvectors[:, desc_idx]

    return sorted_eigenvalues, sorted_eigenvectors


# ============================================================================
# 🛡️ LEVEL 2: SAFE MONADIC WRAPPERS (PYTHON RUNTIME PROTECTOR)
# Digunakan jika orca butuh validasi dimensi sebelum loop berat dimulai.
# ============================================================================

def invert_matrix_safe(matrix: np.ndarray) -> Result[np.ndarray, ValueError]:
    """
    Safe wrapper for matrix inversion. Will route to hyper-optimized 2x2 if applicable.
    """
    if matrix.ndim != 2:
        return Err(ValueError(f"Matrix must be 2D, got {matrix.ndim}D."))

    rows, cols = matrix.shape
    if rows != cols:
        return Err(ValueError(f"Matrix must be square, got {rows}x{cols}."))

    try:
        safe_matrix = matrix.astype(np.float64)
        if rows == 2:
            return Ok(_kernel_inv_2x2(safe_matrix))
        return Ok(_kernel_mat_inv_general(safe_matrix))
    except Exception as e:
        return Err(ValueError(f"Matrix inversion failed (Singular matrix?): {e}"))

def compute_pca_safe(data_matrix: np.ndarray) -> Result[tuple[np.ndarray, np.ndarray], ValueError]:
    """
    Safe wrapper for computing Principal Component Analysis (PCA).
    Validates dimensions, mean-centers the data, and routes to Numba kernels.
    
    Args:
        data_matrix: 2D Numpy array of shape (n_samples, n_features) representing aligned asset returns/prices.

    Returns:
        Ok((eigenvalues, eigenvectors)) sorted in descending order of explained variance,
        or Err(ValueError) if validation fails.
    """
    # 1. Guard Clauses: Validasi Dimensi Dasar
    if data_matrix.ndim != 2:
        return Err(ValueError(f"PCA requires a 2D matrix (samples x features), got {data_matrix.ndim}D."))

    n_samples, n_features = data_matrix.shape

    if n_samples <= 1:
        return Err(ValueError(f"PCA requires > 1 samples to compute covariance, got {n_samples}."))

    if n_features == 0:
        return Err(ValueError("PCA requires at least 1 feature (column) to compute."))

    try:
        # 2. Strict Type Casting
        safe_data = data_matrix.astype(np.float64)

        # 3. Mean Centering (Numpy C-API sangat cepat untuk ini)
        feature_means = np.mean(safe_data, axis=0)
        centered_data = safe_data - feature_means

        # 4. Panggil Kernel Numba untuk Covariance
        cov_matrix = _kernel_covariance_centered(centered_data)

        # 5. Panggil Kernel Numba untuk Eigendecomposition
        eigenvalues, eigenvectors = _kernel_pca_components(cov_matrix)

        return Ok((eigenvalues, eigenvectors))

    except Exception as computation_error:
        return Err(ValueError(f"PCA computation kernel panic: {computation_error}"))

__all__ = [
    "_kernel_pca_components",
    "_kernel_covariance_centered",
    "_kernel_mat_mul",
    "_kernel_inv_2x2",
    "_kernel_mat_inv_general",
    "_kernel_aba_t_add_c",
    "_kernel_a_b_t",
    "_kernel_a_b_inv",
    "_kernel_i_minus_kh_p",
    "invert_matrix_safe",
    "compute_pca_safe",
]
