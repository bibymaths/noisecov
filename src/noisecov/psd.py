"""Positive-semidefinite covariance repair."""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

PSDMethod = Literal["none", "clip", "nearest"]
FloatArray = NDArray[np.float64]


def _symmetric(matrix: FloatArray) -> FloatArray:
    return (matrix + matrix.T) / 2.0


def is_positive_semidefinite(matrix: ArrayLike, *, tolerance: float = 1e-10) -> bool:
    """Return whether a symmetric matrix is positive semidefinite within tolerance."""
    array = np.asarray(matrix, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError("matrix must be square.")
    if not np.allclose(array, array.T, atol=tolerance, rtol=0):
        return False
    return bool(np.linalg.eigvalsh(array)[0] >= -tolerance)


def clip_to_psd(matrix: ArrayLike, *, min_eigenvalue: float = 0.0) -> FloatArray:
    """Project a symmetric matrix onto the PSD cone by clipping eigenvalues.

    This is the nearest positive-semidefinite matrix in Frobenius norm when
    ``min_eigenvalue`` is zero. It does not preserve the input diagonal.
    """
    if min_eigenvalue < 0:
        raise ValueError("min_eigenvalue must be non-negative.")
    array = np.asarray(matrix, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError("matrix must be square.")
    eigenvalues, eigenvectors = np.linalg.eigh(_symmetric(array))
    repaired = (eigenvectors * np.maximum(eigenvalues, min_eigenvalue)) @ eigenvectors.T
    return _symmetric(repaired)


def nearest_covariance(
    matrix: ArrayLike,
    *,
    min_eigenvalue: float = 1e-10,
    max_iterations: int = 100,
    tolerance: float = 1e-8,
) -> FloatArray:
    """Find a PSD covariance while preserving a feasible, floored diagonal.

    The implementation applies Higham's alternating projections with Dykstra's
    correction to the corresponding correlation matrix. Negative input
    variances are first floored because no PSD matrix can preserve them.
    """
    if min_eigenvalue < 0:
        raise ValueError("min_eigenvalue must be non-negative.")
    array = np.asarray(matrix, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError("matrix must be square.")
    array = _symmetric(array)
    diagonal = np.maximum(np.diag(array), max(min_eigenvalue, np.finfo(float).eps))
    scale = np.sqrt(diagonal)
    correlation = array / np.outer(scale, scale)
    np.fill_diagonal(correlation, 1.0)

    correction = np.zeros_like(correlation)
    current = correlation.copy()
    for _ in range(max_iterations):
        previous = current
        residual = current - correction
        projected = clip_to_psd(residual, min_eigenvalue=min_eigenvalue)
        correction = projected - residual
        current = projected
        np.fill_diagonal(current, 1.0)
        relative_change = np.linalg.norm(current - previous, ord="fro") / max(
            np.linalg.norm(previous, ord="fro"), np.finfo(float).eps
        )
        if relative_change < tolerance:
            break

    # A final cone projection removes small negative eigenvalues left by the
    # last unit-diagonal projection. Renormalization restores the diagonal.
    current = clip_to_psd(current, min_eigenvalue=min_eigenvalue)
    corr_scale = np.sqrt(np.maximum(np.diag(current), np.finfo(float).eps))
    current /= np.outer(corr_scale, corr_scale)
    result = current * np.outer(scale, scale)
    return _symmetric(result)


def repair_covariance(
    matrix: ArrayLike,
    *,
    method: PSDMethod = "clip",
    min_eigenvalue: float = 0.0,
) -> FloatArray:
    """Apply the requested PSD policy to a covariance estimate."""
    array = np.asarray(matrix, dtype=np.float64)
    if method == "none":
        return _symmetric(array)
    if method == "clip":
        return clip_to_psd(array, min_eigenvalue=min_eigenvalue)
    if method == "nearest":
        return nearest_covariance(array, min_eigenvalue=max(min_eigenvalue, 1e-12))
    raise ValueError("method must be one of {'none', 'clip', 'nearest'}.")
