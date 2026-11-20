"""Input validation and memory guards."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import sparse

FloatArray = NDArray[np.float64]
FLOAT64 = np.dtype("float64")


def dense_nbytes(shape: tuple[int, ...], dtype: np.dtype[Any] = FLOAT64) -> int:
    """Return the bytes required by a dense array without allocating it."""
    return int(np.prod(shape, dtype=np.int64)) * dtype.itemsize


def require_dense_budget(
    shape: tuple[int, ...],
    *,
    max_bytes: int,
    purpose: str,
    arrays: int = 1,
) -> None:
    """Raise before a requested dense allocation exceeds ``max_bytes``."""
    required = dense_nbytes(shape) * arrays
    if required > max_bytes:
        gib = required / 1024**3
        limit = max_bytes / 1024**3
        raise MemoryError(
            f"{purpose} would require about {gib:.2f} GiB for {arrays} dense "
            f"array(s), above the configured {limit:.2f} GiB limit. Select fewer "
            "features or use HeteroscedasticFactorAnalysis."
        )


def as_dense_2d(
    value: Any,
    *,
    name: str,
    max_bytes: int = 512 * 1024**2,
) -> FloatArray:
    """Convert an array-like or sparse matrix to a finite float64 matrix."""
    if sparse.issparse(value):
        require_dense_budget(value.shape, max_bytes=max_bytes, purpose=f"densifying {name}")
        array = value.toarray()
    else:
        array = np.asarray(value)
    if array.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional; got shape {array.shape}.")
    require_dense_budget(array.shape, max_bytes=max_bytes, purpose=f"using {name}")
    result = np.asarray(array, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values.")
    return result


def validate_X_sd(
    X: Any,
    sd: Any,
    *,
    max_bytes: int = 512 * 1024**2,
) -> tuple[FloatArray, FloatArray]:
    """Validate a cells-by-features matrix and matching standard deviations."""
    X_array = as_dense_2d(X, name="X", max_bytes=max_bytes)
    sd_array = as_dense_2d(sd, name="sd", max_bytes=max_bytes)
    if X_array.shape != sd_array.shape:
        raise ValueError(
            f"X and sd must have the same shape; got {X_array.shape} and {sd_array.shape}."
        )
    if X_array.shape[0] < 2:
        raise ValueError("At least two observations are required.")
    if X_array.shape[1] < 1:
        raise ValueError("At least one feature is required.")
    if np.any(sd_array < 0):
        raise ValueError("sd must be non-negative.")
    return X_array, sd_array


def validate_error_covariance(
    error_cov: Any,
    *,
    n_observations: int,
    n_features: int,
    max_bytes: int = 512 * 1024**2,
    symmetry_tolerance: float = 1e-8,
) -> FloatArray:
    """Validate full per-observation measurement-error covariance matrices."""
    require_dense_budget(
        (n_observations, n_features, n_features),
        max_bytes=max_bytes,
        purpose="using error_cov",
    )
    value = np.asarray(error_cov, dtype=np.float64)
    expected = (n_observations, n_features, n_features)
    if value.shape != expected:
        raise ValueError(f"error_cov must have shape {expected}; got {value.shape}.")
    if not np.all(np.isfinite(value)):
        raise ValueError("error_cov must contain only finite values.")
    if not np.allclose(value, np.swapaxes(value, 1, 2), atol=symmetry_tolerance, rtol=0):
        raise ValueError("Each error_cov matrix must be symmetric.")
    for index, matrix in enumerate(value):
        if np.linalg.eigvalsh(matrix)[0] < -symmetry_tolerance:
            raise ValueError(f"error_cov[{index}] is not positive semidefinite.")
    return value


def measurement_variances(
    X: FloatArray,
    sd: Any | None,
    error_cov: Any | None,
    *,
    max_bytes: int,
) -> tuple[FloatArray | None, FloatArray | None]:
    """Validate exactly one diagonal or full measurement-error representation."""
    if (sd is None) == (error_cov is None):
        raise ValueError("Provide exactly one of sd or error_cov.")
    if sd is not None:
        _, sd_array = validate_X_sd(X, sd, max_bytes=max_bytes)
        return sd_array**2, None
    full = validate_error_covariance(
        error_cov,
        n_observations=X.shape[0],
        n_features=X.shape[1],
        max_bytes=max_bytes,
    )
    return None, full
