"""Covariance and correlation conversions."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def covariance_to_correlation(
    covariance: ArrayLike,
    *,
    min_variance: float = 0.0,
    invalid: str = "raise",
) -> FloatArray:
    """Convert latent covariance to correlation using latent variances.

    Parameters
    ----------
    covariance
        Square latent covariance matrix.
    min_variance
        Variances at or below this threshold are considered invalid.
    invalid
        ``"raise"`` rejects invalid variances. ``"nan"`` returns NaN rows and
        columns for them.
    """
    covariance_array = np.asarray(covariance, dtype=np.float64)
    if covariance_array.ndim != 2 or covariance_array.shape[0] != covariance_array.shape[1]:
        raise ValueError("covariance must be a square matrix.")
    if not np.all(np.isfinite(covariance_array)):
        raise ValueError("covariance must contain only finite values.")
    variances = np.diag(covariance_array)
    valid = variances > min_variance
    if not np.all(valid) and invalid == "raise":
        bad = np.flatnonzero(~valid).tolist()
        raise ValueError(f"Non-positive or too-small latent variances at indices {bad}.")
    if invalid not in {"raise", "nan"}:
        raise ValueError("invalid must be 'raise' or 'nan'.")
    scale = np.sqrt(np.where(valid, variances, np.nan))
    correlation = covariance_array / np.outer(scale, scale)
    correlation = np.clip(correlation, -1.0, 1.0)
    np.fill_diagonal(correlation, np.where(valid, 1.0, np.nan))
    return correlation
