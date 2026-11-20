"""Transparent empirical and measurement-error-corrected covariance estimators."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import (
    as_dense_2d,
    measurement_variances,
    require_dense_budget,
    validate_X_sd,
)
from .preprocessing import residualize
from .psd import PSDMethod, repair_covariance

FloatArray = NDArray[np.float64]
ShrinkageTarget = Literal["diagonal", "scaled_identity"]


def naive_covariance(X: ArrayLike, *, ddof: int = 1) -> FloatArray:
    """Return ordinary empirical covariance, ignoring measurement uncertainty.

    Parameters
    ----------
    X
        Measurements with shape ``(n_observations, n_features)``.
    ddof
        Degrees of freedom removed from the cross-product denominator.

    Notes
    -----
    This is a baseline. Under additive independent measurement error its
    diagonal includes both latent variance and average measurement variance.
    """
    X_array = as_dense_2d(X, name="X")
    denominator = X_array.shape[0] - ddof
    if denominator <= 0:
        raise ValueError("ddof must be smaller than the number of observations.")
    centered = X_array - X_array.mean(axis=0)
    return np.asarray(centered.T @ centered / denominator, dtype=np.float64)


def _shrink_covariance(
    covariance: FloatArray,
    *,
    shrinkage: float,
    target: ShrinkageTarget,
    variance_floor: float,
) -> FloatArray:
    if not 0 <= shrinkage <= 1:
        raise ValueError("shrinkage must lie in [0, 1].")
    if target == "diagonal":
        target_matrix = np.diag(np.maximum(np.diag(covariance), variance_floor))
    elif target == "scaled_identity":
        scale = max(float(np.trace(covariance) / covariance.shape[0]), variance_floor)
        target_matrix = np.eye(covariance.shape[0]) * scale
    else:
        raise ValueError("shrinkage_target must be 'diagonal' or 'scaled_identity'.")
    return (1.0 - shrinkage) * covariance + shrinkage * target_matrix


def corrected_covariance(
    X: ArrayLike,
    sd: ArrayLike | None = None,
    *,
    error_cov: ArrayLike | None = None,
    covariates: ArrayLike | None = None,
    fit_intercept: bool = True,
    psd: PSDMethod = "none",
    ensure_psd: bool | None = None,
    min_eigenvalue: float = 0.0,
    shrinkage: float = 0.0,
    shrinkage_target: ShrinkageTarget = "diagonal",
    max_dense_bytes: int = 512 * 1024**2,
) -> FloatArray:
    """Estimate latent covariance by subtracting known measurement covariance.

    Parameters
    ----------
    X, sd
        Cells/observations by features arrays. ``sd[c, g]`` is the known
        Gaussian measurement standard deviation for ``X[c, g]``.
    error_cov
        Optional full error covariance with shape ``(C, G, G)``. Supply this
        instead of ``sd`` when correlated measurement errors are known.
    covariates
        Optional observed mean-model covariates. When supplied, covariance is
        estimated from OLS residuals and the noise correction accounts for
        residual leverage. This estimates within-mean-model covariance.
    psd
        ``"none"`` preserves the raw unbiased correction, ``"clip"`` clips
        eigenvalues, and ``"nearest"`` uses alternating PSD/diagonal
        projections. PSD repair is regularization, not biological evidence.
    ensure_psd
        Backward-compatible convenience: ``True`` selects ``psd="clip"`` and
        ``False`` selects ``psd="none"``.
    shrinkage
        Explicit convex shrinkage intensity in ``[0, 1]``. It is user chosen:
        ordinary Ledoit-Wolf/OAS intensity formulas do not directly apply to a
        covariance after heteroscedastic noise subtraction.
    shrinkage_target
        ``"diagonal"`` or ``"scaled_identity"``.

    Returns
    -------
    numpy.ndarray
        Estimated latent covariance with shape ``(G, G)``.
    """
    if sd is not None:
        X_array, sd_array = validate_X_sd(X, sd, max_bytes=max_dense_bytes)
        diagonal_variances: FloatArray | None = sd_array**2
        full_errors: FloatArray | None = None
        if error_cov is not None:
            raise ValueError("Provide sd or error_cov, not both.")
    else:
        X_array = as_dense_2d(X, name="X", max_bytes=max_dense_bytes)
        diagonal_variances, full_errors = measurement_variances(
            X_array, None, error_cov, max_bytes=max_dense_bytes
        )
    _, n_features = X_array.shape
    require_dense_budget(
        (n_features, n_features),
        max_bytes=max_dense_bytes,
        purpose="forming the covariance",
        arrays=4,
    )
    if ensure_psd is not None:
        requested: PSDMethod = "clip" if ensure_psd else "none"
        if psd != "none" and psd != requested:
            raise ValueError("ensure_psd and psd specify conflicting policies.")
        psd = requested

    if covariates is None:
        empirical = naive_covariance(X_array)
        if diagonal_variances is not None:
            correction = np.diag(diagonal_variances.mean(axis=0))
        else:
            assert full_errors is not None
            correction = full_errors.mean(axis=0)
    else:
        residualized = residualize(X_array, covariates, fit_intercept=fit_intercept)
        residuals = residualized.residuals
        denominator = residualized.residual_degrees_of_freedom
        empirical = residuals.T @ residuals / denominator
        residual_weights = 1.0 - residualized.leverage
        if diagonal_variances is not None:
            corrected_noise = residual_weights @ diagonal_variances / denominator
            correction = np.diag(corrected_noise)
        else:
            assert full_errors is not None
            correction = np.einsum("c,cij->ij", residual_weights, full_errors) / denominator

    estimate = (empirical - correction + (empirical - correction).T) / 2.0
    if shrinkage:
        estimate = _shrink_covariance(
            estimate,
            shrinkage=shrinkage,
            target=shrinkage_target,
            variance_floor=max(min_eigenvalue, np.finfo(float).eps),
        )
    return repair_covariance(estimate, method=psd, min_eigenvalue=min_eigenvalue)


def estimate_covariance(
    X: ArrayLike,
    sd: ArrayLike | None = None,
    *,
    method: Literal["naive", "corrected"] = "corrected",
    **kwargs: Any,
) -> FloatArray:
    """Dispatch the two transparent sample-covariance estimators.

    Likelihood and factor estimators are stateful classes because their fitted
    parameters and convergence diagnostics are scientifically relevant.
    """
    if method == "naive":
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"naive covariance does not accept: {unexpected}")
        return naive_covariance(X)
    if method == "corrected":
        return corrected_covariance(X, sd, **kwargs)
    raise ValueError("method must be 'naive' or 'corrected'.")
