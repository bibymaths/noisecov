"""Explicit mean-model fitting and residualization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import as_dense_2d

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ResidualizationResult:
    """Output of an ordinary least-squares mean model."""

    residuals: FloatArray
    fitted: FloatArray
    coefficients: FloatArray
    design_matrix: FloatArray
    leverage: FloatArray
    rank: int
    residual_degrees_of_freedom: int


def prepare_design_matrix(
    covariates: ArrayLike | None,
    *,
    n_observations: int,
    fit_intercept: bool = True,
) -> FloatArray:
    """Construct and rank-check an explicit mean-model design matrix."""
    if covariates is None:
        design = np.empty((n_observations, 0), dtype=np.float64)
    else:
        design = as_dense_2d(covariates, name="covariates")
        if design.shape[0] != n_observations:
            raise ValueError("covariates must have the same number of rows as X.")
    if fit_intercept:
        design = np.column_stack((np.ones(n_observations), design))
    return design


def residualize(
    X: ArrayLike,
    covariates: ArrayLike,
    *,
    fit_intercept: bool = True,
) -> ResidualizationResult:
    """Fit ``X = Q B + R`` by OLS and return residualization diagnostics.

    Residualization changes the estimand from total across-cell covariance to
    covariance remaining after the supplied mean model. It is never performed
    implicitly by this function's callers.
    """
    X_array = as_dense_2d(X, name="X")
    design = prepare_design_matrix(
        covariates, n_observations=X_array.shape[0], fit_intercept=fit_intercept
    )
    if design.shape[1] == 0:
        raise ValueError("The mean model has no columns.")
    coefficients, _, rank, _ = np.linalg.lstsq(design, X_array, rcond=None)
    if rank < design.shape[1]:
        raise ValueError("The covariate design matrix is rank deficient.")
    degrees = X_array.shape[0] - int(rank)
    if degrees <= 0:
        raise ValueError("The mean model leaves no residual degrees of freedom.")
    fitted = design @ coefficients
    q, _ = np.linalg.qr(design, mode="reduced")
    leverage = np.sum(q**2, axis=1)
    return ResidualizationResult(
        residuals=X_array - fitted,
        fitted=fitted,
        coefficients=coefficients,
        design_matrix=design,
        leverage=leverage,
        rank=int(rank),
        residual_degrees_of_freedom=degrees,
    )
