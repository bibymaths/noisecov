"""Dense heteroscedastic Gaussian covariance maximum likelihood."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import linalg, optimize

from ._validation import as_dense_2d, measurement_variances, require_dense_budget
from .correlation import covariance_to_correlation
from .covariance import corrected_covariance
from .preprocessing import prepare_design_matrix
from .psd import repair_covariance

FloatArray = NDArray[np.float64]


class HeteroscedasticGaussianCovariance:
    """Estimate a dense latent covariance under known observation errors.

    The fitted model is

    ``x_c ~ Normal(q_c @ B, Sigma + D_c)``,

    where ``D_c`` is known and ``Sigma`` is parameterized as
    ``L @ L.T + min_variance * I``. For each proposed ``Sigma``, the mean-model
    coefficients ``B`` are profiled exactly by generalized least squares.

    Parameters
    ----------
    max_iter
        Maximum L-BFGS-B iterations.
    tolerance
        Optimization tolerance.
    min_variance
        Non-negative numerical floor added to the latent covariance.
    fit_intercept
        Include an intercept in the mean model.
    max_dense_bytes
        Guard for dense intermediate allocations.
    verbose
        Pass optimizer progress to SciPy.
    """

    def __init__(
        self,
        *,
        max_iter: int = 200,
        tolerance: float = 1e-7,
        min_variance: float = 1e-8,
        fit_intercept: bool = True,
        max_dense_bytes: int = 512 * 1024**2,
        verbose: bool = False,
    ) -> None:
        if max_iter < 1:
            raise ValueError("max_iter must be positive.")
        if tolerance <= 0:
            raise ValueError("tolerance must be positive.")
        if min_variance < 0:
            raise ValueError("min_variance must be non-negative.")
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.min_variance = min_variance
        self.fit_intercept = fit_intercept
        self.max_dense_bytes = max_dense_bytes
        self.verbose = verbose

    @staticmethod
    def _pack_cholesky(cholesky: FloatArray) -> FloatArray:
        rows, cols = np.tril_indices(cholesky.shape[0])
        values = cholesky[rows, cols].copy()
        diagonal = rows == cols
        values[diagonal] = np.log(np.maximum(values[diagonal], 1e-12))
        return values

    @staticmethod
    def _unpack_cholesky(parameters: FloatArray, n_features: int) -> FloatArray:
        result = np.zeros((n_features, n_features), dtype=np.float64)
        rows, cols = np.tril_indices(n_features)
        result[rows, cols] = parameters
        diagonal = np.diag_indices(n_features)
        result[diagonal] = np.exp(result[diagonal])
        return result

    @staticmethod
    def _profile_coefficients(
        X: FloatArray,
        design: FloatArray,
        precisions: list[FloatArray],
    ) -> FloatArray:
        n_predictors = design.shape[1]
        n_features = X.shape[1]
        if n_predictors == 0:
            return np.empty((0, n_features), dtype=np.float64)
        dimension = n_predictors * n_features
        normal = np.zeros((dimension, dimension), dtype=np.float64)
        right = np.zeros(dimension, dtype=np.float64)
        for x_c, q_c, precision in zip(X, design, precisions, strict=True):
            qq = np.outer(q_c, q_c)
            normal += np.kron(precision, qq)
            right += np.outer(q_c, precision @ x_c).reshape(-1, order="F")
        coefficients = linalg.solve(normal, right, assume_a="pos", check_finite=False)
        return coefficients.reshape((n_predictors, n_features), order="F")

    def _objective(
        self,
        parameters: FloatArray,
        X: FloatArray,
        design: FloatArray,
        diagonal_errors: FloatArray | None,
        full_errors: FloatArray | None,
    ) -> tuple[float, FloatArray]:
        n_observations, n_features = X.shape
        latent_cholesky = self._unpack_cholesky(parameters, n_features)
        covariance = latent_cholesky @ latent_cholesky.T
        covariance.flat[:: n_features + 1] += self.min_variance
        identity = np.eye(n_features)

        decompositions: list[tuple[FloatArray, bool]] = []
        precisions: list[FloatArray] = []
        for cell in range(n_observations):
            total = covariance.copy()
            if diagonal_errors is not None:
                total.flat[:: n_features + 1] += diagonal_errors[cell]
            else:
                assert full_errors is not None
                total += full_errors[cell]
            factor = linalg.cho_factor(total, lower=True, check_finite=False)
            decompositions.append(factor)
            precisions.append(linalg.cho_solve(factor, identity, check_finite=False))

        coefficients = self._profile_coefficients(X, design, precisions)
        fitted = design @ coefficients if design.shape[1] else np.zeros_like(X)
        gradient_covariance = np.zeros_like(covariance)
        negative_log_likelihood = 0.0
        constant = n_features * np.log(2.0 * np.pi)
        for cell, (factor, precision) in enumerate(zip(decompositions, precisions, strict=True)):
            residual = X[cell] - fitted[cell]
            solved = linalg.cho_solve(factor, residual, check_finite=False)
            log_determinant = 2.0 * np.log(np.diag(factor[0])).sum()
            negative_log_likelihood += 0.5 * (constant + log_determinant + residual @ solved)
            gradient_covariance += 0.5 * (precision - np.outer(solved, solved))

        # The envelope theorem permits differentiating the profiled objective
        # while holding the exact GLS coefficients fixed.
        gradient_cholesky = 2.0 * gradient_covariance @ latent_cholesky
        rows, cols = np.tril_indices(n_features)
        gradient_parameters = gradient_cholesky[rows, cols]
        diagonal = rows == cols
        gradient_parameters[diagonal] *= latent_cholesky[rows[diagonal], cols[diagonal]]
        return (
            float(negative_log_likelihood / n_observations),
            gradient_parameters / n_observations,
        )

    def fit(
        self,
        X: ArrayLike,
        sd: ArrayLike | None = None,
        *,
        error_cov: ArrayLike | None = None,
        covariates: ArrayLike | None = None,
        initial_covariance: ArrayLike | None = None,
    ) -> HeteroscedasticGaussianCovariance:
        """Fit the heteroscedastic marginal likelihood.

        Supply exactly one of ``sd`` or ``error_cov``. Full measurement-error
        covariance is supported for moderate feature counts.
        """
        X_array = as_dense_2d(X, name="X", max_bytes=self.max_dense_bytes)
        n_observations, n_features = X_array.shape
        if n_observations < 2:
            raise ValueError("At least two observations are required.")
        diagonal_errors, full_errors = measurement_variances(
            X_array, sd, error_cov, max_bytes=self.max_dense_bytes
        )
        require_dense_budget(
            (n_features, n_features),
            max_bytes=self.max_dense_bytes,
            purpose="fitting dense heteroscedastic covariance",
            arrays=max(8, min(n_observations, 32)),
        )
        design = prepare_design_matrix(
            covariates,
            n_observations=n_observations,
            fit_intercept=self.fit_intercept,
        )
        if design.shape[1] and np.linalg.matrix_rank(design) < design.shape[1]:
            raise ValueError("The mean-model design matrix is rank deficient.")
        if design.shape[1] * n_features > 2_000:
            raise MemoryError(
                "The profiled GLS mean model is too large. Reduce covariates/features "
                "or residualize with a scientifically justified external model."
            )

        if initial_covariance is None:
            initial = corrected_covariance(
                X_array,
                None if diagonal_errors is None else np.sqrt(diagonal_errors),
                error_cov=full_errors,
                covariates=covariates,
                fit_intercept=self.fit_intercept,
                psd="clip",
                min_eigenvalue=max(self.min_variance, 1e-6),
                max_dense_bytes=self.max_dense_bytes,
            )
        else:
            initial = np.asarray(initial_covariance, dtype=np.float64)
            if initial.shape != (n_features, n_features):
                raise ValueError("initial_covariance has the wrong shape.")
            initial = repair_covariance(
                initial, method="clip", min_eigenvalue=max(self.min_variance, 1e-6)
            )
        initial_cholesky = linalg.cholesky(
            initial + np.eye(n_features) * 1e-8,
            lower=True,
            check_finite=False,
        )
        packed = self._pack_cholesky(initial_cholesky)
        rows, cols = np.tril_indices(n_features)
        bounds = [
            (-20.0, 20.0) if row == col else (None, None)
            for row, col in zip(rows, cols, strict=True)
        ]

        result = optimize.minimize(
            self._objective,
            packed,
            args=(X_array, design, diagonal_errors, full_errors),
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            options={
                "maxiter": self.max_iter,
                "ftol": self.tolerance,
                "gtol": self.tolerance,
            },
        )
        latent_cholesky = self._unpack_cholesky(result.x, n_features)
        covariance = latent_cholesky @ latent_cholesky.T
        covariance.flat[:: n_features + 1] += self.min_variance

        # Recompute profiled coefficients at the optimum for public attributes.
        identity = np.eye(n_features)
        precisions = []
        for cell in range(n_observations):
            total = covariance.copy()
            if diagonal_errors is not None:
                total.flat[:: n_features + 1] += diagonal_errors[cell]
            else:
                assert full_errors is not None
                total += full_errors[cell]
            factor = linalg.cho_factor(total, lower=True, check_finite=False)
            precisions.append(linalg.cho_solve(factor, identity, check_finite=False))
        coefficients = self._profile_coefficients(X_array, design, precisions)
        fitted = design @ coefficients if design.shape[1] else np.zeros_like(X_array)

        self.n_features_in_ = n_features
        self.n_observations_ = n_observations
        self.design_matrix_ = design
        self.coefficients_ = coefficients
        self.fitted_mean_ = fitted
        self.mean_ = fitted.mean(axis=0)
        self.cholesky_ = latent_cholesky
        self.covariance_ = covariance
        self.correlation_ = covariance_to_correlation(covariance)
        self.negative_log_likelihood_ = float(result.fun * n_observations)
        self.converged_ = bool(result.success)
        self.n_iter_ = int(result.nit)
        self.optimization_result_: optimize.OptimizeResult = result
        return self

    def score(
        self,
        X: ArrayLike,
        sd: ArrayLike | None = None,
        *,
        error_cov: ArrayLike | None = None,
        covariates: ArrayLike | None = None,
    ) -> float:
        """Return mean log likelihood under the fitted covariance.

        Mean-model coefficients are re-profiled for the supplied dataset.
        """
        if not hasattr(self, "covariance_"):
            raise RuntimeError("fit must be called before score.")
        X_array = as_dense_2d(X, name="X", max_bytes=self.max_dense_bytes)
        if X_array.shape[1] != self.n_features_in_:
            raise ValueError("X has a different feature count from the fitted model.")
        diagonal_errors, full_errors = measurement_variances(
            X_array, sd, error_cov, max_bytes=self.max_dense_bytes
        )
        design = prepare_design_matrix(
            covariates,
            n_observations=X_array.shape[0],
            fit_intercept=self.fit_intercept,
        )
        objective, _ = self._objective(
            self._pack_cholesky(self.cholesky_),
            X_array,
            design,
            diagonal_errors,
            full_errors,
        )
        return -objective

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Return constructor parameters for estimator interoperability."""
        del deep
        return {
            "max_iter": self.max_iter,
            "tolerance": self.tolerance,
            "min_variance": self.min_variance,
            "fit_intercept": self.fit_intercept,
            "max_dense_bytes": self.max_dense_bytes,
            "verbose": self.verbose,
        }
