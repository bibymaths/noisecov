"""Scalable heteroscedastic Gaussian factor analysis."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import optimize
from scipy.sparse.linalg import svds

from ._validation import require_dense_budget, validate_X_sd
from .correlation import covariance_to_correlation
from .preprocessing import prepare_design_matrix

FloatArray = NDArray[np.float64]


class HeteroscedasticFactorAnalysis:
    """Low-rank latent covariance with known element-wise measurement errors.

    The model is

    ``x_c = q_c B + L f_c + eta_c + epsilon_c``

    with ``f_c ~ N(0, I)``, ``eta_c ~ N(0, diag(psi))`` and known
    ``epsilon_c ~ N(0, diag(sd[c] ** 2))``. Consequently the latent covariance
    is ``L @ L.T + diag(psi)``. Fitting uses an expectation/conditional-
    maximization algorithm and only solves ``n_factors x n_factors`` systems
    in the E-step.
    """

    def __init__(
        self,
        n_factors: int = 10,
        *,
        max_iter: int = 200,
        tolerance: float = 1e-5,
        min_noise_variance: float = 1e-6,
        fit_intercept: bool = True,
        random_state: int = 0,
        max_dense_bytes: int = 512 * 1024**2,
        verbose: bool = False,
    ) -> None:
        if n_factors < 1:
            raise ValueError("n_factors must be positive.")
        if max_iter < 1:
            raise ValueError("max_iter must be positive.")
        if tolerance <= 0 or min_noise_variance <= 0:
            raise ValueError("tolerance and min_noise_variance must be positive.")
        self.n_factors = n_factors
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.min_noise_variance = min_noise_variance
        self.fit_intercept = fit_intercept
        self.random_state = random_state
        self.max_dense_bytes = max_dense_bytes
        self.verbose = verbose

    @staticmethod
    def _e_step(
        residuals: FloatArray,
        sd: FloatArray,
        loadings: FloatArray,
        noise_variance: FloatArray,
    ) -> tuple[FloatArray, FloatArray, float]:
        _, n_features = residuals.shape
        n_factors = loadings.shape[1]
        residual_variance = sd**2 + noise_variance[None, :]
        inverse_variance = 1.0 / residual_variance
        weighted_loadings = inverse_variance[:, :, None] * loadings[None, :, :]
        posterior_precision = np.eye(n_factors)[None, :, :] + np.einsum(
            "gk,cgl->ckl", loadings, weighted_loadings, optimize=True
        )
        right = np.einsum(
            "cg,cgk,cg->ck", residuals, weighted_loadings, np.ones_like(residuals), optimize=True
        )
        posterior_mean = np.linalg.solve(posterior_precision, right[..., None])[..., 0]
        identity = np.broadcast_to(np.eye(n_factors), posterior_precision.shape)
        posterior_covariance = np.linalg.solve(posterior_precision, identity)
        second_moment = posterior_covariance + np.einsum(
            "ck,cl->ckl", posterior_mean, posterior_mean, optimize=True
        )

        cholesky = np.linalg.cholesky(posterior_precision)
        logdet = np.log(residual_variance).sum(axis=1) + 2.0 * np.log(
            np.diagonal(cholesky, axis1=1, axis2=2)
        ).sum(axis=1)
        quadratic_base = np.sum(residuals**2 * inverse_variance, axis=1)
        quadratic_reduction = np.sum(right * posterior_mean, axis=1)
        negative_log_likelihood = 0.5 * np.sum(
            n_features * np.log(2.0 * np.pi) + logdet + quadratic_base - quadratic_reduction
        )
        return posterior_mean, second_moment, float(negative_log_likelihood)

    def _initialize(
        self,
        X: FloatArray,
        sd: FloatArray,
        design: FloatArray,
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        coefficients = (
            np.linalg.lstsq(design, X, rcond=None)[0]
            if design.shape[1]
            else np.empty((0, X.shape[1]))
        )
        residuals = X - design @ coefficients if design.shape[1] else X.copy()
        scaled = residuals / np.sqrt(max(X.shape[0] - 1, 1))
        # Truncated SVD avoids ever forming G x G during initialization.
        _, singular_values, right_vectors = svds(
            scaled,
            k=self.n_factors,
            which="LM",
            random_state=self.random_state,
        )
        order = np.argsort(singular_values)[::-1]
        singular_values = singular_values[order]
        right_vectors = right_vectors[order]
        average_measurement_variance = float(np.mean(sd**2))
        latent_values = np.maximum(
            singular_values**2 - average_measurement_variance,
            self.min_noise_variance,
        )
        loadings = right_vectors.T * np.sqrt(latent_values)
        corrected_variance = np.var(residuals, axis=0, ddof=1) - np.mean(sd**2, axis=0)
        noise_variance = np.maximum(
            corrected_variance - np.sum(loadings**2, axis=1), self.min_noise_variance
        )
        return coefficients, loadings, noise_variance

    def fit(
        self,
        X: ArrayLike,
        sd: ArrayLike,
        *,
        covariates: ArrayLike | None = None,
    ) -> HeteroscedasticFactorAnalysis:
        """Fit the factor model to cells-by-features measurements and errors."""
        X_array, sd_array = validate_X_sd(X, sd, max_bytes=self.max_dense_bytes)
        n_observations, n_features = X_array.shape
        if self.n_factors >= n_features:
            raise ValueError("n_factors must be smaller than n_features.")
        require_dense_budget(
            (n_observations, self.n_factors, self.n_factors),
            max_bytes=self.max_dense_bytes,
            purpose="storing factor posterior moments",
            arrays=2,
        )
        design = prepare_design_matrix(
            covariates,
            n_observations=n_observations,
            fit_intercept=self.fit_intercept,
        )
        if design.shape[1] and np.linalg.matrix_rank(design) < design.shape[1]:
            raise ValueError("The mean-model design matrix is rank deficient.")
        coefficients, loadings, noise_variance = self._initialize(X_array, sd_array, design)
        history: list[float] = []
        converged = False

        for iteration in range(self.max_iter):
            fitted_mean = design @ coefficients if design.shape[1] else np.zeros_like(X_array)
            residuals = X_array - fitted_mean
            factor_mean, factor_second, negative_log_likelihood = self._e_step(
                residuals, sd_array, loadings, noise_variance
            )
            history.append(negative_log_likelihood)
            if self.verbose:
                print(f"iteration={iteration + 1} nll={negative_log_likelihood:.6f}")
            if iteration > 0:
                change = abs(history[-2] - history[-1]) / max(abs(history[-2]), 1.0)
                if change < self.tolerance:
                    converged = True
                    break

            # Conditional maximization of each feature's regression on the
            # observed covariates and latent factor moments.
            for feature in range(n_features):
                weights = 1.0 / (noise_variance[feature] + sd_array[:, feature] ** 2)
                p = design.shape[1]
                normal = np.zeros((p + self.n_factors, p + self.n_factors))
                right = np.zeros(p + self.n_factors)
                if p:
                    normal[:p, :p] = design.T @ (weights[:, None] * design)
                    cross = design.T @ (weights[:, None] * factor_mean)
                    normal[:p, p:] = cross
                    normal[p:, :p] = cross.T
                    right[:p] = design.T @ (weights * X_array[:, feature])
                normal[p:, p:] = np.einsum("c,cij->ij", weights, factor_second, optimize=True)
                right[p:] = factor_mean.T @ (weights * X_array[:, feature])
                ridge = np.eye(p + self.n_factors) * 1e-10
                solution = np.linalg.solve(normal + ridge, right)
                if p:
                    coefficients[:, feature] = solution[:p]
                loadings[feature] = solution[p:]

            fitted_mean = design @ coefficients if design.shape[1] else np.zeros_like(X_array)
            centered = X_array - fitted_mean
            for feature in range(n_features):
                loading = loadings[feature]
                expected_residual_square = (
                    centered[:, feature] ** 2
                    - 2.0 * centered[:, feature] * (factor_mean @ loading)
                    + np.einsum("i,cij,j->c", loading, factor_second, loading, optimize=True)
                )
                measurement_variance = sd_array[:, feature] ** 2

                def uniqueness_objective(
                    value: float,
                    measurement: FloatArray = measurement_variance,
                    expected: FloatArray = expected_residual_square,
                ) -> float:
                    total = value + measurement
                    return float(0.5 * np.sum(np.log(total) + expected / total))

                upper = max(
                    float(np.var(centered[:, feature]) * 10.0),
                    float(noise_variance[feature] * 10.0),
                    self.min_noise_variance * 100.0,
                )
                update = optimize.minimize_scalar(
                    uniqueness_objective,
                    bounds=(self.min_noise_variance, upper),
                    method="bounded",
                    options={"xatol": self.tolerance * 0.1},
                )
                noise_variance[feature] = max(float(update.x), self.min_noise_variance)

        fitted_mean = design @ coefficients if design.shape[1] else np.zeros_like(X_array)
        factor_mean, factor_second, final_nll = self._e_step(
            X_array - fitted_mean, sd_array, loadings, noise_variance
        )
        if not history or final_nll != history[-1]:
            history.append(final_nll)
        if len(history) > 1:
            final_change = abs(history[-2] - history[-1]) / max(abs(history[-2]), 1.0)
            converged = converged or final_change < self.tolerance

        self.n_features_in_ = n_features
        self.n_observations_ = n_observations
        self.design_matrix_ = design
        self.coefficients_ = coefficients
        self.fitted_mean_ = fitted_mean
        self.mean_ = fitted_mean.mean(axis=0)
        self.loadings_ = loadings
        self.noise_variance_ = noise_variance
        self.factor_scores_ = factor_mean
        self.factor_second_moment_ = factor_second
        self.negative_log_likelihood_ = final_nll
        self.log_likelihood_history_ = np.asarray(history)
        self.converged_ = converged
        self.n_iter_ = len(history) - 1
        self._covariance_cache: FloatArray | None = None
        return self

    @property
    def covariance_(self) -> FloatArray:
        """Dense latent covariance, allocated lazily with a memory guard."""
        if not hasattr(self, "loadings_"):
            raise RuntimeError("fit must be called before covariance_ is available.")
        require_dense_budget(
            (self.n_features_in_, self.n_features_in_),
            max_bytes=self.max_dense_bytes,
            purpose="materializing factor-model covariance",
            arrays=1,
        )
        if self._covariance_cache is None:
            covariance = self.loadings_ @ self.loadings_.T
            covariance.flat[:: self.n_features_in_ + 1] += self.noise_variance_
            self._covariance_cache = covariance
        return self._covariance_cache.copy()

    @property
    def correlation_(self) -> FloatArray:
        """Dense latent correlation, allocated on request."""
        return covariance_to_correlation(self.covariance_)

    def transform(
        self,
        X: ArrayLike,
        *,
        sd: ArrayLike,
        covariates: ArrayLike | None = None,
    ) -> FloatArray:
        """Return posterior factor means for new observations."""
        if not hasattr(self, "loadings_"):
            raise RuntimeError("fit must be called before transform.")
        X_array, sd_array = validate_X_sd(X, sd, max_bytes=self.max_dense_bytes)
        if X_array.shape[1] != self.n_features_in_:
            raise ValueError("X has a different feature count from the fitted model.")
        design = prepare_design_matrix(
            covariates,
            n_observations=X_array.shape[0],
            fit_intercept=self.fit_intercept,
        )
        if design.shape[1] != self.coefficients_.shape[0]:
            raise ValueError("covariates do not match the fitted mean model.")
        fitted = design @ self.coefficients_ if design.shape[1] else np.zeros_like(X_array)
        means, _, _ = self._e_step(X_array - fitted, sd_array, self.loadings_, self.noise_variance_)
        return means

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Return constructor parameters for estimator interoperability."""
        del deep
        return {
            "n_factors": self.n_factors,
            "max_iter": self.max_iter,
            "tolerance": self.tolerance,
            "min_noise_variance": self.min_noise_variance,
            "fit_intercept": self.fit_intercept,
            "random_state": self.random_state,
            "max_dense_bytes": self.max_dense_bytes,
            "verbose": self.verbose,
        }
