from __future__ import annotations

import numpy as np
import pytest

from noisecov import (
    HeteroscedasticGaussianCovariance,
    relative_frobenius_error,
    simulate_measurement_error,
)
from noisecov.preprocessing import prepare_design_matrix


def test_analytic_gradient_matches_finite_difference() -> None:
    simulation = simulate_measurement_error(30, 3, random_state=2)
    model = HeteroscedasticGaussianCovariance(max_iter=2)
    initial = np.linalg.cholesky(simulation.Sigma_true)
    parameters = model._pack_cholesky(initial)
    design = prepare_design_matrix(None, n_observations=30)
    variance = simulation.sd**2
    value, gradient = model._objective(parameters, simulation.X, design, variance, None)
    assert np.isfinite(value)
    numerical = np.zeros_like(parameters)
    step = 1e-6
    for index in range(parameters.size):
        plus = parameters.copy()
        minus = parameters.copy()
        plus[index] += step
        minus[index] -= step
        numerical[index] = (
            model._objective(plus, simulation.X, design, variance, None)[0]
            - model._objective(minus, simulation.X, design, variance, None)[0]
        ) / (2 * step)
    assert np.allclose(gradient, numerical, atol=2e-5, rtol=2e-4)


def test_mle_recovers_known_covariance() -> None:
    simulation = simulate_measurement_error(
        1_000,
        4,
        covariance="block",
        noise_scale=0.65,
        random_state=31,
    )
    model = HeteroscedasticGaussianCovariance(max_iter=120, tolerance=1e-7).fit(
        simulation.X, simulation.sd
    )
    assert model.converged_
    assert model.covariance_.shape == (4, 4)
    assert np.linalg.eigvalsh(model.covariance_)[0] > 0
    assert relative_frobenius_error(model.covariance_, simulation.Sigma_true) < 0.16
    assert model.score(simulation.X, simulation.sd) < 0
    assert model.get_params()["max_iter"] == 120


def test_mle_supports_covariates_and_full_error() -> None:
    rng = np.random.default_rng(8)
    n = 220
    covariance = np.array([[1.0, 0.4], [0.4, 0.8]])
    q = rng.normal(size=(n, 1))
    error_matrix = np.array([[0.25, 0.05], [0.05, 0.3]])
    error_cov = np.broadcast_to(error_matrix, (n, 2, 2)).copy()
    latent = rng.multivariate_normal([0, 0], covariance, size=n)
    error = rng.multivariate_normal([0, 0], error_matrix, size=n)
    X = q @ np.array([[2.0, -1.0]]) + latent + error
    model = HeteroscedasticGaussianCovariance(max_iter=80).fit(X, error_cov=error_cov, covariates=q)
    assert model.coefficients_.shape == (2, 2)
    assert np.allclose(model.coefficients_[1], [2.0, -1.0], atol=0.25)
    assert np.isfinite(model.negative_log_likelihood_)


def test_mle_failures_are_clear() -> None:
    X = np.ones((10, 3))
    sd = np.ones_like(X) * 0.1
    with pytest.raises(RuntimeError, match="fit"):
        HeteroscedasticGaussianCovariance().score(X, sd)
    with pytest.raises(ValueError, match="exactly one"):
        HeteroscedasticGaussianCovariance().fit(X)
    q = np.ones((10, 2))
    with pytest.raises(ValueError, match="rank deficient"):
        HeteroscedasticGaussianCovariance().fit(X, sd, covariates=q)
