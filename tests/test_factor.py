from __future__ import annotations

import numpy as np
import pytest

from noisecov import (
    HeteroscedasticFactorAnalysis,
    principal_subspace_angle,
    simulate_measurement_error,
)


def test_factor_model_recovers_low_rank_subspace() -> None:
    simulation = simulate_measurement_error(
        1_200,
        10,
        covariance="low_rank",
        n_factors=2,
        noise_scale=0.45,
        random_state=44,
    )
    model = HeteroscedasticFactorAnalysis(2, max_iter=100, tolerance=2e-5, random_state=1).fit(
        simulation.X, simulation.sd
    )
    assert model.loadings_.shape == (10, 2)
    assert model.factor_scores_.shape == (1_200, 2)
    assert np.all(model.noise_variance_ > 0)
    assert np.all(np.diff(model.log_likelihood_history_[2:]) <= 1e-4)
    assert principal_subspace_angle(model.covariance_, simulation.Sigma_true, rank=2) < 0.25
    assert model.get_params()["n_factors"] == 2


def test_transform_and_covariates() -> None:
    rng = np.random.default_rng(5)
    q = rng.normal(size=(300, 1))
    loadings = rng.normal(size=(6, 2)) * 0.4
    factors = rng.normal(size=(300, 2))
    sd = rng.uniform(0.1, 0.5, size=(300, 6))
    X = q @ np.arange(1.0, 7.0)[None, :] + factors @ loadings.T
    X += rng.normal(size=X.shape) * np.sqrt(sd**2 + 0.2)
    model = HeteroscedasticFactorAnalysis(2, max_iter=60).fit(X, sd, covariates=q)
    scores = model.transform(X[:10], sd=sd[:10], covariates=q[:10])
    assert scores.shape == (10, 2)
    assert model.coefficients_.shape == (2, 6)
    with pytest.raises(ValueError, match="covariates"):
        model.transform(X[:10], sd=sd[:10])


def test_factor_dense_covariance_memory_guard() -> None:
    simulation = simulate_measurement_error(100, 8, covariance="low_rank", random_state=3)
    model = HeteroscedasticFactorAnalysis(2, max_iter=2).fit(simulation.X, simulation.sd)
    model.max_dense_bytes = 100
    with pytest.raises(MemoryError, match="covariance"):
        _ = model.covariance_


def test_factor_validation() -> None:
    X = np.ones((20, 3))
    sd = np.ones_like(X)
    with pytest.raises(ValueError, match="smaller"):
        HeteroscedasticFactorAnalysis(3).fit(X, sd)
    with pytest.raises(RuntimeError, match="fit"):
        HeteroscedasticFactorAnalysis(1).transform(X, sd=sd)
