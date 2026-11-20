from __future__ import annotations

import numpy as np
import pytest

from noisecov import corrected_covariance, estimate_covariance, naive_covariance, residualize


def test_zero_error_equals_sample_covariance() -> None:
    rng = np.random.default_rng(1)
    X = rng.normal(size=(100, 5))
    assert np.allclose(corrected_covariance(X, np.zeros_like(X)), naive_covariance(X))


def test_diagonal_noise_correction_is_exact_formula() -> None:
    rng = np.random.default_rng(2)
    X = rng.normal(size=(80, 4))
    sd = rng.uniform(0.1, 1.0, size=X.shape)
    expected = naive_covariance(X) - np.diag(np.mean(sd**2, axis=0))
    assert np.allclose(corrected_covariance(X, sd), expected)
    assert np.allclose(np.triu(corrected_covariance(X, sd), 1), np.triu(naive_covariance(X), 1))


def test_full_error_covariance_corrects_off_diagonal() -> None:
    rng = np.random.default_rng(3)
    X = rng.normal(size=(50, 3))
    error = np.broadcast_to(
        np.array([[0.5, 0.1, 0.0], [0.1, 0.4, -0.05], [0.0, -0.05, 0.3]]),
        (50, 3, 3),
    ).copy()
    expected = naive_covariance(X) - error.mean(axis=0)
    assert np.allclose(corrected_covariance(X, error_cov=error), expected)


def test_explicit_shrinkage_targets() -> None:
    rng = np.random.default_rng(4)
    X = rng.normal(size=(40, 5))
    sd = np.full_like(X, 0.2)
    diagonal = corrected_covariance(X, sd, shrinkage=1.0, shrinkage_target="diagonal")
    identity = corrected_covariance(X, sd, shrinkage=1.0, shrinkage_target="scaled_identity")
    assert np.allclose(diagonal, np.diag(np.diag(diagonal)))
    assert np.allclose(identity, np.eye(5) * identity[0, 0])
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        corrected_covariance(X, sd, shrinkage=1.1)
    with pytest.raises(ValueError, match="shrinkage_target"):
        corrected_covariance(X, sd, shrinkage=0.2, shrinkage_target="bad")  # type: ignore[arg-type]


def test_covariate_residualization_and_leverage_correction() -> None:
    rng = np.random.default_rng(5)
    n = 120
    q = np.linspace(-2, 2, n)[:, None]
    latent = rng.normal(size=(n, 2))
    sd = np.column_stack((np.linspace(0.1, 0.8, n), np.linspace(0.7, 0.2, n)))
    X = q @ np.array([[3.0, -2.0]]) + latent + rng.normal(size=(n, 2)) * sd
    result = residualize(X, q)
    expected_empirical = result.residuals.T @ result.residuals / result.residual_degrees_of_freedom
    correction = np.diag((1.0 - result.leverage) @ (sd**2) / result.residual_degrees_of_freedom)
    actual = corrected_covariance(X, sd, covariates=q)
    assert np.allclose(actual, expected_empirical - correction)
    assert result.rank == 2
    assert np.allclose(result.fitted + result.residuals, X)


def test_dispatch_and_invalid_inputs() -> None:
    X = np.arange(20.0).reshape(10, 2)
    sd = np.ones_like(X)
    assert np.allclose(estimate_covariance(X, method="naive"), naive_covariance(X))
    assert np.allclose(estimate_covariance(X, sd, method="corrected"), corrected_covariance(X, sd))
    with pytest.raises(ValueError, match="method"):
        estimate_covariance(X, sd, method="unknown")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        estimate_covariance(X, method="naive", psd="clip")
    with pytest.raises(ValueError, match="same shape"):
        corrected_covariance(X, np.ones((9, 2)))
    with pytest.raises(ValueError, match="non-negative"):
        corrected_covariance(X, -sd)
    with pytest.raises(ValueError, match="not both"):
        corrected_covariance(X, sd, error_cov=np.zeros((10, 2, 2)))
    with pytest.raises(ValueError, match="conflicting"):
        corrected_covariance(X, sd, psd="nearest", ensure_psd=True)


def test_residualize_rejects_bad_design() -> None:
    X = np.arange(20.0).reshape(10, 2)
    q = np.ones((10, 2))
    with pytest.raises(ValueError, match="rank deficient"):
        residualize(X, q)
    with pytest.raises(ValueError, match="no columns"):
        residualize(X, np.empty((10, 0)), fit_intercept=False)
