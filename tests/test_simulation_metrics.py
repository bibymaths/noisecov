from __future__ import annotations

import numpy as np
import pytest

from noisecov import (
    correlation_rmse,
    make_covariance,
    principal_subspace_angle,
    relative_frobenius_error,
    simulate_measurement_error,
    spectral_norm_error,
)


@pytest.mark.parametrize("structure", ["block", "low_rank", "sparse", "diagonal"])
def test_covariance_families_are_positive_definite(structure: str) -> None:
    covariance = make_covariance(10, structure=structure, random_state=9)  # type: ignore[arg-type]
    assert covariance.shape == (10, 10)
    assert np.linalg.eigvalsh(covariance)[0] > 0


@pytest.mark.parametrize("noise", ["lognormal", "gene", "cell", "gene_cell"])
def test_simulation_is_reproducible_and_heteroscedastic(noise: str) -> None:
    first = simulate_measurement_error(100, 6, noise=noise, random_state=11)  # type: ignore[arg-type]
    second = simulate_measurement_error(100, 6, noise=noise, random_state=11)  # type: ignore[arg-type]
    assert np.array_equal(first.X, second.X)
    assert np.array_equal(first.sd, second.sd)
    assert first.metadata["seed"] == 11
    assert np.std(first.sd) > 0


def test_metrics_known_values() -> None:
    truth = np.diag([2.0, 1.0, 0.5])
    estimate = truth.copy()
    assert relative_frobenius_error(estimate, truth) == pytest.approx(0.0)
    assert spectral_norm_error(estimate, truth) == pytest.approx(0.0)
    assert correlation_rmse(estimate, truth) == pytest.approx(0.0)
    assert principal_subspace_angle(estimate, truth, rank=1) == pytest.approx(0.0)
    with pytest.raises(ValueError, match="same shape"):
        relative_frobenius_error(np.eye(2), np.eye(3))
    with pytest.raises(ValueError, match="rank"):
        principal_subspace_angle(estimate, truth, rank=0)


def test_invalid_simulation_arguments() -> None:
    with pytest.raises(ValueError):
        make_covariance(0)
    with pytest.raises(ValueError):
        make_covariance(4, n_factors=5)
    with pytest.raises(ValueError):
        simulate_measurement_error(n_cells=1)
    with pytest.raises(ValueError):
        simulate_measurement_error(noise_scale=-1)
    with pytest.raises(ValueError, match="Unknown"):
        simulate_measurement_error(noise="bad")  # type: ignore[arg-type]
