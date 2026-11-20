from __future__ import annotations

import numpy as np

from noisecov import (
    corrected_covariance,
    correlation_rmse,
    covariance_to_correlation,
    naive_covariance,
    relative_frobenius_error,
    simulate_measurement_error,
)


def test_corrected_covariance_converges_and_beats_naive() -> None:
    simulation = simulate_measurement_error(
        n_cells=20_000,
        n_features=6,
        covariance="block",
        noise="gene_cell",
        noise_scale=0.9,
        random_state=17,
    )
    naive = naive_covariance(simulation.X)
    corrected = corrected_covariance(simulation.X, simulation.sd)
    naive_error = relative_frobenius_error(naive, simulation.Sigma_true)
    corrected_error = relative_frobenius_error(corrected, simulation.Sigma_true)
    assert corrected_error < 0.08
    assert corrected_error < naive_error * 0.25
    assert np.mean(np.diag(naive) - np.diag(simulation.Sigma_true)) > 0.5


def test_naive_correlation_is_attenuated_and_correction_recovers() -> None:
    simulation = simulate_measurement_error(
        n_cells=12_000,
        n_features=8,
        covariance="block",
        noise="gene_cell",
        noise_scale=1.0,
        random_state=23,
    )
    naive = naive_covariance(simulation.X)
    corrected = corrected_covariance(simulation.X, simulation.sd, psd="clip")
    true_correlation = covariance_to_correlation(simulation.Sigma_true)
    naive_correlation = covariance_to_correlation(naive)
    upper = np.triu_indices(8, 1)
    signal = np.abs(true_correlation[upper]) > 0.2
    assert np.mean(np.abs(naive_correlation[upper][signal])) < np.mean(
        np.abs(true_correlation[upper][signal])
    )
    assert (
        correlation_rmse(corrected, simulation.Sigma_true)
        < correlation_rmse(naive, simulation.Sigma_true) * 0.35
    )


def test_more_cells_improve_average_recovery() -> None:
    errors: dict[int, list[float]] = {200: [], 4_000: []}
    for n_cells in errors:
        for seed in range(5):
            simulation = simulate_measurement_error(
                n_cells=n_cells,
                n_features=5,
                noise_scale=0.7,
                random_state=seed,
            )
            estimate = corrected_covariance(simulation.X, simulation.sd)
            errors[n_cells].append(relative_frobenius_error(estimate, simulation.Sigma_true))
    assert np.mean(errors[4_000]) < np.mean(errors[200]) * 0.45
