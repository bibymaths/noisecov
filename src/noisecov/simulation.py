"""Reproducible simulations with known latent covariance."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
CovarianceStructure = Literal["block", "low_rank", "sparse", "diagonal"]
NoiseStructure = Literal["lognormal", "gene", "cell", "gene_cell"]


@dataclass(frozen=True)
class SimulationResult:
    """A synthetic measurement-error dataset with known ground truth."""

    X: FloatArray
    sd: FloatArray
    Z_true: FloatArray
    Sigma_true: FloatArray
    mean_true: FloatArray
    metadata: dict[str, object]


def make_covariance(
    n_features: int,
    *,
    structure: CovarianceStructure = "block",
    n_factors: int = 3,
    block_size: int = 5,
    correlation: float = 0.65,
    random_state: int | np.random.Generator = 0,
) -> FloatArray:
    """Construct a deterministic positive-definite covariance family."""
    if n_features < 1:
        raise ValueError("n_features must be positive.")
    if n_factors < 1 or n_factors > n_features:
        raise ValueError("n_factors must be between 1 and n_features.")
    if not -1 < correlation < 1:
        raise ValueError("correlation must lie in (-1, 1).")
    rng = (
        random_state
        if isinstance(random_state, np.random.Generator)
        else np.random.default_rng(random_state)
    )
    if structure == "diagonal":
        return np.diag(np.linspace(0.6, 1.8, n_features))
    if structure == "block":
        covariance = np.eye(n_features)
        for start in range(0, n_features, block_size):
            stop = min(start + block_size, n_features)
            covariance[start:stop, start:stop] = correlation
            np.fill_diagonal(covariance[start:stop, start:stop], 1.0)
        scales = np.linspace(0.8, 1.4, n_features)
        return covariance * np.outer(scales, scales)
    if structure == "low_rank":
        loadings = rng.normal(scale=0.55, size=(n_features, n_factors))
        return loadings @ loadings.T + np.diag(np.linspace(0.25, 0.7, n_features))
    if structure == "sparse":
        covariance = np.eye(n_features)
        mask = rng.random((n_features, n_features)) < min(4 / n_features, 0.25)
        weights = rng.uniform(-0.18, 0.18, size=(n_features, n_features))
        upper = np.triu(mask * weights, 1)
        covariance += upper + upper.T
        minimum = np.linalg.eigvalsh(covariance)[0]
        if minimum <= 0.1:
            covariance += np.eye(n_features) * (0.1 - minimum)
        return covariance
    raise ValueError("Unknown covariance structure.")


def simulate_measurement_error(
    n_cells: int = 1_000,
    n_features: int = 12,
    *,
    covariance: CovarianceStructure = "block",
    noise: NoiseStructure = "gene_cell",
    noise_scale: float = 0.7,
    n_factors: int = 3,
    random_state: int = 0,
) -> SimulationResult:
    """Simulate latent Gaussian cells and heteroscedastic Gaussian errors."""
    if n_cells < 2:
        raise ValueError("n_cells must be at least two.")
    if noise_scale < 0:
        raise ValueError("noise_scale must be non-negative.")
    rng = np.random.default_rng(random_state)
    Sigma_true = make_covariance(
        n_features,
        structure=covariance,
        n_factors=n_factors,
        random_state=rng,
    )
    mean_true = np.linspace(-0.25, 0.25, n_features)
    Z_true = rng.multivariate_normal(mean_true, Sigma_true, size=n_cells)

    if noise == "lognormal":
        relative = rng.lognormal(mean=-0.25, sigma=0.65, size=(n_cells, n_features))
    elif noise == "gene":
        gene_effect = rng.lognormal(mean=-0.25, sigma=0.75, size=n_features)
        relative = np.broadcast_to(gene_effect, (n_cells, n_features)).copy()
    elif noise == "cell":
        cell_effect = rng.lognormal(mean=-0.25, sigma=0.75, size=n_cells)
        relative = np.broadcast_to(cell_effect[:, None], (n_cells, n_features)).copy()
    elif noise == "gene_cell":
        gene_effect = rng.lognormal(mean=-0.2, sigma=0.55, size=n_features)
        cell_effect = rng.lognormal(mean=-0.2, sigma=0.55, size=n_cells)
        local = rng.lognormal(mean=-0.1, sigma=0.25, size=(n_cells, n_features))
        relative = np.sqrt(np.outer(cell_effect, gene_effect)) * local
    else:
        raise ValueError("Unknown noise structure.")
    sd = noise_scale * relative
    X = Z_true + rng.normal(size=Z_true.shape) * sd
    metadata: dict[str, object] = {
        "seed": random_state,
        "n_cells": n_cells,
        "n_features": n_features,
        "covariance_structure": covariance,
        "noise_structure": noise,
        "noise_scale": noise_scale,
    }
    return SimulationResult(X, sd, Z_true, Sigma_true, mean_true, metadata)
