"""Metrics for validation against known covariance structure."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import subspace_angles

from .correlation import covariance_to_correlation


def relative_frobenius_error(estimate: ArrayLike, truth: ArrayLike) -> float:
    """Return ``||estimate - truth||_F / ||truth||_F``."""
    estimate_array = np.asarray(estimate, dtype=np.float64)
    truth_array = np.asarray(truth, dtype=np.float64)
    if estimate_array.shape != truth_array.shape:
        raise ValueError("estimate and truth must have the same shape.")
    denominator = np.linalg.norm(truth_array, ord="fro")
    if denominator == 0:
        raise ValueError("truth must have non-zero Frobenius norm.")
    return float(np.linalg.norm(estimate_array - truth_array, ord="fro") / denominator)


def spectral_norm_error(estimate: ArrayLike, truth: ArrayLike) -> float:
    """Return the spectral norm of the covariance estimation error."""
    difference = np.asarray(estimate, dtype=np.float64) - np.asarray(truth, dtype=np.float64)
    return float(np.linalg.norm(difference, ord=2))


def correlation_rmse(estimate: ArrayLike, truth: ArrayLike) -> float:
    """Return RMSE between off-diagonal latent correlations."""
    estimate_corr = covariance_to_correlation(estimate)
    truth_corr = covariance_to_correlation(truth)
    upper = np.triu_indices_from(truth_corr, k=1)
    return float(np.sqrt(np.mean((estimate_corr[upper] - truth_corr[upper]) ** 2)))


def principal_subspace_angle(
    estimate: ArrayLike,
    truth: ArrayLike,
    *,
    rank: int,
) -> float:
    """Return the largest principal angle in radians between leading subspaces."""
    if rank < 1:
        raise ValueError("rank must be positive.")
    estimate_array = np.asarray(estimate, dtype=np.float64)
    truth_array = np.asarray(truth, dtype=np.float64)
    if estimate_array.shape != truth_array.shape or estimate_array.ndim != 2:
        raise ValueError("estimate and truth must be equally shaped square matrices.")
    if rank > estimate_array.shape[0]:
        raise ValueError("rank cannot exceed the feature count.")
    estimate_vectors = np.linalg.eigh(estimate_array)[1][:, -rank:]
    truth_vectors = np.linalg.eigh(truth_array)[1][:, -rank:]
    return float(np.max(subspace_angles(estimate_vectors, truth_vectors)))
