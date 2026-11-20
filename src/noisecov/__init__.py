"""Noise-aware covariance estimation."""

from .anndata import AnnDataCovarianceResult, from_anndata
from .correlation import covariance_to_correlation
from .covariance import corrected_covariance, estimate_covariance, naive_covariance
from .factor import HeteroscedasticFactorAnalysis
from .integrations import posterior_samples_to_gaussian
from .likelihood import HeteroscedasticGaussianCovariance
from .metrics import (
    correlation_rmse,
    principal_subspace_angle,
    relative_frobenius_error,
    spectral_norm_error,
)
from .preprocessing import ResidualizationResult, residualize
from .simulation import SimulationResult, make_covariance, simulate_measurement_error

__all__ = [
    "AnnDataCovarianceResult",
    "HeteroscedasticFactorAnalysis",
    "HeteroscedasticGaussianCovariance",
    "ResidualizationResult",
    "SimulationResult",
    "corrected_covariance",
    "correlation_rmse",
    "covariance_to_correlation",
    "estimate_covariance",
    "from_anndata",
    "make_covariance",
    "naive_covariance",
    "posterior_samples_to_gaussian",
    "principal_subspace_angle",
    "relative_frobenius_error",
    "residualize",
    "simulate_measurement_error",
    "spectral_norm_error",
]

__version__ = "0.1.0"
