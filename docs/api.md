# API reference

## Functional estimators

- `naive_covariance(X)`: ordinary unbiased empirical covariance.
- `corrected_covariance(X, sd, ...)`: subtract known average error covariance, optionally fit
  a covariate mean model, shrink, and repair PSD.
- `estimate_covariance(...)`: small dispatcher for the two transparent estimators.
- `covariance_to_correlation(covariance)`: normalize by latent variances.

## Models

- `HeteroscedasticGaussianCovariance`: dense marginal MLE with profiled GLS mean.
- `HeteroscedasticFactorAnalysis`: low-rank-plus-diagonal latent covariance.

## Interoperability and validation

- `from_anndata`: layer-aware, subset-aware AnnData workflow.
- `residualize`: explicit OLS mean model with leverage diagnostics.
- `simulate_measurement_error`: deterministic ground-truth simulation.
- `posterior_samples_to_gaussian`: validated, explicitly lossy bridge from posterior draws.
- `relative_frobenius_error`, `spectral_norm_error`, `correlation_rmse`,
  `principal_subspace_angle`: scientific recovery metrics.

All public data matrices use observations × features. Full API signatures and NumPy-style
docstrings are available through Python `help()`.
