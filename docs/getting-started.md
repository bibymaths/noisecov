# Getting started

## Installation

```bash
uv add noisecov
```

For H5AD integration:

```bash
uv add "noisecov[anndata]"
```

## Inputs

`X` and `sd` must both have shape `(n_cells, n_features)`. They must be expressed on the
same scale: if `X` is log expression, `sd` must quantify uncertainty on that log-expression
scale.

```python
cov = corrected_covariance(X, sd, psd="none")
```

`psd="none"` returns the raw moment estimate. Inspect it before deciding whether clipping,
nearest-covariance repair, or shrinkage is scientifically acceptable.

## Full error covariance

When a full known $D_c$ exists, pass `error_cov` instead of `sd`:

```python
cov = corrected_covariance(X, error_cov=error_cov)
model = HeteroscedasticGaussianCovariance().fit(X, error_cov=error_cov)
```

Marginal standard deviations cannot be used to invent these off-diagonal error covariances.
