# noisecov

Ordinary covariance treats all observed variation as real variation. When every
observation × feature measurement has a known uncertainty, that mixes measurement noise with
latent variation. `noisecov` estimates covariance after accounting for that uncertainty.

For observation (c), the model is

$$
z_c \sim \mathcal N(\mu, \Sigma), \qquad
x_c \mid z_c \sim \mathcal N(z_c, D_c),
$$

so (x_c \sim \mathcal N(\mu, \Sigma + D_c)). The target is the latent covariance
(Sigma), not the covariance of the noisy observations. The common case is
(D_c=\operatorname{diag}(\delta_{c1}^2,\ldots,\delta_{cG}^2)).

## Install

```bash
uv add noisecov
```

AnnData support is optional:

```bash
uv add "noisecov[anndata]"
```

## Minimal example

```python
import noisecov as nc

data = nc.simulate_measurement_error(random_state=7)
cov = nc.corrected_covariance(data.X, data.sd, psd="clip")
corr = nc.covariance_to_correlation(cov)

model = nc.HeteroscedasticGaussianCovariance(max_iter=100)
model.fit(data.X, data.sd)
print(model.covariance_, model.converged_)
```

The corrected sample estimator is transparent:

$$
\widehat\Sigma_{\mathrm{corrected}}
= S_X - \frac{1}{C}\sum_c D_c.
$$

It can be indefinite in finite samples. PSD repair and explicit shrinkage are available, but
both are regularization choices rather than new biological information.

## AnnData

```python
result = nc.from_anndata(
    adata,
    x_layer="measurement_mean",
    sd_layer="measurement_sd",
    genes=adata.var_names[adata.var["highly_variable"]],
    method="corrected",
)
```

Use `write=True` to receive a copied AnnData object with covariance and correlation in
`varp`. Input data are not overwritten by default.

## Methods

- `naive_covariance`: baseline that ignores measurement uncertainty.
- `corrected_covariance`: unbiased moment correction under known independent errors.
- `HeteroscedasticGaussianCovariance`: dense marginal maximum likelihood.
- `HeteroscedasticFactorAnalysis`: scalable low-rank-plus-diagonal latent covariance.

Raw UMI counts do not usually satisfy this Gaussian observation model. `noisecov` expects
meaningful measurement means and standard deviations on the analyzed scale; it does not infer
those uncertainties from counts.

## Ground-truth demonstration

The committed reproducible experiment uses 1,500 cells, 10 features, block covariance, and
strong gene-by-cell heteroscedastic errors. Lower is better:

| Estimator | Relative Frobenius error | Correlation RMSE |
|---|---:|---:|
| Naive | 0.441 | 0.174 |
| Corrected | 0.099 | 0.047 |
| Heteroscedastic MLE | 0.080 | 0.035 |
| Rank-2 factor model | 0.083 | 0.030 |

Both iterative models converged. Reproduce with
`uv run python examples/ground_truth_demo.py`; machine-readable results and the comparison
figure are in [`results/`](results/ground_truth_metrics.json).

[Documentation](https://OWNER.github.io/noisecov) ·
[Repository](https://github.com/OWNER/noisecov) ·
[PyPI](https://pypi.org/project/noisecov/) ·
[Citation](CITATION.cff)
