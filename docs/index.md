# Noise-aware covariance

`noisecov` estimates covariance among latent features when every observation has a known,
possibly different Gaussian measurement uncertainty.

The central identity is

$$
\underbrace{\operatorname{Cov}(x)}_{\text{observed variability}}
=
\underbrace{\Sigma}_{\text{latent variability}}
+
\underbrace{\mathbb E[D_c]}_{\text{measurement variability}}.
$$

The package provides a transparent moment correction, a dense heteroscedastic likelihood,
and a low-rank factor model. Every method consumes cells/observations by features arrays.

This identity is useful only when the supplied uncertainty is meaningful on the analyzed
scale. It is not a license to treat raw count data as Gaussian.

## One-screen workflow

```python
import noisecov as nc

sim = nc.simulate_measurement_error(n_cells=2000, n_features=20, random_state=4)
latent_cov = nc.corrected_covariance(sim.X, sim.sd, psd="clip")
latent_corr = nc.covariance_to_correlation(latent_cov)
```

See [Choosing a Method](choosing-a-method.md) before using the result as a biological network.
