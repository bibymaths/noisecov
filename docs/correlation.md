# Correlation

Latent correlation uses corrected latent variances:

$$
R_{ij}=\frac{\Sigma_{ij}}{\sqrt{\Sigma_{ii}\Sigma_{jj}}}.
$$

Independent measurement error does not bias off-diagonal covariance, but it inflates observed
diagonal variances. Therefore naive correlation divides by denominators that are too large and
is attenuated toward zero. “No off-diagonal correction” does **not** mean naive correlation is
unbiased.

`covariance_to_correlation` rejects non-positive latent variances by default. This commonly
happens in an unrepaired small-sample corrected covariance and should be investigated rather
than hidden.
