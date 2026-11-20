# Corrected covariance

Let $S_X$ be the unbiased empirical covariance. For independent cells with a common mean,

$$
\mathbb E[S_X]=\Sigma+\frac1C\sum_{c=1}^C D_c.
$$

This gives the moment estimator

$$
\widehat\Sigma_{\mathrm{corrected}}
=S_X-\frac1C\sum_c D_c.
$$

When only `sd` is supplied, $D_c$ is diagonal: only diagonal measurement variance is
subtracted. Off-diagonal covariance is already unbiased under independent zero-mean errors.

## PSD policies

- `none`: return the raw estimator, including scientifically informative negative eigenvalues.
- `clip`: project onto the PSD cone by eigenvalue clipping. This is Frobenius-nearest without
  a diagonal constraint.
- `nearest`: apply Higham-style alternating projections while preserving the feasible,
  non-negative diagonal.

An indefinite result may reflect small samples, excessive/noisy supplied error bars, or model
misspecification. Repair changes the estimator; it does not validate the assumptions.

## Shrinkage

`shrinkage` is an explicit value in `[0, 1]`, toward a diagonal or scaled-identity target.
Ledoit-Wolf and OAS estimate intensity for an ordinary empirical covariance under their own
sampling assumptions. Applying those formulas unchanged after heteroscedastic noise
subtraction is not justified, so `noisecov` does not present an automatic LW/OAS intensity as
valid here.

## Covariates

If OLS residualization uses residual-maker $M$ and residual degrees of freedom $\nu$, the
independent-error correction for feature $g$ is

$$
\nu^{-1}\sum_c M_{cc}\delta_{cg}^2.
$$

This is implemented rather than simply subtracting the unadjusted average error variance.
