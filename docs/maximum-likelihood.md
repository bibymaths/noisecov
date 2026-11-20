# Heteroscedastic maximum likelihood

The dense model minimizes

$$
-\ell(\mu,\Sigma)=\frac12\sum_c\left[
\log|\Sigma+D_c|+(x_c-\mu)^\top(\Sigma+D_c)^{-1}(x_c-\mu)
\right]+\text{constant}.
$$

`HeteroscedasticGaussianCovariance` uses

$$
\Sigma=LL^\top+\varepsilon I
$$

with a positive diagonal in $L$, so every iterate is positive definite. Each
$\Sigma+D_c$ is factorized by Cholesky decomposition. Quadratic forms and log determinants
use solves and Cholesky diagonals; the implementation does not call a generic matrix inverse.

The mean or covariate coefficient matrix is profiled exactly by generalized least squares at
every covariance proposal. An analytic covariance gradient is then valid by the envelope
theorem. SciPy L-BFGS-B performs the outer optimization.

```python
model = HeteroscedasticGaussianCovariance(max_iter=200).fit(X, sd)
model.covariance_
model.correlation_
model.converged_
model.optimization_result_
```

Cost is approximately $O(CG^3)$, so this is a moderate-feature method, not a 20,000-gene
method. Always inspect convergence; `fit` returns the best finite iterate even when the
optimizer reaches its iteration limit.
