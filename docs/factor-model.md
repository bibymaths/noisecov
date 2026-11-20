# Heteroscedastic factor model

For high-dimensional data, `noisecov` models

$$
\Sigma=LL^\top+\operatorname{diag}(\psi), \qquad K\ll G.
$$

The full observation model is

$$
x_c=q_c^\top B+Lf_c+\eta_c+\epsilon_c,
$$

where $f_c\sim N(0,I)$, $\eta_c\sim N(0,\operatorname{diag}\psi)$, and the known
$\epsilon_c\sim N(0,D_c)$. The current scalable implementation accepts diagonal $D_c$.

```python
model = HeteroscedasticFactorAnalysis(n_factors=20).fit(X, sd)
scores = model.transform(X, sd=sd)
```

Fitting uses expectation/conditional maximization. The E-step uses the matrix determinant
lemma and Woodbury form, solving only $K\times K$ systems. Initialization uses a truncated
SVD and does not form $G\times G$. `loadings_` and `noise_variance_` remain scalable.

`covariance_` is a lazy dense property and checks the configured memory budget before
allocation. For very large $G$, work with the factor representation directly.

Factor loadings are identifiable only up to rotation; validate their subspace or reconstructed
covariance, not element-by-element loading signs.
