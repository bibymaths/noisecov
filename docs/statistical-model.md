# Statistical model

For cell $c$, let $z_c\in\mathbb R^G$ be the unobserved biological feature vector:

$$
z_c \sim \mathcal N(\mu,\Sigma), \qquad
x_c\mid z_c \sim \mathcal N(z_c,D_c).
$$

Therefore

$$
x_c \sim \mathcal N(\mu,\Sigma+D_c).
$$

The estimand is $\Sigma$. The default error model is

$$
D_c=\operatorname{diag}(\delta_{c1}^2,\ldots,\delta_{cG}^2),
$$

which asserts that measurement errors are conditionally independent across features. The
software can use a supplied full $D_c$ in the corrected and dense-likelihood estimators,
but it never infers error correlations from standard deviations.

## Mean structure

With covariates $q_c$, the model becomes

$$
x_c=B^\top q_c+z_c+\epsilon_c.
$$

The corrected estimator uses an explicit OLS residual model and adjusts the noise correction
for leverage. The dense likelihood profiles $B$ by generalized least squares. The factor
model updates $B$ jointly with its loadings. In every case the target becomes residual
latent covariance conditional on the specified mean model.
