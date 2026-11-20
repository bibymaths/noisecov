# Existing tools and dependency decisions

Investigation date: **2026-09-23**. The question used for this audit was narrow: can the tool
estimate one latent feature covariance when every observation has a different, known
feature-level measurement variance (D_c)?

## Numerical and statistical foundations

| Tool | Reusable capability | Why it does not solve the full problem | Decision |
|---|---|---|---|
| [NumPy](https://numpy.org/doc/stable/reference/generated/numpy.cov.html) | Arrays, empirical covariance, eigendecomposition, batched solves and simulation | `numpy.cov` has observation/frequency weights, not a known matrix (D_c) to subtract or include in a likelihood | Required core dependency |
| [SciPy](https://docs.scipy.org/doc/scipy/reference/linalg.html) | Cholesky factors/solves, L-BFGS-B, bounded scalar optimization, sparse truncated SVD | Supplies algorithms, not the heteroscedastic covariance model or its profiled likelihood | Required core dependency; all dense optimization and factor initialization use it |
| [scikit-learn covariance](https://scikit-learn.org/stable/modules/covariance.html) | Empirical covariance, explicit shrinkage, Ledoit-Wolf, OAS and sparse precision estimators | LW/OAS intensities assume an ordinary empirical covariance. They are not automatically valid after subtracting unequal known errors | Not required. `noisecov` exposes explicit shrinkage and documents why intensity is user controlled |
| [scikit-learn FactorAnalysis](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FactorAnalysis.html) | Mature homoscedastic-observation factor analysis API and likelihood | Estimates one feature-specific uniqueness shared across samples; it cannot add the known cell × feature variance `sd**2` to that uniqueness | Not required; the model structure is different |
| [statsmodels](https://www.statsmodels.org/dev/generated/statsmodels.stats.correlation_tools.cov_nearest.html) | `cov_nearest` and correlation-matrix repair | Does not estimate latent covariance with known (D_c). Pulling the full dependency only for one repair function would enlarge the core install | Not required. The package implements documented Higham-style alternating projections using NumPy eigensolvers |

`LedoitWolf` and `OAS` remain useful comparators for ordinary noisy observations, not drop-in
estimators of the corrected target.

## scverse

| Tool | Reusable capability | Boundary | Decision |
|---|---|---|---|
| [AnnData](https://anndata.readthedocs.io/en/stable/generated/anndata.AnnData.html) | Axis-aligned `X`, `layers`, `varm`, and `varp`; H5AD I/O | A container, not an uncertainty-aware estimator | Optional `anndata` extra and the primary single-cell interface |
| [Scanpy](https://scanpy.readthedocs.io/en/stable/generated/scanpy.pp.pca.html) | Filtering, highly-variable-gene selection, standard PCA and public example datasets | Standard PCA/covariance does not consume per-cell/per-gene Gaussian error bars | Notebook/development group only, not imported by the package |
| [scvi-tools](https://docs.scvi-tools.org/en/stable/api/reference/scvi.model.base.RNASeqMixin.html) | Count-level generative models, posterior and posterior-predictive samples | scVI does not directly solve the supplied-(x,delta) covariance problem. Posterior uncertainty, count noise and this project's Gaussian observation error are distinct | Optional `scvi` extra and documented bridge only |

The core package intentionally does not depend on Scanpy or scvi-tools.

## Probabilistic programming and autodiff

| Tool | What it provides | Decision |
|---|---|---|
| [PyMC](https://www.pymc.io/projects/docs/en/stable/api/distributions/generated/pymc.MvNormal.html) | Bayesian multivariate normals, LKJ/Cholesky covariance priors and posterior inference | Excellent for a user who needs a prior and uncertainty over (Sigma), but too heavy for the deterministic core and not a ready-made (D_c)-aware estimator |
| [NumPyro](https://num.pyro.ai/en/stable/distributions.html) | JAX-backed multivariate distributions, NUTS/SVI and Cholesky parameterizations | Same modeling opportunity as PyMC; not needed for the requested frequentist estimators |
| [JAX](https://docs.jax.dev/en/latest/automatic-differentiation.html) | Automatic differentiation, batched linear algebra and accelerator execution | Would duplicate the required NumPy array/backend surface. The dense model has a tractable analytic gradient, while the factor ECM has closed conditional updates | Not a dependency in 0.1 |
| [JAXopt](https://github.com/google/jaxopt) | Historically provided JAX optimization wrappers | Its repository states that it is no longer maintained or developed | Rejected as a dependency |

The dense MLE uses an analytic gradient with SciPy L-BFGS-B. This keeps NumPy/AnnData
interoperability direct and avoids requiring accelerator-specific wheels.

## Specialized implementations found

- [HePPCAT.jl](https://github.com/dahong67/HePPCAT.jl) is a maintained Julia implementation of
  probabilistic PCA for samples with heterogeneous quality. Its noise model is not the known
  element-wise (C\times G) variance used here, and it is not a Python dependency.
- [ALPCAH](https://github.com/javiersc1/ALPCAH) addresses sample-wise heteroscedastic subspace
  learning and estimates sample noise; it does not expose this known cell-feature error model.
- [empca](https://github.com/sbailey/empca) implements weighted expectation-maximization PCA for
  noisy/missing entries. It is useful prior art, but does not provide the complete covariance,
  uniqueness, covariate, AnnData, PSD and likelihood API required here.
- Searches for a maintained Python package specifically implementing dense covariance MLE with
  a separate known diagonal (D_c) per observation did not identify a suitable composable
  library. This is why the small amount of custom statistical machinery is necessary.

The custom code is limited to the model-specific objective/gradient, leverage-aware moment
correction, and heteroscedastic factor ECM. Standard eigendecomposition, Cholesky solves,
optimization, truncated SVD, storage and plotting remain delegated to maintained libraries.

## Name audit

The working name `uncertaincov` was not retained. On 2026-09-23, the PyPI JSON endpoint for
`noisecov` returned 404 and GitHub repository search showed no exact project using that name.
`noisecov` was selected because it is short and describes covariance separation from known
noise without implying a single-cell-only scope. Availability must be rechecked immediately
before publication because names can be claimed at any time.
