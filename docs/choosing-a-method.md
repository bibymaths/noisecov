# Choosing a method

| Regime | Recommended starting point | Reason |
|---|---|---|
| $C\gg G$ | Corrected covariance | Transparent, fast, directly checks the noise correction |
| Moderate $G$ | Dense heteroscedastic MLE | Uses every cell-specific $D_c$ in the likelihood |
| $G\approx C$ or $G>C$ | Explicit shrinkage or factor model | Unregularized covariance is unstable or singular |
| Very large $G$ | Gene selection and/or factor model | Never allocate a dense all-gene covariance casually |

Start with the raw corrected estimate even if the final model is likelihood-based. Its diagonal,
eigenvalues, and comparison with naive covariance reveal how strongly the supplied errors drive
the answer.

Choose factor rank by held-out marginal likelihood, stability of the recovered subspace, and
scientific interpretability—not by maximizing in-sample fit alone.

A conditional-dependence network requires a precision matrix, which is a separate regularized
estimation problem. `noisecov` does not equate covariance with direct interaction.
