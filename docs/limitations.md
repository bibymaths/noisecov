# Assumptions and limitations

- Supplied `sd` values must be meaningful measurement standard deviations on the same scale as
  `X`.
- Standard deviations alone cannot identify correlated measurement errors.
- Raw UMI counts are discrete; Gaussian errors are commonly inappropriate without a defensible
  transformation or upstream posterior quantity.
- Covariance depends on normalization, transformation, filtering, and feature selection.
- Pooling cell populations produces covariance driven partly by population composition.
- Batch, donor, condition, cell type, library size, and other covariates can induce apparent
  correlation. Include an explicit mean model or stratify when appropriate.
- Noise subtraction can produce an indefinite matrix. PSD repair is regularization, not
  additional biological information.
- Covariance does not imply direct interaction or causality. Precision-matrix estimation is a
  separate problem requiring its own regularization.
- Treating `sd` as known does not propagate uncertainty in the error bars themselves.
- The Gaussian latent distribution is a working model. Heavy tails, mixtures, and nonlinear
  structure can change the estimand and inference.
- Factor loadings are rotationally non-identifiable.

Calibration should be checked with replicates, controls, posterior predictive diagnostics, or
domain-specific uncertainty validation whenever possible.
