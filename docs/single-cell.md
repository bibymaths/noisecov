# Single-cell and AnnData

AnnData follows cells × genes, which matches the public API.

```python
result = from_anndata(
    adata,
    x_layer="measurement_mean",
    sd_layer="measurement_sd",
    genes=adata.var_names[adata.var["highly_variable"]],
    method="corrected",
)
```

Use `write=True` to produce an output AnnData object. Dense covariance and correlation are
stored in `varp`; factor loadings are stored in `varm`. The original object is not modified by
default. When a gene subset is requested, output is a subset copy so `varp` remains aligned.

Sparse input layers are accepted, but the selected measurement/error blocks are densified.
The package estimates memory before dense covariance allocation. Select highly variable or
otherwise hypothesis-relevant genes before dense estimation.

## What scVI does and does not provide

scvi-tools models raw counts with count-based likelihoods and can return posterior or posterior
predictive samples. Those are not automatically Gaussian measurement errors for expression.
If a scientifically chosen posterior quantity has samples `s[m, c, g]`, one may calculate an
approximate mean and standard deviation across posterior draws and use them here:

```python
measurement_mean = posterior_samples.mean(axis=0)
measurement_sd = posterior_samples.std(axis=0, ddof=1)
```

Equivalently, `posterior_samples_to_gaussian(posterior_samples)` performs this validated
summary for a `(draws, cells, genes)` array. An upstream scvi-tools call can request multiple
normalized-expression samples with `n_samples > 1` and `return_mean=False`; verify the returned
axis order for the installed scvi-tools version before passing the array.

That approximation targets uncertainty in that posterior quantity, not raw count sampling
noise, and it discards posterior error correlations. Document the chosen quantity, scale, and
conditioning. `scvi-tools` therefore remains an optional integration, not a covariance engine
or core dependency.
