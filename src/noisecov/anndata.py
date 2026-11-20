"""Optional AnnData interoperability without a mandatory scverse dependency."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from ._validation import as_dense_2d, dense_nbytes, require_dense_budget
from .correlation import covariance_to_correlation
from .covariance import corrected_covariance, naive_covariance
from .factor import HeteroscedasticFactorAnalysis
from .likelihood import HeteroscedasticGaussianCovariance

FloatArray = NDArray[np.float64]
AnnDataMethod = Literal["naive", "corrected", "mle", "factor"]


def _require_anndata() -> Any:
    try:
        import anndata as ad
    except ImportError as error:  # pragma: no cover - environment dependent
        raise ImportError(
            "AnnData support requires the optional dependency: uv add 'noisecov[anndata]'"
        ) from error
    return ad


def covariance_memory_bytes(n_features: int, *, matrices: int = 1) -> int:
    """Estimate memory required for float64 dense feature covariance matrices."""
    if n_features < 0 or matrices < 1:
        raise ValueError("n_features must be non-negative and matrices positive.")
    return dense_nbytes((n_features, n_features)) * matrices


@dataclass
class AnnDataCovarianceResult:
    """Result of an AnnData covariance workflow."""

    covariance: FloatArray | None
    correlation: FloatArray | None
    genes: NDArray[np.str_]
    method: str
    model: Any | None = None
    loadings: FloatArray | None = None
    adata: Any | None = None


def _select_genes(adata: Any, genes: Sequence[str] | Sequence[int] | NDArray[Any] | None) -> Any:
    if genes is None:
        return adata
    values = np.asarray(genes)
    if values.dtype == bool:
        if values.shape != (adata.n_vars,):
            raise ValueError("A boolean genes mask must have length adata.n_vars.")
        return adata[:, values]
    if np.issubdtype(values.dtype, np.integer):
        return adata[:, values.astype(int)]
    missing = set(values.astype(str)) - set(adata.var_names.astype(str))
    if missing:
        preview = ", ".join(sorted(missing)[:5])
        raise KeyError(f"Unknown genes: {preview}")
    return adata[:, values.astype(str)]


def _matrix(adata: Any, layer: str | None, *, label: str, max_dense_bytes: int) -> FloatArray:
    if layer is None:
        source = adata.X
    else:
        if layer not in adata.layers:
            raise KeyError(f"AnnData layer {layer!r} does not exist.")
        source = adata.layers[layer]
    return as_dense_2d(source, name=label, max_bytes=max_dense_bytes)


def from_anndata(
    adata: Any,
    *,
    x_layer: str | None = None,
    sd_layer: str,
    genes: Sequence[str] | Sequence[int] | NDArray[Any] | None = None,
    method: AnnDataMethod = "corrected",
    covariates: NDArray[Any] | None = None,
    n_factors: int = 20,
    psd: str = "clip",
    shrinkage: float = 0.0,
    write: bool = False,
    copy: bool = True,
    overwrite: bool = False,
    max_dense_bytes: int = 512 * 1024**2,
    **model_kwargs: Any,
) -> AnnDataCovarianceResult:
    """Estimate feature covariance from an AnnData measurement/error pair.

    ``adata`` is never modified unless ``write=True`` and ``copy=False``. When
    a gene subset is requested, writing always targets a subset copy because
    ``varp`` matrices must match the object's complete variable axis.
    """
    _require_anndata()
    selected = _select_genes(adata, genes)
    n_features = selected.n_vars
    if method != "factor":
        require_dense_budget(
            (n_features, n_features),
            max_bytes=max_dense_bytes,
            purpose="AnnData feature covariance",
            arrays=2,
        )
    X = _matrix(selected, x_layer, label="AnnData measurements", max_dense_bytes=max_dense_bytes)
    sd = _matrix(selected, sd_layer, label="AnnData uncertainties", max_dense_bytes=max_dense_bytes)
    if X.shape != sd.shape:
        raise ValueError("Measurement and uncertainty layers must have the same shape.")

    model: Any | None = None
    loadings: FloatArray | None = None
    covariance: FloatArray | None
    if method == "naive":
        covariance = naive_covariance(X)
    elif method == "corrected":
        covariance = corrected_covariance(
            X,
            sd,
            covariates=covariates,
            psd=psd,  # type: ignore[arg-type]
            shrinkage=shrinkage,
            max_dense_bytes=max_dense_bytes,
            **model_kwargs,
        )
    elif method == "mle":
        model = HeteroscedasticGaussianCovariance(
            max_dense_bytes=max_dense_bytes, **model_kwargs
        ).fit(X, sd, covariates=covariates)
        try:
            covariance = model.covariance_
        except MemoryError:
            covariance = None
    elif method == "factor":
        model = HeteroscedasticFactorAnalysis(
            n_factors=n_factors,
            max_dense_bytes=max_dense_bytes,
            **model_kwargs,
        ).fit(X, sd, covariates=covariates)
        loadings = model.loadings_
        covariance = model.covariance_
    else:
        raise ValueError("method must be one of {'naive', 'corrected', 'mle', 'factor'}.")
    correlation = covariance_to_correlation(covariance) if covariance is not None else None
    result = AnnDataCovarianceResult(
        covariance=covariance,
        correlation=correlation,
        genes=np.asarray(selected.var_names, dtype=str),
        method=method,
        model=model,
        loadings=loadings,
    )

    if write:
        is_subset = selected.n_vars != adata.n_vars or not np.array_equal(
            np.asarray(selected.var_names), np.asarray(adata.var_names)
        )
        if is_subset and not copy:
            raise ValueError("Writing a gene subset requires copy=True.")
        target = selected.copy() if copy or is_subset else adata
        keys = ["latent_covariance", "latent_correlation"]
        if loadings is not None:
            keys.append("latent_factor_loadings")
        occupied = [key for key in keys if key in target.varp or key in target.varm]
        if occupied and not overwrite:
            raise KeyError(f"Refusing to overwrite existing AnnData keys: {occupied}")
        if covariance is not None:
            target.varp["latent_covariance"] = covariance
            target.varp["latent_correlation"] = correlation
        if loadings is not None:
            target.varm["latent_factor_loadings"] = loadings
        target.uns["noisecov"] = {
            "method": method,
            "x_layer": x_layer,
            "sd_layer": sd_layer,
            "n_features": int(n_features),
        }
        result.adata = target
    return result
