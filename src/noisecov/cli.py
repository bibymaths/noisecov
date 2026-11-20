"""Typer command-line interface."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import numpy as np
import typer

from .anndata import covariance_memory_bytes, from_anndata
from .simulation import simulate_measurement_error

app = typer.Typer(
    help="Estimate latent covariance with known heteroscedastic measurement error.",
    no_args_is_help=True,
)


def _read_h5ad(path: Path):  # type: ignore[no-untyped-def]
    try:
        import anndata as ad
    except ImportError as error:
        raise typer.BadParameter("H5AD commands require: uv add 'noisecov[anndata]'") from error
    return ad.read_h5ad(path)


@app.command()
def estimate(
    data: Annotated[Path, typer.Argument(exists=True, readable=True)],
    output: Annotated[Path, typer.Option("--output", "-o")],
    x_layer: Annotated[str | None, typer.Option()] = None,
    sd_layer: Annotated[str, typer.Option()] = "measurement_sd",
    method: Annotated[str, typer.Option()] = "corrected",
    genes_key: Annotated[
        str | None, typer.Option(help="Boolean adata.var column for selection.")
    ] = None,
    n_factors: Annotated[int, typer.Option()] = 20,
    psd: Annotated[str, typer.Option()] = "clip",
    shrinkage: Annotated[float, typer.Option()] = 0.0,
    overwrite: Annotated[bool, typer.Option()] = False,
) -> None:
    """Estimate covariance from H5AD layers and write a new H5AD file."""
    adata = _read_h5ad(data)
    genes = None
    if genes_key is not None:
        if genes_key not in adata.var:
            raise typer.BadParameter(f"adata.var[{genes_key!r}] does not exist.")
        genes = np.asarray(adata.var[genes_key], dtype=bool)
    result = from_anndata(
        adata,
        x_layer=x_layer,
        sd_layer=sd_layer,
        genes=genes,
        method=method,  # type: ignore[arg-type]
        n_factors=n_factors,
        psd=psd,
        shrinkage=shrinkage,
        write=True,
        copy=True,
        overwrite=overwrite,
    )
    assert result.adata is not None
    result.adata.write_h5ad(output)
    typer.echo(f"Wrote {method} estimate for {len(result.genes)} features to {output}")


@app.command()
def simulate(
    output: Annotated[Path, typer.Option("--output", "-o")] = Path("simulation.npz"),
    n_cells: Annotated[int, typer.Option()] = 1_000,
    n_features: Annotated[int, typer.Option()] = 12,
    covariance: Annotated[str, typer.Option()] = "block",
    noise: Annotated[str, typer.Option()] = "gene_cell",
    noise_scale: Annotated[float, typer.Option()] = 0.7,
    seed: Annotated[int, typer.Option()] = 0,
) -> None:
    """Generate an NPZ dataset with known latent covariance."""
    result = simulate_measurement_error(
        n_cells=n_cells,
        n_features=n_features,
        covariance=covariance,  # type: ignore[arg-type]
        noise=noise,  # type: ignore[arg-type]
        noise_scale=noise_scale,
        random_state=seed,
    )
    np.savez_compressed(
        output,
        X=result.X,
        sd=result.sd,
        Z_true=result.Z_true,
        Sigma_true=result.Sigma_true,
        mean_true=result.mean_true,
        metadata=json.dumps(result.metadata),
    )
    typer.echo(f"Wrote reproducible simulation to {output}")


@app.command()
def inspect(data: Annotated[Path, typer.Argument(exists=True, readable=True)]) -> None:
    """Inspect an H5AD or noisecov simulation without estimating covariance."""
    if data.suffix == ".h5ad":
        adata = _read_h5ad(data)
        memory = covariance_memory_bytes(adata.n_vars, matrices=2) / 1024**2
        typer.echo(f"observations: {adata.n_obs}")
        typer.echo(f"features: {adata.n_vars}")
        typer.echo(f"layers: {', '.join(adata.layers.keys()) or '(none)'}")
        typer.echo(f"two dense feature matrices: {memory:.1f} MiB")
        return
    with np.load(data, allow_pickle=False) as archive:
        if "X" not in archive or "sd" not in archive:
            raise typer.BadParameter("NPZ must contain X and sd arrays.")
        typer.echo(f"X shape: {archive['X'].shape}")
        typer.echo(f"sd range: [{archive['sd'].min():.4g}, {archive['sd'].max():.4g}]")


if __name__ == "__main__":  # pragma: no cover
    app()
