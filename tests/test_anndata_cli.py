from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pytest
from typer.testing import CliRunner

from noisecov.anndata import covariance_memory_bytes, from_anndata
from noisecov.cli import app


def make_adata() -> ad.AnnData:
    rng = np.random.default_rng(6)
    X = rng.normal(size=(60, 6))
    adata = ad.AnnData(X=np.zeros_like(X))
    adata.var_names = [f"g{i}" for i in range(6)]
    adata.layers["measurement_mean"] = X
    adata.layers["measurement_sd"] = rng.uniform(0.1, 0.4, size=X.shape)
    adata.var["highly_variable"] = [True, True, True, False, False, False]
    return adata


def test_anndata_round_trip_without_mutation() -> None:
    adata = make_adata()
    result = from_anndata(
        adata,
        x_layer="measurement_mean",
        sd_layer="measurement_sd",
        genes=["g0", "g1", "g2"],
        write=True,
    )
    assert "latent_covariance" not in adata.varp
    assert result.adata.shape == (60, 3)
    assert result.adata.varp["latent_covariance"].shape == (3, 3)
    assert covariance_memory_bytes(3, matrices=2) == 144


def test_anndata_factor_writes_loadings() -> None:
    adata = make_adata()
    result = from_anndata(
        adata,
        x_layer="measurement_mean",
        sd_layer="measurement_sd",
        genes=["g0", "g1", "g2", "g3"],
        method="factor",
        n_factors=2,
        max_iter=3,
        write=True,
    )
    assert result.loadings.shape == (4, 2)
    assert "latent_factor_loadings" in result.adata.varm


def test_anndata_guards() -> None:
    adata = make_adata()
    with pytest.raises(KeyError, match="Unknown"):
        from_anndata(adata, sd_layer="measurement_sd", genes=["missing"])
    with pytest.raises(ValueError, match="subset"):
        from_anndata(
            adata,
            x_layer="measurement_mean",
            sd_layer="measurement_sd",
            genes=["g0", "g1"],
            write=True,
            copy=False,
        )
    with pytest.raises(MemoryError):
        from_anndata(
            adata,
            x_layer="measurement_mean",
            sd_layer="measurement_sd",
            max_dense_bytes=100,
        )


def test_cli_simulate_inspect_and_estimate(tmp_path: Path) -> None:
    runner = CliRunner()
    simulation_path = tmp_path / "simulation.npz"
    result = runner.invoke(
        app,
        ["simulate", "--output", str(simulation_path), "--n-cells", "30", "--n-features", "4"],
    )
    assert result.exit_code == 0, result.output
    inspected = runner.invoke(app, ["inspect", str(simulation_path)])
    assert inspected.exit_code == 0
    assert "X shape" in inspected.output

    input_path = tmp_path / "input.h5ad"
    output_path = tmp_path / "output.h5ad"
    make_adata().write_h5ad(input_path)
    estimated = runner.invoke(
        app,
        [
            "estimate",
            str(input_path),
            "--output",
            str(output_path),
            "--x-layer",
            "measurement_mean",
            "--sd-layer",
            "measurement_sd",
            "--genes-key",
            "highly_variable",
        ],
    )
    assert estimated.exit_code == 0, estimated.output
    written = ad.read_h5ad(output_path)
    assert written.n_vars == 3
    assert "latent_covariance" in written.varp
