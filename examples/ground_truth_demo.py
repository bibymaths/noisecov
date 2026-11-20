"""Ground-truth covariance recovery experiment used in the documentation."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

from noisecov import (
    HeteroscedasticFactorAnalysis,
    HeteroscedasticGaussianCovariance,
    corrected_covariance,
    correlation_rmse,
    covariance_to_correlation,
    naive_covariance,
    relative_frobenius_error,
    simulate_measurement_error,
)


def main() -> None:
    output = Path("results")
    output.mkdir(exist_ok=True)
    sim = simulate_measurement_error(
        n_cells=1_500,
        n_features=10,
        covariance="block",
        noise="gene_cell",
        noise_scale=0.9,
        random_state=2026,
    )
    estimates = {
        "naive": naive_covariance(sim.X),
        "corrected": corrected_covariance(sim.X, sim.sd, psd="clip"),
    }
    mle = HeteroscedasticGaussianCovariance(max_iter=100, tolerance=1e-6).fit(sim.X, sim.sd)
    factor = HeteroscedasticFactorAnalysis(n_factors=2, max_iter=100, tolerance=1e-5).fit(
        sim.X, sim.sd
    )
    estimates["mle"] = mle.covariance_
    estimates["factor"] = factor.covariance_

    metrics = {
        name: {
            "relative_frobenius_error": relative_frobenius_error(value, sim.Sigma_true),
            "correlation_rmse": correlation_rmse(value, sim.Sigma_true),
        }
        for name, value in estimates.items()
    }
    metrics["mle"]["converged"] = mle.converged_
    metrics["factor"]["converged"] = factor.converged_
    (output / "ground_truth_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )

    matrices = {"truth": sim.Sigma_true, **estimates}
    figure, axes = plt.subplots(2, len(matrices), figsize=(3 * len(matrices), 5.6))
    covariance_limit = max(abs(matrix).max() for matrix in matrices.values())
    for column, (name, matrix) in enumerate(matrices.items()):
        axes[0, column].imshow(
            matrix, cmap="coolwarm", vmin=-covariance_limit, vmax=covariance_limit
        )
        axes[0, column].set_title(name)
        axes[0, column].set_xticks([])
        axes[0, column].set_yticks([])
        axes[1, column].imshow(covariance_to_correlation(matrix), cmap="coolwarm", vmin=-1, vmax=1)
        axes[1, column].set_xticks([])
        axes[1, column].set_yticks([])
    axes[0, 0].set_ylabel("covariance")
    axes[1, 0].set_ylabel("correlation")
    figure.suptitle("Known truth versus noise-aware estimates")
    figure.tight_layout()
    figure.savefig(output / "ground_truth_comparison.png", dpi=180)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
