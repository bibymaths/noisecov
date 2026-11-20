"""Reproducible runtime and peak-Python-memory benchmarks.

Run a quick suite with:
    uv run python benchmarks/benchmark_estimators.py --quick

The full predefined dimensions are intentionally opt-in.
"""

from __future__ import annotations

import argparse
import json
import time
import tracemalloc
from collections.abc import Callable
from pathlib import Path
from typing import Any

from noisecov import (
    HeteroscedasticFactorAnalysis,
    HeteroscedasticGaussianCovariance,
    corrected_covariance,
    naive_covariance,
    simulate_measurement_error,
)
from noisecov.psd import repair_covariance


def measure(function: Callable[[], Any]) -> dict[str, float]:
    """Measure wall time and peak Python-tracked allocation."""
    tracemalloc.start()
    start = time.perf_counter()
    function()
    elapsed = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {"seconds": elapsed, "peak_mib": peak / 1024**2}


def run_case(n_cells: int, n_features: int) -> dict[str, Any]:
    """Run feasible estimators for one dimension pair."""
    sim = simulate_measurement_error(
        n_cells=n_cells,
        n_features=n_features,
        covariance="low_rank",
        n_factors=min(5, n_features - 1),
        random_state=42,
    )
    result: dict[str, Any] = {"n_cells": n_cells, "n_features": n_features}
    result["naive"] = measure(lambda: naive_covariance(sim.X))
    result["corrected"] = measure(lambda: corrected_covariance(sim.X, sim.sd))
    raw = corrected_covariance(sim.X, sim.sd)
    result["psd_clip"] = measure(lambda: repair_covariance(raw, method="clip"))
    rank = min(5, n_features - 1)
    result["factor_5_iterations"] = measure(
        lambda: HeteroscedasticFactorAnalysis(rank, max_iter=5).fit(sim.X, sim.sd)
    )
    if n_features <= 30 and n_cells <= 2_000:
        result["mle_10_iterations"] = measure(
            lambda: HeteroscedasticGaussianCovariance(max_iter=10).fit(sim.X, sim.sd)
        )
    else:
        result["mle"] = "skipped: dense O(C G^3) case is not representative"
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("benchmark-results.json"))
    arguments = parser.parse_args()
    cases = [(300, 15)] if arguments.quick else [(1_000, 50), (5_000, 100), (10_000, 500)]
    results = [run_case(*case) for case in cases]
    arguments.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
