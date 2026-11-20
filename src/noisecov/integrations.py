"""Small bridges from upstream uncertainty models."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def posterior_samples_to_gaussian(
    posterior_samples: ArrayLike,
    *,
    ddof: int = 1,
) -> tuple[FloatArray, FloatArray]:
    """Summarize posterior draws as cell-feature Gaussian means and SDs.

    Parameters
    ----------
    posterior_samples
        Array with shape ``(n_draws, n_cells, n_features)``. The sampled
        quantity and its scale must be scientifically defined upstream.
    ddof
        Delta degrees of freedom for the standard deviation.

    Returns
    -------
    mean, sd
        Arrays with shape ``(n_cells, n_features)``.

    Notes
    -----
    This lossy approximation discards posterior error correlations and does not
    turn count-level variability into Gaussian measurement error. It is useful
    only when marginal Gaussian summaries are appropriate for the chosen
    posterior quantity.
    """
    samples = np.asarray(posterior_samples, dtype=np.float64)
    if samples.ndim != 3:
        raise ValueError("posterior_samples must have shape (draws, cells, features).")
    if samples.shape[0] <= ddof:
        raise ValueError("n_draws must be greater than ddof.")
    if not np.all(np.isfinite(samples)):
        raise ValueError("posterior_samples must contain only finite values.")
    return samples.mean(axis=0), samples.std(axis=0, ddof=ddof)
