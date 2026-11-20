from __future__ import annotations

import numpy as np
import pytest

from noisecov import posterior_samples_to_gaussian


def test_posterior_sample_summary() -> None:
    samples = np.arange(60.0).reshape(5, 4, 3)
    mean, sd = posterior_samples_to_gaussian(samples)
    assert np.allclose(mean, samples.mean(axis=0))
    assert np.allclose(sd, samples.std(axis=0, ddof=1))
    with pytest.raises(ValueError, match="shape"):
        posterior_samples_to_gaussian(samples[0])
    with pytest.raises(ValueError, match="greater"):
        posterior_samples_to_gaussian(samples[:1])
