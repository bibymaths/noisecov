from __future__ import annotations

import numpy as np
import pytest

from noisecov import covariance_to_correlation
from noisecov.psd import (
    clip_to_psd,
    is_positive_semidefinite,
    nearest_covariance,
    repair_covariance,
)


@pytest.mark.parametrize("method", ["clip", "nearest"])
def test_psd_repairs_indefinite_matrix(method: str) -> None:
    matrix = np.array([[1.0, 1.4, 0.2], [1.4, 1.0, -0.5], [0.2, -0.5, -0.1]])
    repaired = repair_covariance(matrix, method=method)  # type: ignore[arg-type]
    assert is_positive_semidefinite(repaired, tolerance=1e-8)
    assert np.allclose(repaired, repaired.T)


def test_nearest_preserves_positive_diagonal() -> None:
    matrix = np.array([[2.0, 2.0], [2.0, 1.0]])
    repaired = nearest_covariance(matrix)
    assert np.allclose(np.diag(repaired), np.diag(matrix), atol=1e-7)


def test_clip_is_eigen_projection() -> None:
    matrix = np.array([[1.0, 2.0], [2.0, 1.0]])
    repaired = clip_to_psd(matrix)
    assert np.allclose(np.linalg.eigvalsh(repaired), [0.0, 3.0], atol=1e-10)
    assert np.allclose(repair_covariance(matrix, method="none"), matrix)


def test_correlation_uses_latent_variances() -> None:
    covariance = np.array([[4.0, 1.0], [1.0, 9.0]])
    correlation = covariance_to_correlation(covariance)
    assert np.allclose(correlation, [[1.0, 1 / 6], [1 / 6, 1.0]])
    with pytest.raises(ValueError, match="variances"):
        covariance_to_correlation(np.diag([1.0, 0.0]))
    with_nan = covariance_to_correlation(np.diag([1.0, 0.0]), invalid="nan")
    assert np.isnan(with_nan[1]).all()
    with pytest.raises(ValueError, match="invalid"):
        covariance_to_correlation(covariance, invalid="ignore")


def test_psd_validation_errors() -> None:
    with pytest.raises(ValueError, match="square"):
        clip_to_psd(np.ones((2, 3)))
    with pytest.raises(ValueError, match="non-negative"):
        clip_to_psd(np.eye(2), min_eigenvalue=-1)
    with pytest.raises(ValueError, match="method"):
        repair_covariance(np.eye(2), method="bad")  # type: ignore[arg-type]
    assert not is_positive_semidefinite(np.array([[1.0, 0.2], [0.1, 1.0]]))
