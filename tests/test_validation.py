from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from noisecov import covariance_to_correlation, principal_subspace_angle, relative_frobenius_error
from noisecov._validation import (
    as_dense_2d,
    dense_nbytes,
    measurement_variances,
    require_dense_budget,
    validate_error_covariance,
)


def test_dense_conversion_and_memory_budget() -> None:
    matrix = sparse.eye(4, format="csr")
    assert np.array_equal(as_dense_2d(matrix, name="matrix"), np.eye(4))
    assert dense_nbytes((4, 4)) == 128
    with pytest.raises(MemoryError, match="configured"):
        require_dense_budget((100, 100), max_bytes=10, purpose="test")
    with pytest.raises(ValueError, match="two-dimensional"):
        as_dense_2d(np.ones(4), name="vector")
    with pytest.raises(ValueError, match="finite"):
        as_dense_2d([[1.0, np.nan]], name="bad")


def test_full_error_covariance_validation() -> None:
    valid = np.broadcast_to(np.eye(2), (3, 2, 2)).copy()
    assert np.array_equal(validate_error_covariance(valid, n_observations=3, n_features=2), valid)
    with pytest.raises(ValueError, match="shape"):
        validate_error_covariance(valid[:2], n_observations=3, n_features=2)
    asymmetric = valid.copy()
    asymmetric[0, 0, 1] = 0.5
    with pytest.raises(ValueError, match="symmetric"):
        validate_error_covariance(asymmetric, n_observations=3, n_features=2)
    indefinite = valid.copy()
    indefinite[1] = np.array([[1.0, 2.0], [2.0, 1.0]])
    with pytest.raises(ValueError, match="not positive"):
        validate_error_covariance(indefinite, n_observations=3, n_features=2)
    with pytest.raises(ValueError, match="exactly one"):
        measurement_variances(np.ones((3, 2)), None, None, max_bytes=1_000)


def test_remaining_public_validation() -> None:
    with pytest.raises(ValueError, match="square"):
        covariance_to_correlation(np.ones((2, 3)))
    with pytest.raises(ValueError, match="finite"):
        covariance_to_correlation(np.array([[1.0, np.nan], [np.nan, 1.0]]))
    with pytest.raises(ValueError, match="non-zero"):
        relative_frobenius_error(np.zeros((2, 2)), np.zeros((2, 2)))
    with pytest.raises(ValueError, match="exceed"):
        principal_subspace_angle(np.eye(2), np.eye(2), rank=3)
