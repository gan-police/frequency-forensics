"""Code testing the welford estimator against numpy functions."""

import numpy as np
import torch
from src.freqdect.prepare_dataset import WelfordEstimator


def test_welford() -> None:
    """Test the welford estimator."""
    test_data = np.random.randn(2000, 128, 128, 3)
    np_mean = np.mean(test_data, axis=(0, 1, 2))
    np_std = np.std(test_data, axis=(0, 1, 2))

    welford = WelfordEstimator()
    for test_el in test_data:
        welford.update(torch.from_numpy(test_el))

    welford_mean, welford_std = welford.finalize()
    assert np.allclose(np_mean, welford_mean)  # noqa: S101
    assert np.allclose(np_std, welford_std)  # noqa: S101
