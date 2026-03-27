"""ECT backend adapter backed by the `trailed` package."""

from __future__ import annotations

import numpy as np
from trailed.tabular import compute_ect_from_numpy


def compute_ect_descriptor(
    points: np.ndarray,
    num_thetas: int,
    radius: float,
    resolution: int,
    scale: float,
    seed: int,
) -> np.ndarray:
    """Compute a single-sample ECT descriptor with stable output shape.

    Returns an array shaped [num_thetas, resolution].
    """
    points = np.asarray(points, dtype=np.float32)
    ect = compute_ect_from_numpy(
        points=points,
        num_thetas=num_thetas,
        resolution=resolution,
        radius=radius,
        scale=scale,
        seed=seed,
        normalized=False,
        parallel=True,
    )
    ect = np.asarray(ect, dtype=np.float32)
    if ect.shape == (resolution, num_thetas):
        return ect.T
    if ect.shape == (num_thetas, resolution):
        return ect
    raise ValueError(f"Unexpected ECT shape from trailed backend: {ect.shape}")
