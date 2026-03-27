"""ECT backend adapter backed by the `trailed` package."""

from __future__ import annotations

import importlib

import numpy as np


def _load_backend():
    try:
        return importlib.import_module("trailed")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("No ECT backend found. Install `trailed`.") from exc


_BACKEND = _load_backend()


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
    ect = _BACKEND.compute_ect_from_numpy(
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
