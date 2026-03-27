"""
Euler Characteristic Transform implementation backed by Rust.
"""

from typing import List

import numpy as np

from . import rust_backend
from .base import Magic
from .config import ECTConfig


class ECT(Magic):
    def __init__(self, config: ECTConfig):
        self.config = config
        self.configure(**config.model_dump())

    def configure(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid configuration key: {key}")

    def generate(self, X: List[np.ndarray]) -> List[np.ndarray]:
        if not isinstance(X, list):
            raise ValueError("Input must be a list of numpy arrays")
        if not X or any(x.size == 0 for x in X):
            raise ValueError("Input cannot be empty")

        scale = float(self.scale)
        descriptors: List[np.ndarray] = []
        for sample in X:
            if np.ndim(sample) != 2:
                raise ValueError("Each sample must be a 2D numpy array")
            cloud = np.asarray(sample, dtype=np.float32)
            descriptor = rust_backend.compute_ect_descriptor(
                points=cloud,
                num_thetas=self.num_thetas,
                radius=self.radius,
                resolution=self.resolution,
                scale=scale,
                seed=self.seed,
            )
            if self.normalize:
                descriptor = self._normalize(descriptor)
            descriptors.append(descriptor)

        return descriptors

    @staticmethod
    def _normalize(arr: np.ndarray) -> np.ndarray:
        lo = np.min(arr)
        hi = np.max(arr)
        if hi <= lo:
            return np.zeros_like(arr)
        return (arr - lo) / (hi - lo)
