"""
Base interfaces for descriptor generation methods.
"""

from abc import ABC, abstractmethod

import numpy as np
from pydantic import BaseModel


class Magic(ABC):
    def __init__(self, config: BaseModel):
        self.config = config

    @abstractmethod
    def generate(self, data: list[np.ndarray]) -> list[np.ndarray]:
        pass
