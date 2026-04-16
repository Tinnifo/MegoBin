from typing import Protocol

import numpy as np


class Binner(Protocol):
    def cluster(self, embeddings: np.ndarray) -> np.ndarray:
        """(N, d) → (N,) integer bin assignments"""
        ...
