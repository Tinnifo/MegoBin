from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Binner(Protocol):
    def cluster(self, embeddings: np.ndarray) -> np.ndarray:
        """(N, d) → (N,) integer bin assignments"""
        ...
