from typing import Protocol

import numpy as np


class Representation(Protocol):
    def encode(self, kmer_profiles: np.ndarray) -> np.ndarray:
        """(N, input_dim) → (N, embedding_dim)"""
        ...

    @property
    def embedding_dim(self) -> int: ...
