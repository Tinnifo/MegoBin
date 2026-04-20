from typing import Protocol

import numpy as np


class Representation(Protocol):
    def encode(self, features: np.ndarray) -> np.ndarray:
        """(N, input_dim) → (N, embedding_dim)

        `features` is the encoder's input matrix. It may be k-mer profiles
        alone, or k-mer profiles concatenated with abundance / coverage
        vectors — encoders document their own restrictions.
        """
        ...

    @property
    def embedding_dim(self) -> int: ...
