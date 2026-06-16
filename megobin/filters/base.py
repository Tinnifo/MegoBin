from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Filter(Protocol):
    def fit_transform(
        self,
        embeddings: np.ndarray,
        contig_ids: np.ndarray,
        side_outputs: dict[str, np.ndarray] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...
