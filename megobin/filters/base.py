from typing import Mapping, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Filter(Protocol):
    @property
    def needs(self) -> Mapping[str, frozenset[str]]:
        """Contract on port ``z``: the labels the filter reads."""
        ...

    def fit_transform(
        self,
        embeddings: np.ndarray,
        contig_ids: np.ndarray,
        side_outputs: dict[str, np.ndarray] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...
