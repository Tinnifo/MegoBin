from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Filter(Protocol):
    """Pre-clustering filter slot.

    A Filter consumes encoder embeddings (and any side-outputs the
    encoder produces — e.g. UncertainGen's per-contig covariance) and
    returns a subset to pass to the binner. The default implementation,
    :class:`NoOpFilter`, keeps everything.
    """

    def fit_transform(
        self,
        embeddings: np.ndarray,
        contig_ids: np.ndarray,
        side_outputs: dict[str, np.ndarray] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns ``(kept_embeddings, kept_ids, dropped_ids)``."""
        ...
