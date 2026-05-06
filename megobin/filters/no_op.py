import numpy as np


class NoOpFilter:
    """Pass-through filter — keeps all contigs.

    Used when no Filter is configured, or for encoders that don't
    expose side outputs.
    """

    def fit_transform(
        self,
        embeddings: np.ndarray,
        contig_ids: np.ndarray,
        side_outputs: dict[str, np.ndarray] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        empty = np.empty(0, dtype=contig_ids.dtype)
        return embeddings, contig_ids, empty
