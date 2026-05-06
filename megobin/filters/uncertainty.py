import numpy as np


class UncertaintyFilter:
    """Drop the most uncertain ``ratio`` fraction of contigs by their covariance.

    Expects ``side_outputs["covariance"]`` of shape ``(N, d)`` produced
    by an encoder that exposes ``encode_with_uncertainty`` (e.g.
    :class:`UncertainGenEncoder`). The per-contig uncertainty score
    collapses ``cov`` along the embedding axis using ``metric``
    (``"max_variance"`` or ``"mean_variance"``); the highest-scoring
    ``ratio`` are dropped.
    """

    def __init__(self, ratio: float = 0.10, metric: str = "max_variance"):
        if not 0.0 <= ratio < 1.0:
            raise ValueError(f"ratio must be in [0, 1); got {ratio}")
        if metric not in ("max_variance", "mean_variance"):
            raise ValueError(
                f"metric must be 'max_variance' or 'mean_variance'; got {metric}"
            )
        self.ratio = ratio
        self.metric = metric

    def fit_transform(
        self,
        embeddings: np.ndarray,
        contig_ids: np.ndarray,
        side_outputs: dict[str, np.ndarray] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if side_outputs is None or "covariance" not in side_outputs:
            raise ValueError(
                "UncertaintyFilter needs side_outputs['covariance'] from an "
                "encoder with encode_with_uncertainty (e.g. UncertainGenEncoder)."
            )
        cov = side_outputs["covariance"]
        if cov.shape[0] != embeddings.shape[0]:
            raise ValueError(
                f"covariance rows ({cov.shape[0]}) must match embeddings rows "
                f"({embeddings.shape[0]})."
            )

        score = (
            cov.max(axis=1) if self.metric == "max_variance" else cov.mean(axis=1)
        )
        n = len(score)
        n_keep = max(1, int(round(n * (1 - self.ratio))))
        keep_idx = np.argsort(score, kind="stable")[:n_keep]
        keep_idx.sort()
        keep_mask = np.zeros(n, dtype=bool)
        keep_mask[keep_idx] = True

        kept_embeddings = embeddings[keep_mask]
        kept_ids = contig_ids[keep_mask]
        dropped_ids = contig_ids[~keep_mask]
        return kept_embeddings, kept_ids, dropped_ids
