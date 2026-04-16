import numpy as np
from sklearn.cluster import DBSCAN


class DBSCANEnsembleBinner:
    """DBSCAN ensemble binner (SemiBin long reads).

    Runs DBSCAN at multiple ``eps`` values and selects the best set of
    bins.  By default the 12 eps values span 0.01–0.55.

    Without marker-gene information the selection heuristic keeps the
    run that produces the most non-noise bins of reasonable size.  When
    a ``bin_scorer`` callable is provided it is used instead.
    """

    def __init__(
        self,
        eps_values: list[float] | None = None,
        min_samples: int = 5,
        min_bin_size: int = 1,
        bin_scorer: object | None = None,
    ):
        if eps_values is None:
            # 12 values from 0.01 to 0.55 (matching SemiBin defaults)
            self.eps_values = np.linspace(0.01, 0.55, 12).tolist()
        else:
            self.eps_values = eps_values
        self.min_samples = min_samples
        self.min_bin_size = min_bin_size
        self.bin_scorer = bin_scorer

    def cluster(self, embeddings: np.ndarray) -> np.ndarray:
        """(N, d) → (N,) integer bin assignments."""
        best_labels = None
        best_score = -1.0

        for eps in self.eps_values:
            db = DBSCAN(eps=eps, min_samples=self.min_samples, metric="euclidean")
            labels = db.fit_predict(embeddings)

            if self.bin_scorer is not None:
                score = float(self.bin_scorer(labels))
            else:
                score = self._default_score(labels)

            if score > best_score:
                best_score = score
                best_labels = labels

        # Map noise (-1) to its own bin and re-number to 0..K-1
        result = best_labels.copy()
        if (result == -1).any():
            result[result == -1] = result.max() + 1
        _, result = np.unique(result, return_inverse=True)
        return result.astype(np.int64)

    def _default_score(self, labels: np.ndarray) -> float:
        """Heuristic when marker genes are unavailable.

        Score = n_valid_bins × (fraction of points in valid bins).
        Prefers eps values that form multiple bins while assigning most
        points (penalises both over-fragmentation and excessive noise).
        """
        if len(labels) == 0:
            return 0.0
        valid_mask = labels >= 0
        if not valid_mask.any():
            return 0.0
        unique, counts = np.unique(labels[valid_mask], return_counts=True)
        good = counts >= self.min_bin_size
        n_good_bins = good.sum()
        n_assigned = counts[good].sum()
        coverage = n_assigned / len(labels)
        return float(n_good_bins * coverage)
