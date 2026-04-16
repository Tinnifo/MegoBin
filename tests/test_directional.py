"""Perfect-input sanity: perfectly separated embeddings fed into every
binner must produce near-perfect bins (ARI > 0.95).

If the binner can't recover obvious clusters, it's broken.

The threshold_percentile is set per-test so it falls between inter- and
intra-cluster similarities.  With n clusters of equal size the intra-
cluster pair fraction ≈ 1/n, so the percentile must be > 100*(1 − 1/n).
"""

import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score

from src.binners.kmedoids import KMedoidsBinner


def _make_perfect_clusters(
    n_clusters: int = 3,
    points_per_cluster: int = 50,
    separation: float = 10.0,
    noise: float = 0.1,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate well-separated clusters with axis-aligned centers.

    Dimension is set to n_clusters so each center occupies its own axis,
    guaranteeing maximum separation.
    """
    rng = np.random.default_rng(seed)
    dim = n_clusters

    centers = np.zeros((n_clusters, dim))
    for i in range(n_clusters):
        centers[i, i] = separation

    embeddings = []
    labels = []
    for cid in range(n_clusters):
        pts = centers[cid] + rng.normal(scale=noise, size=(points_per_cluster, dim))
        embeddings.append(pts)
        labels.append(np.full(points_per_cluster, cid))

    return np.vstack(embeddings), np.concatenate(labels)


def _threshold_for(n_clusters: int) -> float:
    """Return a percentile that sits between inter and intra pairs."""
    return 100.0 * (1.0 - 0.5 / n_clusters)


class TestKMedoidsDirectional:
    def test_perfect_clusters_l1(self):
        n = 3
        embeddings, gt = _make_perfect_clusters(n_clusters=n)
        binner = KMedoidsBinner(
            metric="cityblock",
            similarity="exp_neg",
            threshold_percentile=_threshold_for(n),
            min_bin_size=10,
        )
        labels = binner.cluster(embeddings)
        ari = adjusted_rand_score(gt, labels)
        assert ari > 0.95, f"ARI = {ari:.3f} (expected > 0.95)"

    def test_perfect_clusters_l2(self):
        n = 3
        embeddings, gt = _make_perfect_clusters(n_clusters=n)
        binner = KMedoidsBinner(
            metric="euclidean",
            similarity="exp_neg",
            threshold_percentile=_threshold_for(n),
            min_bin_size=10,
        )
        labels = binner.cluster(embeddings)
        ari = adjusted_rand_score(gt, labels)
        assert ari > 0.95, f"ARI = {ari:.3f} (expected > 0.95)"

    def test_perfect_clusters_dot(self):
        n = 3
        embeddings, gt = _make_perfect_clusters(n_clusters=n, separation=30.0)
        binner = KMedoidsBinner(
            metric="euclidean",
            similarity="dot",
            threshold_percentile=_threshold_for(n),
            min_bin_size=10,
        )
        labels = binner.cluster(embeddings)
        ari = adjusted_rand_score(gt, labels)
        assert ari > 0.95, f"ARI = {ari:.3f} (expected > 0.95)"

    def test_correct_number_of_clusters(self):
        n = 4
        embeddings, gt = _make_perfect_clusters(n_clusters=n)
        binner = KMedoidsBinner(
            metric="cityblock",
            similarity="exp_neg",
            threshold_percentile=_threshold_for(n),
            min_bin_size=10,
        )
        labels = binner.cluster(embeddings)
        found = len(np.unique(labels))
        assert found == n, f"Found {found} clusters, expected {n}"

    @pytest.mark.parametrize("n_clusters", [2, 3, 5, 7])
    def test_various_cluster_counts(self, n_clusters):
        embeddings, gt = _make_perfect_clusters(n_clusters=n_clusters)
        binner = KMedoidsBinner(
            metric="cityblock",
            similarity="exp_neg",
            threshold_percentile=_threshold_for(n_clusters),
            min_bin_size=10,
        )
        labels = binner.cluster(embeddings)
        ari = adjusted_rand_score(gt, labels)
        assert ari > 0.95, (
            f"k={n_clusters}: ARI = {ari:.3f} (expected > 0.95)"
        )
