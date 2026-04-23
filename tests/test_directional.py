"""Perfect-input sanity: perfectly separated embeddings fed into every
binner must produce reasonable bins.

If a binner can't recover obvious clusters, it's broken.
"""

import numpy as np
from sklearn.metrics import adjusted_rand_score

from megobin.binners.dbscan_ensemble import DBSCANEnsembleBinner
from megobin.binners.infomap import InfomapBinner


def _make_perfect_clusters(
    n_clusters: int = 3,
    points_per_cluster: int = 50,
    separation: float = 10.0,
    noise: float = 0.1,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate well-separated clusters with axis-aligned centers."""
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


class TestInfomapDirectional:
    def test_perfect_clusters(self):
        n = 3
        embeddings, gt = _make_perfect_clusters(n_clusters=n, points_per_cluster=30)
        binner = InfomapBinner(k_neighbours=10, n_trials=5)
        labels = binner.cluster(embeddings)
        ari = adjusted_rand_score(gt, labels)
        assert ari > 0.9, f"ARI = {ari:.3f} (expected > 0.9)"


class TestDBSCANEnsembleDirectional:
    def test_perfect_clusters(self):
        n = 3
        embeddings, gt = _make_perfect_clusters(
            n_clusters=n, points_per_cluster=30, separation=5.0
        )
        binner = DBSCANEnsembleBinner(
            eps_values=np.linspace(0.5, 3.0, 8).tolist(),
            min_samples=5,
            min_bin_size=10,
        )
        labels = binner.cluster(embeddings)
        ari = adjusted_rand_score(gt, labels)
        assert ari > 0.9, f"ARI = {ari:.3f} (expected > 0.9)"
