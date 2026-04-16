import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


class KMedoidsBinner:
    """Greedy K-Medoid clustering (RevisitingKmers / UncertainGen).

    1. Build full N×N similarity matrix from embeddings.
    2. Threshold at a percentile of the calibration similarities.
    3. Iteratively pick the densest seed, refine the medoid, assign the
       cluster, remove assigned points, and repeat.
    """

    def __init__(
        self,
        metric: str = "cityblock",
        similarity: str = "exp_neg",
        threshold_percentile: float = 70.0,
        medoid_refine_iters: int = 3,
        max_clusters: int = 1000,
        min_bin_size: int = 10,
    ):
        self.metric = metric
        self.similarity = similarity
        self.threshold_percentile = threshold_percentile
        self.medoid_refine_iters = medoid_refine_iters
        self.max_clusters = max_clusters
        self.min_bin_size = min_bin_size

    # ------------------------------------------------------------------
    # Similarity matrix
    # ------------------------------------------------------------------

    def _build_similarity(self, embeddings: np.ndarray) -> np.ndarray:
        """(N, d) → (N, N) similarity matrix."""
        if self.similarity == "dot":
            return embeddings @ embeddings.T

        dist = cdist(embeddings, embeddings, metric=self.metric)

        if self.similarity == "exp_neg":
            return np.exp(-dist)

        raise ValueError(f"Unknown similarity: {self.similarity}")

    # ------------------------------------------------------------------
    # Core greedy loop
    # ------------------------------------------------------------------

    def cluster(self, embeddings: np.ndarray) -> np.ndarray:
        """(N, d) → (N,) integer bin assignments."""
        n = embeddings.shape[0]
        sim = self._build_similarity(embeddings)

        # Threshold: values below this are "not similar"
        upper_vals = sim[np.triu_indices(n, k=1)]
        threshold = np.percentile(upper_vals, self.threshold_percentile)

        labels = np.full(n, -1, dtype=np.int64)
        remaining = np.ones(n, dtype=bool)
        cluster_id = 0

        for _ in range(self.max_clusters):
            rem_idx = np.where(remaining)[0]
            if len(rem_idx) == 0:
                break

            # Sub-matrix of similarities among remaining points
            sub_sim = sim[np.ix_(rem_idx, rem_idx)]

            # Density = number of neighbours above threshold per point
            density = (sub_sim >= threshold).sum(axis=1)

            # Pick the densest seed
            seed_local = int(np.argmax(density))
            seed_global = rem_idx[seed_local]

            # Refine medoid: move towards mean of nearby points
            medoid = seed_global
            for _ in range(self.medoid_refine_iters):
                neighbours = rem_idx[sub_sim[seed_local] >= threshold]
                if len(neighbours) == 0:
                    break
                centroid = embeddings[neighbours].mean(axis=0)
                dists = cdist(centroid[np.newaxis, :], embeddings[neighbours], metric=self.metric)[0]
                medoid = neighbours[int(np.argmin(dists))]
                # Update seed_local for next refinement iteration
                seed_local = int(np.where(rem_idx == medoid)[0][0])

            # Assign cluster: all remaining points similar enough to medoid
            member_mask = sim[medoid, rem_idx] >= threshold
            members = rem_idx[member_mask]

            if len(members) < self.min_bin_size:
                # Not enough members — mark remaining as noise and stop
                break

            labels[members] = cluster_id
            remaining[members] = False
            cluster_id += 1

        # Assign unclustered points to nearest medoid
        unassigned = np.where(labels == -1)[0]
        if len(unassigned) > 0 and cluster_id > 0:
            medoids = []
            for cid in range(cluster_id):
                members = np.where(labels == cid)[0]
                centroid = embeddings[members].mean(axis=0)
                dists = cdist(centroid[np.newaxis, :], embeddings[members], metric=self.metric)[0]
                medoids.append(members[int(np.argmin(dists))])
            medoid_embeds = embeddings[np.array(medoids)]
            dists = cdist(embeddings[unassigned], medoid_embeds, metric=self.metric)
            labels[unassigned] = np.argmin(dists, axis=1)

        return labels

    # ------------------------------------------------------------------
    # Label alignment (for evaluation against ground truth)
    # ------------------------------------------------------------------

    @staticmethod
    def align_labels(
        predicted: np.ndarray, ground_truth: np.ndarray
    ) -> np.ndarray:
        """Remap predicted cluster IDs to best-match ground-truth IDs
        using the Hungarian algorithm.

        Returns relabelled copy of *predicted*.
        """
        pred_ids = np.unique(predicted)
        gt_ids = np.unique(ground_truth)

        # Build cost matrix: -overlap for each (pred, gt) pair
        cost = np.zeros((len(pred_ids), len(gt_ids)), dtype=np.int64)
        for i, pid in enumerate(pred_ids):
            for j, gid in enumerate(gt_ids):
                cost[i, j] = -np.sum((predicted == pid) & (ground_truth == gid))

        row_ind, col_ind = linear_sum_assignment(cost)

        mapping = {pred_ids[r]: gt_ids[c] for r, c in zip(row_ind, col_ind)}
        return np.array([mapping.get(p, -1) for p in predicted], dtype=np.int64)
