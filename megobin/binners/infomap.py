import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors


class InfomapBinner:
    """Infomap community-detection binner (SemiBin short reads).

    Pipeline:
      1. Build dual k-NN graphs: one on embeddings, one on raw k-mer profiles.
      2. Binarize the k-mer graph, element-wise multiply (intersection).
      3. Convert distances to similarities: w = 1 − min(d, 1).
      4. Run Infomap community detection on the fused graph.
      5. Recluster over-large communities with K-Means.
    """

    def __init__(
        self,
        k_neighbours: int = 200,
        n_trials: int = 10,
        max_bin_size: int | None = None,
        contig_lengths: np.ndarray | None = None,
        kmer_profiles: np.ndarray | None = None,
    ):
        self.k_neighbours = k_neighbours
        self.n_trials = n_trials
        self.max_bin_size = max_bin_size
        self.contig_lengths = contig_lengths
        self.kmer_profiles = kmer_profiles

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    @staticmethod
    def _knn_graph(X: np.ndarray, k: int) -> np.ndarray:
        """Build a k-NN distance matrix (N, N).  Non-neighbour entries are 0."""
        k_actual = min(k, len(X) - 1)
        nn = NearestNeighbors(n_neighbors=k_actual + 1, metric="euclidean")
        nn.fit(X)
        dist, idx = nn.kneighbors(X)

        n = len(X)
        graph = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j_pos in range(1, k_actual + 1):  # skip self (pos 0)
                j = idx[i, j_pos]
                d = dist[i, j_pos]
                graph[i, j] = d
                graph[j, i] = d  # symmetrise
        return graph

    def _build_fused_graph(self, embeddings: np.ndarray) -> np.ndarray:
        """Build the fused similarity graph.

        If kmer_profiles were provided, intersect the two k-NN graphs.
        Otherwise fall back to the embedding k-NN graph alone.
        """
        k = min(self.k_neighbours, len(embeddings) - 1)

        emb_graph = self._knn_graph(embeddings, k)

        if self.kmer_profiles is not None:
            kmer_graph = self._knn_graph(self.kmer_profiles, k)
            # Binarize k-mer graph (edge present / absent)
            kmer_binary = (kmer_graph > 0).astype(np.float64)
            # Intersection: keep embedding distances only where k-mer edge exists
            fused_dist = emb_graph * kmer_binary
        else:
            fused_dist = emb_graph

        # Distance → similarity: w = 1 − min(d, 1)
        fused_sim = np.where(fused_dist > 0, 1.0 - np.minimum(fused_dist, 1.0), 0.0)
        return fused_sim

    # ------------------------------------------------------------------
    # Infomap
    # ------------------------------------------------------------------

    @staticmethod
    def _run_infomap(
        sim: np.ndarray,
        n_trials: int,
        vertex_weights: np.ndarray | None,
    ) -> np.ndarray:
        """Run Infomap community detection and return labels."""
        import igraph as ig

        n = sim.shape[0]
        edges = []
        weights = []
        for i in range(n):
            for j in range(i + 1, n):
                if sim[i, j] > 0:
                    edges.append((i, j))
                    weights.append(sim[i, j])

        g = ig.Graph(n=n, edges=edges, directed=False)
        g.es["weight"] = weights

        if vertex_weights is not None:
            g.vs["weight"] = vertex_weights.tolist()

        membership = g.community_infomap(
            edge_weights="weight",
            vertex_weights="weight" if vertex_weights is not None else None,
            trials=n_trials,
        ).membership

        return np.array(membership, dtype=np.int64)

    # ------------------------------------------------------------------
    # Reclustering
    # ------------------------------------------------------------------

    def _recluster(
        self, labels: np.ndarray, embeddings: np.ndarray
    ) -> np.ndarray:
        """Split over-large bins via K-Means."""
        if self.max_bin_size is None:
            return labels

        new_labels = labels.copy()
        next_id = labels.max() + 1

        for cid in np.unique(labels):
            members = np.where(labels == cid)[0]
            if len(members) <= self.max_bin_size:
                continue
            n_sub = max(2, len(members) // self.max_bin_size)
            km = KMeans(n_clusters=n_sub, n_init=10, random_state=0)
            sub_labels = km.fit_predict(embeddings[members])
            for s in range(n_sub):
                mask = sub_labels == s
                new_labels[members[mask]] = next_id
                next_id += 1

        # Re-number labels to 0..K-1
        _, new_labels = np.unique(new_labels, return_inverse=True)
        return new_labels

    # ------------------------------------------------------------------
    # Binner Protocol
    # ------------------------------------------------------------------

    def cluster(self, embeddings: np.ndarray) -> np.ndarray:
        """(N, d) → (N,) integer bin assignments."""
        sim = self._build_fused_graph(embeddings)
        labels = self._run_infomap(sim, self.n_trials, self.contig_lengths)
        labels = self._recluster(labels, embeddings)
        return labels
