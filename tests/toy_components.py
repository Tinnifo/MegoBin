"""Toy components used by the contract tests (not part of the package)."""

import numpy as np

from megobin.ports import Bins


class GraphBinner:
    """Needs an assembly graph; labels every contig 0."""

    needs = {"z_kept": frozenset({"mu"}), "graph": frozenset({"assembly_graph"})}
    provides = {"bins": frozenset({"one_label_per_contig"})}
    embedding = ("z_kept", "mu")
    bindings = {"graph": "graph"}

    def __init__(self, graph=None):
        self.graph = graph

    def cluster(self, embeddings):
        assert self.graph is not None and self.graph.path.exists()
        return np.zeros(len(embeddings), dtype=np.int64)


class RawFeatureBinner:
    """Clusters raw features (no embedding): label = sign of the first k-mer column
    relative to its median."""

    needs = {"z": frozenset({"raw_features"})}
    provides = {"bins": frozenset({"one_label_per_contig"})}
    embedding = ("z", "raw_features")
    bindings: dict = {}

    def cluster(self, embeddings):
        col = embeddings[:, 0]
        return (col > np.median(col)).astype(np.int64)


class GraphRefiner:
    needs = {
        "bins": frozenset({"one_label_per_contig"}),
        "graph": frozenset({"assembly_graph"}),
    }
    provides = {"bins_refined": frozenset({"one_label_per_contig"})}
    bindings = {"graph": "graph"}

    def __init__(self, graph=None):
        self.graph = graph

    def refine(self, bins: Bins, **inputs) -> Bins:
        return bins
