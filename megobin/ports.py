"""Typed values that travel between stages (one class per port)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from megobin.z import Z

log = logging.getLogger(__name__)


@dataclass
class Features:
    """Port ``features``: contig feature matrix as written by feature_merge.

    ``x`` is ``[k-mer block | depth block]`` per contig. ``split_path`` points
    at ``data_split.csv`` (loaded on demand, ``[left halves; right halves]``).
    """

    x: np.ndarray
    names: np.ndarray
    kmer_dim: int
    split_path: Path | None = None

    @property
    def n_abund(self) -> int:
        return int(self.x.shape[1] - self.kmer_dim)

    def load_split(self) -> np.ndarray:
        """Read ``data_split.csv`` into the ``[left; right]`` layout.

        feature_merge writes rows interleaved ``[c1_1, c1_2, c2_1, ...]``
        (names suffixed ``_1`` / ``_2``); samplers expect all left halves first.
        """
        if self.split_path is None:
            raise FileNotFoundError("no data_split.csv available for this dataset")
        df = pd.read_csv(self.split_path, index_col=0)
        idx = df.index.astype(str)
        left = df.loc[idx.str.endswith("_1")].values
        right = df.loc[idx.str.endswith("_2")].values
        if left.shape[0] != right.shape[0]:
            raise ValueError(
                f"data_split.csv at {self.split_path} has mismatched halves: "
                f"{left.shape[0]} '_1' rows vs {right.shape[0]} '_2' rows"
            )
        return np.concatenate([left, right], axis=0).astype(np.float32)


@dataclass
class Embedding:
    """Port ``z`` / ``z_kept``: labelled embedding plus the contig names of its rows."""

    z: Z[np.ndarray]
    names: np.ndarray
    dropped: np.ndarray | None = None


@dataclass
class Bins:
    """Port ``bins`` / ``bins_refined``: one integer label per contig; -1 = unbinned."""

    labels: np.ndarray
    names: np.ndarray


@dataclass
class ContigSet:
    """Port ``contigs``: the assembly FASTA of the dataset."""

    fasta: Path
    _seqs: dict[str, str] | None = field(default=None, repr=False)

    @property
    def sequences(self) -> dict[str, str]:
        if self._seqs is None:
            from megobin.utils.fasta import fasta_iter

            self._seqs = dict(fasta_iter(str(self.fasta)))
        return self._seqs

    def lengths_for(self, names: np.ndarray) -> np.ndarray | None:
        """Contig lengths aligned to ``names``; ``None`` if any contig is missing."""
        seqs = self.sequences
        lens = [len(seqs.get(str(n), "")) for n in names]
        if all(ell > 0 for ell in lens):
            return np.asarray(lens)
        log.warning(
            "Some contigs missing from %s — DBSCAN length-weighting disabled.",
            self.fasta,
        )
        return None


@dataclass
class GraphFile:
    """Port ``graph``: an assembly graph on disk (consumed by graph-aware binners/refiners)."""

    path: Path


def write_bins(bins: Bins, contigs: ContigSet | None, out_dir: Path) -> int:
    """Write one FASTA per bin (label -1 is unbinned and skipped). Returns #bins."""
    out_dir.mkdir(parents=True, exist_ok=True)
    n_bins = len(np.unique(bins.labels[bins.labels >= 0]))
    if contigs is None:
        return n_bins
    name_to_seq = contigs.sequences
    if bins.names is not None and name_to_seq:
        for bin_id in np.unique(bins.labels):
            if bin_id < 0:
                continue
            members = bins.names[bins.labels == bin_id]
            with open(out_dir / f"bin_{bin_id:04d}.fasta", "w") as f:
                for name in members:
                    seq = name_to_seq.get(str(name), "")
                    if seq:
                        f.write(f">{name}\n{seq}\n")
    return n_bins
