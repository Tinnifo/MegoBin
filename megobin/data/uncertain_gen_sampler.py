import numpy as np
import torch
from torch.utils.data import Dataset


class UncertainGenPairSampler(Dataset):
    """UncertainGen contrastive pair sampler (paper-faithful).

    Reproduces the pair-construction scheme from
    "UncertainGen: Uncertainty-Aware Representations of DNA Sequences for
    Metagenomic Binning" (arXiv:2509.26116):

    - Positive pairs: the two halves of the same contig.
    - Negative pairs: a random half of one contig paired with a random
      half of a *different* contig (200 per positive in the paper).

    ``features_split`` is laid out as ``[left_halves; right_halves]`` so
    row ``i`` and row ``i + N`` are the two halves of contig ``i``.
    """

    def __init__(
        self,
        features_split: np.ndarray,
        neg_per_pos: int = 200,
        seed: int = 0,
    ):
        if features_split.shape[0] % 2 != 0:
            raise ValueError(
                f"features_split rows must be even; got {features_split.shape[0]}"
            )

        rng = np.random.default_rng(seed)
        n = features_split.shape[0] // 2
        if n < 2:
            raise ValueError(
                f"need >=2 contigs to draw distinct-contig negatives; got n={n}"
            )

        pos = np.stack([np.arange(n), np.arange(n) + n], axis=1)

        m = neg_per_pos * n
        contig_a = rng.integers(0, n, size=m)
        contig_b = rng.integers(0, n, size=m)
        collisions = contig_a == contig_b
        while collisions.any():
            contig_b[collisions] = rng.integers(0, n, size=int(collisions.sum()))
            collisions = contig_a == contig_b

        half_a = rng.integers(0, 2, size=m)
        half_b = rng.integers(0, 2, size=m)
        neg = np.stack([contig_a + half_a * n, contig_b + half_b * n], axis=1)

        self._x = torch.from_numpy(features_split).float()
        self._pos = pos
        self._neg = neg
        self._n_pos = len(pos)

    def __len__(self) -> int:
        return self._n_pos + len(self._neg)

    def __getitem__(self, idx):
        if idx < self._n_pos:
            i, j = self._pos[idx]
            return self._x[i], self._x[j], torch.tensor(1.0)
        i, j = self._neg[idx - self._n_pos]
        return self._x[i], self._x[j], torch.tensor(0.0)
