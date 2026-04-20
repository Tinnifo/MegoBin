import numpy as np
import torch
from torch.utils.data import Dataset


class UncertainGenPairSampler(Dataset):
    """UncertainGen-style contrastive pairs.

    Positive pairs: two halves of the same contig (``features_split``
    layout ``[left_halves; right_halves]``).
    Negative pairs: uniform multinomial sampling over all half-contigs.
    """

    def __init__(
        self,
        features_split: np.ndarray,
        neg_per_pos: int = 1000,
        seed: int = 0,
    ):
        if features_split.shape[0] % 2 != 0:
            raise ValueError(
                f"features_split rows must be even; got {features_split.shape[0]}"
            )

        rng = np.random.default_rng(seed)
        n = features_split.shape[0] // 2
        total = features_split.shape[0]

        pos = np.stack([np.arange(n), np.arange(n) + n], axis=1)
        neg = np.stack(
            [
                rng.integers(0, total, size=neg_per_pos * n),
                rng.integers(0, total, size=neg_per_pos * n),
            ],
            axis=1,
        )

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
