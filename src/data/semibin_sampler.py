import numpy as np
import torch
from torch.utils.data import Dataset


class SemiBinPairSampler(Dataset):
    """SemiBin-style contrastive pair sampler.

    Positive pairs: two halves of the same contig (must-link), drawn from
    ``features_split`` which is laid out as ``[left_halves; right_halves]``
    so row ``i`` and row ``i + N`` form one positive pair.

    Negative pairs: cannot-link pairs mined from mmseqs2 taxonomy, provided
    as (M, 2) indices into ``features_whole``.
    """

    def __init__(
        self,
        features_split: np.ndarray,
        features_whole: np.ndarray,
        cannot_link_pairs: np.ndarray,
        neg_per_pos: int = 500,
        max_pairs: int = 4_000_000,
        seed: int = 0,
    ):
        if features_split.shape[0] % 2 != 0:
            raise ValueError(
                f"features_split rows must be even (left halves then right halves); "
                f"got {features_split.shape[0]}"
            )
        if cannot_link_pairs.ndim != 2 or cannot_link_pairs.shape[1] != 2:
            raise ValueError(
                f"cannot_link_pairs must have shape (M, 2); got {cannot_link_pairs.shape}"
            )
        if features_split.shape[1] != features_whole.shape[1]:
            raise ValueError(
                f"feature dims must match: split={features_split.shape[1]} "
                f"vs whole={features_whole.shape[1]}"
            )

        rng = np.random.default_rng(seed)
        n = features_split.shape[0] // 2

        pos = np.stack([np.arange(n), np.arange(n) + n], axis=1)

        n_neg = min(neg_per_pos * n, cannot_link_pairs.shape[0], max_pairs)
        neg_idx = rng.choice(cannot_link_pairs.shape[0], size=n_neg, replace=False)
        neg = cannot_link_pairs[neg_idx]

        self._x_split = torch.from_numpy(features_split).float()
        self._x_whole = torch.from_numpy(features_whole).float()
        self._pos = pos
        self._neg = neg
        self._n_pos = len(pos)

    def __len__(self) -> int:
        return self._n_pos + len(self._neg)

    def __getitem__(self, idx):
        if idx < self._n_pos:
            i, j = self._pos[idx]
            return self._x_split[i], self._x_split[j], torch.tensor(1.0)
        i, j = self._neg[idx - self._n_pos]
        return self._x_whole[i], self._x_whole[j], torch.tensor(0.0)
