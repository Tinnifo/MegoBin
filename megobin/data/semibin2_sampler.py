import numpy as np
import torch
from torch.utils.data import Dataset


class SemiBin2PairSampler(Dataset):
    def __init__(
        self,
        features_split: np.ndarray,
        ratio: int = 1000,
        max_pairs: int = 4_000_000,
        seed: int = 0,
    ):
        if features_split.shape[0] % 2 != 0:
            raise ValueError(
                f"features_split rows must be even; got {features_split.shape[0]}"
            )
        n = features_split.shape[0] // 2
        if n < 2:
            raise ValueError(
                f"need >=2 contigs to draw distinct-row negatives; got n={n}"
            )

        self._x = torch.from_numpy(features_split).float()
        self._rows = features_split.shape[0]  # 2N
        self._pos = np.stack([np.arange(n), np.arange(n) + n], axis=1)
        self._n_pos = len(self._pos)
        # SemiBin: n_cannot_link = min(n_must_link * ratio // 2, max_pairs),
        # with n_must_link == number of split rows (2N).
        self._n_neg = min(self._rows * ratio // 2, max_pairs)
        self._seed = int(seed)
        self._neg: np.ndarray | None = None
        self.set_epoch(0)

    def set_epoch(self, epoch: int) -> None:
        """Redraw the random cannot-link pairs for ``epoch`` (deterministic)."""
        rng = np.random.default_rng([self._seed, int(epoch)])
        rows = self._rows
        m = self._n_neg
        idx1 = rng.integers(0, rows, size=m)
        # idx2 != idx1 (SemiBin's offset trick): (idx1 + 1 + [0..rows-2]) % rows
        idx2 = (idx1 + 1 + rng.integers(0, rows - 1, size=m)) % rows
        self._neg = np.stack([idx1, idx2], axis=1)

    def __len__(self) -> int:
        return self._n_pos + self._n_neg

    def __getitem__(self, idx):
        if idx < self._n_pos:
            i, j = self._pos[idx]
            return self._x[i], self._x[j], torch.tensor(1.0)
        assert self._neg is not None
        i, j = self._neg[idx - self._n_pos]
        return self._x[i], self._x[j], torch.tensor(0.0)
