import numpy as np
import torch
from torch.utils.data import Dataset


class SemiBin2PairSampler(Dataset):
    """SemiBin2 self-supervised pairs.

    Must-link pairs are the two halves of one contig (rows of
    ``features_split``, laid out ``[left halves; right halves]``).

    Cannot-link pairs are two different whole contigs (rows of
    ``features_whole``), redrawn every epoch. This mirrors SemiBin's
    ``train_self``, which indexes the whole-contig matrix of ``data.csv``
    for cannot-links and ``data_split.csv`` for must-links.
    """

    def __init__(
        self,
        features_split: np.ndarray,
        features_whole: np.ndarray,
        ratio: int = 1000,
        max_pairs: int = 4_000_000,
        seed: int = 0,
    ):
        if features_split.shape[0] % 2 != 0:
            raise ValueError(
                f"features_split rows must be even; got {features_split.shape[0]}"
            )
        n = features_split.shape[0] // 2
        if n < 1:
            raise ValueError("need >=1 split contig to form a must-link pair; got 0")

        dim = features_split.shape[1]
        if features_whole.shape[1] < dim:
            raise ValueError(
                f"features_whole has {features_whole.shape[1]} columns, fewer than "
                f"features_split ({dim})"
            )
        # SemiBin feeds the network the same columns for both kinds of pair.
        # With fewer than five samples data.csv carries extra depth columns
        # that data_split.csv lacks; SemiBin drops them (train_data[:, :136]).
        whole = np.ascontiguousarray(features_whole[:, :dim])
        if whole.shape[0] < 2:
            raise ValueError(
                "need >=2 whole contigs to draw distinct cannot-link pairs; "
                f"got {whole.shape[0]}"
            )

        self._x = torch.from_numpy(features_split).float()
        self._whole = torch.from_numpy(whole).float()
        self._rows = features_split.shape[0]  # 2N split rows
        self._n_whole = whole.shape[0]  # whole contigs: the cannot-link pool
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
        rows = self._n_whole
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
        return self._whole[i], self._whole[j], torch.tensor(0.0)
