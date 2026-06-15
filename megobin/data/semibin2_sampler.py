import numpy as np
import torch
from torch.utils.data import Dataset


class SemiBin2PairSampler(Dataset):
    """SemiBin2 self-supervised pair sampler with per-epoch resampling.

    Reproduces SemiBin's ``train_self`` pair construction:

    - **Must-link (positive)** pairs: the two halves of each contig — fixed
      across epochs. With ``features_split`` laid out as
      ``[left_halves; right_halves]``, row ``i`` and row ``i + N`` are the
      two halves of contig ``i``.
    - **Cannot-link (negative)** pairs: uniformly random pairs of distinct
      *half-contig rows*, **redrawn every epoch** via :meth:`set_epoch`
      (called by the trainer at the start of each epoch). The count is
      ``min(rows * ratio // 2, max_pairs)`` where ``rows == 2N`` — i.e.
      SemiBin's ``min(n_must_link * 1000 // 2, 4_000_000)``. Distinct rows
      are guaranteed by SemiBin's offset trick
      ``idx2 = (idx1 + 1 + rand(rows - 1)) % rows``.

    This is the only piece that differs from :class:`UncertainGenPairSampler`,
    which draws one fixed negative set at construction. Drop-in compatible:
    both consume ``features_split`` and emit ``(x_i, x_j, label)``, so either
    sampler can be paired with either encoder / preprocessing via the
    ``pair_sampler`` config slot.

    Note: per-epoch resampling mutates this in-process Dataset, so it only
    reaches the DataLoader when ``num_workers == 0`` (the SinglePhaseTrainer
    warns otherwise).
    """

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
