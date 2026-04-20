import numpy as np
import torch
from torch.utils.data import Dataset


class HybridPairSampler(Dataset):
    """Hybrid SemiBin + UncertainGen contrastive pairs.

    Positive pairs: contig halves (must-link), from ``features_split``.
    Negative pairs: a mixture controlled by ``taxonomy_fraction`` ∈ [0, 1]
      - fraction α drawn from taxonomy cannot-link pairs (indices into
        ``features_whole``),
      - fraction (1 - α) drawn by uniform multinomial over half-contigs
        (indices into ``features_split``).

    Index spaces are deliberately mixed: taxonomy operates at
    whole-contig granularity, random negatives benefit from the larger
    half-contig pool. ``__getitem__`` routes each pair to the correct
    feature array based on its bucket.
    """

    def __init__(
        self,
        features_split: np.ndarray,
        features_whole: np.ndarray,
        cannot_link_pairs: np.ndarray,
        neg_per_pos: int = 1000,
        taxonomy_fraction: float = 0.5,
        seed: int = 0,
    ):
        if not 0.0 <= taxonomy_fraction <= 1.0:
            raise ValueError(
                f"taxonomy_fraction must be in [0, 1]; got {taxonomy_fraction}"
            )
        if features_split.shape[0] % 2 != 0:
            raise ValueError(
                f"features_split rows must be even; got {features_split.shape[0]}"
            )
        if features_split.shape[1] != features_whole.shape[1]:
            raise ValueError(
                f"feature dims must match: split={features_split.shape[1]} "
                f"vs whole={features_whole.shape[1]}"
            )

        rng = np.random.default_rng(seed)
        n = features_split.shape[0] // 2
        total = features_split.shape[0]

        total_neg = neg_per_pos * n
        n_tax = min(int(taxonomy_fraction * total_neg), cannot_link_pairs.shape[0])
        n_rand = total_neg - n_tax

        pos = np.stack([np.arange(n), np.arange(n) + n], axis=1)

        tax_idx = rng.choice(cannot_link_pairs.shape[0], size=n_tax, replace=False)
        tax_neg = cannot_link_pairs[tax_idx]

        rand_neg = np.stack(
            [
                rng.integers(0, total, size=n_rand),
                rng.integers(0, total, size=n_rand),
            ],
            axis=1,
        )

        self._x_split = torch.from_numpy(features_split).float()
        self._x_whole = torch.from_numpy(features_whole).float()
        self._pos = pos
        self._tax_neg = tax_neg
        self._rand_neg = rand_neg
        self._n_pos = len(pos)
        self._n_tax = len(tax_neg)

    def __len__(self) -> int:
        return self._n_pos + self._n_tax + len(self._rand_neg)

    def __getitem__(self, idx):
        if idx < self._n_pos:
            i, j = self._pos[idx]
            return self._x_split[i], self._x_split[j], torch.tensor(1.0)
        idx -= self._n_pos
        if idx < self._n_tax:
            i, j = self._tax_neg[idx]
            return self._x_whole[i], self._x_whole[j], torch.tensor(0.0)
        i, j = self._rand_neg[idx - self._n_tax]
        return self._x_split[i], self._x_split[j], torch.tensor(0.0)
