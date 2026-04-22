import numpy as np
import torch
from torch.utils.data import Dataset


class CooccurrencePairSampler(Dataset):
    """K-mer co-occurrence pair sampler for the Poisson encoder.

    Yields ``(kmer_idx_i, kmer_idx_j, count)`` triples drawn from the
    upper triangle (including diagonal) of a co-occurrence matrix. All
    pairs are included — not just observed ones — so the Poisson NLL
    can both drive observed λ toward its count and push unobserved λ
    toward zero.
    """

    def __init__(self, cooccurrence: np.ndarray):
        if cooccurrence.ndim != 2 or cooccurrence.shape[0] != cooccurrence.shape[1]:
            raise ValueError(
                f"cooccurrence must be square; got {cooccurrence.shape}"
            )
        i, j = np.triu_indices_from(cooccurrence, k=0)
        self._i = torch.from_numpy(i).long()
        self._j = torch.from_numpy(j).long()
        self._c = torch.from_numpy(cooccurrence[i, j]).float()

    def __len__(self) -> int:
        return len(self._c)

    def __getitem__(self, idx):
        return self._i[idx], self._j[idx], self._c[idx]
