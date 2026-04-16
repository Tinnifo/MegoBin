import random

import numpy as np
import torch
import torch.nn as nn

from src.features.kmer_profiles import generate_kmers


class PoissonRepresentation(nn.Module):
    """Poisson co-occurrence k-mer embedding (RevisitingKmers).

    Learnable embedding table E of shape (num_kmers, embedding_dim).
    Inference: z = f^T @ E  (weighted average by k-mer frequency).
    """

    def __init__(self, num_kmers: int = 256, embedding_dim: int = 256):
        super().__init__()
        self._embedding_dim = embedding_dim
        self.embeddings = nn.Embedding(num_kmers, embedding_dim)
        nn.init.uniform_(self.embeddings.weight, -1.0, 1.0)

    def encode(self, kmer_profiles: np.ndarray) -> np.ndarray:
        """(N, num_kmers) L1-normalized profiles → (N, embedding_dim)."""
        E = self.embeddings.weight.detach().cpu().numpy()
        return kmer_profiles @ E

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim


# ---------------------------------------------------------------------------
# Training data: co-occurrence matrix from reads
# ---------------------------------------------------------------------------


def compute_cooccurrence(
    reads: list[str],
    k: int = 4,
    window: int = 4,
    max_reads: int = 10_000,
    alphabet: str = "ACGT",
) -> np.ndarray:
    """Compute k-mer pair co-occurrence counts from reads.

    For each read, extract valid k-mers in order and count pairs that fall
    within a context *window* (in k-mer positions, like word2vec).

    Args:
        reads: Raw read sequences.
        k: K-mer length.
        window: Context window size (number of neighbouring k-mers).
        max_reads: Subsample to this many reads.
        alphabet: Valid bases.

    Returns:
        Upper-triangle co-occurrence matrix of shape (num_kmers, num_kmers).
    """
    all_kmers = generate_kmers(k, alphabet)
    kmer_to_idx = {kmer: i for i, kmer in enumerate(all_kmers)}
    num_kmers = len(all_kmers)
    valid_bases = set(alphabet)

    cooc = np.zeros((num_kmers, num_kmers), dtype=np.float64)

    if len(reads) > max_reads:
        reads = random.sample(reads, max_reads)

    for read in reads:
        read = read.upper()

        # Collect indices of valid k-mers along the read
        indices: list[int] = []
        for pos in range(len(read) - k + 1):
            kmer = read[pos : pos + k]
            if all(b in valid_bases for b in kmer):
                idx = kmer_to_idx.get(kmer)
                if idx is not None:
                    indices.append(idx)

        # Count co-occurrences within context window (upper triangle)
        for a in range(len(indices)):
            for b in range(a + 1, min(a + window + 1, len(indices))):
                i, j = min(indices[a], indices[b]), max(indices[a], indices[b])
                cooc[i, j] += 1

    return cooc
