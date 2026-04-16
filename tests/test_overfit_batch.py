"""Smoke test: each encoder must overfit a tiny batch.

If a model can't memorise 100 contigs, something is fundamentally wrong.
For the Poisson encoder this means the Poisson NLL on a small co-occurrence
matrix decreases steadily over a handful of optimisation steps.
"""

import numpy as np
import torch

from src.features.kmer_profiles import compute_kmer_profiles
from src.losses.poisson_nll import PoissonNLLLoss
from src.representations.poisson import PoissonRepresentation, compute_cooccurrence


def _random_dna(length: int, rng: np.random.Generator) -> str:
    return "".join(rng.choice(list("ACGT"), size=length))


class TestPoissonOverfit:
    """Train the Poisson embedding on a tiny co-occurrence matrix and verify
    that the loss decreases."""

    def test_loss_decreases(self):
        rng = np.random.default_rng(42)

        # Generate 100 short synthetic reads
        reads = [_random_dna(100, rng) for _ in range(100)]

        # Build co-occurrence matrix
        cooc = compute_cooccurrence(reads, k=4, window=4, max_reads=100)

        # Extract non-zero upper-triangle pairs for training
        rows, cols = np.triu_indices(256, k=1)
        mask = cooc[rows, cols] > 0
        pair_i = rows[mask]
        pair_j = cols[mask]
        counts = cooc[pair_i, pair_j]

        assert len(counts) > 0, "No co-occurrence pairs found"

        pair_i_t = torch.from_numpy(pair_i).long()
        pair_j_t = torch.from_numpy(pair_j).long()
        counts_t = torch.from_numpy(counts).float()

        # Model + loss
        model = PoissonRepresentation(num_kmers=256, embedding_dim=32)
        loss_fn = PoissonNLLLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        losses = []
        for _ in range(50):
            z_i = model.embeddings(pair_i_t)
            z_j = model.embeddings(pair_j_t)
            loss = loss_fn(z_i, z_j, counts_t)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        # Loss must decrease: final < initial
        assert losses[-1] < losses[0], (
            f"Loss did not decrease: {losses[0]:.4f} → {losses[-1]:.4f}"
        )

        # Sanity: loss should drop by at least 10%
        drop = (losses[0] - losses[-1]) / abs(losses[0])
        assert drop > 0.10, f"Loss only dropped {drop:.1%}"

    def test_encode_after_training(self):
        """After training, encode() should still produce valid output."""
        rng = np.random.default_rng(99)
        reads = [_random_dna(80, rng) for _ in range(100)]

        profiles = compute_kmer_profiles(reads, k=4, canonical=False)
        model = PoissonRepresentation(num_kmers=256, embedding_dim=32)

        z = model.encode(profiles)
        assert z.shape == (100, 32)
        assert np.isfinite(z).all()
