"""Smoke test: each encoder must overfit a tiny batch.

If a model can't memorise ~100 contigs / pairs, something is fundamentally
wrong. These tests go through the full (encoder, trainer, sampler, loss)
stack so a regression anywhere in the pipeline shows up here.
"""

from functools import partial

import numpy as np
import torch

from src.data.cooccurrence_sampler import CooccurrencePairSampler
from src.features.kmer_profiles import compute_kmer_profiles
from src.losses.bce_contrastive import BCEContrastiveLoss
from src.losses.poisson_nll import PoissonNLLLoss
from src.representations.contrastive_mlp import ContrastiveMLP
from src.representations.poisson import PoissonRepresentation, compute_cooccurrence
from src.trainers.single_phase import SinglePhaseTrainer


def _random_dna(length: int, rng: np.random.Generator) -> str:
    return "".join(rng.choice(list("ACGT"), size=length))


class TestPoissonOverfit:
    """Drive the Poisson encoder through SinglePhaseTrainer on a tiny
    co-occurrence matrix and verify the loss decreases."""

    def test_loss_decreases(self):
        rng = np.random.default_rng(42)
        reads = [_random_dna(100, rng) for _ in range(100)]
        cooc = compute_cooccurrence(reads, k=4, window=4, max_reads=100)
        assert cooc.sum() > 0, "degenerate reads produced zero co-occurrence"

        encoder = PoissonRepresentation(num_kmers=256, embedding_dim=32)
        loss_fn = PoissonNLLLoss()
        sampler = CooccurrencePairSampler(cooc)

        initial = _mean_loss_on_sampler(encoder, loss_fn, sampler)

        trainer = SinglePhaseTrainer(
            optimizer=partial(torch.optim.Adam, lr=1e-2),
            epochs=10,
            batch_size=1024,
            device="cpu",
        )
        trainer.fit(encoder, sampler, loss_fn)

        final = _mean_loss_on_sampler(encoder, loss_fn, sampler)

        assert final < initial, f"loss did not decrease: {initial:.4f} → {final:.4f}"
        drop = (initial - final) / abs(initial)
        assert drop > 0.10, f"loss only dropped {drop:.1%}"

    def test_encode_after_training(self):
        rng = np.random.default_rng(99)
        reads = [_random_dna(80, rng) for _ in range(100)]
        profiles = compute_kmer_profiles(reads, k=4, canonical=False)

        encoder = PoissonRepresentation(num_kmers=256, embedding_dim=32)
        z = encoder.encode(profiles)
        assert z.shape == (100, 32)
        assert np.isfinite(z).all()


class TestContrastiveMLPOverfit:
    """ContrastiveMLP should drive BCE down on a small set of pairs."""

    def test_loss_decreases(self):
        torch.manual_seed(0)
        rng = np.random.default_rng(0)

        # Separable pairs: positives close, negatives far.
        n = 100
        d = 16
        base = rng.standard_normal((n, d)).astype("float32")
        x_i = np.concatenate([base, base], axis=0)
        x_j = np.concatenate(
            [base + 0.01 * rng.standard_normal((n, d)).astype("float32"),
             rng.standard_normal((n, d)).astype("float32") * 5.0],
            axis=0,
        )
        y = np.concatenate([np.ones(n), np.zeros(n)]).astype("float32")

        class _Pairs(torch.utils.data.Dataset):
            def __len__(self): return 2 * n
            def __getitem__(self, i):
                return (
                    torch.from_numpy(x_i[i]),
                    torch.from_numpy(x_j[i]),
                    torch.tensor(y[i]),
                )

        encoder = ContrastiveMLP(input_dim=d, hidden_dim=16, embedding_dim=8)
        loss_fn = BCEContrastiveLoss()
        sampler = _Pairs()

        initial = _mean_loss_on_sampler(encoder, loss_fn, sampler)

        trainer = SinglePhaseTrainer(
            optimizer=partial(torch.optim.Adam, lr=1e-2),
            epochs=10,
            batch_size=32,
            device="cpu",
        )
        trainer.fit(encoder, sampler, loss_fn)

        final = _mean_loss_on_sampler(encoder, loss_fn, sampler)
        assert final < initial, f"loss did not decrease: {initial:.4f} → {final:.4f}"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _mean_loss_on_sampler(encoder, loss_fn, sampler) -> float:
    encoder.eval()
    total = 0.0
    n = 0
    with torch.no_grad():
        for i in range(len(sampler)):
            batch = tuple(t.unsqueeze(0) for t in sampler[i])
            total += encoder.training_step(batch, loss_fn).item()
            n += 1
    return total / n
