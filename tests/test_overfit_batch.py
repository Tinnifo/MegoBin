"""Smoke test: each encoder must overfit a tiny batch.

If a model can't memorise ~100 contigs / pairs, something is fundamentally
wrong. These tests go through the full (encoder, trainer, sampler, loss)
stack so a regression anywhere in the pipeline shows up here.
"""

from functools import partial

import numpy as np
import torch
from torch.utils.data import Dataset

from megobin.losses.hinge_contrastive import HingeContrastiveLoss
from megobin.losses.mahalanobis_bce import MahalanobisBCELoss
from megobin.representations.semibin_encoder import SemiBinEncoder
from megobin.representations.uncertain_gen import UncertainGenRepresentation
from megobin.trainers.single_phase import SinglePhaseTrainer
from megobin.trainers.two_phase import TwoPhaseTrainer


class _SeparablePairs(Dataset):
    """Positive pairs close in feature space, negatives far apart."""

    def __init__(self, n: int = 100, d: int = 16, seed: int = 0):
        rng = np.random.default_rng(seed)
        base = rng.standard_normal((n, d)).astype("float32")
        self.x_i = np.concatenate([base, base], axis=0)
        self.x_j = np.concatenate(
            [
                base + 0.01 * rng.standard_normal((n, d)).astype("float32"),
                rng.standard_normal((n, d)).astype("float32") * 5.0,
            ],
            axis=0,
        )
        self.y = np.concatenate([np.ones(n), np.zeros(n)]).astype("float32")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return (
            torch.from_numpy(self.x_i[i]),
            torch.from_numpy(self.x_j[i]),
            torch.tensor(self.y[i]),
        )


class TestSemiBinOverfit:
    def test_loss_decreases(self):
        torch.manual_seed(0)
        d = 16
        encoder = SemiBinEncoder(input_dim=d, embedding_dim=8, dropout=0.0)
        loss_fn = HingeContrastiveLoss()
        sampler = _SeparablePairs(n=100, d=d)

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


class TestUncertainGenOverfit:
    def test_phase1_loss_decreases(self):
        torch.manual_seed(0)
        d = 16
        encoder = UncertainGenRepresentation(
            input_dim=d, hidden_dim=16, embedding_dim=8, include_std=False
        )
        loss_fn = MahalanobisBCELoss(include_std=False)
        sampler = _SeparablePairs(n=100, d=d)

        initial = _mean_loss_on_sampler(encoder, loss_fn, sampler)

        phases = [
            {
                "params": "mean",
                "epochs": 5,
                "batch_size": 32,
                "optimizer": partial(torch.optim.Adam, lr=1e-2),
                "encoder_attrs": {"include_std": False},
                "loss_attrs": {"include_std": False},
            },
        ]
        trainer = TwoPhaseTrainer(phases=phases, device="cpu")
        trainer.fit(encoder, sampler, loss_fn)

        final = _mean_loss_on_sampler(encoder, loss_fn, sampler)
        assert final < initial, f"loss did not decrease: {initial:.4f} → {final:.4f}"


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
