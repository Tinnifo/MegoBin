"""Trainer smoke tests.

- SinglePhaseTrainer + SemiBinEncoder: loss must drop over a few epochs
- TwoPhaseTrainer + UncertainGen: mean trains in phase 1, cov in phase 2,
  and the frozen group is left untouched in each phase
- Invalid parameter group name raises
"""

from functools import partial

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

from megobin.losses.hinge_contrastive import HingeContrastiveLoss
from megobin.losses.mahalanobis_bce import MahalanobisBCELoss
from megobin.representations.semibin_encoder import SemiBinEncoder
from megobin.representations.uncertain_gen import UncertainGenRepresentation
from megobin.trainers.single_phase import SinglePhaseTrainer
from megobin.trainers.two_phase import TwoPhaseTrainer


class _ToyPairs(Dataset):
    """Synthetic (x_i, x_j, label) pairs for smoke testing."""

    def __init__(self, n: int = 256, d: int = 32, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.x_i = torch.from_numpy(rng.standard_normal((n, d)).astype("float32"))
        self.x_j = torch.from_numpy(rng.standard_normal((n, d)).astype("float32"))
        self.y = torch.from_numpy((rng.random(n) > 0.5).astype("float32"))

    def __len__(self):
        return len(self.x_i)

    def __getitem__(self, i):
        return self.x_i[i], self.x_j[i], self.y[i]


class _SeparablePairs(Dataset):
    """Positives are near-copies of each other; negatives are scaled apart."""

    def __init__(self, n: int = 100, d: int = 32, seed: int = 0):
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


class TestSinglePhaseTrainerSemiBin:
    def test_loss_decreases(self):
        torch.manual_seed(0)
        encoder = SemiBinEncoder(input_dim=16, embedding_dim=8, dropout=0.0)
        loss_fn = HingeContrastiveLoss()
        sampler = _SeparablePairs(n=100, d=16)

        initial = _eval_mean_loss(encoder, loss_fn, sampler)

        trainer = SinglePhaseTrainer(
            optimizer=partial(torch.optim.Adam, lr=1e-2),
            epochs=10,
            batch_size=32,
            device="cpu",
        )
        trainer.fit(encoder, sampler, loss_fn)

        final = _eval_mean_loss(encoder, loss_fn, sampler)
        assert final < initial, f"loss did not decrease: {initial:.4f} → {final:.4f}"

    def test_invalid_parameter_group_raises(self):
        encoder = SemiBinEncoder(input_dim=8, embedding_dim=4, dropout=0.0)
        sampler = _ToyPairs(n=8, d=8)
        trainer = SinglePhaseTrainer(
            optimizer=partial(torch.optim.Adam, lr=1e-3),
            epochs=1,
            batch_size=4,
            params="no_such_group",
            device="cpu",
        )
        with pytest.raises(ValueError, match="parameter group"):
            trainer.fit(encoder, sampler, HingeContrastiveLoss())


class TestTwoPhaseTrainerUncertainGen:
    def test_phase1_trains_mean_only_phase2_trains_cov_only(self):
        torch.manual_seed(0)
        encoder = UncertainGenRepresentation(
            input_dim=16, hidden_dim=16, embedding_dim=8, include_std=False
        )
        loss_fn = MahalanobisBCELoss(include_std=False)
        sampler = _ToyPairs(n=128, d=16)

        mean_before = _clone(encoder.mean_head)
        cov_before = _clone(encoder.cov_head)

        phases = [
            {
                "params": "mean",
                "epochs": 2,
                "batch_size": 16,
                "optimizer": partial(torch.optim.Adam, lr=1e-2),
                "encoder_attrs": {"include_std": False},
                "loss_attrs": {"include_std": False},
            },
            {
                "params": "cov",
                "epochs": 2,
                "batch_size": 16,
                "optimizer": partial(torch.optim.Adam, lr=1e-2),
                "encoder_attrs": {"include_std": True},
                "loss_attrs": {"include_std": True},
            },
        ]
        trainer = TwoPhaseTrainer(phases=phases, device="cpu")
        trainer.fit(encoder, sampler, loss_fn)

        mean_after = _clone(encoder.mean_head)
        cov_after = _clone(encoder.cov_head)

        assert _params_changed(mean_before, mean_after), "mean should have trained"
        assert _params_changed(cov_before, cov_after), "cov should have trained (phase 2)"

        assert encoder.include_std is True
        assert loss_fn.include_std is True

        assert all(p.requires_grad for p in encoder.parameters())

    def test_only_selected_group_updates_per_phase(self):
        torch.manual_seed(1)
        encoder = UncertainGenRepresentation(
            input_dim=16, hidden_dim=16, embedding_dim=8, include_std=False
        )
        loss_fn = MahalanobisBCELoss(include_std=False)
        sampler = _ToyPairs(n=64, d=16)

        cov_before = _clone(encoder.cov_head)

        phases = [
            {
                "params": "mean",
                "epochs": 2,
                "batch_size": 16,
                "optimizer": partial(torch.optim.Adam, lr=1e-2),
                "encoder_attrs": {"include_std": False},
                "loss_attrs": {"include_std": False},
            },
        ]
        trainer = TwoPhaseTrainer(phases=phases, device="cpu")
        trainer.fit(encoder, sampler, loss_fn)

        cov_after = _clone(encoder.cov_head)
        assert not _params_changed(cov_before, cov_after), "cov must be frozen in phase 1"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _eval_mean_loss(encoder, loss_fn, sampler) -> float:
    encoder.eval()
    total = 0.0
    n = 0
    with torch.no_grad():
        for i in range(len(sampler)):
            x_i, x_j, y = sampler[i]
            loss = encoder.training_step(
                (x_i.unsqueeze(0), x_j.unsqueeze(0), y.unsqueeze(0)), loss_fn
            )
            total += loss.item()
            n += 1
    return total / n


def _clone(module: torch.nn.Module) -> list[torch.Tensor]:
    return [p.detach().clone() for p in module.parameters()]


def _params_changed(before: list[torch.Tensor], after: list[torch.Tensor]) -> bool:
    return any(not torch.allclose(a, b) for a, b in zip(before, after))
