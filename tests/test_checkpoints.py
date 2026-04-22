"""Checkpoint save/load tests.

Covers:
- Round-trip: save an encoder's weights, load them into a fresh encoder,
  and verify produced embeddings match.
- load_checkpoint on a missing path raises FileNotFoundError.
- SinglePhaseTrainer writes a final checkpoint when `checkpoint_path` is set.
- SinglePhaseTrainer writes interim checkpoints every N epochs.
- TwoPhaseTrainer writes a final checkpoint + per-phase snapshots when
  `checkpoint_per_phase=True`.
"""

from functools import partial

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

from src.losses.bce_contrastive import BCEContrastiveLoss
from src.losses.mahalanobis_bce import MahalanobisBCELoss
from src.representations.contrastive_mlp import ContrastiveMLP
from src.representations.uncertain_gen import UncertainGenRepresentation
from src.trainers.single_phase import SinglePhaseTrainer
from src.trainers.two_phase import TwoPhaseTrainer
from src.utils.checkpoints import load_checkpoint, save_checkpoint


class _ToyPairs(Dataset):
    def __init__(self, n=32, d=16):
        rng = np.random.default_rng(0)
        self.x_i = torch.from_numpy(rng.standard_normal((n, d)).astype("float32"))
        self.x_j = torch.from_numpy(rng.standard_normal((n, d)).astype("float32"))
        self.y = torch.from_numpy((rng.random(n) > 0.5).astype("float32"))

    def __len__(self):
        return len(self.x_i)

    def __getitem__(self, i):
        return self.x_i[i], self.x_j[i], self.y[i]


class TestRoundTrip:
    def test_save_load_preserves_embeddings(self, tmp_path):
        torch.manual_seed(0)
        encoder_a = ContrastiveMLP(input_dim=16, hidden_dim=8, embedding_dim=4)
        path = save_checkpoint(encoder_a, tmp_path / "nested" / "enc.pt")
        assert path.exists()

        encoder_b = ContrastiveMLP(input_dim=16, hidden_dim=8, embedding_dim=4)
        rng = np.random.default_rng(1)
        features = rng.standard_normal((20, 16)).astype("float32")
        z_b_before = encoder_b.encode(features)

        load_checkpoint(encoder_b, path)
        z_a = encoder_a.encode(features)
        z_b_after = encoder_b.encode(features)

        assert not np.allclose(z_b_before, z_a), "fresh encoder should differ from trained"
        assert np.allclose(z_a, z_b_after), "reloaded encoder must match source"

    def test_load_missing_raises(self, tmp_path):
        encoder = ContrastiveMLP(input_dim=4, hidden_dim=4, embedding_dim=2)
        with pytest.raises(FileNotFoundError):
            load_checkpoint(encoder, tmp_path / "does_not_exist.pt")


class TestSinglePhaseCheckpointing:
    def test_writes_final_checkpoint(self, tmp_path):
        encoder = ContrastiveMLP(input_dim=16, hidden_dim=8, embedding_dim=4)
        trainer = SinglePhaseTrainer(
            optimizer=partial(torch.optim.Adam, lr=1e-2),
            epochs=2,
            batch_size=8,
            device="cpu",
            checkpoint_path=tmp_path / "enc.pt",
        )
        trainer.fit(encoder, _ToyPairs(n=32, d=16), BCEContrastiveLoss())

        assert (tmp_path / "enc.pt").exists()

    def test_writes_interim_checkpoints(self, tmp_path):
        encoder = ContrastiveMLP(input_dim=16, hidden_dim=8, embedding_dim=4)
        trainer = SinglePhaseTrainer(
            optimizer=partial(torch.optim.Adam, lr=1e-2),
            epochs=4,
            batch_size=8,
            device="cpu",
            checkpoint_path=tmp_path / "enc.pt",
            checkpoint_every=2,
        )
        trainer.fit(encoder, _ToyPairs(n=32, d=16), BCEContrastiveLoss())

        assert (tmp_path / "enc_epoch2.pt").exists()
        # epoch 4 is the final epoch → final checkpoint, not an interim one
        assert not (tmp_path / "enc_epoch4.pt").exists()
        assert (tmp_path / "enc.pt").exists()

    def test_no_checkpoint_when_path_null(self, tmp_path):
        encoder = ContrastiveMLP(input_dim=16, hidden_dim=8, embedding_dim=4)
        trainer = SinglePhaseTrainer(
            optimizer=partial(torch.optim.Adam, lr=1e-2),
            epochs=1,
            batch_size=8,
            device="cpu",
            checkpoint_path=None,
        )
        trainer.fit(encoder, _ToyPairs(n=16, d=16), BCEContrastiveLoss())
        assert list(tmp_path.iterdir()) == []


class TestTwoPhaseCheckpointing:
    def test_writes_final_and_per_phase(self, tmp_path):
        encoder = UncertainGenRepresentation(
            input_dim=16, hidden_dim=16, embedding_dim=8, include_std=False
        )
        phases = [
            {
                "params": "mean",
                "epochs": 1,
                "batch_size": 16,
                "optimizer": partial(torch.optim.Adam, lr=1e-2),
                "encoder_attrs": {"include_std": False},
                "loss_attrs": {"include_std": False},
            },
            {
                "params": "cov",
                "epochs": 1,
                "batch_size": 16,
                "optimizer": partial(torch.optim.Adam, lr=1e-2),
                "encoder_attrs": {"include_std": True},
                "loss_attrs": {"include_std": True},
            },
        ]
        trainer = TwoPhaseTrainer(
            phases=phases,
            device="cpu",
            checkpoint_path=tmp_path / "enc.pt",
            checkpoint_per_phase=True,
        )
        trainer.fit(encoder, _ToyPairs(n=32, d=16), MahalanobisBCELoss(include_std=False))

        assert (tmp_path / "enc_phase1.pt").exists()
        # phase 2 is final → saved as the main final path, not enc_phase2
        assert not (tmp_path / "enc_phase2.pt").exists()
        assert (tmp_path / "enc.pt").exists()
