"""End-to-end integration test (checkpoint resume).

Train → save → load → verify the loaded encoder produces identical
embeddings. Protocol contracts live in ``tests/test_interfaces.py``.
"""

from functools import partial

import numpy as np
import torch

from megobin.data.uncertain_gen_sampler import UncertainGenPairSampler
from megobin.losses.mahalanobis_bce import MahalanobisBCELoss
from megobin.encoders.uncertaingen import UncertainGenEncoder
from megobin.trainers.single_phase import SinglePhaseTrainer


def _make_genome_cluster_features(
    n_genomes: int = 3,
    contigs_per_genome: int = 30,
    dim: int = 32,
    seed: int = 0,
    noise: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build synthetic features with known genome labels."""
    rng = np.random.default_rng(seed)
    genome_profiles = rng.dirichlet(np.ones(dim) * 0.5, size=n_genomes)

    labels = np.repeat(np.arange(n_genomes), contigs_per_genome)
    whole = np.stack(
        [genome_profiles[g] + rng.normal(0, noise, size=dim) for g in labels]
    ).astype("float32")

    first = np.stack(
        [genome_profiles[g] + rng.normal(0, noise, size=dim) for g in labels]
    ).astype("float32")
    second = np.stack(
        [genome_profiles[g] + rng.normal(0, noise, size=dim) for g in labels]
    ).astype("float32")
    split = np.concatenate([first, second], axis=0)

    return whole, split, labels


class TestCheckpointResume:
    """Train → save → load → verify the loaded encoder produces identical
    embeddings."""

    def test_resume_produces_identical_embeddings(self, tmp_path):
        from megobin.utils.checkpoints import load_checkpoint

        torch.manual_seed(0)
        whole, split, _ = _make_genome_cluster_features(
            n_genomes=3, contigs_per_genome=20, dim=32, seed=0
        )
        sampler = UncertainGenPairSampler(features_split=split, neg_per_pos=5, seed=0)

        encoder_a = UncertainGenEncoder(
            input_dim=32, hidden_dim=32, embedding_dim=16, include_std=False
        )
        trainer = SinglePhaseTrainer(
            optimizer=partial(torch.optim.Adam, lr=5e-3),
            epochs=2,
            batch_size=64,
            device="cpu",
            params="mean",
            checkpoint_path=tmp_path / "enc.pt",
        )
        trainer.fit(encoder_a, sampler, MahalanobisBCELoss(include_std=False))

        z_a = encoder_a.encode(whole)

        encoder_b = UncertainGenEncoder(
            input_dim=32, hidden_dim=32, embedding_dim=16, include_std=False
        )
        load_checkpoint(encoder_b, tmp_path / "enc.pt")
        z_b = encoder_b.encode(whole)

        assert np.allclose(z_a, z_b, atol=1e-6)
