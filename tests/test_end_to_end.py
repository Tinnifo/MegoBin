"""End-to-end integration test.

Protects against regression across the whole pipeline. Uses synthetic
features built to separate into known clusters so we can assert on ARI
without needing real assemblies / BAM files / CheckM2.

Two passes:
- UncertainGen + TwoPhaseTrainer (primary)
- ContrastiveMLP + SinglePhaseTrainer (simpler regression guard)

Both must run in under a minute each on CPU so they stay in the
default `pytest` suite.
"""

import time
from functools import partial

import numpy as np
import pytest
import torch
from sklearn.metrics import adjusted_rand_score

from src.binners.kmedoids import KMedoidsBinner
from src.data.semibin_sampler import SemiBinPairSampler
from src.data.uncertain_gen_sampler import UncertainGenPairSampler
from src.losses.bce_contrastive import BCEContrastiveLoss
from src.losses.mahalanobis_bce import MahalanobisBCELoss
from src.representations.contrastive_mlp import ContrastiveMLP
from src.representations.uncertain_gen import UncertainGenRepresentation
from src.trainers.single_phase import SinglePhaseTrainer
from src.trainers.two_phase import TwoPhaseTrainer


# ---------------------------------------------------------------------------
# Synthetic data fixture: N contigs from G genomes with distinct profiles
# ---------------------------------------------------------------------------


def _make_genome_cluster_features(
    n_genomes: int = 3,
    contigs_per_genome: int = 30,
    dim: int = 32,
    seed: int = 0,
    noise: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build synthetic features with known genome labels.

    Returns
    -------
    features_whole : (N, dim) — one row per contig
    features_split : (2N, dim) — contig halves (first half of row i
        paired with row i+N for each contig, UncertainGen-style)
    labels : (N,) — genome id for each contig
    """
    rng = np.random.default_rng(seed)
    genome_profiles = rng.dirichlet(np.ones(dim) * 0.5, size=n_genomes)

    n = n_genomes * contigs_per_genome
    labels = np.repeat(np.arange(n_genomes), contigs_per_genome)
    whole = np.stack(
        [
            genome_profiles[g] + rng.normal(0, noise, size=dim)
            for g in labels
        ]
    ).astype("float32")

    # split halves: simulate first+second half of each contig by
    # sampling two slightly different noised profiles per contig
    first = np.stack(
        [genome_profiles[g] + rng.normal(0, noise, size=dim) for g in labels]
    ).astype("float32")
    second = np.stack(
        [genome_profiles[g] + rng.normal(0, noise, size=dim) for g in labels]
    ).astype("float32")
    split = np.concatenate([first, second], axis=0)

    return whole, split, labels


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestUncertainGenEndToEnd:
    def test_pipeline_recovers_cluster_structure(self):
        torch.manual_seed(0)
        start = time.perf_counter()

        whole, split, labels = _make_genome_cluster_features(
            n_genomes=3, contigs_per_genome=30, dim=32, seed=0, noise=0.03
        )

        sampler = UncertainGenPairSampler(
            features_split=split, neg_per_pos=5, seed=0
        )

        encoder = UncertainGenRepresentation(
            input_dim=32, hidden_dim=32, embedding_dim=16, include_std=False
        )
        loss_fn = MahalanobisBCELoss(include_std=False)

        phases = [
            {
                "params": "mean",
                "epochs": 3,
                "batch_size": 64,
                "optimizer": partial(torch.optim.Adam, lr=5e-3),
                "encoder_attrs": {"include_std": False},
                "loss_attrs": {"include_std": False},
            },
            {
                "params": "cov",
                "epochs": 2,
                "batch_size": 64,
                "optimizer": partial(torch.optim.Adam, lr=5e-3),
                "encoder_attrs": {"include_std": True},
                "loss_attrs": {"include_std": True},
            },
        ]
        trainer = TwoPhaseTrainer(phases=phases, device="cpu")
        trainer.fit(encoder, sampler, loss_fn)

        embeddings = encoder.encode(whole)
        assert embeddings.shape == (whole.shape[0], 16)
        assert np.isfinite(embeddings).all()

        binner = KMedoidsBinner(min_bin_size=5)
        predicted = binner.cluster(embeddings)
        assert len(predicted) == len(labels)

        ari = adjusted_rand_score(labels, predicted)
        assert ari > 0.3, f"ARI {ari:.3f} is not above the random baseline"

        elapsed = time.perf_counter() - start
        assert elapsed < 60, f"took {elapsed:.1f}s, needs to stay under 60s"


class TestContrastiveMLPEndToEnd:
    def test_pipeline_recovers_cluster_structure(self):
        torch.manual_seed(1)
        start = time.perf_counter()

        whole, split, labels = _make_genome_cluster_features(
            n_genomes=3, contigs_per_genome=30, dim=32, seed=1, noise=0.03
        )
        n = whole.shape[0]

        cannot_link = np.array(
            [[i, j] for i in range(n) for j in range(n) if labels[i] != labels[j]]
        )
        sampler = SemiBinPairSampler(
            features_split=split,
            features_whole=whole,
            cannot_link_pairs=cannot_link,
            neg_per_pos=5,
            max_pairs=10_000,
            seed=1,
        )

        encoder = ContrastiveMLP(input_dim=32, hidden_dim=32, embedding_dim=16)
        loss_fn = BCEContrastiveLoss()

        trainer = SinglePhaseTrainer(
            optimizer=partial(torch.optim.Adam, lr=5e-3),
            epochs=5,
            batch_size=64,
            device="cpu",
        )
        trainer.fit(encoder, sampler, loss_fn)

        embeddings = encoder.encode(whole)
        predicted = KMedoidsBinner(min_bin_size=5).cluster(embeddings)

        ari = adjusted_rand_score(labels, predicted)
        assert ari > 0.3, f"ARI {ari:.3f} is not above the random baseline"

        elapsed = time.perf_counter() - start
        assert elapsed < 60, f"took {elapsed:.1f}s, needs to stay under 60s"


class TestCheckpointResume:
    """Train → save → load → verify the loaded encoder produces identical
    embeddings. Exercises the OPE-458 save/load path against the trainer."""

    def test_resume_produces_identical_embeddings(self, tmp_path):
        from src.utils.checkpoints import load_checkpoint

        torch.manual_seed(0)
        whole, split, _ = _make_genome_cluster_features(
            n_genomes=3, contigs_per_genome=20, dim=32, seed=0
        )
        sampler = UncertainGenPairSampler(features_split=split, neg_per_pos=5, seed=0)

        encoder_a = UncertainGenRepresentation(
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

        encoder_b = UncertainGenRepresentation(
            input_dim=32, hidden_dim=32, embedding_dim=16, include_std=False
        )
        load_checkpoint(encoder_b, tmp_path / "enc.pt")
        z_b = encoder_b.encode(whole)

        assert np.allclose(z_a, z_b, atol=1e-6)
