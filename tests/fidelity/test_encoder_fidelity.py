"""Encoder-slot fidelity (CPU, tiny net, seconds).

Variance comes from the training seed. A faithful re-implementation (a fresh
seeded training of the same encoder) lands inside the band and recovers the
planted genomes; a perturbed "encoder" that loses the genome signal does not.
The encoder is routed through a fixed downstream binner so the comparison is on
the partition it induces (embeddings are only defined up to an isometry).
"""

from functools import partial

import numpy as np
import torch
from sklearn.cluster import KMeans

from megobin.data.uncertain_gen_sampler import UncertainGenPairSampler
from megobin.encoders.uncertaingen import UncertainGenEncoder
from megobin.fidelity import (
    EncoderTrainingOracle,
    FidelityHarness,
    binner_ari,
)
from megobin.losses.mahalanobis_bce import MahalanobisBCELoss
from megobin.trainers.single_phase import SinglePhaseTrainer

_SEEDS = range(8)
_EPOCHS = 5


def _reference_embed(fx, seed: int) -> np.ndarray:
    """Train the reference encoder with ``seed`` and encode the features."""
    torch.manual_seed(seed)
    enc = UncertainGenEncoder(
        input_dim=fx.input_dim, hidden_dim=32, embedding_dim=16, include_std=False
    )
    sampler = UncertainGenPairSampler(
        features_split=fx.features_split, neg_per_pos=5, seed=seed
    )
    trainer = SinglePhaseTrainer(
        optimizer=partial(torch.optim.Adam, lr=5e-3),
        epochs=_EPOCHS,
        batch_size=64,
        device="cpu",
        params="mean",
        checkpoint_path=None,
    )
    trainer.fit(enc, sampler, MahalanobisBCELoss(include_std=False))
    return enc.encode(fx.features)


def _perturbed_embed(fx, seed: int) -> np.ndarray:
    """A broken encoder: embeddings uncorrelated with the genome signal."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((fx.features.shape[0], 16)).astype("float32")


def _cluster(embedding: np.ndarray) -> np.ndarray:
    # fixed downstream binner: KMeans at the known genome count
    return KMeans(n_clusters=3, n_init=10, random_state=0).fit_predict(embedding)


def _oracle() -> EncoderTrainingOracle:
    return EncoderTrainingOracle(_reference_embed, _cluster, seeds=_SEEDS)


def test_faithful_encoder_passes(encoder_set):
    report = FidelityHarness(binner_ari, z=2.0).run(
        candidate=_reference_embed, oracle=_oracle(), fixture=encoder_set
    )
    assert report.passed, report.to_markdown()


def test_perturbed_encoder_fails(encoder_set):
    report = FidelityHarness(binner_ari, z=2.0).run(
        candidate=_perturbed_embed, oracle=_oracle(), fixture=encoder_set
    )
    assert not report.passed, report.to_markdown()
    assert report.vs_reference < report.reference_self.floor


def test_distance_agreement_smoke(encoder_set):
    # The cheap, binner-free geometry comparator agrees across training seeds.
    from megobin.fidelity import encoder_distance_agreement

    a = _reference_embed(encoder_set, 0)
    b = _reference_embed(encoder_set, 1)
    assert encoder_distance_agreement(a, b, sample=512) > 0.5
