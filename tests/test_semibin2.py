"""Tests for the SemiBin2 long-read port: encoder, binner minfasta, feature gen."""

import numpy as np
import pytest
import torch

from megobin.binners.semibin_2 import DBSCANEnsembleBinner
from megobin.data.semibin2_sampler import SemiBin2PairSampler
from megobin.encoders.semibin_2 import SemiBin2Encoder
from megobin.features.feature_merge import generate_sequence_features_single
from megobin.losses.hinge_contrastive import HingeContrastiveLoss


# ---- Encoder ---------------------------------------------------------------


class TestSemiBin2Encoder:
    def test_single_sample_concats_log_mean_depth(self):
        enc = SemiBin2Encoder(
            input_dim=136, kmer_dim=136, output_dim=8, is_combined=False
        )
        n = 12
        kmer = np.random.rand(n, 136).astype("float32")
        # 2 BAMs, single-sample layout: [mean, var, mean, var]
        depth = np.abs(np.random.randn(n, 4)).astype("float32") + 0.1
        feats = np.concatenate([kmer, depth], axis=1)

        out = enc.encode(feats)
        # output = embedding (8) + per-BAM mean (even cols -> 2)
        assert out.shape == (n, 10)
        expected_log_depth = np.log(np.clip(depth[:, ::2], 1e-6, None))
        assert np.allclose(out[:, 8:], expected_log_depth, atol=1e-5)

    def test_single_sample_clips_zero_depth(self):
        enc = SemiBin2Encoder(
            input_dim=136, kmer_dim=136, output_dim=4, is_combined=False
        )
        kmer = np.random.rand(5, 136).astype("float32")
        depth = np.zeros((5, 2), dtype="float32")  # zero coverage
        out = enc.encode(np.concatenate([kmer, depth], axis=1))
        # log(clip(0, 1e-6)) = log(1e-6), finite (no -inf)
        assert np.isfinite(out).all()

    def test_combined_no_concat(self):
        enc = SemiBin2Encoder(
            input_dim=141, kmer_dim=136, output_dim=8, is_combined=True
        )
        feats = np.random.rand(12, 141).astype("float32")
        out = enc.encode(feats)
        assert out.shape == (12, 8)  # no depth concat in combined mode

    def test_training_step_backprops(self):
        enc = SemiBin2Encoder(input_dim=136, kmer_dim=136, output_dim=8)
        loss = enc.training_step(
            (torch.randn(8, 136), torch.randn(8, 136), torch.rand(8)),
            HingeContrastiveLoss(),
        )
        assert loss.shape == ()
        loss.backward()

    def test_no_encode_with_uncertainty(self):
        # The pipeline routes encoders exposing this into the covariance path.
        enc = SemiBin2Encoder()
        assert not hasattr(enc, "encode_with_uncertainty")
        assert enc.consumes_depth is True


# ---- Binner minfasta -------------------------------------------------------


def _binner(n, minfasta, lengths, c2m, n_markers):
    names = np.array([f"c{i}" for i in range(n)])
    return DBSCANEnsembleBinner(
        eps_values=[0.05, 0.1, 0.3],
        min_samples=3,
        minfasta=minfasta,
        contig_names=names,
        contig_lengths=lengths,
        contig_to_marker=c2m,
        n_total_markers=n_markers,
    )


def _one_cluster_plus_noise():
    """10 tight large contigs (one bin) + 6 scattered small contigs (noise)."""
    rng = np.random.default_rng(0)
    core = rng.normal(0, 0.01, size=(10, 6))
    noise = np.stack([np.full(6, 50.0 * (i + 1)) for i in range(6)]).astype(float)
    emb = np.concatenate([core, noise], axis=0)
    lengths = np.array([30000] * 10 + [5000] * 6)  # core 300k bp, noise 5k each
    # distinct markers on the core so it scores F1 = 1; noise unmarked
    c2m = {f"c{i}": [f"g{i}"] for i in range(10)}
    for i in range(10, 16):
        c2m[f"c{i}"] = []
    return emb, lengths, c2m


def test_minfasta_requires_contig_lengths():
    b = DBSCANEnsembleBinner(
        minfasta=200000,
        contig_names=np.array(["c0", "c1"]),
        contig_to_marker={"c0": ["g0"], "c1": ["g1"]},
        n_total_markers=2,
    )
    with pytest.raises(ValueError, match="contig_lengths"):
        b.cluster(np.random.randn(2, 4))


def test_minfasta_drops_small_bins_and_leftovers():
    emb, lengths, c2m = _one_cluster_plus_noise()
    binner = _binner(16, minfasta=200000, lengths=lengths, c2m=c2m, n_markers=10)
    labels = binner.cluster(emb)

    # Invariant: every surviving bin totals >= minfasta bp.
    for lab in np.unique(labels[labels >= 0]):
        bp = lengths[labels == lab].sum()
        assert bp >= 200000, f"bin {lab} has {bp} bp < minfasta"
    # The 5k-bp noise contigs cannot form a >=200kb bin -> dropped to -1.
    assert (labels[10:] == -1).all()
    # The 300kb core survives as a real bin.
    assert (labels[:10] >= 0).all()


def test_minfasta_zero_keeps_singletons():
    emb, lengths, c2m = _one_cluster_plus_noise()
    binner = _binner(16, minfasta=0, lengths=lengths, c2m=c2m, n_markers=10)
    labels = binner.cluster(emb)
    assert (labels >= 0).all()  # backward-compatible: no -1


# ---- Single-sample feature generation (kmer-only path, no BAM needed) ------


def test_generate_features_single_only_kmer(tmp_path):
    fasta = tmp_path / "contigs.fasta"
    rng = np.random.default_rng(1)
    with open(fasta, "w") as f:
        for i in range(4):
            seq = "".join(rng.choice(list("ACGT"), size=5000))
            f.write(f">contig_{i}\n{seq}\n")

    import logging

    generate_sequence_features_single(
        logging.getLogger("t"),
        str(fasta),
        bams=None,
        binned_length=1000,
        must_link_threshold=4000,
        num_process=1,
        output=str(tmp_path),
        abundances=None,
        only_kmer=True,
    )
    import pandas as pd

    data = pd.read_csv(tmp_path / "data.csv", index_col=0)
    assert data.shape == (4, 136)  # 4 contigs, 136 canonical 4-mers


# ---- SemiBin2 per-epoch resampling sampler ---------------------------------


def _split(n=20, dim=16):
    rng = np.random.default_rng(0)
    return rng.standard_normal((2 * n, dim)).astype("float32")


class TestSemiBin2PairSampler:
    def test_positives_are_contig_halves(self):
        s = SemiBin2PairSampler(_split(n=10), ratio=4, seed=0)
        _, _, label = s[0]
        assert float(label) == 1.0
        assert len(s) == 10 + min((2 * 10) * 4 // 2, 4_000_000)

    def test_negatives_distinct_rows_and_labeled_zero(self):
        s = SemiBin2PairSampler(_split(n=10), ratio=10, seed=1)
        assert (s._neg[:, 0] != s._neg[:, 1]).all()
        _, _, label = s[s._n_pos]
        assert float(label) == 0.0

    def test_resamples_across_epochs(self):
        s = SemiBin2PairSampler(_split(n=30), ratio=10, seed=0)
        s.set_epoch(0)
        neg0 = s._neg.copy()
        s.set_epoch(1)
        assert not np.array_equal(neg0, s._neg)

    def test_deterministic_given_seed_and_epoch(self):
        a = SemiBin2PairSampler(_split(n=30), ratio=10, seed=7)
        a.set_epoch(3)
        b = SemiBin2PairSampler(_split(n=30), ratio=10, seed=7)
        b.set_epoch(3)
        assert np.array_equal(a._neg, b._neg)

    def test_count_matches_semibin_formula(self):
        s = SemiBin2PairSampler(_split(n=50), ratio=1000, max_pairs=4_000_000)
        assert s._n_neg == min((2 * 50) * 1000 // 2, 4_000_000)

    def test_trainer_calls_set_epoch_each_epoch(self):
        from functools import partial

        from megobin.trainers.single_phase import SinglePhaseTrainer

        s = SemiBin2PairSampler(_split(n=12, dim=136), ratio=4, seed=0)
        seen: list[int] = []
        orig = s.set_epoch
        s.set_epoch = lambda e: (seen.append(e), orig(e)) and None  # type: ignore[method-assign]
        enc = SemiBin2Encoder(input_dim=136, kmer_dim=136, output_dim=8)
        trainer = SinglePhaseTrainer(
            optimizer=partial(torch.optim.Adam, lr=1e-3),
            epochs=3,
            batch_size=64,
            device="cpu",
            params="all",
            checkpoint_path=None,
        )
        trainer.fit(enc, s, HingeContrastiveLoss())
        assert seen == [0, 1, 2]
