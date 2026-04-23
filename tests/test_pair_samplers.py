"""Tests for the PairSampler slot."""

import numpy as np
import pytest
import torch

from megobin.data.hybrid_sampler import HybridPairSampler
from megobin.data.semibin_sampler import SemiBinPairSampler
from megobin.data.uncertain_gen_sampler import UncertainGenPairSampler


D = 8      # feature dimension used across tests
N = 50     # number of whole contigs (and split-pairs)
M = 10_000  # cannot-link pool size


def _features_split(value: float = 1.0) -> np.ndarray:
    return np.full((2 * N, D), value, dtype=np.float64)


def _features_whole(value: float = 0.0) -> np.ndarray:
    return np.full((N, D), value, dtype=np.float64)


def _cannot_link_pairs() -> np.ndarray:
    rng = np.random.default_rng(0)
    a = rng.integers(0, N, size=M)
    b = rng.integers(0, N, size=M)
    return np.stack([a, b], axis=1)


# ---- Protocol compliance ---------------------------------------------------


def _assert_triple_shapes(sampler):
    x_i, x_j, label = sampler[0]
    assert isinstance(x_i, torch.Tensor)
    assert isinstance(x_j, torch.Tensor)
    assert isinstance(label, torch.Tensor)
    assert x_i.shape == (D,)
    assert x_j.shape == (D,)
    assert label.shape == ()
    assert label.item() in (0.0, 1.0)


class TestSemiBinSampler:
    def setup_method(self):
        self.sampler = SemiBinPairSampler(
            features_split=_features_split(),
            features_whole=_features_whole(),
            cannot_link_pairs=_cannot_link_pairs(),
            neg_per_pos=20,
            seed=0,
        )

    def test_triple_shapes(self):
        _assert_triple_shapes(self.sampler)

    def test_positive_pair_indices(self):
        for k in range(min(N, 5)):
            i, j = self.sampler._pos[k]
            assert i == k
            assert j == k + N

    def test_length(self):
        assert len(self.sampler) == N + 20 * N


class TestUncertainGenSampler:
    def setup_method(self):
        self.sampler = UncertainGenPairSampler(
            features_split=_features_split(),
            neg_per_pos=20,
            seed=0,
        )

    def test_triple_shapes(self):
        _assert_triple_shapes(self.sampler)

    def test_positive_pair_indices(self):
        for k in range(min(N, 5)):
            i, j = self.sampler._pos[k]
            assert i == k
            assert j == k + N

    def test_length(self):
        assert len(self.sampler) == N + 20 * N


class TestHybridSampler:
    def setup_method(self):
        self.sampler = HybridPairSampler(
            features_split=_features_split(1.0),
            features_whole=_features_whole(0.0),
            cannot_link_pairs=_cannot_link_pairs(),
            neg_per_pos=20,
            taxonomy_fraction=0.5,
            seed=0,
        )

    def test_triple_shapes(self):
        _assert_triple_shapes(self.sampler)

    def test_positive_pair_indices(self):
        for k in range(min(N, 5)):
            i, j = self.sampler._pos[k]
            assert i == k
            assert j == k + N

    def test_index_space_routing(self):
        """Positives and random negatives come from features_split (ones);
        taxonomy negatives come from features_whole (zeros)."""
        n_pos = self.sampler._n_pos
        n_tax = self.sampler._n_tax

        # Positive: both rows from features_split (ones) with label 1
        x_i, x_j, label = self.sampler[0]
        assert torch.all(x_i == 1.0)
        assert torch.all(x_j == 1.0)
        assert label.item() == 1.0

        # Taxonomy negative: rows from features_whole (zeros) with label 0
        x_i, x_j, label = self.sampler[n_pos]
        assert torch.all(x_i == 0.0)
        assert torch.all(x_j == 0.0)
        assert label.item() == 0.0

        # Random negative: rows from features_split (ones) with label 0
        x_i, x_j, label = self.sampler[n_pos + n_tax]
        assert torch.all(x_i == 1.0)
        assert torch.all(x_j == 1.0)
        assert label.item() == 0.0

    def test_taxonomy_fraction_zero_matches_uncertain_gen(self):
        """With taxonomy_fraction=0 all negatives come from the random pool,
        matching UncertainGenPairSampler's sampling strategy."""
        sampler = HybridPairSampler(
            features_split=_features_split(1.0),
            features_whole=_features_whole(0.0),
            cannot_link_pairs=_cannot_link_pairs(),
            neg_per_pos=20,
            taxonomy_fraction=0.0,
            seed=0,
        )
        assert sampler._n_tax == 0
        assert len(sampler) == N + 20 * N

        # Every negative should come from features_split (ones), label 0
        for idx in range(sampler._n_pos, len(sampler), max(1, (20 * N) // 20)):
            x_i, x_j, label = sampler[idx]
            assert torch.all(x_i == 1.0)
            assert torch.all(x_j == 1.0)
            assert label.item() == 0.0


# ---- Validation errors -----------------------------------------------------


def test_semibin_rejects_odd_split_rows():
    with pytest.raises(ValueError, match="rows must be even"):
        SemiBinPairSampler(
            features_split=np.zeros((3, D)),
            features_whole=_features_whole(),
            cannot_link_pairs=_cannot_link_pairs(),
        )


def test_hybrid_rejects_bad_taxonomy_fraction():
    with pytest.raises(ValueError, match="taxonomy_fraction"):
        HybridPairSampler(
            features_split=_features_split(),
            features_whole=_features_whole(),
            cannot_link_pairs=_cannot_link_pairs(),
            taxonomy_fraction=1.5,
        )


def test_uncertain_gen_rejects_odd_split_rows():
    with pytest.raises(ValueError, match="rows must be even"):
        UncertainGenPairSampler(features_split=np.zeros((3, D)))
