"""SemiBin2PairSampler: where must-link and cannot-link pairs come from."""

import numpy as np
import pytest
import torch

from megobin.data.semibin2_sampler import SemiBin2PairSampler


def _data(n_whole: int = 7, n_split: int = 3, dim: int = 4, extra_cols: int = 2):
    """Whole rows are 100 + contig index; split rows are small. So a tensor's
    value tells which matrix it came from."""
    whole = np.zeros((n_whole, dim + extra_cols), dtype=np.float32)
    whole[:, :dim] = 100.0 + np.arange(n_whole, dtype=np.float32)[:, None]
    whole[:, dim:] = -1.0  # depth columns that must not reach the network
    left = np.arange(n_split, dtype=np.float32)[:, None] + np.zeros((1, dim), np.float32)
    right = left + 0.5
    split = np.concatenate([left, right], axis=0)
    return split, whole


def test_cannot_link_pairs_are_two_different_whole_contigs():
    split, whole = _data()
    s = SemiBin2PairSampler(features_split=split, features_whole=whole, ratio=10, seed=1)
    assert s._n_pos == 3
    assert s._n_neg == split.shape[0] * 10 // 2
    for epoch in range(3):
        s.set_epoch(epoch)
        assert s._neg is not None
        assert s._neg.min() >= 0 and s._neg.max() < whole.shape[0]
        assert (s._neg[:, 0] != s._neg[:, 1]).all()
    assert set(np.unique(s._neg)) == set(range(whole.shape[0]))
    for idx in range(s._n_pos, len(s)):
        x_i, x_j, label = s[idx]
        assert label.item() == 0.0
        assert x_i.shape == (4,) and x_j.shape == (4,)
        assert x_i.min() >= 100.0 and x_j.min() >= 100.0  # whole rows, depth dropped
        assert not torch.equal(x_i, x_j)


def test_must_link_pairs_are_the_two_halves_of_one_contig():
    split, whole = _data()
    s = SemiBin2PairSampler(features_split=split, features_whole=whole, ratio=10, seed=1)
    for idx in range(s._n_pos):
        x_i, x_j, label = s[idx]
        assert label.item() == 1.0
        assert torch.allclose(x_j - x_i, torch.full((4,), 0.5))


def test_cannot_link_count_is_capped():
    split, whole = _data()
    s = SemiBin2PairSampler(
        features_split=split, features_whole=whole, ratio=1000, max_pairs=50, seed=0
    )
    assert s._n_neg == 50


def test_draws_are_deterministic_per_seed_and_epoch():
    split, whole = _data()
    a = SemiBin2PairSampler(features_split=split, features_whole=whole, ratio=10, seed=3)
    b = SemiBin2PairSampler(features_split=split, features_whole=whole, ratio=10, seed=3)
    a.set_epoch(2)
    b.set_epoch(2)
    assert a._neg is not None and b._neg is not None
    assert np.array_equal(a._neg, b._neg)
    b.set_epoch(3)
    assert not np.array_equal(a._neg, b._neg)


def test_whole_contig_matrix_is_required():
    split, whole = _data()
    with pytest.raises(TypeError):
        SemiBin2PairSampler(features_split=split)  # type: ignore[call-arg]
    with pytest.raises(ValueError):
        SemiBin2PairSampler(features_split=split, features_whole=whole[:1])
    with pytest.raises(ValueError):
        SemiBin2PairSampler(features_split=split, features_whole=whole[:, :2])
