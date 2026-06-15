"""Unit tests for the pure fidelity comparators."""

import numpy as np
import pytest

from megobin.fidelity.metrics import (
    array_close,
    binner_ari,
    binner_bp_match_f1,
    binner_bp_match_f1_details,
    set_agreement,
)


# ---- binner_ari ------------------------------------------------------------


def test_ari_identical_is_one():
    labels = np.array([0, 0, 1, 1, 2, 2])
    assert binner_ari(labels, labels) == pytest.approx(1.0)


def test_ari_is_label_permutation_invariant():
    a = np.array([0, 0, 1, 1, 2, 2])
    b = np.array([5, 5, 9, 9, 7, 7])  # same partition, renamed labels
    assert binner_ari(a, b) == pytest.approx(1.0)


def test_ari_clips_negative_to_zero():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 5, size=200)
    b = rng.integers(0, 5, size=200)
    score = binner_ari(a, b)
    assert 0.0 <= score < 0.1


# ---- binner_bp_match_f1 ----------------------------------------------------


def test_bp_f1_identical_is_one():
    labels = np.array([0, 0, 0, 1, 1, 1])
    lengths = np.full(6, 1000.0)
    assert binner_bp_match_f1(labels, labels, lengths) == pytest.approx(1.0)


def test_bp_f1_shatter_collapses_recall():
    # Reference: one bin of 10 contigs. Candidate: 10 singleton bins.
    ref = np.zeros(10, dtype=int)
    cand = np.arange(10, dtype=int)
    lengths = np.full(10, 30000.0)
    d = binner_bp_match_f1_details(ref, cand, lengths)
    assert d.recall == pytest.approx(0.1)  # best match = a single contig
    assert d.precision == pytest.approx(1.0)  # each singleton is pure
    assert d.f1 < 0.2
    assert d.shatter_index == pytest.approx(10.0)


def test_bp_f1_merge_collapses_precision():
    # Reference: two bins. Candidate: merged into one.
    ref = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    cand = np.zeros(10, dtype=int)
    lengths = np.full(10, 2000.0)
    d = binner_bp_match_f1_details(ref, cand, lengths)
    assert d.recall == pytest.approx(1.0)
    assert d.precision == pytest.approx(0.5)
    assert d.f1 == pytest.approx(2 * 0.5 * 1.0 / 1.5)


def test_bp_f1_penalizes_binning_reference_unbinned():
    # The minfasta divergence: faithful ref leaves small noise contigs -1;
    # the divergent candidate gives each its own singleton bin.
    ref = np.array([0] * 10 + [-1] * 6)
    cand = np.array([0] * 10 + [1, 2, 3, 4, 5, 6])
    lengths = np.array([30000.0] * 10 + [5000.0] * 6)
    d = binner_bp_match_f1_details(ref, cand, lengths)
    assert d.recall == pytest.approx(1.0)
    # candidate bins 30k bp that the reference left unbinned -> precision < 1
    assert d.precision == pytest.approx(300000 / 330000)
    assert d.shatter_index > 1.0


def test_bp_f1_both_empty_agree():
    ref = np.array([-1, -1, -1])
    lengths = np.full(3, 100.0)
    assert binner_bp_match_f1(ref, ref, lengths) == pytest.approx(1.0)


def test_bp_f1_identical_zero_length_is_one():
    # Pathological zero-length contigs must not score perfect agreement as 0.0.
    labels = np.array([0, 0, 1, 1])
    assert binner_bp_match_f1(labels, labels, np.zeros(4)) == pytest.approx(1.0)


def test_bp_f1_shape_mismatch_raises():
    with pytest.raises(ValueError, match="share shape"):
        binner_bp_match_f1(np.zeros(3), np.zeros(4), np.zeros(3))


# ---- generic comparators ---------------------------------------------------


def test_set_agreement():
    assert set_agreement([1, 2, 3], [2, 3, 4]) == pytest.approx(2 / 4)
    assert set_agreement([], []) == pytest.approx(1.0)
    assert set_agreement([1, 2], [3, 4]) == pytest.approx(0.0)


def test_array_close():
    a = np.array([1.0, 2.0, 3.0])
    assert array_close(a, a) == pytest.approx(1.0)
    assert array_close(a, a + 1.0) == pytest.approx(0.0)
    assert array_close(np.zeros(3), np.zeros(4)) == pytest.approx(0.0)
    assert array_close(np.array([1.0, 2.0]), np.array([1.0, 99.0])) == pytest.approx(0.5)
