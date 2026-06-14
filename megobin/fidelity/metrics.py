"""Slot-appropriate fidelity comparators.

Every comparator is a pure function returning an agreement score in ``[0, 1]``
(1.0 = perfect agreement) and is invariant to the nuisance symmetry of its slot:

* **binner** — invariant to label permutation (ARI) and, for the bp-weighted
  match, scored on *base pairs* not contig count so a candidate that shatters a
  reference bin into singletons is penalised.
* **encoder** — embeddings are only defined up to an isometry, so we compare the
  *geometry* (pairwise-distance ranks) or route both embeddings through a fixed
  binner and compare the partitions.

No I/O, no global state, no torch.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score


class _Clusterer(Protocol):
    """Structural type for anything with a ``cluster`` method (a Binner)."""

    def cluster(self, embeddings: np.ndarray) -> np.ndarray: ...


# --------------------------------------------------------------------------- #
# Binner comparators
# --------------------------------------------------------------------------- #


def binner_ari(labels_a: ArrayLike, labels_b: ArrayLike) -> float:
    """Adjusted Rand Index, clipped to ``[0, 1]``.

    Label-permutation invariant. ARI can be slightly negative for
    worse-than-random partitions; for fidelity we treat anything ``< 0`` as
    zero agreement. ``-1`` (unbinned) is treated as an ordinary label, so two
    partitions that unbin the *same* contigs are rewarded for agreeing.
    """
    a = np.asarray(labels_a)
    b = np.asarray(labels_b)
    if a.shape != b.shape:
        raise ValueError(f"label arrays differ in shape: {a.shape} vs {b.shape}")
    if a.size == 0:
        return 1.0
    return max(0.0, float(adjusted_rand_score(a, b)))


@dataclass(frozen=True)
class BpMatchDetails:
    """Diagnostics for :func:`binner_bp_match_f1`."""

    precision: float
    recall: float
    f1: float
    n_bins_a: int
    n_bins_b: int
    shatter_index: float


def _bins_by_label(
    labels: np.ndarray, unbinned_label: int
) -> dict[int, list[int]]:
    bins: dict[int, list[int]] = defaultdict(list)
    for row, lab in enumerate(labels):
        ilab = int(lab)
        if ilab != unbinned_label:
            bins[ilab].append(row)
    return bins


def binner_bp_match_f1_details(
    labels_a: ArrayLike,
    labels_b: ArrayLike,
    contig_lengths: ArrayLike,
    *,
    unbinned_label: int = -1,
) -> BpMatchDetails:
    """Base-pair-weighted best-match bin pairing.

    Convention: **A is the reference/truth side, B is the candidate.** For each
    reference bin we credit the base pairs captured by its single best-matching
    candidate bin (and symmetrically for precision). Weighting by base pairs —
    not contig count — is what makes this sensitive to the "many tiny singleton
    bins" failure mode: shattering a reference bin collapses every candidate
    overlap down to one contig, so recall craters even though ARI barely moves.
    """
    a = np.asarray(labels_a)
    b = np.asarray(labels_b)
    lengths = np.asarray(contig_lengths, dtype=float)
    if not (a.shape == b.shape == lengths.shape):
        raise ValueError(
            "labels_a, labels_b, contig_lengths must share shape: "
            f"{a.shape}, {b.shape}, {lengths.shape}"
        )

    bins_a = _bins_by_label(a, unbinned_label)
    bins_b = _bins_by_label(b, unbinned_label)
    n_a, n_b = len(bins_a), len(bins_b)

    if n_a == 0 and n_b == 0:
        # Both sides agree there is nothing binnable.
        return BpMatchDetails(1.0, 1.0, 1.0, 0, 0, 0.0)

    total_a = float(sum(lengths[rows].sum() for rows in bins_a.values()))
    total_b = float(sum(lengths[rows].sum() for rows in bins_b.values()))

    if total_a == 0.0 and total_b == 0.0:
        # Both sides bin only zero-length contigs: agreement is undefined by base
        # pairs, so fall back to label-permutation agreement (identical
        # partitions still score 1.0, disagreeing ones less).
        agree = binner_ari(a, b)
        return BpMatchDetails(agree, agree, agree, n_a, n_b, n_b / max(n_a, 1))

    def _best_match_bp(
        bins_src: dict[int, list[int]], other: np.ndarray
    ) -> float:
        captured = 0.0
        for rows in bins_src.values():
            overlap: dict[int, float] = defaultdict(float)
            for row in rows:
                lab = int(other[row])
                if lab != unbinned_label:
                    overlap[lab] += lengths[row]
            captured += max(overlap.values(), default=0.0)
        return captured

    recall = _best_match_bp(bins_a, b) / total_a if total_a > 0 else 0.0
    precision = _best_match_bp(bins_b, a) / total_b if total_b > 0 else 0.0
    denom = precision + recall
    f1 = 2 * precision * recall / denom if denom > 0 else 0.0
    shatter = n_b / max(n_a, 1)
    return BpMatchDetails(precision, recall, f1, n_a, n_b, shatter)


def binner_bp_match_f1(
    labels_a: ArrayLike,
    labels_b: ArrayLike,
    contig_lengths: ArrayLike,
    *,
    unbinned_label: int = -1,
) -> float:
    """Base-pair-weighted bin-match F1 in ``[0, 1]`` (see details variant)."""
    return binner_bp_match_f1_details(
        labels_a, labels_b, contig_lengths, unbinned_label=unbinned_label
    ).f1


# --------------------------------------------------------------------------- #
# Encoder comparators
# --------------------------------------------------------------------------- #


def encoder_distance_agreement(
    emb_a: ArrayLike,
    emb_b: ArrayLike,
    *,
    sample: int = 2000,
    seed: int = 0,
) -> float:
    """Spearman correlation of subsampled pairwise distances.

    Invariant to rotation, reflection, permutation and uniform scaling of the
    embedding — it compares *neighbour geometry* (what is close to what), which
    is the part of an embedding the downstream binner actually consumes. The
    same row-pairs are used for both embeddings, so it is a paired test.
    """
    a = np.asarray(emb_a, dtype=float)
    b = np.asarray(emb_b, dtype=float)
    if a.shape[0] != b.shape[0]:
        raise ValueError(
            f"embeddings differ in row count: {a.shape[0]} vs {b.shape[0]}"
        )
    n = a.shape[0]
    if n < 3:
        return 1.0  # too few rows to form a meaningful distance distribution
    max_pairs = n * (n - 1) // 2
    if max_pairs <= sample:
        # Small n: enumerate every unordered pair deterministically. Rejection
        # sampling would leave too few pairs after self-pair removal here.
        iu, ju = np.triu_indices(n, k=1)
        i, j = iu, ju
    else:
        rng = np.random.default_rng(seed)
        k = sample
        draw = max(k * 3, 16)  # margin for self-pair rejection (rate 1/n)
        ii = rng.integers(0, n, size=draw)
        jj = rng.integers(0, n, size=draw)
        keep = ii != jj
        i, j = ii[keep][:k], jj[keep][:k]
    if i.size < 2:  # degenerate sampling: fall back to exact equality
        return 1.0 if np.allclose(a, b) else 0.0
    da = np.linalg.norm(a[i] - a[j], axis=1)
    db = np.linalg.norm(b[i] - b[j], axis=1)
    rho = spearmanr(da, db).statistic
    if not np.isfinite(rho):
        # constant distances on at least one side: agree only if truly identical
        return 1.0 if np.allclose(a, b) else 0.0
    return max(0.0, float(rho))


def encoder_through_binner(
    emb_a: ArrayLike, emb_b: ArrayLike, binner: _Clusterer
) -> float:
    """Cluster both embeddings with the *same fixed* binner, then ARI.

    Tests the encoder through its real downstream contract: do the two
    embeddings induce the same partition once a binner has run on them?
    """
    labels_a = binner.cluster(np.asarray(emb_a, dtype=float))
    labels_b = binner.cluster(np.asarray(emb_b, dtype=float))
    return binner_ari(labels_a, labels_b)


# --------------------------------------------------------------------------- #
# Generic comparators
# --------------------------------------------------------------------------- #


def set_agreement(a: ArrayLike, b: ArrayLike) -> float:
    """Jaccard overlap of two id sets; two empty sets agree perfectly."""
    sa = set(np.asarray(a).ravel().tolist())
    sb = set(np.asarray(b).ravel().tolist())
    union = sa | sb
    if not union:
        return 1.0
    return len(sa & sb) / len(union)


def array_close(
    a: ArrayLike, b: ArrayLike, *, rtol: float = 1e-5, atol: float = 1e-8
) -> float:
    """Fraction of elementwise-close entries; shape mismatch scores 0."""
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    if aa.shape != bb.shape:
        return 0.0
    if aa.size == 0:
        return 1.0
    return float(np.isclose(aa, bb, rtol=rtol, atol=atol).mean())
