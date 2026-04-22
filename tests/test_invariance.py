"""Reverse-complement invariance at the feature level.

With canonical k-mers, a sequence and its reverse complement share the
same profile — this is a property of the feature computation and does
not depend on any particular encoder. With full (non-canonical) k-mers
the profiles are related by the complement permutation, so their sorted
versions must match.
"""

import numpy as np

from src.features.kmer_profiles import compute_kmer_profiles
from src.features.reverse_complement import reverse_complement


def _random_dna(length: int, rng: np.random.Generator) -> str:
    return "".join(rng.choice(list("ACGT"), size=length))


class TestCanonicalInvariance:
    """With canonical k-mers, a sequence and its RC have identical profiles."""

    def test_canonical_profiles_identical(self):
        rng = np.random.default_rng(42)
        seqs = [_random_dna(500, rng) for _ in range(20)]
        rc_seqs = [reverse_complement(s) for s in seqs]

        profiles = compute_kmer_profiles(seqs, k=4, canonical=True)
        rc_profiles = compute_kmer_profiles(rc_seqs, k=4, canonical=True)

        np.testing.assert_allclose(profiles, rc_profiles, atol=1e-10)


class TestFullKmerInvariance:
    """With full (non-canonical) k-mers, a sequence and its RC share the
    same multiset of k-mer counts — just permuted by the complement map."""

    def test_full_kmer_rc_similarity(self):
        rng = np.random.default_rng(7)
        seqs = [_random_dna(1000, rng) for _ in range(20)]
        rc_seqs = [reverse_complement(s) for s in seqs]

        profiles = compute_kmer_profiles(seqs, k=4, canonical=False)
        rc_profiles = compute_kmer_profiles(rc_seqs, k=4, canonical=False)

        for i in range(len(seqs)):
            np.testing.assert_allclose(
                np.sort(profiles[i]), np.sort(rc_profiles[i]), atol=1e-10
            )
