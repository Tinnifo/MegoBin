"""Data unit tests for feature computation.

- K-mer profiles sum to 1.0 (±1e-6) after L1 normalisation
- No all-zero profiles (empty sequences produce zero rows, but real
  sequences of sufficient length must not)
- No NaN or Inf in any feature matrix
- Abundance values non-negative
- Correct dimensions: 256 (full) / 136 (canonical) for k=4
"""

import numpy as np
import pytest

from src.features.kmer_profiles import (
    build_canonical_map,
    compute_kmer_profiles,
    generate_kmers,
)


def _random_dna(length: int, rng: np.random.Generator) -> str:
    return "".join(rng.choice(list("ACGT"), size=length))


# ---- Dimensions ------------------------------------------------------------


class TestKmerDimensions:
    def test_full_kmer_dims(self):
        kmers = generate_kmers(4, "ACGT")
        assert len(kmers) == 256

    def test_canonical_kmer_dims(self):
        canon, _ = build_canonical_map(4, "ACGT")
        assert len(canon) == 136

    def test_canonical_kmer_dims_atgc(self):
        canon, _ = build_canonical_map(4, "ATGC")
        assert len(canon) == 136

    def test_full_profile_shape(self):
        rng = np.random.default_rng(0)
        seqs = [_random_dna(500, rng) for _ in range(10)]
        profiles = compute_kmer_profiles(seqs, k=4, canonical=False)
        assert profiles.shape == (10, 256)

    def test_canonical_profile_shape(self):
        rng = np.random.default_rng(0)
        seqs = [_random_dna(500, rng) for _ in range(10)]
        profiles = compute_kmer_profiles(seqs, k=4, canonical=True)
        assert profiles.shape == (10, 136)


# ---- L1 normalisation -----------------------------------------------------


class TestL1Normalisation:
    def test_rows_sum_to_one_full(self):
        rng = np.random.default_rng(1)
        seqs = [_random_dna(500, rng) for _ in range(20)]
        profiles = compute_kmer_profiles(seqs, k=4, canonical=False)
        np.testing.assert_allclose(profiles.sum(axis=1), 1.0, atol=1e-6)

    def test_rows_sum_to_one_canonical(self):
        rng = np.random.default_rng(2)
        seqs = [_random_dna(500, rng) for _ in range(20)]
        profiles = compute_kmer_profiles(seqs, k=4, canonical=True)
        np.testing.assert_allclose(profiles.sum(axis=1), 1.0, atol=1e-6)

    def test_rows_sum_to_one_with_pseudocount(self):
        rng = np.random.default_rng(3)
        seqs = [_random_dna(500, rng) for _ in range(20)]
        profiles = compute_kmer_profiles(
            seqs, k=4, canonical=True, pseudocount=1e-5
        )
        np.testing.assert_allclose(profiles.sum(axis=1), 1.0, atol=1e-6)


# ---- No degenerate values --------------------------------------------------


class TestNoDegenerate:
    def test_no_nan(self):
        rng = np.random.default_rng(4)
        seqs = [_random_dna(500, rng) for _ in range(20)]
        profiles = compute_kmer_profiles(seqs, k=4, canonical=False)
        assert not np.isnan(profiles).any()

    def test_no_inf(self):
        rng = np.random.default_rng(5)
        seqs = [_random_dna(500, rng) for _ in range(20)]
        profiles = compute_kmer_profiles(seqs, k=4, canonical=False)
        assert not np.isinf(profiles).any()

    def test_no_allzero_for_real_sequences(self):
        """A sequence of sufficient length (≥ k) should not have an all-zero
        profile row."""
        rng = np.random.default_rng(6)
        seqs = [_random_dna(500, rng) for _ in range(20)]
        profiles = compute_kmer_profiles(seqs, k=4, canonical=False)
        row_sums = profiles.sum(axis=1)
        assert (row_sums > 0).all()

    def test_non_negative(self):
        rng = np.random.default_rng(7)
        seqs = [_random_dna(500, rng) for _ in range(20)]
        profiles = compute_kmer_profiles(seqs, k=4, canonical=False)
        assert (profiles >= 0).all()


# ---- Edge cases ------------------------------------------------------------


class TestEdgeCases:
    def test_short_sequence_below_k(self):
        """Sequence shorter than k produces a zero row (no valid k-mers)."""
        profiles = compute_kmer_profiles(["AC"], k=4, canonical=False)
        assert profiles.shape == (1, 256)
        # Row is zero → normalisation leaves it at zero
        assert profiles.sum() == 0.0

    def test_sequence_with_n_bases(self):
        """N bases should be skipped, remaining k-mers still counted."""
        profiles = compute_kmer_profiles(
            ["ACGTNNNNACGT"], k=4, canonical=False
        )
        assert profiles.shape == (1, 256)
        assert not np.isnan(profiles).any()

    def test_empty_list(self):
        profiles = compute_kmer_profiles([], k=4, canonical=False)
        assert profiles.shape == (0, 256)

    @pytest.mark.parametrize("k", [2, 3, 4, 5])
    def test_various_k(self, k):
        rng = np.random.default_rng(8)
        seqs = [_random_dna(200, rng) for _ in range(5)]
        profiles = compute_kmer_profiles(seqs, k=k, canonical=False)
        assert profiles.shape == (5, 4**k)
        np.testing.assert_allclose(profiles.sum(axis=1), 1.0, atol=1e-6)
