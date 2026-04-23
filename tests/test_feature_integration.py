"""End-to-end integration between k-mer features and abundance features.

Bugs that live between two modules — inconsistent contig ordering, shape
mismatch at concat, NaNs introduced where none existed before — are
invisible to unit tests on each side. These cases exercise the whole
k-mer → abundance → concat path on a synthetic FASTA + fake BAMs.

The fake pysam plumbing is the same as ``test_abundance.py``; kept local
here so each file can run in isolation.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

from megobin.features.kmer_profiles import (
    compute_kmer_profiles,
    compute_kmer_profiles_with_splits,
    compute_profiles_from_fasta,
    read_fasta,
)


# ---------------------------------------------------------------------------
# Fake pysam (local copy — see test_abundance.py)
# ---------------------------------------------------------------------------


class _FakeAlignmentFile:
    _registry: dict[str, "_FakeAlignmentFile"] = {}

    def __init__(self, path: str, mode: str = "rb"):
        cfg = self._registry[path]
        self._lengths = cfg._lengths
        self._depths = cfg._depths

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def get_reference_length(self, contig: str) -> int:
        return self._lengths.get(contig, 0)

    def count_coverage(self, contig: str, start: int = 0, stop: int | None = None):
        length = self._lengths.get(contig, 0)
        if stop is None:
            stop = length
        window = stop - start
        depth = self._depths.get(contig, 0.0)
        total = np.full(window, float(depth), dtype=np.float64)
        q = total / 4.0
        return q, q, q, q


def _register_bam(path: str, lengths, depths):
    cfg = object.__new__(_FakeAlignmentFile)
    cfg._lengths = lengths
    cfg._depths = depths
    _FakeAlignmentFile._registry[path] = cfg


@pytest.fixture
def fake_pysam(monkeypatch):
    _FakeAlignmentFile._registry.clear()
    fake = types.ModuleType("pysam")
    fake.AlignmentFile = _FakeAlignmentFile
    monkeypatch.setitem(sys.modules, "pysam", fake)
    yield
    _FakeAlignmentFile._registry.clear()


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------


def _random_dna(rng: np.random.Generator, length: int) -> str:
    return "".join(rng.choice(list("ACGT"), size=length))


@pytest.fixture
def synthetic_dataset(tmp_path):
    """Build a FASTA with contigs of mixed lengths and return paths + metadata.

    Lengths are picked so that some contigs fall below common split/min
    thresholds:

        c0: 3000 bp  (passes min=1000, passes split_min=2000)
        c1: 1500 bp  (passes min=1000, below split_min=2000)
        c2: 2500 bp  (passes both)
        c3:  800 bp  (below min=1000 — should be dropped)
        c4: 4000 bp  (passes both)
    """
    rng = np.random.default_rng(123)
    contigs = {
        "c0": _random_dna(rng, 3000),
        "c1": _random_dna(rng, 1500),
        "c2": _random_dna(rng, 2500),
        "c3": _random_dna(rng, 800),
        "c4": _random_dna(rng, 4000),
    }

    fasta_path = tmp_path / "contigs.fasta"
    with open(fasta_path, "w") as f:
        for name, seq in contigs.items():
            f.write(f">{name}\n{seq}\n")

    return {"fasta": fasta_path, "contigs": contigs}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCombinedShape:
    def test_canonical_kmer_plus_abundance_shape(self, fake_pysam, synthetic_dataset):
        from megobin.features.abundance import compute_abundance

        kmer, names = compute_profiles_from_fasta(
            synthetic_dataset["fasta"],
            k=4,
            canonical=True,
            alphabet="ATGC",
            min_length=1000,
        )
        assert kmer.shape[1] == 136
        assert len(names) == 4  # c3 dropped

        num_bams = 3
        bam_paths = []
        for b in range(num_bams):
            path = f"/tmp/bam_{b}.bam"
            _register_bam(
                path,
                lengths={
                    n: len(synthetic_dataset["contigs"][n]) for n in names
                },
                depths={n: float(b + 1) for n in names},
            )
            bam_paths.append(Path(path))

        abundance = compute_abundance(bam_paths, names)
        assert abundance.shape == (len(names), 2 * num_bams)

        combined = np.concatenate([kmer, abundance], axis=1)
        assert combined.shape == (len(names), 136 + 2 * num_bams)

    def test_full_kmer_plus_abundance_shape(self, fake_pysam, synthetic_dataset):
        from megobin.features.abundance import compute_abundance

        kmer, names = compute_profiles_from_fasta(
            synthetic_dataset["fasta"],
            k=4,
            canonical=False,
            min_length=1000,
        )
        assert kmer.shape[1] == 256

        _register_bam(
            "/tmp/one.bam",
            lengths={n: len(synthetic_dataset["contigs"][n]) for n in names},
            depths={n: 2.0 for n in names},
        )
        abundance = compute_abundance([Path("/tmp/one.bam")], names)

        combined = np.concatenate([kmer, abundance], axis=1)
        assert combined.shape == (len(names), 256 + 2)


class TestCombinedCleanliness:
    def test_no_nan_or_inf_in_combined(self, fake_pysam, synthetic_dataset):
        from megobin.features.abundance import compute_abundance

        kmer, names = compute_profiles_from_fasta(
            synthetic_dataset["fasta"],
            k=4,
            canonical=True,
            alphabet="ATGC",
            pseudocount=1e-5,
            min_length=1000,
        )
        _register_bam(
            "/tmp/one.bam",
            lengths={n: len(synthetic_dataset["contigs"][n]) for n in names},
            depths={n: 7.5 for n in names},
        )
        abundance = compute_abundance([Path("/tmp/one.bam")], names)
        combined = np.concatenate([kmer, abundance], axis=1)

        assert not np.isnan(combined).any()
        assert not np.isinf(combined).any()

    def test_kmer_slice_remains_l1_normalized(self, fake_pysam, synthetic_dataset):
        """Concatenating abundance must not alter the k-mer portion."""
        from megobin.features.abundance import compute_abundance

        kmer, names = compute_profiles_from_fasta(
            synthetic_dataset["fasta"],
            k=4,
            canonical=True,
            alphabet="ATGC",
            pseudocount=1e-5,
            min_length=1000,
        )
        _register_bam(
            "/tmp/one.bam",
            lengths={n: len(synthetic_dataset["contigs"][n]) for n in names},
            depths={n: 4.0 for n in names},
        )
        abundance = compute_abundance([Path("/tmp/one.bam")], names)
        combined = np.concatenate([kmer, abundance], axis=1)

        kmer_slice = combined[:, :136]
        row_sums = kmer_slice.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

    def test_abundance_slice_non_negative(self, fake_pysam, synthetic_dataset):
        from megobin.features.abundance import compute_abundance

        kmer, names = compute_profiles_from_fasta(
            synthetic_dataset["fasta"],
            k=4,
            canonical=True,
            alphabet="ATGC",
            min_length=1000,
        )
        _register_bam(
            "/tmp/one.bam",
            lengths={n: len(synthetic_dataset["contigs"][n]) for n in names},
            depths={n: 6.0 for n in names},
        )
        abundance = compute_abundance([Path("/tmp/one.bam")], names)
        combined = np.concatenate([kmer, abundance], axis=1)

        assert (combined[:, 136:] >= 0.0).all()


class TestContigOrdering:
    def test_kmer_abundance_row_alignment(self, fake_pysam, synthetic_dataset):
        """Row i refers to the same contig across kmer, abundance, and names.

        Gives each contig a unique depth signature so we can verify that
        the abundance row matches the name at the same index.
        """
        from megobin.features.abundance import compute_abundance

        _, names = compute_profiles_from_fasta(
            synthetic_dataset["fasta"],
            k=4,
            canonical=True,
            alphabet="ATGC",
            min_length=1000,
        )

        per_contig_depth = {n: float(10 * (i + 1)) for i, n in enumerate(names)}
        _register_bam(
            "/tmp/one.bam",
            lengths={n: len(synthetic_dataset["contigs"][n]) for n in names},
            depths=per_contig_depth,
        )
        abundance = compute_abundance([Path("/tmp/one.bam")], names)

        for i, n in enumerate(names):
            assert abundance[i, 0] == pytest.approx(per_contig_depth[n])


class TestSplitLayout:
    def test_split_profiles_count_and_halves(self, synthetic_dataset):
        """Split profiles have shape (2M, D) where M is the count of contigs
        passing split_min_length, and rows 0..M-1 / M..2M-1 are left/right
        halves of the same contigs."""
        all_names, all_seqs = read_fasta(synthetic_dataset["fasta"])

        min_length = 1000
        split_min_length = 2000
        alphabet = "ATGC"

        whole, split = compute_kmer_profiles_with_splits(
            all_seqs,
            k=4,
            canonical=True,
            alphabet=alphabet,
            pseudocount=1e-5,
            min_length=min_length,
            split_min_length=split_min_length,
        )

        kept = [s for s in all_seqs if len(s) >= min_length]
        splittable = [s for s in kept if len(s) >= split_min_length]
        M = len(splittable)

        assert whole.shape == (len(kept), 136)
        assert split.shape == (2 * M, 136)

        # Recompute left/right halves independently and compare.
        lefts_expected = compute_kmer_profiles(
            [s[: len(s) // 2] for s in splittable],
            k=4,
            canonical=True,
            alphabet=alphabet,
            pseudocount=1e-5,
        )
        rights_expected = compute_kmer_profiles(
            [s[len(s) // 2 :] for s in splittable],
            k=4,
            canonical=True,
            alphabet=alphabet,
            pseudocount=1e-5,
        )

        np.testing.assert_allclose(split[:M], lefts_expected, atol=1e-12)
        np.testing.assert_allclose(split[M:], rights_expected, atol=1e-12)

    def test_split_profiles_empty_when_all_contigs_short(self):
        """No contigs ≥ split_min_length → split array has shape (0, D)."""
        rng = np.random.default_rng(0)
        short_seqs = ["".join(rng.choice(list("ACGT"), 1200)) for _ in range(3)]

        whole, split = compute_kmer_profiles_with_splits(
            short_seqs,
            k=4,
            canonical=True,
            alphabet="ATGC",
            pseudocount=1e-5,
            min_length=1000,
            split_min_length=2000,
        )
        assert whole.shape == (3, 136)
        assert split.shape == (0, 136)
