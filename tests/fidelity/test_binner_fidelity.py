"""Binner-slot fidelity on the synthetic genome set (no external tool).

Demonstrates the harness's discriminating power: the faithful-config binner
lands inside the reference's own variance band and recovers the planted
genomes, while the divergent config (minfasta off, length-weighting off) does
not. Bands are calibrated from the reference's run-to-run variance, so this is
a genuine self-consistency + ground-truth check.
"""

import pytest

from megobin.fidelity import (
    DegenerateBandError,
    FidelityHarness,
    MegoBinSelfBinnerOracle,
    binner_ari,
    binner_bp_match_f1,
)
from megobin.fidelity.oracle import ReferenceOracle

from .conftest import divergent_binner, faithful_binner


def _ari_harness() -> FidelityHarness:
    return FidelityHarness(binner_ari, z=2.0)


def test_faithful_config_passes(genome_set):
    oracle = MegoBinSelfBinnerOracle(faithful_binner)
    candidate = faithful_binner(genome_set)
    report = _ari_harness().run(candidate=candidate, oracle=oracle, fixture=genome_set)
    assert report.passed, report.to_markdown()
    assert report.vs_reference >= report.reference_self.floor
    assert report.vs_truth is not None and report.vs_truth >= report.reference_truth.lower


def test_divergent_config_fails(genome_set):
    oracle = MegoBinSelfBinnerOracle(faithful_binner)
    candidate = divergent_binner(genome_set)
    report = _ari_harness().run(candidate=candidate, oracle=oracle, fixture=genome_set)
    assert not report.passed, report.to_markdown()
    assert report.vs_reference < report.reference_self.floor


def test_groundtruth_recovery(genome_set):
    # The faithful binner recovers the planted genomes near-perfectly.
    binner = faithful_binner(genome_set)
    labels = binner.cluster(genome_set.embedding(0))
    assert binner_ari(genome_set.labels, labels) > 0.95


def test_divergent_shatters_into_many_bins(genome_set):
    # bp-weighted F1 + shatter_index expose the "many tiny singleton bins" mode.
    from megobin.fidelity import binner_bp_match_f1_details

    emb = genome_set.embedding(7)
    faithful_labels = faithful_binner(genome_set).cluster(emb)
    divergent_labels = divergent_binner(genome_set).cluster(emb)
    d = binner_bp_match_f1_details(faithful_labels, divergent_labels, genome_set.lengths)
    assert d.shatter_index > 3.0
    assert d.precision < 0.95  # divergent bins bp the faithful run left unbinned


def test_bp_f1_comparator_also_discriminates(genome_set):
    # Same two-way acceptance using the bp-weighted comparator.
    def cmp(a, b):
        return binner_bp_match_f1(a, b, genome_set.lengths)

    harness = FidelityHarness(cmp, comparator_name="binner_bp_match_f1", z=2.0)
    oracle = MegoBinSelfBinnerOracle(faithful_binner)
    assert harness.run(
        candidate=faithful_binner(genome_set), oracle=oracle, fixture=genome_set
    ).passed
    assert not harness.run(
        candidate=divergent_binner(genome_set), oracle=oracle, fixture=genome_set
    ).passed


def test_parse_bins_to_labels_round_trip(tmp_path):
    # The reusable subprocess-oracle core: per-bin FASTAs -> labels (input order).
    from megobin.fidelity import parse_bins_to_labels

    (tmp_path / "bin.0.fa").write_text(">c0\nACGT\n>c1\nACGT\n")
    (tmp_path / "bin.1.fa").write_text(">c2\nACGT\n")
    names = ["c0", "c1", "c2", "c3"]  # c3 is in no bin -> -1
    labels = parse_bins_to_labels(tmp_path, names)
    assert labels.tolist() == [0, 0, 1, -1]


def test_degenerate_band_tripwire(genome_set):
    # An oracle that returns the SAME labels K times has no variance -> the
    # harness refuses rather than silently passing only identical candidates.
    fixed = faithful_binner(genome_set).cluster(genome_set.embedding(0))

    class _ZeroVarianceOracle(ReferenceOracle):
        slot = "binner"

        def runs(self, fixture):
            return [fixed.copy() for _ in range(10)]

        def candidate_run(self, candidate, fixture):
            return fixed.copy()

    with pytest.raises(DegenerateBandError):
        _ari_harness().run(
            candidate=faithful_binner(genome_set),
            oracle=_ZeroVarianceOracle(),
            fixture=genome_set,
        )


def test_mismatched_reference_run_shapes_raise(genome_set):
    # An oracle whose runs disagree in shape is caught up front with a clear
    # error, not an opaque downstream comparator failure.
    import numpy as np

    class _RaggedOracle(ReferenceOracle):
        slot = "binner"

        def runs(self, fixture):
            return [np.zeros(10, dtype=int), np.zeros(9, dtype=int)]

        def candidate_run(self, candidate, fixture):
            return np.zeros(10, dtype=int)

    with pytest.raises(ValueError, match="disagree in shape"):
        _ari_harness().run(
            candidate=faithful_binner(genome_set),
            oracle=_RaggedOracle(),
            fixture=genome_set,
        )
