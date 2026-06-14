"""Genuine SemiBin2 oracle: the true-oracle binner acceptance demo.

Skipped automatically when SemiBin is not installed (it is an optional
``[fidelity]`` dependency), so the default suite stays green without it. When
present, this is the *non-circular* fidelity check — the MegoBin binner is
measured against SemiBin's real ``get_best_bin``, not against itself.
"""

import numpy as np
import pytest

pytest.importorskip("SemiBin", reason="install the [fidelity] extra to run the SemiBin2 oracle")

from megobin.binners.semibin_2 import DBSCANEnsembleBinner  # noqa: E402
from megobin.fidelity import (  # noqa: E402
    SEMIBIN_N_TOTAL_MARKERS,
    FidelityHarness,
    SemiBin2GetBestBinOracle,
    binner_ari,
    semibin_cluster_labels,
)
from megobin.fidelity.oracles_semibin2 import _LenStr  # noqa: E402

_SEEDS = range(8)


def _mego_binner(fx, *, minfasta, lengths_on):
    # n_total_markers=107 matches SemiBin's hardcoded recall divisor.
    return DBSCANEnsembleBinner(
        minfasta=minfasta,
        contig_names=fx.contig_names,
        contig_lengths=fx.lengths if lengths_on else None,
        contig_to_marker=fx.contig_to_marker,
        n_total_markers=SEMIBIN_N_TOTAL_MARKERS,
    )


def _faithful(fx):
    return _mego_binner(fx, minfasta=200000, lengths_on=True)


def _divergent(fx):
    return _mego_binner(fx, minfasta=0, lengths_on=False)


def test_port_matches_semibin_exactly_on_same_embedding(genome_set):
    # The strongest claim: on the same embedding + markers, the MegoBin port and
    # genuine SemiBin produce identical bins.
    emb = genome_set.embedding(0)
    names = [str(c) for c in genome_set.contig_names]
    contig_dict = {n: _LenStr(int(length)) for n, length in zip(names, genome_set.lengths)}
    c2m = {str(k): list(v) for k, v in genome_set.contig_to_marker.items()}
    sb_labels = semibin_cluster_labels(
        emb, names, c2m, contig_dict, np.asarray(genome_set.lengths, float), 200000
    )
    mego_labels = _faithful(genome_set).cluster(emb)
    assert binner_ari(sb_labels, mego_labels) == pytest.approx(1.0)


def test_faithful_passes_against_genuine_semibin(genome_set):
    oracle = SemiBin2GetBestBinOracle(minfasta=200000, seeds=_SEEDS)
    report = FidelityHarness(binner_ari, z=2.0).run(
        candidate=_faithful(genome_set), oracle=oracle, fixture=genome_set
    )
    assert report.passed, report.to_markdown()
    assert report.vs_reference >= report.reference_self.floor


def test_divergent_fails_against_genuine_semibin(genome_set):
    oracle = SemiBin2GetBestBinOracle(minfasta=200000, seeds=_SEEDS)
    report = FidelityHarness(binner_ari, z=2.0).run(
        candidate=_divergent(genome_set), oracle=oracle, fixture=genome_set
    )
    assert not report.passed, report.to_markdown()
    assert report.vs_reference < report.reference_self.floor


def test_semibin_oracle_runs_are_stable(genome_set):
    # genuine SemiBin recovers exactly the planted genomes (6) every seed
    oracle = SemiBin2GetBestBinOracle(minfasta=200000, seeds=range(4))
    runs = oracle.runs(genome_set)
    for labels in runs:
        assert len(set(labels[labels >= 0])) == 6
