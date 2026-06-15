"""Shared fixtures and binner factories for the fidelity suite."""

from __future__ import annotations

import numpy as np
import pytest

from megobin.binners.semibin_2 import DBSCANEnsembleBinner
from megobin.fidelity import SEMIBIN_N_TOTAL_MARKERS
from megobin.fidelity.fixtures import (
    SyntheticGenomeSet,
    canonical_genome_fixture,
)


@pytest.fixture(scope="session")
def genome_set() -> SyntheticGenomeSet:
    return canonical_genome_fixture()


def faithful_binner(fx: SyntheticGenomeSet) -> DBSCANEnsembleBinner:
    """SemiBin-faithful config: bp gating on, length-weighting on.

    ``n_total_markers=107`` matches SemiBin's hardcoded recall divisor so the
    comparison against genuine SemiBin is apples-to-apples.
    """
    return DBSCANEnsembleBinner(
        minfasta=200000,
        contig_names=fx.contig_names,
        contig_lengths=fx.lengths,
        contig_to_marker=fx.contig_to_marker,
        n_total_markers=SEMIBIN_N_TOTAL_MARKERS,
    )


def divergent_binner(fx: SyntheticGenomeSet) -> DBSCANEnsembleBinner:
    """Divergent config: minfasta off (tiny singletons) + length-weighting off."""
    return DBSCANEnsembleBinner(
        minfasta=0,
        contig_names=fx.contig_names,
        contig_lengths=None,
        contig_to_marker=fx.contig_to_marker,
        n_total_markers=SEMIBIN_N_TOTAL_MARKERS,
    )


def labels_array(binner: DBSCANEnsembleBinner, embedding: np.ndarray) -> np.ndarray:
    return binner.cluster(embedding)
