"""Shared fixtures and binner factories for the fidelity suite."""

from __future__ import annotations

import numpy as np
import pytest

from megobin.binners.semibin_2 import DBSCANEnsembleBinner
from megobin.fidelity import SEMIBIN_N_TOTAL_MARKERS
from megobin.fidelity.fixtures import (
    EncoderFixture,
    GaussianBlobs,
    SyntheticGenomeSet,
    canonical_genome_fixture,
    make_blobs_fixture,
    make_encoder_fixture,
)


@pytest.fixture(scope="session")
def genome_set() -> SyntheticGenomeSet:
    # Shared with the committed SemiBin2 cache via canonical_genome_fixture().
    return canonical_genome_fixture()


@pytest.fixture(scope="session")
def blobs() -> GaussianBlobs:
    return make_blobs_fixture()


@pytest.fixture(scope="session")
def encoder_set() -> EncoderFixture:
    # Tuned for a robust, non-degenerate band: trained recovery ~0.97 with
    # enough seed-to-seed spread that the self band is non-degenerate.
    return make_encoder_fixture(n_genomes=3, contigs_per_genome=40, noise=0.04)


def faithful_binner(fx: SyntheticGenomeSet) -> DBSCANEnsembleBinner:
    """SemiBin-faithful config: bp gating on, length-weighting on.

    ``n_total_markers=107`` matches SemiBin's hardcoded recall divisor so the
    comparison against the committed SemiBin cache is apples-to-apples.
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
