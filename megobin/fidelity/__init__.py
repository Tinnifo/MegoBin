"""Behavioral fidelity testing for MegoBin slot re-implementations.

A generic, tool-agnostic framework for checking that a MegoBin re-implementation
of an external tool is *behaviorally faithful* to the original ("oracle"),
calibrated to the oracle's own run-to-run variance and compared on
slot-appropriate invariants. SemiBin2 is the first concrete oracle; any future
tool follows the same recipe (subclass :class:`ReferenceOracle`).

See ``docs/06-fidelity/fidelity-harness.md`` for the extension recipe.
"""

from __future__ import annotations

from megobin.fidelity.metrics import (
    BpMatchDetails,
    array_close,
    binner_ari,
    binner_bp_match_f1,
    binner_bp_match_f1_details,
    set_agreement,
)
from megobin.fidelity.fixtures import (
    Fixture,
    GaussianBlobs,
    RealGenomeSet,
    SyntheticGenomeSet,
    canonical_genome_fixture,
    load_real_genome_set,
    make_blobs_fixture,
    make_genome_fixture,
)
from megobin.fidelity.harness import FidelityHarness
from megobin.fidelity.oracle import (
    MegoBinSelfBinnerOracle,
    OracleUnavailable,
    ReferenceOracle,
)
from megobin.fidelity.oracles_semibin2 import (
    SEMIBIN_N_TOTAL_MARKERS,
    SemiBin2BinLongOracle,
    SemiBin2GetBestBinOracle,
    parse_bins_to_labels,
    semibin_cluster_labels,
)
from megobin.fidelity.report import BandStats, DegenerateBandError, FidelityReport

__all__ = [
    # metrics
    "binner_ari",
    "binner_bp_match_f1",
    "binner_bp_match_f1_details",
    "BpMatchDetails",
    "set_agreement",
    "array_close",
    # report
    "BandStats",
    "FidelityReport",
    "DegenerateBandError",
    # harness + oracle
    "FidelityHarness",
    "ReferenceOracle",
    "MegoBinSelfBinnerOracle",
    "OracleUnavailable",
    "SemiBin2GetBestBinOracle",
    "SemiBin2BinLongOracle",
    "parse_bins_to_labels",
    "semibin_cluster_labels",
    "SEMIBIN_N_TOTAL_MARKERS",
    # fixtures
    "Fixture",
    "GaussianBlobs",
    "SyntheticGenomeSet",
    "RealGenomeSet",
    "make_blobs_fixture",
    "make_genome_fixture",
    "canonical_genome_fixture",
    "load_real_genome_set",
]
