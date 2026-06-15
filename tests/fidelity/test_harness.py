"""Band math and the acceptance predicate (the harness orchestrator is tested
in test_binner_fidelity.py once oracles exist)."""

import numpy as np
import pytest

from megobin.fidelity.report import (
    BandStats,
    DegenerateBandError,
    FidelityReport,
)


# ---- BandStats -------------------------------------------------------------


def test_from_scores_basic_stats():
    scores = [0.8, 0.85, 0.9, 0.95, 1.0]
    band = BandStats.from_scores(scores, z=2.0)
    assert band.n == 5
    assert band.mean == pytest.approx(np.mean(scores))
    assert band.std == pytest.approx(np.std(scores))
    assert band.p05 == pytest.approx(np.percentile(scores, 5))
    assert band.floor == pytest.approx(max(band.p05, band.mean - 2.0 * band.std))


def test_accepts_and_accepts_p05():
    band = BandStats.from_scores([0.7, 0.8, 0.9], z=1.0)
    assert band.accepts(band.floor)
    assert band.accepts(band.floor + 0.01)
    assert not band.accepts(band.floor - 0.01)
    assert band.accepts_p05(band.p05)
    assert not band.accepts_p05(band.p05 - 0.01)


def test_degenerate_band_raises():
    with pytest.raises(DegenerateBandError):
        BandStats.from_scores([1.0] * 10, z=2.0)


def test_degenerate_guard_can_be_disabled():
    band = BandStats.from_scores([1.0] * 10, z=2.0, degenerate_guard=False)
    assert band.floor == pytest.approx(1.0)


def test_near_perfect_band_with_real_variance_is_not_degenerate():
    # Genuine sub-floor variance (floor < 1.0) must NOT trip the guard, even
    # though std < std_floor.
    band = BandStats.from_scores([1.0, 1.0, 0.999998], z=2.0)
    assert band.floor < 1.0


def test_truth_lower_floor_is_clamped_nonnegative():
    # A fat lower tail must not make the recovery gate vacuous: lower >= 0.
    band = BandStats.from_scores([0.9, 0.9, 0.9, 0.0], z=2.0)
    assert band.lower >= 0.0
    assert not band.accepts_lower(-0.01)


def test_zero_variance_below_one_is_not_degenerate():
    # A tight band at 0.5 is a valid (if strict) band, not the vacuous case.
    band = BandStats.from_scores([0.5] * 8, z=2.0)
    assert band.floor == pytest.approx(0.5)


def test_empty_scores_raises():
    with pytest.raises(ValueError, match="at least one"):
        BandStats.from_scores([])


# ---- FidelityReport predicate ----------------------------------------------


def _band(scores, **kw):
    return BandStats.from_scores(scores, **kw)


def test_report_passes_when_inside_both_bands():
    self_band = _band([0.8, 0.85, 0.9, 0.95], z=2.0)
    truth_band = _band([0.7, 0.75, 0.8], z=2.0)
    rep = FidelityReport.build(
        slot="binner",
        comparator="binner_ari",
        vs_reference=0.92,
        reference_self=self_band,
        vs_truth=0.78,
        reference_truth=truth_band,
    )
    assert rep.passed


def test_report_fails_on_reference_band():
    self_band = _band([0.9, 0.92, 0.94, 0.96], z=1.0)
    rep = FidelityReport.build(
        slot="binner",
        comparator="binner_ari",
        vs_reference=0.2,  # well below the floor
        reference_self=self_band,
        vs_truth=None,
        reference_truth=None,
    )
    assert not rep.passed


def test_report_fails_on_truth_band_even_if_reference_ok():
    self_band = _band([0.8, 0.85, 0.9], z=2.0)
    truth_band = _band([0.7, 0.75, 0.8], z=2.0)
    rep = FidelityReport.build(
        slot="binner",
        comparator="binner_ari",
        vs_reference=0.95,  # inside reference band
        reference_self=self_band,
        vs_truth=0.1,  # below truth p05
        reference_truth=truth_band,
    )
    assert not rep.passed


def test_report_without_truth_uses_reference_only():
    self_band = _band([0.8, 0.85, 0.9], z=2.0)
    rep = FidelityReport.build(
        slot="binner",
        comparator="binner_ari",
        vs_reference=0.9,
        reference_self=self_band,
        vs_truth=None,
        reference_truth=None,
    )
    assert rep.passed
    assert "PASS" in rep.to_markdown()
