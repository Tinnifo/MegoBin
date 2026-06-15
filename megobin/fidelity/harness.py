"""The fidelity harness — slot-agnostic orchestration.

The harness owns one algorithm: gather K reference runs, build the
self-agreement band and (if truth exists) the recovery band, run the candidate,
and decide acceptance. *What a run is* and *where its variance comes from* live
entirely in the :class:`~megobin.fidelity.oracle.ReferenceOracle`, so the same
harness validates a binner (or any future slot) by swapping the oracle +
comparator.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from megobin.fidelity.oracle import ReferenceOracle
from megobin.fidelity.report import BandStats, FidelityReport

# A comparator is ``(a, b) -> float in [0, 1]``. The harness always calls it with
# the reference (or truth) in slot ``a`` and the candidate in slot ``b``, so an
# asymmetric comparator (e.g. a directional recall) is scored under one fixed
# role convention. The self-agreement band is reference-vs-reference, so it is
# symmetric by construction regardless.
Comparator = Callable[[np.ndarray, np.ndarray], float]


class FidelityHarness:
    """Calibrate a band from a reference's own variance, then accept-or-reject.

    Parameters
    ----------
    comparator:
        ``(a, b) -> float in [0, 1]`` agreement in the oracle's output space
        (e.g. ``binner_ari``, or ``binner_bp_match_f1`` with lengths bound).
    z:
        Std-multiplier for the self-agreement floor ``max(p05, mean - z*std)``.
    aggregate:
        How to reduce the candidate-vs-K-references scores (default median).
    """

    def __init__(
        self,
        comparator: Comparator,
        *,
        comparator_name: str | None = None,
        z: float = 2.0,
        std_floor: float = 1e-6,
        aggregate: Callable[[Any], Any] = np.median,
        degenerate_guard: bool = True,
    ) -> None:
        self.comparator = comparator
        self.comparator_name = str(
            comparator_name or getattr(comparator, "__name__", "comparator")
        )
        self.z = z
        self.std_floor = std_floor
        self.aggregate = aggregate
        self.degenerate_guard = degenerate_guard

    def run(
        self, *, candidate: Any, oracle: ReferenceOracle, fixture: Any
    ) -> FidelityReport:
        ref_runs = oracle.runs(fixture)
        if len(ref_runs) < 2:
            raise ValueError(
                "need >= 2 reference runs to form a self-agreement band; "
                f"got {len(ref_runs)}"
            )
        shapes = [np.shape(r) for r in ref_runs]
        if len(set(shapes)) > 1:
            raise ValueError(
                f"reference runs disagree in shape (got {shapes}); an oracle "
                "run dropped or added rows."
            )

        self_scores = [
            self.comparator(ref_runs[i], ref_runs[j])
            for i in range(len(ref_runs))
            for j in range(i + 1, len(ref_runs))
        ]
        reference_self = BandStats.from_scores(
            self_scores,
            z=self.z,
            std_floor=self.std_floor,
            degenerate_guard=self.degenerate_guard,
        )

        truth = oracle.truth(fixture)
        reference_truth = None
        if truth is not None:
            truth_scores = [self.comparator(truth, r) for r in ref_runs]
            # Recovery is a one-sided floor; never guard it as degenerate.
            reference_truth = BandStats.from_scores(
                truth_scores, z=self.z, std_floor=self.std_floor, degenerate_guard=False
            )

        cand_run = oracle.candidate_run(candidate, fixture)
        # Reference always occupies slot A (matching the truth side and the
        # bp-F1 "A = reference, B = candidate" convention) so an asymmetric
        # comparator is scored under the same role convention as the band.
        vs_reference = float(
            self.aggregate([self.comparator(r, cand_run) for r in ref_runs])
        )
        vs_truth = float(self.comparator(truth, cand_run)) if truth is not None else None

        details: dict[str, Any] = {
            "n_reference_runs": len(ref_runs),
            "reference_self_mean": round(reference_self.mean, 4),
            "reference_self_floor": round(reference_self.floor, 4),
            "fixture_id": getattr(fixture, "fixture_id", None),
        }
        if reference_truth is not None:
            details["reference_truth_lower"] = round(reference_truth.lower, 4)

        return FidelityReport.build(
            slot=oracle.slot,
            comparator=self.comparator_name,
            vs_reference=vs_reference,
            reference_self=reference_self,
            vs_truth=vs_truth,
            reference_truth=reference_truth,
            details=details,
        )
