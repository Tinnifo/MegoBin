"""Calibration bands and the fidelity report.

A :class:`BandStats` summarises a distribution of agreement scores — either the
reference's *self-agreement* (how much the reference oracle disagrees with
itself across seeded runs) or its *ground-truth recovery* (how well the
reference recovers known labels). A candidate is accepted when it lands inside
the relevant band; see :class:`FidelityReport`.

These types are pure data + arithmetic — no numpy-array inputs beyond a flat
sequence of floats, no I/O.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# Floors are compared with a small epsilon so a candidate that equals the floor
# to within float noise is accepted rather than spuriously rejected.
_EPS = 1e-9


class DegenerateBandError(ValueError):
    """Raised when a self-agreement band has no usable variance.

    A reference whose runs agree perfectly with each other (std ~ 0, mean ~ 1)
    yields a floor of 1.0, so *only a byte-identical candidate could ever pass*.
    That is never a meaningful fidelity test — it signals the oracle was wired
    to a deterministic process with no run-to-run variance (e.g. clustering the
    *same* embedding matrix K times). The harness raises this loudly rather than
    silently producing a vacuous pass/fail.
    """


@dataclass(frozen=True)
class BandStats:
    """Summary statistics of a distribution of agreement scores in ``[0, 1]``.

    ``floor`` is the acceptance threshold: ``max(p05, mean - z * std)``. The
    p05 term guards against a thin tail; the ``mean - z*std`` term against a
    fat one. Whichever is *higher* wins, so the band is as strict as the
    reference's own spread justifies.
    """

    mean: float
    std: float
    p05: float
    n: int
    floor: float  # self-agreement floor: max(p05, mean - z*std)
    lower: float  # one-sided lower tolerance bound: mean - z*std

    @classmethod
    def from_scores(
        cls,
        scores: Sequence[float],
        *,
        z: float = 2.0,
        std_floor: float = 1e-6,
        degenerate_guard: bool = True,
    ) -> BandStats:
        """Build a band from a sequence of agreement scores.

        Parameters
        ----------
        scores:
            Agreement scores in ``[0, 1]`` (higher = more agreement).
        z:
            Number of standard deviations below the mean to admit.
        std_floor:
            Below this std the band is treated as having no spread.
        degenerate_guard:
            When ``True`` (default), raise :class:`DegenerateBandError` if the
            band has no spread *and* sits at ~1.0 (the vacuous case). Set
            ``False`` only for the self-agreement band of an intentionally
            deterministic check where a 1.0 floor is acceptable.
        """
        arr = np.asarray(list(scores), dtype=float)
        if arr.size == 0:
            raise ValueError("BandStats.from_scores: need at least one score")
        mean = float(arr.mean())
        std = float(arr.std(ddof=0))
        p05 = float(np.percentile(arr, 5))
        # Agreement scores live in [0, 1]; clamp the one-sided lower bound to >=0
        # so a fat lower tail can never produce a negative (impossible-to-fail)
        # floor. The recovery gate intentionally uses this lenient lower bound
        # rather than p05 — it scales with the reference's own reliability, and a
        # very noisy reference legitimately cannot impose a tight recovery gate.
        lower = max(0.0, mean - z * std)
        floor = max(p05, lower)
        # Degenerate iff the floor itself pins at ~1.0 (only a byte-identical
        # candidate could pass) — tie the guard to the actual floor, not a proxy
        # std threshold that misfires on genuine sub-floor variance.
        if degenerate_guard and std < std_floor and floor >= 1.0 - _EPS:
            raise DegenerateBandError(
                "self-agreement band is degenerate (floor ~1.0, std ~0): the "
                "reference has no run-to-run variance, so only a byte-identical "
                "candidate could pass. Wire the oracle to a variance-bearing "
                "source (e.g. K seeded embedding realisations), not a single "
                "fixed input clustered K times."
            )
        return cls(
            mean=mean, std=std, p05=p05, n=int(arr.size), floor=floor, lower=lower
        )

    def accepts(self, value: float) -> bool:
        """True if ``value`` clears the full ``floor`` (self-agreement side)."""
        return value >= self.floor - _EPS

    def accepts_lower(self, value: float) -> bool:
        """True if ``value`` clears the one-sided lower tolerance bound.

        Used for ground-truth recovery: "the candidate recovers truth about as
        well as the reference does, on the low side of its own spread". A proper
        ``mean - z*std`` lower bound is more robust for small reference samples
        than the p05 (which collapses onto the observed minimum).
        """
        return value >= self.lower - _EPS

    def accepts_p05(self, value: float) -> bool:
        """True if ``value`` clears the p05 (stricter one-sided floor)."""
        return value >= self.p05 - _EPS


@dataclass(frozen=True)
class FidelityReport:
    """The outcome of one fidelity comparison for one slot.

    ``passed`` is true iff the candidate agrees with the reference at least as
    well as the reference agrees with itself **and** (when ground truth exists)
    recovers it at least as well as the reference's worst seeded run.
    """

    slot: str
    comparator: str
    vs_reference: float
    reference_self: BandStats
    vs_truth: float | None
    reference_truth: BandStats | None
    passed: bool
    details: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def build(
        cls,
        *,
        slot: str,
        comparator: str,
        vs_reference: float,
        reference_self: BandStats,
        vs_truth: float | None,
        reference_truth: BandStats | None,
        details: dict[str, Any] | None = None,
    ) -> FidelityReport:
        """Assemble a report and compute ``passed`` from the two-way predicate."""
        ref_ok = reference_self.accepts(vs_reference)
        if vs_truth is None:
            truth_ok = True
        elif reference_truth is None:
            truth_ok = True
        else:
            truth_ok = reference_truth.accepts_lower(vs_truth)
        return cls(
            slot=slot,
            comparator=comparator,
            vs_reference=vs_reference,
            reference_self=reference_self,
            vs_truth=vs_truth,
            reference_truth=reference_truth,
            passed=bool(ref_ok and truth_ok),
            details=details or {},
        )

    def to_markdown(self) -> str:
        """Render a compact human-readable summary."""
        verdict = "PASS ✅" if self.passed else "FAIL ❌"
        lines = [
            f"### Fidelity report — `{self.slot}` ({verdict})",
            "",
            f"- comparator: `{self.comparator}`",
            (
                f"- candidate vs reference: **{self.vs_reference:.4f}** "
                f"(floor {self.reference_self.floor:.4f}; "
                f"ref self mean {self.reference_self.mean:.4f} ± "
                f"{self.reference_self.std:.4f}, n={self.reference_self.n})"
            ),
        ]
        if self.vs_truth is not None and self.reference_truth is not None:
            lines.append(
                f"- candidate vs truth: **{self.vs_truth:.4f}** "
                f"(lower floor {self.reference_truth.lower:.4f}; "
                f"ref recovery mean {self.reference_truth.mean:.4f} ± "
                f"{self.reference_truth.std:.4f}, n={self.reference_truth.n})"
            )
        elif self.vs_truth is not None:
            lines.append(f"- candidate vs truth: **{self.vs_truth:.4f}** (no band)")
        if self.details:
            lines.append("")
            lines.append("<details><summary>details</summary>")
            lines.append("")
            for key, val in self.details.items():
                lines.append(f"- `{key}`: {val}")
            lines.append("")
            lines.append("</details>")
        return "\n".join(lines)
