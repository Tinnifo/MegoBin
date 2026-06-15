"""Reference oracles — the tool-agnostic extensibility seam.

A :class:`ReferenceOracle` wraps a reference (the source of "ground truth
behaviour") for one slot and answers three questions for the harness:

* ``runs(fixture)`` — K reference outputs **in the comparator's space**
  (cluster labels for a binner check). The K outputs must carry genuine
  run-to-run variance, or the calibration band is degenerate.
* ``truth(fixture)`` — the ground-truth array in the same space, or ``None``.
* ``candidate_run(candidate, fixture)`` — the candidate's output in the same
  space, produced the same way as a reference run.

Adding a new tool is one subclass. SemiBin2 oracles live in
``oracles_semibin2``; a future ComeBin oracle would be another subclass here
or alongside it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np


class OracleUnavailable(RuntimeError):
    """Raised when a live oracle's external tool/data is not installed."""


class ReferenceOracle(ABC):
    slot: str = "unknown"

    def __init__(self) -> None:
        self.fixture_id: str | None = None

    @abstractmethod
    def runs(self, fixture: Any) -> list[np.ndarray]:
        """K reference outputs in the comparator's space."""

    def truth(self, fixture: Any) -> np.ndarray | None:
        """Ground-truth array (default: the fixture's planted ``labels``)."""
        labels = getattr(fixture, "labels", None)
        return None if labels is None else np.asarray(labels)

    @abstractmethod
    def candidate_run(self, candidate: Any, fixture: Any) -> np.ndarray:
        """The candidate's output in the comparator's space."""


class MegoBinSelfBinnerOracle(ReferenceOracle):
    """Reference = the **faithful-config MegoBin binner** over K seeded embedding
    realisations.

    This is NOT a true external oracle. It calibrates *self-consistency* and
    *discriminating power* (a faithful candidate lands inside the band; a
    divergent one does not) with zero external dependencies — it is what gives
    the default suite teeth, proving the harness accepts a faithful config and
    rejects a divergent one. The genuine, non-circular check against real
    SemiBin2 lives in
    :class:`~megobin.fidelity.oracles_semibin2.SemiBin2GetBestBinOracle`, run on
    the real-genome fixture (``real_data``-gated).

    ``binner_factory(fixture) -> Binner`` builds the faithful binner; variance
    comes from ``fixture.embedding(seed)`` redrawing the per-contig noise.
    """

    slot = "binner"

    def __init__(
        self,
        binner_factory: Callable[[Any], Any],
        *,
        seeds: Sequence[int] = tuple(range(10)),
        candidate_seed: int = 99,
    ) -> None:
        super().__init__()
        self.binner_factory = binner_factory
        self.seeds = list(seeds)
        self.candidate_seed = candidate_seed

    def runs(self, fixture: Any) -> list[np.ndarray]:
        self.fixture_id = getattr(fixture, "fixture_id", None)
        return [
            self.binner_factory(fixture).cluster(fixture.embedding(seed))
            for seed in self.seeds
        ]

    def candidate_run(self, candidate: Any, fixture: Any) -> np.ndarray:
        return np.asarray(candidate.cluster(fixture.embedding(self.candidate_seed)))
