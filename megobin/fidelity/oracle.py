"""Reference oracles — the tool-agnostic extensibility seam.

A :class:`ReferenceOracle` wraps a reference (the source of "ground truth
behaviour") for one slot and answers three questions for the harness:

* ``runs(fixture)`` — K reference outputs **in the comparator's space**
  (cluster labels for a binner check, embeddings for a geometry check). The K
  outputs must carry genuine run-to-run variance, or the calibration band is
  degenerate.
* ``truth(fixture)`` — the ground-truth array in the same space, or ``None``.
* ``candidate_run(candidate, fixture)`` — the candidate's output in the same
  space, produced the same way as a reference run.

Adding a new tool is one subclass. SemiBin2 oracles live in
``oracles_semibin2``; a future ComeBin oracle would be another subclass here
or alongside it.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np


class OracleUnavailable(RuntimeError):
    """Raised when a live oracle's external tool/data is not installed."""


class StaleCacheError(RuntimeError):
    """Raised when a cached oracle's provenance no longer matches the fixture."""


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
    divergent one does not) with zero external dependencies — the default-path
    workhorse. For a genuine, non-circular oracle use
    :class:`~megobin.fidelity.oracles_semibin2.SemiBin2GetBestBinOracle`.

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


class CachedReferenceOracle(ReferenceOracle):
    """Reference runs loaded from disk — no live tool at test time.

    Loads ``run_*.csv`` (``contig_name,label``) + ``provenance.json`` from a
    cache directory produced by ``megobin.fidelity.refresh_cache``. The cache is
    text (survives ``.gitignore``) and can be committed. ``provenance.json``
    records the fixture manifest hash; if it no longer matches the live fixture
    the oracle raises :class:`StaleCacheError` rather than comparing against a
    stale reference. This is how the default suite gets a genuine SemiBin2
    comparison without SemiBin installed.
    """

    slot = "binner"

    def __init__(
        self,
        cache_dir: str | Path,
        *,
        candidate_seed: int = 99,
        check_manifest: bool = True,
    ) -> None:
        super().__init__()
        self.cache_dir = Path(cache_dir)
        self.candidate_seed = candidate_seed
        self.check_manifest = check_manifest

    @property
    def provenance(self) -> dict[str, Any]:
        path = self.cache_dir / "provenance.json"
        if not path.exists():
            raise OracleUnavailable(f"no cache provenance at {path}")
        with open(path) as fh:
            return dict(json.load(fh))

    def runs(self, fixture: Any) -> list[np.ndarray]:
        import pandas as pd

        prov = self.provenance
        self.fixture_id = prov.get("fixture_id")
        if self.check_manifest:
            from megobin.fidelity.fixtures import _manifest_sha256

            manifest = getattr(fixture, "manifest", None)
            if manifest is None:
                raise StaleCacheError(
                    f"cannot verify cache at {self.cache_dir}: fixture exposes "
                    "no manifest to hash; pass check_manifest=False to bypass."
                )
            want = _manifest_sha256(manifest)
            got = prov.get("fixture_manifest_sha256")
            if got is None or want != got:
                raise StaleCacheError(
                    f"cache at {self.cache_dir} was generated for a different "
                    f"fixture (manifest {got} != {want}); regenerate with "
                    "`python -m megobin.fidelity.refresh_cache`."
                )
        names = [str(c) for c in fixture.contig_names]
        runs: list[np.ndarray] = []
        for csv_path in sorted(self.cache_dir.glob("run_*.csv")):
            df = pd.read_csv(csv_path)
            mapping = dict(
                zip(df["contig_name"].astype(str), df["label"].astype(int))
            )
            runs.append(np.array([mapping[n] for n in names], dtype=np.int64))
        if not runs:
            raise OracleUnavailable(f"no run_*.csv files in {self.cache_dir}")
        return runs

    def candidate_run(self, candidate: Any, fixture: Any) -> np.ndarray:
        return np.asarray(candidate.cluster(fixture.embedding(self.candidate_seed)))


class EncoderTrainingOracle(ReferenceOracle):
    """Encoder-slot oracle: variance from the **training seed**.

    Each reference run trains the reference encoder with a fresh seed, encodes
    the fixture features, and clusters the embedding through a *fixed* downstream
    binner (``cluster_fn``) — so everything lives in label space and the
    comparator is :func:`~megobin.fidelity.metrics.binner_ari`, exactly as for
    the binner slot. This routes the encoder through its real contract (does the
    embedding induce the right partition?) rather than comparing raw embeddings,
    which are only defined up to an isometry.

    ``reference_embed(fixture, seed) -> (N, d)`` produces a reference embedding;
    the ``candidate`` passed to :meth:`candidate_run` is itself a callable
    ``(fixture, seed) -> (N, d)`` (a faithful re-implementation's train+encode,
    or a perturbed one).
    """

    slot = "encoder"

    def __init__(
        self,
        reference_embed: Callable[[Any, int], np.ndarray],
        cluster_fn: Callable[[np.ndarray], np.ndarray],
        *,
        seeds: Sequence[int] = tuple(range(8)),
        candidate_seed: int = 99,
    ) -> None:
        super().__init__()
        self.reference_embed = reference_embed
        self.cluster_fn = cluster_fn
        self.seeds = list(seeds)
        self.candidate_seed = candidate_seed

    def runs(self, fixture: Any) -> list[np.ndarray]:
        self.fixture_id = getattr(fixture, "fixture_id", None)
        return [
            np.asarray(self.cluster_fn(self.reference_embed(fixture, seed)))
            for seed in self.seeds
        ]

    def candidate_run(
        self, candidate: Callable[[Any, int], np.ndarray], fixture: Any
    ) -> np.ndarray:
        return np.asarray(self.cluster_fn(candidate(fixture, self.candidate_seed)))
