"""Unit tests for src/features/abundance.py.

The real pysam wheel is heavy and optional (HPC-only — see environment.yml).
These tests inject a minimal fake `pysam` into ``sys.modules`` so the BAM
code path is exercised without needing a compiled pysam.

The fake mirrors the two methods ``compute_abundance`` calls:
  - ``get_reference_length(contig) -> int``
  - ``count_coverage(contig, start, stop) -> (A, C, G, T)`` arrays.

Scope: what the current implementation actually guarantees — output shape,
column layout, multi-sample ordering, variance math, and NaN-free output.
Edge trimming, pseudocounts, and short-contig filtering are **not**
implemented today; asserting them would bake wishful behavior into the
test suite, so those cases are deliberately omitted.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Fake pysam
# ---------------------------------------------------------------------------


class _FakeAlignmentFile:
    """Stand-in for ``pysam.AlignmentFile`` configured per-test.

    Each instance is driven by two dicts keyed by contig name:
      ``reference_lengths`` — int length per contig.
      ``depths``            — per-base total depth (A+C+G+T) per contig.
                              Entries may be a scalar (broadcast) or an array
                              matching the contig length.

    The alignment file is keyed by its path string so a single fake module
    instance can serve multiple BAMs with distinct behavior.
    """

    _registry: dict[str, "_FakeAlignmentFile"] = {}

    def __init__(self, path: str, mode: str = "rb"):
        cfg = self._registry[path]
        self._lengths = cfg._lengths
        self._depths = cfg._depths

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def get_reference_length(self, contig: str) -> int:
        return self._lengths.get(contig, 0)

    def count_coverage(self, contig: str, start: int = 0, stop: int | None = None):
        length = self._lengths.get(contig, 0)
        if stop is None:
            stop = length
        window = stop - start

        depth = self._depths.get(contig, 0)
        if np.isscalar(depth):
            total = np.full(window, float(depth), dtype=np.float64)
        else:
            total = np.asarray(depth, dtype=np.float64)[start:stop]

        # Split evenly across A/C/G/T so sum equals total depth.
        quarter = total / 4.0
        return quarter, quarter, quarter, quarter


def _register_fake_bam(
    path: str, lengths: dict[str, int], depths: dict[str, float | np.ndarray]
) -> None:
    cfg = object.__new__(_FakeAlignmentFile)
    cfg._lengths = lengths
    cfg._depths = depths
    _FakeAlignmentFile._registry[path] = cfg


@pytest.fixture
def fake_pysam(monkeypatch):
    """Install a throwaway fake ``pysam`` for the duration of the test."""
    _FakeAlignmentFile._registry.clear()
    fake = types.ModuleType("pysam")
    fake.AlignmentFile = _FakeAlignmentFile
    monkeypatch.setitem(sys.modules, "pysam", fake)
    yield
    _FakeAlignmentFile._registry.clear()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestComputeAbundanceShape:
    def test_output_shape_single_bam(self, fake_pysam):
        from src.features.abundance import compute_abundance

        _register_fake_bam(
            "/tmp/one.bam",
            lengths={"c0": 100, "c1": 200, "c2": 150},
            depths={"c0": 5.0, "c1": 3.0, "c2": 7.0},
        )
        out = compute_abundance([Path("/tmp/one.bam")], ["c0", "c1", "c2"])
        assert out.shape == (3, 2)

    def test_output_shape_multi_bam(self, fake_pysam):
        from src.features.abundance import compute_abundance

        for path, depth in [("/tmp/a.bam", 4.0), ("/tmp/b.bam", 8.0), ("/tmp/c.bam", 2.0)]:
            _register_fake_bam(
                path,
                lengths={"c0": 100, "c1": 200},
                depths={"c0": depth, "c1": depth},
            )
        out = compute_abundance(
            [Path("/tmp/a.bam"), Path("/tmp/b.bam"), Path("/tmp/c.bam")],
            ["c0", "c1"],
        )
        assert out.shape == (2, 6)


class TestColumnLayout:
    def test_mean_var_interleaved_per_bam(self, fake_pysam):
        """Columns are [mean_0, var_0, mean_1, var_1, ...]."""
        from src.features.abundance import compute_abundance

        _register_fake_bam(
            "/tmp/uniform.bam",
            lengths={"c0": 100},
            depths={"c0": 4.0},
        )
        # Non-uniform depth → non-zero variance on BAM #2.
        non_uniform = np.concatenate([np.full(50, 2.0), np.full(50, 10.0)])
        _register_fake_bam(
            "/tmp/bumpy.bam",
            lengths={"c0": 100},
            depths={"c0": non_uniform},
        )

        out = compute_abundance(
            [Path("/tmp/uniform.bam"), Path("/tmp/bumpy.bam")], ["c0"]
        )

        assert out[0, 0] == pytest.approx(4.0)
        assert out[0, 1] == pytest.approx(0.0, abs=1e-12)
        assert out[0, 2] == pytest.approx(6.0)
        assert out[0, 3] == pytest.approx(non_uniform.var())

    def test_bam_order_preserved(self, fake_pysam):
        """Reordering the BAM list permutes the column pairs accordingly."""
        from src.features.abundance import compute_abundance

        _register_fake_bam(
            "/tmp/a.bam", lengths={"c0": 50}, depths={"c0": 1.0}
        )
        _register_fake_bam(
            "/tmp/b.bam", lengths={"c0": 50}, depths={"c0": 9.0}
        )

        ab = compute_abundance(
            [Path("/tmp/a.bam"), Path("/tmp/b.bam")], ["c0"]
        )
        ba = compute_abundance(
            [Path("/tmp/b.bam"), Path("/tmp/a.bam")], ["c0"]
        )

        assert ab[0, 0] == pytest.approx(1.0)
        assert ab[0, 2] == pytest.approx(9.0)
        assert ba[0, 0] == pytest.approx(9.0)
        assert ba[0, 2] == pytest.approx(1.0)


class TestCoverageMath:
    def test_uniform_coverage_has_zero_variance(self, fake_pysam):
        from src.features.abundance import compute_abundance

        _register_fake_bam(
            "/tmp/uniform.bam",
            lengths={"c0": 500, "c1": 500},
            depths={"c0": 10.0, "c1": 25.0},
        )
        out = compute_abundance([Path("/tmp/uniform.bam")], ["c0", "c1"])

        assert out[0, 0] == pytest.approx(10.0)
        assert out[0, 1] == pytest.approx(0.0, abs=1e-12)
        assert out[1, 0] == pytest.approx(25.0)
        assert out[1, 1] == pytest.approx(0.0, abs=1e-12)

    def test_empty_bam_yields_zero_coverage(self, fake_pysam):
        """No aligned reads → depth 0 everywhere → no NaN, no Inf."""
        from src.features.abundance import compute_abundance

        _register_fake_bam(
            "/tmp/empty.bam",
            lengths={"c0": 100, "c1": 200},
            depths={"c0": 0.0, "c1": 0.0},
        )
        out = compute_abundance([Path("/tmp/empty.bam")], ["c0", "c1"])

        assert np.all(out == 0.0)
        assert not np.isnan(out).any()
        assert not np.isinf(out).any()

    def test_zero_length_contig_skipped(self, fake_pysam):
        """Contigs with reference length 0 must not crash and must leave
        their row at its zero-initialized value."""
        from src.features.abundance import compute_abundance

        _register_fake_bam(
            "/tmp/mixed.bam",
            lengths={"real": 100, "empty": 0},
            depths={"real": 3.0, "empty": 0.0},
        )
        out = compute_abundance([Path("/tmp/mixed.bam")], ["real", "empty"])

        assert out[0, 0] == pytest.approx(3.0)
        assert out[0, 1] == pytest.approx(0.0, abs=1e-12)
        assert np.all(out[1] == 0.0)
        assert not np.isnan(out).any()


class TestContigOrdering:
    def test_row_order_matches_input_names(self, fake_pysam):
        from src.features.abundance import compute_abundance

        depths = {"c0": 1.0, "c1": 2.0, "c2": 3.0, "c3": 4.0}
        _register_fake_bam(
            "/tmp/one.bam",
            lengths={c: 100 for c in depths},
            depths=depths,
        )

        names = ["c3", "c0", "c2", "c1"]
        out = compute_abundance([Path("/tmp/one.bam")], names)
        for i, name in enumerate(names):
            assert out[i, 0] == pytest.approx(depths[name])


class TestNoNaN:
    def test_no_nan_in_combined_output(self, fake_pysam):
        """Combination of varied contig lengths, depths, empty contigs and
        multiple BAMs must not produce NaN or Inf anywhere."""
        from src.features.abundance import compute_abundance

        for path, depth in [("/tmp/a.bam", 0.0), ("/tmp/b.bam", 5.5)]:
            _register_fake_bam(
                path,
                lengths={"c0": 100, "c1": 300, "c2": 0},
                depths={"c0": depth, "c1": depth, "c2": 0.0},
            )
        out = compute_abundance(
            [Path("/tmp/a.bam"), Path("/tmp/b.bam")], ["c0", "c1", "c2"]
        )
        assert not np.isnan(out).any()
        assert not np.isinf(out).any()
