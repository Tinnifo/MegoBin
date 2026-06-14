"""SemiBin2 reference oracles — the first concrete tool wrappers.

Two flavours, both subclassing :class:`~megobin.fidelity.oracle.ReferenceOracle`:

* :class:`SemiBin2GetBestBinOracle` — **in-process, genuine, laptop**. Imports
  SemiBin's real ``get_best_bin`` and replicates its exact DBSCAN sweep + greedy
  loop, run on the *same* embeddings + injected marker dict the MegoBin binner
  receives. Non-circular (the reference is genuine SemiBin code), needs no
  BAM/HMM/real-DNA, runs in seconds. This is the true-oracle binner check.
* :class:`SemiBin2BinLongOracle` — **subprocess template** (PR6). Runs
  ``SemiBin2 bin_long`` and parses its per-bin FASTAs back to labels. Needs real
  contigs + abundance, so it is marker-gated; it is the pattern a future tool
  oracle (e.g. ComeBin) copies.

The SemiBin import is lazy so importing this module never requires SemiBin to be
installed; the oracle raises :class:`OracleUnavailable` only when actually run.
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.cluster import dbscan
from sklearn.neighbors import kneighbors_graph

from megobin.fidelity.oracle import OracleUnavailable, ReferenceOracle

# SemiBin's exact long-read eps grid and parameters (verified against
# SemiBin 2.3.0 SemiBin/long_read_cluster.py::cluster_long_read).
_EPS_GRID = (0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55)
_MIN_SAMPLES = 5
_K_NEIGHBOURS = 200
# get_best_bin hardcodes recall = |unique markers| / 107, so a candidate must be
# built with n_total_markers=107 for an apples-to-apples comparison.
SEMIBIN_N_TOTAL_MARKERS = 107


def _import_get_best_bin() -> Any:
    try:
        from SemiBin.long_read_cluster import get_best_bin
    except ImportError as exc:  # pragma: no cover - exercised only without SemiBin
        raise OracleUnavailable(
            "SemiBin is not installed; `pip install -e '.[fidelity]'` (or "
            "`pip install SemiBin>=2.3`) to run the genuine SemiBin2 oracle."
        ) from exc
    return get_best_bin


def semibin_cluster_labels(
    embedding: np.ndarray,
    contig_names: Sequence[str],
    contig_to_marker: dict[str, list[str]],
    contig_dict: dict[str, Any],
    length_weight: np.ndarray,
    minfasta: int,
) -> np.ndarray:
    """Genuine SemiBin clustering of ``embedding`` -> labels (input order).

    A faithful transcription of ``cluster_long_read``'s post-embedding flow
    (kNN graph -> 12-eps DBSCAN sweep with bp ``sample_weight`` -> greedy loop
    calling the real ``get_best_bin``), so the only behaviour under test is the
    candidate's, not ours.
    """
    get_best_bin = _import_get_best_bin()
    n = embedding.shape[0]
    dist_matrix = kneighbors_graph(
        embedding,
        n_neighbors=min(_K_NEIGHBOURS, n - 1),
        mode="distance",
        p=2,
        n_jobs=1,
    )
    results: list[list[int]] = []
    for eps in _EPS_GRID:
        _, labels = dbscan(
            dist_matrix,
            eps=eps,
            min_samples=_MIN_SAMPLES,
            n_jobs=1,
            metric="precomputed",
            sample_weight=length_weight,
        )
        results.append(labels.tolist())

    contig_list = [str(c) for c in contig_names]
    extracted: list[list[str]] = []
    while sum(len(contig_dict[c]) for c in contig_list) >= minfasta:
        if len(contig_list) == 1:
            extracted.append(list(contig_list))
            break
        max_bin = get_best_bin(
            results, contig_to_marker, contig_list, contig_dict, minfasta
        )
        if not max_bin:
            break
        extracted.append(max_bin)
        for clustered in max_bin:
            ix = contig_list.index(clustered)
            contig_list.pop(ix)
            for row in results:
                row.pop(ix)

    contig2ix = {c: i for i, cs in enumerate(extracted) for c in cs}
    return np.array(
        [contig2ix.get(str(c), -1) for c in contig_names], dtype=np.int64
    )


class _LenStr:
    """Length-only stand-in so ``len(contig_dict[name])`` returns bp without
    materialising multi-kb DNA strings."""

    __slots__ = ("_n",)

    def __init__(self, n: int) -> None:
        self._n = n

    def __len__(self) -> int:
        return self._n


class SemiBin2GetBestBinOracle(ReferenceOracle):
    """Genuine, in-process SemiBin2 binner oracle (gated by SemiBin install)."""

    slot = "binner"

    def __init__(
        self,
        *,
        minfasta: int = 200000,
        seeds: Sequence[int] = tuple(range(10)),
        candidate_seed: int = 99,
    ) -> None:
        super().__init__()
        self.minfasta = minfasta
        self.seeds = list(seeds)
        self.candidate_seed = candidate_seed

    def _contig_dict(self, fixture: Any) -> dict[str, _LenStr]:
        return {
            str(name): _LenStr(int(length))
            for name, length in zip(fixture.contig_names, fixture.lengths)
        }

    def runs(self, fixture: Any) -> list[np.ndarray]:
        self.fixture_id = getattr(fixture, "fixture_id", None)
        names = [str(c) for c in fixture.contig_names]
        contig_dict = self._contig_dict(fixture)
        c2m = {str(k): list(v) for k, v in fixture.contig_to_marker.items()}
        length_weight = np.asarray(fixture.lengths, dtype=float)
        return [
            semibin_cluster_labels(
                fixture.embedding(seed),
                names,
                c2m,
                contig_dict,
                length_weight,
                self.minfasta,
            )
            for seed in self.seeds
        ]

    def candidate_run(self, candidate: Any, fixture: Any) -> np.ndarray:
        return np.asarray(candidate.cluster(fixture.embedding(self.candidate_seed)))


def parse_bins_to_labels(
    bins_dir: str | Path,
    contig_names: Sequence[str],
    *,
    unbinned_label: int = -1,
    patterns: Sequence[str] = ("*.fa", "*.fasta", "*.fa.gz", "*.fna"),
) -> np.ndarray:
    """Map a directory of per-bin FASTAs back to a labels array in input order.

    The reusable heart of *any* subprocess oracle: a tool writes one FASTA per
    bin; we assign each contig the index of its bin file (sorted for
    determinism) and ``-1`` to contigs in no bin. Reused by
    :class:`SemiBin2BinLongOracle` and the template a future tool oracle copies.
    """
    from megobin.utils.fasta import fasta_iter

    bins_path = Path(bins_dir)
    files: list[Path] = []
    for pat in patterns:
        files.extend(bins_path.glob(pat))
    files = sorted(set(files))
    name_to_label: dict[str, int] = {}
    for label, bin_file in enumerate(files):
        for header, _ in fasta_iter(str(bin_file)):
            name_to_label[header] = label
    return np.array(
        [name_to_label.get(str(n), unbinned_label) for n in contig_names],
        dtype=np.int64,
    )


class SemiBin2BinLongOracle(ReferenceOracle):
    """Subprocess wrapper around ``SemiBin2 bin_long`` (the generic template).

    This is the pattern a future tool oracle (e.g. ComeBin) follows: run the
    downloaded tool's CLI on a fixture it can consume, then parse its per-bin
    FASTA outputs back into a slot-canonical labels array via
    :func:`parse_bins_to_labels`. It needs real contigs + precomputed
    abundance/features and the ``SemiBin2`` binary, so it is ``slow``/
    ``real_data`` gated; on a machine without those it raises
    :class:`OracleUnavailable`.
    """

    slot = "binner"

    def __init__(
        self,
        *,
        semibin_bin: str = "SemiBin2",
        environment: str = "global",
        candidate_seed: int = 99,
    ) -> None:
        super().__init__()
        self.semibin_bin = semibin_bin
        self.environment = environment
        self.candidate_seed = candidate_seed

    def _require_binary(self) -> str:
        path = shutil.which(self.semibin_bin)
        if path is None:
            raise OracleUnavailable(
                f"`{self.semibin_bin}` is not on PATH; install the [fidelity] "
                "extra to run the bin_long subprocess oracle."
            )
        return path

    def run_bin_long(
        self,
        contigs_fasta: str | Path,
        data_csv: str | Path,
        out_dir: str | Path,
        *,
        minfasta: int = 200000,
    ) -> Path:
        """Invoke ``SemiBin2 bin_long`` and return the output bins directory.

        ``data_csv`` is SemiBin's precomputed feature/abundance table (so no read
        mapping / BAM is needed here). Raises :class:`OracleUnavailable` if the
        binary is missing and surfaces the subprocess error otherwise.
        """
        binary = self._require_binary()
        out = Path(out_dir)
        cmd = [
            binary,
            "bin_long",
            "-i",
            str(contigs_fasta),
            "--data",
            str(data_csv),
            "--environment",
            self.environment,
            "--minfasta-kbs",
            str(minfasta // 1000),
            "-o",
            str(out),
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:  # pragma: no cover - gated
            raise OracleUnavailable(
                f"SemiBin2 bin_long failed: {exc.stderr[-500:] if exc.stderr else exc}"
            ) from exc
        return out / "output_bins"

    def runs(self, fixture: Any) -> list[np.ndarray]:  # pragma: no cover - gated
        raise OracleUnavailable(
            "SemiBin2BinLongOracle.runs needs a precomputed data table + abundance "
            "for the fixture; use run_bin_long(...) + parse_bins_to_labels(...) "
            "from a real-data harness, or the in-process SemiBin2GetBestBinOracle."
        )

    def candidate_run(
        self, candidate: Any, fixture: Any
    ) -> np.ndarray:  # pragma: no cover - gated
        return np.asarray(candidate.cluster(fixture.embedding(self.candidate_seed)))
