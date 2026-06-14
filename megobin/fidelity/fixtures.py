"""Deterministic, laptop-sized fidelity fixtures.

Fixtures are generated from a seed + a small manifest — never committed as
binary arrays (``data/*`` and ``*.npy`` are git-ignored). Each fixture exposes
ground-truth ``labels`` plus an ``embedding(seed)`` method that redraws a fresh
noise realisation of the embedding matrix while keeping the planted structure
(labels, lengths, markers) fixed. Those K realisations are the genuine
run-to-run variance source the calibration band needs — a single fixed matrix
clustered K times by a deterministic binner would give a degenerate band.

Two generators:

* :class:`GaussianBlobs` — a handful of well-separated blobs, ~100 rows, for
  fast comparator/harness unit tests.
* :class:`SyntheticGenomeSet` — synthetic "genomes" as well-separated unit-sphere
  directions chopped into contigs with per-contig lengths and a synthetic
  single-copy-marker dict, so the DBSCAN-ensemble binner recovers the planted
  genomes and ``minfasta`` bp-gating is observable. Mirrors the statistical
  structure of the existing ``test_end_to_end`` / ``test_semibin2`` fixtures.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

_UNBINNED = -1


class Fixture(Protocol):
    """Structural contract the harness and oracles consume."""

    fixture_id: str
    labels: np.ndarray  # ground-truth label per contig (-1 = not in any genome)
    contig_names: np.ndarray  # str ids aligned to rows
    lengths: np.ndarray  # base pairs per contig
    contig_to_marker: dict[str, list[str]]
    n_total_markers: int
    manifest: dict[str, Any]

    def embedding(self, seed: int) -> np.ndarray:
        """A fresh (N, d) embedding realisation; structure fixed, noise varies."""
        ...


def _manifest_sha256(manifest: dict[str, Any]) -> str:
    blob = json.dumps(manifest, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()


# --------------------------------------------------------------------------- #
# Gaussian blobs
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class GaussianBlobs:
    """Well-separated isotropic blobs with one distinct marker per cluster."""

    fixture_id: str
    labels: np.ndarray
    contig_names: np.ndarray
    lengths: np.ndarray
    contig_to_marker: dict[str, list[str]]
    n_total_markers: int
    manifest: dict[str, Any]
    _centers: np.ndarray
    _noise: float

    def embedding(self, seed: int) -> np.ndarray:
        rng = np.random.default_rng([self.manifest["base_seed"], seed])
        base = self._centers[self.labels]
        return (base + rng.standard_normal(base.shape) * self._noise).astype(
            np.float32
        )


def make_blobs_fixture(
    *,
    n_clusters: int = 4,
    per_cluster: int = 25,
    dim: int = 8,
    sep: float = 10.0,
    noise: float = 0.3,
    base_seed: int = 0,
    fixture_id: str = "blobs_v1",
) -> GaussianBlobs:
    rng = np.random.default_rng(base_seed)
    centers = rng.standard_normal((n_clusters, dim)) * sep
    labels = np.repeat(np.arange(n_clusters), per_cluster)
    n = labels.size
    contig_names = np.array([f"c{i}" for i in range(n)])
    lengths = np.full(n, 1000.0)
    # one distinct marker per cluster so the binner's marker-F1 has signal
    contig_to_marker = {
        contig_names[i]: [f"m{int(labels[i])}"] for i in range(n)
    }
    manifest = {
        "fixture_id": fixture_id,
        "kind": "gaussian_blobs",
        "n_clusters": n_clusters,
        "per_cluster": per_cluster,
        "dim": dim,
        "sep": sep,
        "noise": noise,
        "base_seed": base_seed,
    }
    return GaussianBlobs(
        fixture_id=fixture_id,
        labels=labels,
        contig_names=contig_names,
        lengths=lengths,
        contig_to_marker=contig_to_marker,
        n_total_markers=n_clusters,
        manifest=manifest,
        _centers=centers,
        _noise=noise,
    )


# --------------------------------------------------------------------------- #
# Synthetic genome set
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SyntheticGenomeSet:
    """Synthetic genomes as unit-sphere directions chopped into contigs.

    Genome contigs cluster tightly around their genome's direction; "noise"
    contigs sit at isolated directions so DBSCAN leaves them out (-1 in the
    sweep). Genome totals exceed ``minfasta``; noise singletons do not, so the
    ``minfasta`` divergence is measurable. ``labels`` is the genome id for
    genome contigs and ``-1`` for noise (i.e. ground truth also leaves noise
    unbinned).
    """

    fixture_id: str
    labels: np.ndarray
    contig_names: np.ndarray
    lengths: np.ndarray
    contig_to_marker: dict[str, list[str]]
    n_total_markers: int
    manifest: dict[str, Any]
    sequences: dict[str, str] | None
    _centers: np.ndarray  # (n_genomes, dim) unit directions
    _noise_base: np.ndarray  # (n_noise, dim) isolated directions
    _noise_std: float
    _genome_rows: np.ndarray  # genome index per row, -1 for noise rows

    @property
    def manifest_sha256(self) -> str:
        return _manifest_sha256(self.manifest)

    def embedding(self, seed: int) -> np.ndarray:
        rng = np.random.default_rng([self.manifest["base_seed"], seed])
        n, dim = self._genome_rows.size, self._centers.shape[1]
        out = np.empty((n, dim), dtype=float)
        is_noise = self._genome_rows < 0
        genome_idx = self._genome_rows[~is_noise]
        out[~is_noise] = self._centers[genome_idx]
        out[is_noise] = self._noise_base
        out += rng.standard_normal((n, dim)) * self._noise_std
        return out.astype(np.float32)

    def write_fasta(self, path: str | Path) -> Path:
        """Write the synthetic DNA sequences to a FASTA (for round-trip / live
        oracle tests). Requires the fixture to have been built ``with_dna=True``.
        """
        if self.sequences is None:
            raise ValueError(
                "fixture built without DNA; pass with_dna=True to make_genome_fixture"
            )
        path = Path(path)
        with open(path, "w") as fh:
            for name in self.contig_names:
                fh.write(f">{name}\n{self.sequences[str(name)]}\n")
        return path


def canonical_genome_fixture() -> "SyntheticGenomeSet":
    """The committed-cache / shared-test genome set.

    A single definition used by the test conftest *and*
    ``megobin.fidelity.refresh_cache`` so the committed SemiBin2 cache always
    matches the fixture the tests run on (the provenance manifest hash guards
    against drift). ``n_noise=60`` gives a comfortable faithful-vs-divergent
    margin.
    """
    return make_genome_fixture(n_noise=60)


@dataclass(frozen=True)
class EncoderFixture:
    """Synthetic features for encoder-slot tests: a fixed (N, input_dim) feature
    matrix with genome structure plus the split halves a contrastive sampler
    consumes. Encoder variance comes from the *training seed*, not the features,
    so the features are fixed.
    """

    fixture_id: str
    features: np.ndarray  # (N, input_dim) "whole" features
    features_split: np.ndarray  # (2N, input_dim) must-link halves
    labels: np.ndarray  # genome id per contig
    n_genomes: int
    input_dim: int
    manifest: dict[str, Any]


def make_encoder_fixture(
    *,
    n_genomes: int = 3,
    contigs_per_genome: int = 20,
    dim: int = 32,
    noise: float = 0.05,
    base_seed: int = 0,
    fixture_id: str = "encoder_set_v1",
) -> EncoderFixture:
    """Dirichlet genome profiles chopped into contigs (mirrors
    ``test_end_to_end._make_genome_cluster_features``)."""
    rng = np.random.default_rng(base_seed)
    profiles = rng.dirichlet(np.ones(dim) * 0.5, size=n_genomes)
    labels = np.repeat(np.arange(n_genomes), contigs_per_genome)
    whole = np.stack(
        [profiles[g] + rng.normal(0, noise, size=dim) for g in labels]
    ).astype("float32")
    first = np.stack(
        [profiles[g] + rng.normal(0, noise, size=dim) for g in labels]
    ).astype("float32")
    second = np.stack(
        [profiles[g] + rng.normal(0, noise, size=dim) for g in labels]
    ).astype("float32")
    split = np.concatenate([first, second], axis=0)
    manifest = {
        "fixture_id": fixture_id,
        "kind": "encoder_set",
        "n_genomes": n_genomes,
        "contigs_per_genome": contigs_per_genome,
        "dim": dim,
        "noise": noise,
        "base_seed": base_seed,
    }
    return EncoderFixture(
        fixture_id=fixture_id,
        features=whole,
        features_split=split,
        labels=labels,
        n_genomes=n_genomes,
        input_dim=dim,
        manifest=manifest,
    )


@dataclass(frozen=True)
class RealGenomeSet:
    """Real reference genomes chopped into contigs, with real single-copy markers.

    Unlike the synthetic generators this loads downloaded DNA and calls the real
    marker pipeline (``fast-naive`` ORF finder + ``hmmsearch``), so it needs
    those tools and the downloaded FASTA — hence the ``real_data``/``slow`` gate.
    ``n_total_markers`` is 107 to match SemiBin's recall divisor. ``embedding``
    adds a small per-seed jitter to the (fixed) k-mer composition so a
    calibration band is non-degenerate; ``kmer_embedding`` is the noise-free
    version for direct port-agreement checks.
    """

    fixture_id: str
    contig_names: np.ndarray
    labels: np.ndarray
    lengths: np.ndarray
    sequences: dict[str, str]
    kmer_embedding: np.ndarray
    contig_to_marker: dict[str, list[str]]
    n_total_markers: int
    manifest: dict[str, Any]

    def embedding(self, seed: int) -> np.ndarray:
        rng = np.random.default_rng([self.manifest["base_seed"], seed])
        noise = rng.standard_normal(self.kmer_embedding.shape) * self.manifest["noise_std"]
        return (self.kmer_embedding + noise).astype(np.float32)

    def write_fasta(self, path: str | Path) -> Path:
        path = Path(path)
        with open(path, "w") as fh:
            for name in self.contig_names:
                fh.write(f">{name}\n{self.sequences[str(name)]}\n")
        return path


def load_real_genome_set(
    fasta_path: str | Path,
    *,
    chunk: int = 20000,
    min_len: int = 2500,
    noise_std: float = 0.01,
    base_seed: int = 0,
    num_process: int = 1,
    fixture_id: str = "real_genomes_v1",
) -> RealGenomeSet:
    """Chop downloaded genomes into contigs and compute k-mers + real markers.

    Each genome (FASTA record) becomes a labelled group of ``chunk``-bp contigs.
    Marker calling shells out to ``hmmsearch`` (``num_process=1`` keeps the ORF
    finder serial, avoiding multiprocessing-spawn issues).
    """
    import tempfile

    from megobin.features.generate_kmers import generate_kmer_features_from_fasta
    from megobin.utils.fasta import fasta_iter
    from megobin.utils.markers import estimate_seeds, get_marker

    names: list[str] = []
    labels: list[int] = []
    sequences: dict[str, str] = {}
    for genome_idx, (header, seq) in enumerate(fasta_iter(str(fasta_path))):
        for start in range(0, len(seq), chunk):
            piece = seq[start : start + chunk]
            if len(piece) < min_len:
                continue
            name = f"{header}__{start // chunk}"
            names.append(name)
            labels.append(genome_idx)
            sequences[name] = piece
    if not names:
        raise ValueError(f"no contigs >= {min_len} bp in {fasta_path}")

    lengths = np.array([len(sequences[n]) for n in names], dtype=float)

    with tempfile.TemporaryDirectory() as tdir:
        contigs_fa = Path(tdir) / "contigs.fasta"
        with open(contigs_fa, "w") as fh:
            for name in names:
                fh.write(f">{name}\n{sequences[name]}\n")

        kmer_df = generate_kmer_features_from_fasta(str(contigs_fa), min_len, 4)
        kmer = kmer_df.reindex(names).to_numpy(dtype=float)
        # L2-normalise rows so distances live on a scale the eps grid resolves.
        norms = np.linalg.norm(kmer, axis=1, keepdims=True)
        kmer_embedding = (kmer / np.clip(norms, 1e-12, None)).astype(np.float32)

        estimate_seeds(
            str(contigs_fa), min_len, num_process, output=tdir, orf_finder="fast-naive"
        )
        c2m = get_marker(
            str(Path(tdir) / "markers.hmmout"),
            fasta_path=str(contigs_fa),
            min_contig_len=min_len,
            orf_finder="fast-naive",
            contig_to_marker=True,
        )
        if not isinstance(c2m, dict):
            c2m = {}
    contig_to_marker = {n: list(c2m.get(n, [])) for n in names}

    manifest = {
        "fixture_id": fixture_id,
        "kind": "real_genome_set",
        "chunk": chunk,
        "min_len": min_len,
        "noise_std": noise_std,
        "base_seed": base_seed,
        "n_contigs": len(names),
        "n_genomes": int(max(labels)) + 1,
    }
    return RealGenomeSet(
        fixture_id=fixture_id,
        contig_names=np.array(names),
        labels=np.array(labels, dtype=int),
        lengths=lengths,
        sequences=sequences,
        kmer_embedding=kmer_embedding,
        contig_to_marker=contig_to_marker,
        n_total_markers=107,
        manifest=manifest,
    )


def make_genome_fixture(
    *,
    n_genomes: int = 6,
    contigs_per_genome: int = 40,
    n_noise: int = 30,
    dim: int = 8,
    n_total_markers: int = 20,
    noise_std: float = 0.03,
    genome_len_range: tuple[int, int] = (5000, 8000),
    noise_len_range: tuple[int, int] = (2000, 4000),
    base_seed: int = 0,
    with_dna: bool = False,
    fixture_id: str = "genome_set_v1",
) -> SyntheticGenomeSet:
    """Build a deterministic synthetic genome set.

    Each genome's contigs collectively carry markers ``m0..m{M-1}`` exactly
    once (recall 1, contamination 0 -> marker-F1 1 for a pure genome bin; a bin
    merging two genomes doubles every marker -> contamination 0.5). Genome bp
    totals exceed 200 kb; noise contigs are isolated and small.
    """
    rng = np.random.default_rng(base_seed)

    # well-separated genome directions on the unit sphere
    centers = rng.standard_normal((n_genomes, dim))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    noise_base = rng.standard_normal((n_noise, dim))
    noise_base /= np.linalg.norm(noise_base, axis=1, keepdims=True)

    genome_rows: list[int] = []
    names: list[str] = []
    lengths: list[float] = []
    contig_to_marker: dict[str, list[str]] = {}
    labels: list[int] = []

    for g in range(n_genomes):
        for c in range(contigs_per_genome):
            name = f"g{g}_c{c}"
            names.append(name)
            genome_rows.append(g)
            labels.append(g)
            lengths.append(float(rng.integers(*genome_len_range)))
            # one distinct marker per contig for the first M contigs; the rest
            # are unmarked but still cluster with their genome.
            contig_to_marker[name] = [f"m{c}"] if c < n_total_markers else []

    for k in range(n_noise):
        name = f"noise_{k}"
        names.append(name)
        genome_rows.append(-1)
        labels.append(_UNBINNED)
        lengths.append(float(rng.integers(*noise_len_range)))
        contig_to_marker[name] = []

    sequences: dict[str, str] | None = None
    if with_dna:
        alphabet = np.array(list("ACGT"))
        sequences = {}
        for name, length in zip(names, lengths, strict=True):
            seq = "".join(rng.choice(alphabet, size=int(length)))
            sequences[name] = seq

    manifest = {
        "fixture_id": fixture_id,
        "kind": "synthetic_genome_set",
        "n_genomes": n_genomes,
        "contigs_per_genome": contigs_per_genome,
        "n_noise": n_noise,
        "dim": dim,
        "n_total_markers": n_total_markers,
        "noise_std": noise_std,
        "genome_len_range": list(genome_len_range),
        "noise_len_range": list(noise_len_range),
        "base_seed": base_seed,
        "with_dna": with_dna,
    }

    return SyntheticGenomeSet(
        fixture_id=fixture_id,
        labels=np.array(labels, dtype=int),
        contig_names=np.array(names),
        lengths=np.array(lengths, dtype=float),
        contig_to_marker=contig_to_marker,
        n_total_markers=n_total_markers,
        manifest=manifest,
        sequences=sequences,
        _centers=centers,
        _noise_base=noise_base,
        _noise_std=noise_std,
        _genome_rows=np.array(genome_rows, dtype=int),
    )
