"""The deterministic fixtures load fast, CPU-only, with the planted structure."""

import time

import numpy as np

from megobin.fidelity.fixtures import make_blobs_fixture, make_genome_fixture


def test_blobs_shapes_and_markers():
    fx = make_blobs_fixture(n_clusters=4, per_cluster=25, dim=8)
    assert fx.labels.shape == (100,)
    assert fx.contig_names.shape == (100,)
    assert fx.embedding(0).shape == (100, 8)
    assert set(np.unique(fx.labels)) == {0, 1, 2, 3}
    # one distinct marker per cluster
    assert fx.contig_to_marker[str(fx.contig_names[0])] == [f"m{int(fx.labels[0])}"]


def test_genome_set_structure():
    fx = make_genome_fixture(n_genomes=6, contigs_per_genome=40, n_noise=30)
    n = 6 * 40 + 30
    assert fx.labels.shape == (n,)
    assert fx.lengths.shape == (n,)
    # genomes labelled 0..5, noise labelled -1
    assert set(np.unique(fx.labels)) == {-1, 0, 1, 2, 3, 4, 5}
    # every genome clears minfasta; noise contigs do not
    for g in range(6):
        assert fx.lengths[fx.labels == g].sum() > 200000
    assert fx.lengths[fx.labels == -1].max() < 200000


def test_embedding_is_deterministic_per_seed():
    fx = make_genome_fixture(n_noise=20)
    a = fx.embedding(3)
    b = fx.embedding(3)
    np.testing.assert_array_equal(a, b)
    assert a.dtype == np.float32


def test_embedding_varies_across_seeds():
    fx = make_genome_fixture(n_noise=20)
    a = fx.embedding(0)
    b = fx.embedding(1)
    # structure preserved (same shape) but noise realisation differs
    assert a.shape == b.shape
    assert not np.array_equal(a, b)


def test_genome_fixture_is_fast():
    t = time.time()
    fx = make_genome_fixture(n_noise=60)
    fx.embedding(0)
    assert time.time() - t < 2.0
    assert fx.manifest_sha256 == fx.manifest_sha256  # stable


def test_with_dna_round_trips(tmp_path):
    fx = make_genome_fixture(n_genomes=2, contigs_per_genome=3, n_noise=2, with_dna=True)
    path = fx.write_fasta(tmp_path / "contigs.fasta")
    from megobin.utils.fasta import fasta_iter

    seqs = dict(fasta_iter(str(path)))
    assert set(seqs) == set(str(n) for n in fx.contig_names)
    name0 = str(fx.contig_names[0])
    assert len(seqs[name0]) == int(fx.lengths[0])
