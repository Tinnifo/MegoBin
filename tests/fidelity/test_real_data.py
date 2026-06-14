"""Real-genome fidelity (gated: ``real_data``).

Loads downloaded reference genomes, computes *real* single-copy markers
(fast-naive ORF finder + hmmsearch), and checks that the MegoBin binner
reproduces genuine SemiBin ``get_best_bin`` on real markers + real k-mer
composition. Skips unless the FASTA, hmmsearch, and SemiBin are all present.

Prerequisites (developer machine):
    python -m megobin.fidelity.download_fixture
    pip install -e '.[fidelity]'   # SemiBin
    # hmmsearch on PATH (e.g. brew install hmmer)
"""

import shutil
from pathlib import Path

import numpy as np
import pytest

pytestmark = [pytest.mark.real_data, pytest.mark.slow]

_FASTA = Path(__file__).parent / "data" / "real" / "genomes.fasta"

if not _FASTA.exists():
    pytest.skip("real-genome FASTA absent; run download_fixture", allow_module_level=True)
if shutil.which("hmmsearch") is None:
    pytest.skip("hmmsearch not on PATH", allow_module_level=True)
pytest.importorskip("SemiBin", reason="install [fidelity] extra")

from megobin.binners.semibin_2 import DBSCANEnsembleBinner  # noqa: E402
from megobin.fidelity import (  # noqa: E402
    SEMIBIN_N_TOTAL_MARKERS,
    FidelityHarness,
    SemiBin2GetBestBinOracle,
    binner_ari,
    load_real_genome_set,
    semibin_cluster_labels,
)
from megobin.fidelity.oracles_semibin2 import _LenStr  # noqa: E402


@pytest.fixture(scope="module")
def real_genomes():
    return load_real_genome_set(_FASTA, num_process=1)


def _faithful(fx):
    return DBSCANEnsembleBinner(
        minfasta=200000,
        contig_names=fx.contig_names,
        contig_lengths=fx.lengths,
        contig_to_marker=fx.contig_to_marker,
        n_total_markers=SEMIBIN_N_TOTAL_MARKERS,
    )


def test_real_markers_are_called(real_genomes):
    marked = sum(1 for n in real_genomes.contig_names if real_genomes.contig_to_marker[str(n)])
    assert marked > 0, "no real single-copy markers were detected"


def test_port_matches_semibin_on_real_markers(real_genomes):
    fx = real_genomes
    emb = fx.kmer_embedding
    names = [str(c) for c in fx.contig_names]
    contig_dict = {n: _LenStr(int(length)) for n, length in zip(names, fx.lengths)}
    c2m = {str(k): list(v) for k, v in fx.contig_to_marker.items()}
    sb = semibin_cluster_labels(
        emb, names, c2m, contig_dict, np.asarray(fx.lengths, float), 200000
    )
    mego = _faithful(fx).cluster(emb)
    assert len(set(sb[sb >= 0])) > 0, "SemiBin found no bins on the real fixture"
    assert binner_ari(sb, mego) == pytest.approx(1.0)


def test_faithful_passes_band_on_real_data(real_genomes):
    # noise-augmented embeddings give the band genuine variance
    oracle = SemiBin2GetBestBinOracle(minfasta=200000, seeds=range(6))
    report = FidelityHarness(binner_ari, z=2.0).run(
        candidate=_faithful(real_genomes), oracle=oracle, fixture=real_genomes
    )
    assert report.passed, report.to_markdown()
