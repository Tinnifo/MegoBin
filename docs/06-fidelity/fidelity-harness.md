# The Fidelity Harness

`mypy` proves the **shape** half of a slot contract (types/dims). The **semantic**
half — geometry, normalization, ordering, bin-size semantics — is invisible to the
type system and is where silent porting bugs hide. `megobin/fidelity/` checks the
semantic half empirically: it validates a MegoBin re-implementation of an external
tool against a reference **oracle**, calibrated to the oracle's own run-to-run
variance and compared on slot-appropriate invariants.

The fidelity claim itself — "does the port match the original tool?" — is checked
**only against the real tool on real data**, by the person re-implementing the
tool. It is correctness-first, not speed-first: there is no cached or
synthetic-data stand-in for the genuine comparison.

It is **generic**. SemiBin2's binner is the first oracle; any future tool (e.g.
ComeBin) follows the same recipe — wrap the downloaded tool as a `ReferenceOracle`,
re-implement the slot, validate against the wrapper.

## The idea in four points

1. **The oracle is a distribution, not a point.** Re-running a tool varies (seed,
   float, thread order). "Faithful" can't mean byte-identical, so the harness
   calibrates a tolerance **band** from the reference's own repeated runs.
2. **Compare on invariants.** ARI (label-permutation-invariant) and a bp-weighted
   bin-match F1 (scored on *base pairs*, so bin-shattering is penalised) for the
   binner slot.
3. **Anchor to ground truth.** The genome fixtures carry known genome labels, so
   the harness measures recovery against truth, not only against the oracle.
4. **Two-way acceptance.** A candidate passes iff it agrees with the reference at
   least as well as the reference agrees with itself **and** recovers ground truth
   within the reference's own recovery band.

## Quick start — the genuine SemiBin2 fidelity check

This is the real check, run on real reference genomes against real SemiBin2:

```python
from megobin.fidelity import (
    FidelityHarness, SemiBin2GetBestBinOracle, binner_ari, load_real_genome_set,
)
from megobin.binners.semibin_2 import DBSCANEnsembleBinner

# One-time prerequisites:
#   python -m megobin.fidelity.download_fixture   # real reference genomes
#   pip install -e '.[fidelity]'                   # genuine SemiBin2
#   hmmsearch on PATH (e.g. brew install hmmer)    # real single-copy markers
fixture = load_real_genome_set("tests/fidelity/data/real/genomes.fasta")
candidate = DBSCANEnsembleBinner(
    minfasta=200000, contig_names=fixture.contig_names, contig_lengths=fixture.lengths,
    contig_to_marker=fixture.contig_to_marker, n_total_markers=107,
)
report = FidelityHarness(binner_ari, z=2.0).run(
    candidate=candidate,
    oracle=SemiBin2GetBestBinOracle(minfasta=200000, seeds=range(6)),
    fixture=fixture,
)
print(report.to_markdown())   # PASS/FAIL + the bands
```

## Oracles

| Oracle | Reference | When it runs | True-oracle? |
|--------|-----------|--------------|--------------|
| `MegoBinSelfBinnerOracle` | faithful-config MegoBin binner | default suite | no — self-consistency + discrimination (gives the harness its teeth) |
| `SemiBin2GetBestBinOracle` | SemiBin's real `get_best_bin`, in-process on real genomes | `real_data` (needs `[fidelity]` + downloaded genomes) | **yes** |
| `SemiBin2BinLongOracle` | `SemiBin2 bin_long` subprocess | `real_data` (CLI-only template) | **yes** (heavy) |

The **default suite proves the harness itself works**: metric correctness, band
math, self-consistency, ground-truth recovery, and discriminating power (the
faithful config passes and a divergent config is *measurably rejected*). It makes
**no** claim of fidelity to SemiBin2 — it has no real tool in it.

The **fidelity claim** is the `real_data` path: the genuine SemiBin2
`get_best_bin` run on real reference genomes with real single-copy markers. That
is the check the person re-implementing the tool runs locally — correctness over
speed.

## Adding a new tool oracle (the extension recipe)

To validate a re-implementation of some tool `Foo` for the binner slot:

```python
from megobin.fidelity import ReferenceOracle, FidelityHarness, binner_ari

class FooOracle(ReferenceOracle):
    slot = "binner"
    def runs(self, fixture):            # K reference label arrays (genuine Foo)
        return [foo_cluster(fixture.embedding(s), fixture) for s in range(10)]
    def candidate_run(self, candidate, fixture):
        return candidate.cluster(fixture.embedding(99))

FidelityHarness(binner_ari).run(candidate=my_foo_port, oracle=FooOracle(), fixture=fx)
```

`runs()` must carry genuine run-to-run variance (e.g. K seeded embedding
realisations) or the harness raises `DegenerateBandError`. If `Foo` only ships a
CLI, copy `SemiBin2BinLongOracle`: run it and map its per-bin FASTAs back with
`parse_bins_to_labels`.

## Running

```bash
pytest tests/fidelity                 # default: harness-validation, CPU-only, no external tools
pytest tests/fidelity -m real_data    # the genuine SemiBin2 fidelity check (real genomes + SemiBin)
pip install -e '.[fidelity]'          # genuine SemiBin2 (required for the real_data check)

# download the real-genome fixture (needs network), once:
python -m megobin.fidelity.download_fixture
```

## Runtime budget (measured, Apple Silicon, CPU-only)

- Default harness-validation suite (`pytest tests/fidelity`, excludes
  `slow`/`real_data`): **~2 s**, < 2 GB RAM, no GPU/SLURM/CheckM2.
- Real-genome fidelity path (`-m real_data`: download once + `[fidelity]` extra +
  hmmsearch, then markers + genuine SemiBin2 on 3 Mycoplasma genomes / 126
  contigs): **~4 s** after download.

## What it found

The MegoBin SemiBin2 port reproduces genuine SemiBin `get_best_bin` **exactly**
(ARI = 1.0) on the same embeddings + markers, on real reference genomes. The
divergent config (`minfasta=0`, length-weighting off) falls outside the band,
confirming the harness rejects a genuinely-divergent re-implementation.
