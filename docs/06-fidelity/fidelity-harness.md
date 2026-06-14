# The Fidelity Harness

`mypy` proves the **shape** half of a slot contract (types/dims). The **semantic**
half — geometry, normalization, ordering, bin-size semantics — is invisible to the
type system and is where silent porting bugs hide. `megobin/fidelity/` checks the
semantic half empirically: it validates a MegoBin re-implementation of an external
tool against a reference **oracle**, calibrated to the oracle's own run-to-run
variance and compared on slot-appropriate invariants.

It is **generic**. SemiBin2 is the first oracle; any future tool (e.g. ComeBin)
follows the same recipe — wrap the downloaded tool as a `ReferenceOracle`,
re-implement the slot, validate against the wrapper.

## The idea in four points

1. **The oracle is a distribution, not a point.** Re-running a tool varies (seed,
   float, thread order). "Faithful" can't mean byte-identical, so the harness
   calibrates a tolerance **band** from the reference's own repeated runs.
2. **Compare on invariants.** ARI (label-permutation-invariant) and a bp-weighted
   bin-match F1 for binners; pairwise-distance Spearman (isometry-invariant) or
   through-a-fixed-binner ARI for encoders.
3. **Anchor to ground truth.** The synthetic fixtures carry known genome labels, so
   the harness measures recovery against truth, not only against the oracle.
4. **Two-way acceptance.** A candidate passes iff it agrees with the reference at
   least as well as the reference agrees with itself **and** recovers ground truth
   within the reference's own recovery band.

## Quick start

```python
from megobin.fidelity import (
    FidelityHarness, SemiBin2GetBestBinOracle, binner_ari, canonical_genome_fixture,
)
from megobin.binners.semibin_2 import DBSCANEnsembleBinner

fixture = canonical_genome_fixture()
candidate = DBSCANEnsembleBinner(
    minfasta=200000, contig_names=fixture.contig_names, contig_lengths=fixture.lengths,
    contig_to_marker=fixture.contig_to_marker, n_total_markers=107,
)
report = FidelityHarness(binner_ari, z=2.0).run(
    candidate=candidate, oracle=SemiBin2GetBestBinOracle(), fixture=fixture,
)
print(report.to_markdown())   # PASS/FAIL + the bands
```

## Oracles

| Oracle | Reference | Default path? | True-oracle? |
|--------|-----------|---------------|--------------|
| `MegoBinSelfBinnerOracle` | faithful-config MegoBin binner | yes | no — self-consistency + teeth |
| `CachedReferenceOracle` | committed genuine-SemiBin runs (CSV) | yes (cache committed) | **yes** (loaded, no SemiBin needed) |
| `SemiBin2GetBestBinOracle` | SemiBin's real `get_best_bin`, in-process | if `[fidelity]` installed | **yes** |
| `SemiBin2BinLongOracle` | `SemiBin2 bin_long` subprocess | no (`real_data`) | **yes** (heavy) |
| `EncoderTrainingOracle` | K seeded encoder trainings | yes | n/a (self-consistency) |

The default suite is **honest about what it checks**: it proves self-consistency,
ground-truth recovery, and discriminating power (faithful config passes, divergent
config fails), plus a genuine SemiBin comparison via the committed cache. The live
`SemiBin2*` oracles and the real-genome path are gated.

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
pytest tests/fidelity                 # default: CPU-only, no external tools
pytest tests/fidelity -m real_data    # gated: real genomes, hmmsearch, bin_long
pip install -e '.[fidelity]'          # optional SemiBin oracle (self-skips if absent)

# regenerate the committed SemiBin cache (needs SemiBin installed):
python -m megobin.fidelity.refresh_cache
# download the real-genome fixture (needs network):
python -m megobin.fidelity.download_fixture
```

## Runtime budget (measured, Apple Silicon, CPU-only)

- Default suite (`pytest tests/fidelity`, excludes `slow`/`real_data`): **~7 s**,
  < 2 GB RAM, no GPU/SLURM/CheckM2.
- Genuine SemiBin2 oracle (8 seeds, synthetic): **~1.2 s**.
- Real-genome path (`-m real_data`: download once, then markers + oracle on 3
  Mycoplasma genomes / 126 contigs): **~4 s** after download.

## What it found

The MegoBin SemiBin2 port reproduces genuine SemiBin `get_best_bin` **exactly**
(ARI = 1.0) on the same embeddings + markers — synthetic *and* real. The divergent
config (`minfasta=0`, length-weighting off) falls outside the band, confirming the
harness rejects a genuinely-divergent re-implementation.
