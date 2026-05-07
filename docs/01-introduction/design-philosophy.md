# Design philosophy

Four rules.

## 1. Slots before code

Every swappable piece is a Python `Protocol` (`@runtime_checkable`), not a base class. No registry, no decorator, no inheritance.

- `Encoder` — `megobin/encoders/base.py`
- `Binner`, `Evaluator`, `ContrastiveLoss`, `PairSampler`, `Trainer`, `Logger` — same pattern.

Adding an encoder = ~100 lines of Python + ~5 lines of YAML. `pipeline.py` is 230 lines of code.

## 2. Features are shared

Feature computation lives in [megobin/features/](https://github.com/Tinnifo/Metagenomic-Binning/tree/main/megobin/features) and runs **once** per dataset. Encoders read shared `data.csv` (whole-contig kmer + abundance) and `data_split.csv` (half-contig kmer for pair sampling). See [data-layout.md](../02-getting-started/data-layout.md).

Why: removes confounds, makes ablations cheap. `_check_signal_compatibility` in `pipeline.py` aborts before file I/O if `features.required_signals` ⊄ `dataset.signals`.

## 3. Hydra for composition

Experiment configs declare their slots via `defaults:`:

```yaml
defaults:
  - /dataset: example
  - /features: canonical_kmer_abundance
  - /encoder: uncertain_gen
  - /loss: mahalanobis_bce
  - /binner: dbscan_ensemble
  - /evaluator: checkm2
  - /pair_sampler: uncertain_gen
  - /trainer: two_phase
  - /logger: tensorboard
  - _self_
```

`_self_` last — so per-experiment `encoder:` / `trainer:` blocks override the slot defaults rather than the other way around.

One flavour for now: `configs/experiment/*.yaml` (override-friendly).

Optimizers/schedulers use `_partial_: true` so the trainer can rebuild them per phase.

## 4. Components don't import each other

UncertainGen does not import the contrastive loss it composes with. Hinge loss does not import Mahalanobis. Copy code, pay the duplication cost. The win: delete any component without cascading breakage.

## Why

Optimized for cheap "what if X?" experiments. That iteration rate is the rate-limiting variable.
