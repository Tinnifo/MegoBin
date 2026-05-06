# Design philosophy

Four rules.

## 1. Slots before code

Every swappable piece is a Python `Protocol` (`@runtime_checkable`), not a base class. No registry, no decorator, no inheritance.

- `Encoder` — `megobin/encoders/base.py`
- `Binner`, `Evaluator`, `ContrastiveLoss`, `PairSampler`, `Trainer`, `Logger` — same pattern.

Adding an encoder = ~100 lines of Python + ~5 lines of YAML. `pipeline.py` is 230 lines of straight-line code.

## 2. Features are shared

Feature computation lives in [megobin/features/](https://github.com/Tinnifo/Metagenomic-Binning/tree/main/megobin/features) and runs **once** per dataset. Encoders read shared `kmer_profiles.npy`, `abundance.npy`, `contig_names.npy`.

Why: removes confounds, makes ablations cheap. `_check_signal_compatibility` in `pipeline.py` aborts before file I/O if `features.required_signals` ⊄ `dataset.signals`.

## 3. Hydra for composition

Experiment configs declare their slots via `defaults:`:

```yaml
defaults:
  - _self_
  - /dataset: CAMI_medium
  - /features: canonical_kmer_abundance
  - /encoder: uncertain_gen
  - /loss: mahalanobis_bce
  - /binner: infomap
  - /evaluator: checkm2
  - /pair_sampler: hybrid
  - /trainer: two_phase
  - /logger: tensorboard
```

Two flavours:
- `configs/experiment/*.yaml` — ad-hoc, override-friendly.
- `configs/experiment/training/*.yaml` — pinned, reproducible.

Optimizers/schedulers use `_partial_: true` so the trainer can rebuild them per phase.

## 4. Components don't import each other

SemiBin does not import UncertainGen. Hinge loss does not import Mahalanobis. Copy code, pay the duplication cost. The win: delete any component without cascading breakage.

## Why

Optimized for cheap "what if X?" experiments. That iteration rate is the rate-limiting variable.
