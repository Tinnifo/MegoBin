# Design philosophy

Four rules.

## 1. Slots before code

Every swappable piece is a Python `Protocol` (`@runtime_checkable`), not a base class. No registry, no decorator, no inheritance.

- `Encoder` — `megobin/encoders/base.py`
- `Binner`, `Evaluator`, `ContrastiveLoss`, `PairSampler`, `Trainer`, `Logger` — same pattern.

Adding an encoder = ~100 lines of Python + ~5 lines of YAML. `pipeline.py` is 230 lines of code.

## 2. Features are shared

Feature computation lives in [megobin/features/](https://github.com/Tinnifo/Metagenomic-Binning/tree/main/megobin/features) and runs **once** per dataset. Encoders read shared `data.csv` (whole-contig kmer + abundance) and `data_split.csv` (half-contig kmer for pair sampling). 

Why: removes confounds, makes ablations cheap. `_check_signal_compatibility` in `pipeline.py` aborts before file I/O if `features.required_signals` ⊄ `dataset.signals`.

## 3. Hydra for composition

Experiment configs declare their slots via [`defaults:`](https://hydra.cc/docs/advanced/defaults_list/):

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

Experiment configs use Hydra composition with `_self_` placed last in the `defaults:` list. This ensures that shared slot configs such as the encoder or trainer are loaded first, and any values defined directly in the experiment config are applied afterward. As a result, per-experiment `encoder:` and `trainer:` blocks override slot defaults, rather than being overwritten by them. 

At the moment, the project uses a single experiment-config layout under `configs/experiment/*.yaml`. This layout is designed to be override-friendly: experiments inherit common building blocks through `defaults:` and only redefine the settings that need to change.

Optimizers and schedulers are declared with `_partial_: true`. This means the config provides a callable factory instead of constructing the object immediately, allowing the trainer to instantiate fresh optimizers or schedulers separately for each training phase. See the [Hydra object instantiation docs](https://hydra.cc/docs/advanced/instantiate_objects/overview/).






