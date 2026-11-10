# Design philosophy

Five rules.

## 1. Slots before code

Every swappable piece is a Python `Protocol` (`@runtime_checkable`), not a base class. No registry, no decorator, no inheritance.

- `Encoder` — `megobin/encoders/base.py`
- `Binner`, `Evaluator`, `ContrastiveLoss`, `PairSampler`, `Trainer`, `Logger` — same pattern.

Adding an encoder = ~100 lines of Python + ~5 lines of YAML, plus the encoder's contract (`needs`, `provides`).

## 2. Features are shared

Feature computation lives in [megobin/features/](https://github.com/Tinnifo/Metagenomic-Binning/tree/main/megobin/features) and runs **once** per dataset. Encoders read shared `data.csv` (whole-contig kmer + abundance) and `data_split.csv` (half-contig kmer for pair sampling). 

Why: removes confounds, makes ablations cheap. The features stage *needs* `signals.<features.required_signals>` and the dataset *offers* `signals.<dataset.signals>`; the contract check aborts before any compute if one is missing.

## 3. Hydra for composition

Experiment configs declare their slots via [`defaults:`](https://hydra.cc/docs/advanced/defaults_list/):

```yaml
defaults:
  - /dataset: example
  - /pipeline: embed_and_bin
  - /features: canonical_kmer_abundance
  - /encoder: uncertain_gen
  - /loss: mahalanobis_bce
  - /pair_sampler: uncertain_gen
  - /trainer: two_phase
  - /filter: no_op
  - /binner: dbscan_ensemble
  - /logger: tensorboard
  - _self_
```

Experiment configs use Hydra composition with `_self_` placed last in the `defaults:` list. This ensures that shared slot configs such as the encoder or trainer are loaded first, and any values defined directly in the experiment config are applied afterward. As a result, per-experiment `encoder:` and `trainer:` blocks override slot defaults, rather than being overwritten by them. 

At the moment, the project uses a single experiment-config layout under `configs/experiment/*.yaml`. This layout is designed to be override-friendly: experiments inherit common building blocks through `defaults:` and only redefine the settings that need to change.

Optimizers and schedulers are declared with `_partial_: true`. This means the config provides a callable factory instead of constructing the object immediately, allowing the trainer to instantiate fresh optimizers or schedulers separately for each training phase. See the [Hydra object instantiation docs](https://hydra.cc/docs/advanced/instantiate_objects/overview/).

## 4. Contracts, checked before compute

Every stage and every inner component (sampler, encoder, loss) declares what it **needs** and what it **provides**, by port *and* by content label (`z: {mu, var}`, `batch: {views=2, label}`, `graph: {assembly_graph}`). The stage order is resolved backwards from the requested output, so stages nobody needs are not run (the optional refiner, for instance). All contracts are checked at pipeline start; a failure names the component, the port and the missing labels. One fact has one owner: the encoder owns *what it sends*, the loss owns *what it needs*.

## 5. The judge is outside the method

Scoring is a separate entry point (`python -m megobin.evaluate`). The method run writes `bins/` and `status.json` (resolved order, contract result, config hash, interventions ledger); the evaluator scores `bins/`. Swappable parts never include the thing that judges them.
