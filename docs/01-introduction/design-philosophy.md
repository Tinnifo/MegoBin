# Design philosophy

MegoBin makes four structural decisions up front. Each of them has implications you will feel every time you add a component, and every time you try to compare two experiments. This chapter spells them out so the rest of the guide feels obvious rather than arbitrary.

## 1. Slots before code

Every swappable piece of the pipeline is defined by a Python `Protocol`, not a base class or an abstract method. Concretely, `megobin/encoders/base.py` defines the `Encoder` Protocol; `megobin/binners/base.py` defines `Binner`; and so on for `Evaluator`, `ContrastiveLoss`, `PairSampler`, `Trainer`, and `Logger`.

An encoder is **any** Python object that quacks like an `Encoder` — it has `encode(features)`, `training_step(batch, loss_fn)`, `parameter_groups()`, and an `embedding_dim` property. There is no registration step, no parent class to inherit, no decorator. You write a file, Hydra instantiates it, the pipeline calls its methods. The `@runtime_checkable` decorator on each Protocol makes this type-checkable if you want, but nothing enforces it at import time — the test suite does that via `test_interfaces.py`, which `isinstance`-checks every implementation against its Protocol.

The consequence: adding a new encoder means writing one ~100-line Python file plus one ~5-line YAML. There is no "plugin system" to learn, no lifecycle, no registry. The pipeline is 230 lines of straight-line code that instantiates objects and calls methods on them.

## 2. Features are shared, not owned

A tempting design for a modular pipeline is to put feature computation inside each encoder: "my encoder knows what k-mer size it wants." MegoBin explicitly rejects that. Feature computation lives in [megobin/features/](https://github.com/Tinnifo/Metagenomic-Binning/tree/main/megobin/features) and runs **once** per dataset, writing `kmer_profiles.npy`, `abundance.npy`, and `contig_names.npy` to disk. Every encoder reads from the same files.

This matters for two reasons. First, it removes a confound: if two encoders produce different bins, you know it is the encoder, not a subtly different k-mer pipeline. Second, it makes ablations cheap — swapping the feature set is a config change, not a rerun of DIAMOND or BAM parsing.

A small machinery detail supports this. The `features` config declares `required_signals: [kmers, abundance]` and each `dataset` config declares `signals: [kmers, abundance, taxonomy]`. When you run the pipeline, `_check_signal_compatibility` in [pipeline.py](https://github.com/Tinnifo/Metagenomic-Binning/blob/main/megobin/pipeline.py) compares the two and aborts with a clear error if your features need a signal the dataset does not provide — **before** any file I/O.

## 3. Hydra for composition, YAML for pinning

All configuration is YAML, composed by [Hydra](https://hydra.cc/). An experiment config under `configs/experiment/` declares its defaults — which dataset, features, encoder, loss, binner, trainer, logger, pair sampler, evaluator — and Hydra resolves the graph:

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

There are two kinds of experiment config and it is worth understanding the split. `configs/experiment/hybrid_uncertain_gen.yaml` is an **ad-hoc exploration** config — it picks reasonable defaults but lets you override things from the CLI. `configs/experiment/training/uncertain_gen_cami_toy.yaml` is a **pinned reproducible run** config — every slot is specified, every hyperparameter inherits from its dedicated file, and running it twice gives you the same thing twice.

The rule of thumb: experiment with the top-level configs, publish with the training configs.

Optimizers and schedulers use Hydra's `_partial_` pattern. `configs/optimizer/adam.yaml` looks like this:

```yaml
_target_: torch.optim.Adam
_partial_: true
lr: 1e-3
betas: [0.9, 0.999]
weight_decay: 0.0
```

`_partial_: true` means Hydra instantiates a **factory** that takes parameters as an argument, not a fully-constructed optimizer. The trainer calls `factory(encoder.parameter_groups()[phase_name])` when it is ready to bind the optimizer to a specific parameter group. That detail matters for two-phase training (Chapter 4 on two-phase training), where the optimizer needs to be rebuilt from scratch between phases.

## 4. Protocols do not reach into each other

Every contract in the codebase is one-directional. The pipeline depends on the Protocols; implementations depend only on PyTorch, NumPy, and standard library. **No component imports from another component.** SemiBin does not import UncertainGen. Infomap does not import DBSCAN. The Hinge loss does not import the Mahalanobis loss.

This is the rule you will break first if you are not careful. The temptation is to "reuse the covariance projection from UncertainGen" or "borrow the must-link sampler from SemiBin." Do not. Copy the code; pay the small duplication cost. The benefit is that you can delete any component without a cascade of broken imports, and you can compare two encoders knowing for a fact that they do not share hidden state.

The `Logger` Protocol is a particularly good example. Trainers call `logger.log_scalars({...}, step=N)` without knowing or caring whether that logger writes to TensorBoard, to a file, or to `/dev/null`. Swapping `logger=tensorboard` for `logger=none` in a CLI override is a full replacement of the logging backend, with zero code changes.

## Why this shape?

The design is optimized for one thing: **making "what if we tried X?" cheap**. X is usually a new encoder, a new loss, a new binner, or an ablation of one of those. In a research codebase, the number of such "what if" experiments you run per week is the rate-limiting variable for progress. Every structural decision above is in service of keeping that rate high.
