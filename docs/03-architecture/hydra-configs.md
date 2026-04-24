# Hydra configs

MegoBin uses [Hydra](https://hydra.cc/) for all configuration. If you have not used Hydra before, this chapter is enough to be productive: you will understand how `configs/experiment/hybrid_uncertain_gen.yaml` turns into a runnable pipeline, how to override any field from the CLI, and what the `_target_` and `_partial_` keys do.

## The config tree

The full `configs/` tree is about 20 files. Every subfolder maps to one slot in the pipeline:

```
configs/
├── dataset/                 CAMI_medium.yaml, CAMI_toy.yaml
├── features/                canonical_kmer.yaml, canonical_kmer_abundance.yaml
├── representation/          uncertain_gen.yaml, semibin_encoder.yaml
├── loss/                    hinge.yaml, mahalanobis_bce.yaml
├── binner/                  infomap.yaml, dbscan_ensemble.yaml
├── evaluator/               checkm2.yaml
├── pair_sampler/            hybrid.yaml, semibin.yaml, uncertain_gen.yaml
├── trainer/                 single_phase.yaml, two_phase.yaml
├── optimizer/               adam.yaml, adamw.yaml, sgd.yaml
├── scheduler/               constant.yaml, cosine.yaml, step_lr.yaml
├── logger/                  tensorboard.yaml, none.yaml
└── experiment/              hybrid_uncertain_gen.yaml, random_pairs_only.yaml,
                             semibin_pairs_only.yaml, training/
```

Each leaf file specifies exactly one instantiable object: a dataset descriptor, an encoder, a loss, a sampler, and so on. The top-level files in `configs/experiment/` compose them.

## A single YAML file — what's inside

The simplest shape is a slot config. `configs/representation/uncertain_gen.yaml` is four lines:

```yaml
_target_: megobin.representations.uncertain_gen.UncertainGenRepresentation
input_dim: 256
hidden_dim: 512
embedding_dim: 256
dropout: 0.2
```

`_target_` is the fully-qualified Python class name Hydra will import and instantiate. Everything else is keyword arguments passed to that class's `__init__`. When `pipeline.py` calls `hydra.utils.instantiate(cfg.representation)`, it runs:

```python
from megobin.representations.uncertain_gen import UncertainGenRepresentation
UncertainGenRepresentation(input_dim=256, hidden_dim=512, embedding_dim=256, dropout=0.2)
```

That's all there is to component instantiation.

## `_partial_: true` — factories, not instances

Optimizers and schedulers use a variant. `configs/optimizer/adam.yaml`:

```yaml
_target_: torch.optim.Adam
_partial_: true
lr: 1e-3
betas: [0.9, 0.999]
weight_decay: 0.0
```

With `_partial_: true`, `hydra.utils.instantiate` returns a **factory** — a `functools.partial(torch.optim.Adam, lr=1e-3, ...)` — not a bound optimizer. The trainer decides when (and for which parameters) to call the factory. That's what lets the two-phase trainer instantiate a fresh Adam for each phase, bound to a different `parameter_group`, without the config knowing which parameters will end up in each phase.

## Composing an experiment

An experiment config is a single YAML that imports the leaf configs and stitches them together. `configs/experiment/hybrid_uncertain_gen.yaml`:

```yaml
# @package _global_

defaults:
  - _self_
  - /dataset: CAMI_medium
  - /features: canonical_kmer_abundance
  - /representation: uncertain_gen
  - /loss: mahalanobis_bce
  - /binner: infomap
  - /evaluator: checkm2
  - /pair_sampler: hybrid
  - /trainer: two_phase
  - /logger: tensorboard

seed: 42
use_abundance: true

representation:
  input_dim: 236
  embedding_dim: 256
```

Three things are happening. First, the `# @package _global_` directive tells Hydra "merge my keys into the top-level config" rather than nesting under an `experiment:` key. Second, the `defaults:` list pulls in one file from each slot folder — Hydra resolves the paths relative to `configs/` and merges them into a single DictConfig. Third, the final block overrides `seed`, `use_abundance`, and two fields of the `representation` section. The `representation.input_dim` override in particular says "start from `configs/representation/uncertain_gen.yaml`, then replace `input_dim: 256` with `input_dim: 236`" — that 236 is 136 canonical k-mers + 2 × 50 BAM coverage features.

## CLI overrides

Hydra's dot-notation overrides let you change any value from the command line. Four common patterns:

**Swap a component wholesale.** Replace `representation: uncertain_gen` with `representation: semibin_encoder`:

```bash
python megobin/pipeline.py --config-name experiment/hybrid_uncertain_gen representation=semibin_encoder
```

**Change a field inside a component.**

```bash
python megobin/pipeline.py --config-name experiment/hybrid_uncertain_gen representation.dropout=0.3
```

**Add a new key.** Use `+` to add a key that doesn't exist in the base config:

```bash
python megobin/pipeline.py --config-name experiment/hybrid_uncertain_gen +debug=true
```

**Sweep (multirun mode).** `-m` or `--multirun`:

```bash
python megobin/pipeline.py -m --config-name experiment/hybrid_uncertain_gen seed=1,2,3
```

That last one runs three experiments with seeds 1, 2, 3, each under its own `multirun/<date>/<time>/<N>/` subdirectory.

## Two flavours of experiment config

The project distinguishes two styles of experiment:

**Ad-hoc exploration** lives in `configs/experiment/*.yaml` — `hybrid_uncertain_gen.yaml`, `random_pairs_only.yaml`, `semibin_pairs_only.yaml`. These inherit most hyperparameters from their component defaults and are fine to override on the CLI. Use them while you're still figuring out the experiment.

**Pinned reproducible runs** live in `configs/experiment/training/` — `uncertain_gen_cami_toy.yaml`, `semibin_cami_toy.yaml`. These specify every slot, cite their hyperparameter sources in inline comments, and are the canonical "re-run this exactly" configs. Convention: one file per (encoder, dataset) pair. Use them when you're ready to publish a result.

The naming rule is simple: `{encoder}_{dataset}.yaml`. To add a new one — say SemiBin on CAMI_medium — copy `semibin_cami_toy.yaml`, swap `/dataset: CAMI_toy` for `/dataset: CAMI_medium`, and adjust `representation.input_dim` to match the new BAM count.

## The resolved config, in one command

Hydra's `--cfg job` mode is the best single debugging tool. It prints the fully-resolved config without running the pipeline:

```bash
python megobin/pipeline.py \
  --config-name experiment/hybrid_uncertain_gen \
  representation.dropout=0.5 \
  --cfg job
```

Use this any time the pipeline is doing something surprising — nine times out of ten a mis-override is visible in the printed config before anything else.

## Interpolations

Values can reference each other. `configs/pair_sampler/hybrid.yaml`:

```yaml
_target_: megobin.data.hybrid_sampler.HybridPairSampler
neg_per_pos: 1000
taxonomy_fraction: 0.5
seed: ${seed}
```

`${seed}` pulls from the top-level `seed` field in the composed config. `${hydra:runtime.output_dir}` (used in `configs/trainer/*.yaml` for `checkpoint_path` and in `configs/logger/tensorboard.yaml` for `logdir`) resolves to Hydra's auto-generated per-run output directory.

## A mental model for debugging configs

Three failure modes cover most of what you'll hit:

**"Field not found" errors** mean you're using `.` syntax for a key that doesn't exist in the base. Add `+` before the override, or look at `--cfg job` to see the real field name.

**"Cannot instantiate" errors** mean `_target_` points at a class that either doesn't exist or has a different `__init__` signature than the YAML supplies. Grep the target FQN and check the constructor.

**Silent wrong behavior** is the worst one — the config parses but does the wrong thing. Run `--cfg job` and diff against a known-good config. If your override changed something unexpected because of merge semantics, this is where you'll spot it.

Chapter 4's tutorials use this machinery everywhere. If the above feels shaky, try them first — swapping an encoder is mostly a CLI override, and you'll develop intuition for Hydra by doing.
