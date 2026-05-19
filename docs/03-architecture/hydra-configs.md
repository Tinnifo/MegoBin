# Hydra configs

Built on [Hydra](https://hydra.cc/docs/intro/) + [OmegaConf](https://omegaconf.readthedocs.io/). If anything below is unfamiliar, the linked sections are the canonical reference.

## Tree

```
configs/
├── dataset/         example (placeholder; copy and edit for your data)
├── features/        canonical_kmer, canonical_kmer_abundance
├── encoder/         uncertain_gen
├── loss/            hinge, mahalanobis_bce
├── binner/          dbscan_ensemble
├── filter/          no_op, uncertainty
├── evaluator/       checkm2
├── pair_sampler/    uncertain_gen
├── trainer/         single_phase, two_phase
├── optimizer/       adam, adamw, sgd
├── scheduler/       constant, cosine, step_lr
├── logger/          tensorboard, none
└── experiment/      composed runs
```

Each subdir is a [Hydra config group](https://hydra.cc/docs/tutorials/basic/your_first_app/config_groups/). CLI: `<group>=<config_name>`.

## A slot config

```yaml
# configs/encoder/uncertain_gen.yaml
_target_: megobin.encoders.uncertain_gen.UncertainGenEncoder
input_dim: 256
hidden_dim: 512
embedding_dim: 256
dropout: 0.2
```

[`hydra.utils.instantiate(cfg.encoder)`](https://hydra.cc/docs/advanced/instantiate_objects/overview/) becomes `UncertainGenEncoder(**kwargs)`.

## `_partial_: true`

Optimizers and schedulers return a **factory**, not an instance ([partial instantiation](https://hydra.cc/docs/advanced/instantiate_objects/overview/#partial-instantiation)):

```yaml
# configs/optimizer/adam.yaml
_target_: torch.optim.Adam
_partial_: true
lr: 1e-3
betas: [0.9, 0.999]
weight_decay: 0.0
```

Trainer calls `factory(parameter_group)` when ready. Lets two-phase rebuild Adam per phase.

## Composing an experiment

```yaml
# @package _global_

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

seed: 42

encoder:
  input_dim: 136
  embedding_dim: 256
```

[`# @package _global_`](https://hydra.cc/docs/advanced/overriding_packages/) flattens keys to top level.

[`_self_`](https://hydra.cc/docs/advanced/defaults_list/#composition-order-of-primary-config) goes **last** in `defaults:` so the experiment's own keys (e.g. the `encoder:` block) override the slot defaults pulled in above. Putting `_self_` first silently inverts that — the slot YAML wins and your overrides are dropped.

`encoder.input_dim` must equal `data_split.csv` column count (kmer features). In single-sample mode that's 136; in multi-sample mode the [norm_abundance gate](../02-getting-started/data-layout.md#the-norm_abundance-gate) keeps abundance and the dim grows to `136 + n_bams * 2`. `pipeline.py` trims `data.csv` automatically when the gate fires so training and inference dims align.

## CLI overrides

See [How to run experiments from CLI](../04-tutorials/how-to-run-experiments-from-cli.md)

```bash
# Swap component
filter=uncertainty

# Field inside a component
encoder.dropout=0.3

# Add a missing key
+debug=true

# Sweep — see https://hydra.cc/docs/tutorials/basic/running_your_app/multi-run/
-m seed=1,2,3
```

## Two flavours

- `configs/experiment/*.yaml` — ad-hoc; override-friendly.
- Pinned reproducible configs can live alongside or in their own subdir; the only contract is that they bind every slot.

## Debug a config

```bash
python megobin/pipeline.py --config-name <name> --cfg job
```

Prints resolved config without running. Best single debugging tool. See [Hydra command-line flags](https://hydra.cc/docs/advanced/hydra-command-line-flags/) for `--cfg`, `--info`, `--resolve`, etc.

## Interpolations

[OmegaConf variable interpolation](https://omegaconf.readthedocs.io/en/latest/usage.html#variable-interpolation):

- `${seed}` — top-level field
- `${hydra:runtime.output_dir}` — per-run dir (see [Hydra runtime variables](https://hydra.cc/docs/configure_hydra/intro/))

## Failure modes

| Symptom | Cause |
|---------|-------|
| "Field not found" | `.` syntax for missing key. Use `+`, or check `--cfg job`. |
| "Cannot instantiate" | `_target_` wrong, or kwargs don't match `__init__`. |
| Silent wrong behavior | `--cfg job` and diff against known-good. |
