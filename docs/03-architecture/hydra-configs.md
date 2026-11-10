# Hydra configs

Built on [Hydra](https://hydra.cc/docs/intro/) + [OmegaConf](https://omegaconf.readthedocs.io/). If anything below is unfamiliar, the linked sections are the canonical reference.

## Tree

```
configs/
├── dataset/         example (placeholder; copy and edit for your data)
├── pipeline/        embed_and_bin, bin_raw_features (stage wiring + target)
├── features/        canonical_kmer, canonical_kmer_abundance
├── encoder/         semibin2, uncertain_gen
├── loss/            hinge, mahalanobis_bce
├── binner/          dbscan_ensemble, dbscan_ensemble_semibin
├── filter/          no_op, uncertainty
├── evaluator/       checkm2 (used by `python -m megobin.evaluate`)
├── pair_sampler/    semibin2, uncertain_gen
├── trainer/         single_phase, two_phase
├── optimizer/       adam, adamw, sgd
├── scheduler/       constant, cosine, step_lr
├── logger/          tensorboard, none
├── evaluate.yaml    root config of the standalone judge
└── experiment/      composed runs (semibin2_longread, uncertain_gen_dbscan)
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
  - /pipeline: embed_and_bin     # stage wiring: features -> embedder -> filter -> binner
  - /features: canonical_kmer_abundance
  - /encoder: uncertain_gen
  - /loss: mahalanobis_bce
  - /pair_sampler: uncertain_gen
  - /trainer: two_phase
  - /filter: no_op
  - /binner: dbscan_ensemble
  - /logger: tensorboard
  - _self_

seed: 42

pipeline:
  name: uncertain_gen_dbscan
  target: bins                   # bins_refined also pulls in the optional refiner
  interventions: []              # ledger: [{name, level: L0..L3, note}]

encoder:
  input_dim: 136
  embedding_dim: 256
```

`# @package _global_` (first line of the experiment YAML; see [Hydra Packages](https://hydra.cc/docs/advanced/overriding_packages/)) — Without this, Hydra would nest this file under an `experiment:` key (because it lives in `configs/experiment/`). With it, `seed`, `encoder`, and your overrides land at the **top level** of the merged config, which is what the pipeline expects (`cfg.encoder`, `cfg.seed`, etc.).

**Without `_global_` (conceptually):**

```yaml
experiment:
  seed: 42
  encoder:
    input_dim: 136
```

**With `_global_` (what MegoBin uses):**

```yaml
seed: 42
encoder:
  input_dim: 136
  # ...rest from configs/encoder/uncertain_gen.yaml
trainer:
  # ...from configs/trainer/two_phase.yaml
```

`_self_` (see [Hydra defaults list — composition order](https://hydra.cc/docs/advanced/defaults_list/#composition-order)) goes **last** in `defaults:` so the experiment's own keys (e.g. the `encoder:` block) override the slot defaults pulled in above. Putting `_self_` first silently inverts that — the slot YAML wins and your overrides are dropped.

`encoder.input_dim` must equal `data_split.csv` column count (kmer features). In single-sample mode that's 136; in multi-sample mode the [norm_abundance gate](../../megobin/features/views.py) keeps abundance and the dim grows to `136 + n_bams * 2`. The encoder's feature view (`KmerGateView`) trims `data.csv` automatically when the gate fires so training and inference dims align.

The `pipeline` group (`configs/pipeline/*.yaml`) wires the component groups into stages by interpolation (`encoder: ${encoder}`); stage entries use `_recursive_: false` so a stage can instantiate its own parts later (a sampler needs data that only exists at run time). See [Contracts and stages](contracts-and-stages.md).

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
| `ContractError` / exit code 2 | A contract failed before any compute. Read the message (component, port, missing labels) and `status.json`. |
| Silent wrong behavior | `--cfg job` and diff against known-good. |
