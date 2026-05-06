# Hydra configs

## Tree

```
configs/
├── dataset/         CAMI_medium, CAMI_toy
├── features/        canonical_kmer, canonical_kmer_abundance
├── encoder/         uncertain_gen, semibin_encoder
├── loss/            hinge, mahalanobis_bce
├── binner/          infomap, dbscan_ensemble
├── evaluator/       checkm2
├── pair_sampler/    hybrid, semibin, uncertain_gen
├── trainer/         single_phase, two_phase
├── optimizer/       adam, adamw, sgd
├── scheduler/       constant, cosine, step_lr
├── logger/          tensorboard, none
└── experiment/      composed runs (+ training/ for pinned runs)
```

Each subdir is a Hydra group. CLI: `<group>=<config_name>`.

## A slot config

```yaml
# configs/encoder/uncertain_gen.yaml
_target_: megobin.encoders.uncertain_gen.UncertainGenEncoder
input_dim: 256
hidden_dim: 512
embedding_dim: 256
dropout: 0.2
```

`hydra.utils.instantiate(cfg.encoder)` becomes `UncertainGenEncoder(**kwargs)`.

## `_partial_: true`

Optimizers and schedulers return a **factory**, not an instance:

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

seed: 42
use_abundance: true

encoder:
  input_dim: 236
  embedding_dim: 256
```

`# @package _global_` flattens keys to top level.

## CLI overrides

```bash
# Swap component
encoder=semibin_encoder

# Field inside a component
encoder.dropout=0.3

# Add a missing key
+debug=true

# Sweep
-m seed=1,2,3
```

## Two flavours

- `configs/experiment/*.yaml` — ad-hoc; override-friendly.
- `configs/experiment/training/*.yaml` — pinned, reproducible. Naming: `{encoder}_{dataset}.yaml`.

## Debug a config

```bash
python megobin/pipeline.py --config-name <name> --cfg job
```

Prints resolved config without running. Best single debugging tool.

## Interpolations

- `${seed}` — top-level field
- `${hydra:runtime.output_dir}` — per-run dir

## Failure modes

| Symptom | Cause |
|---------|-------|
| "Field not found" | `.` syntax for missing key. Use `+`, or check `--cfg job`. |
| "Cannot instantiate" | `_target_` wrong, or kwargs don't match `__init__`. |
| Silent wrong behavior | `--cfg job` and diff against known-good. |
