# Config reference

Lookup table for every shipped config. For composition mechanics, see [Hydra configs](../03-architecture/hydra-configs.md).

```
configs/
├── dataset/        ├── encoder/        ├── binner/         ├── trainer/
├── features/       ├── loss/           ├── pair_sampler/   ├── optimizer/
├── evaluator/      ├── logger/         ├── scheduler/      └── experiment/
```

CLI: `<group>=<config_name>`.

## `dataset/`

| File | Purpose |
|------|---------|
| `CAMI_toy.yaml` | 50-BAM CAMI for smoke tests |
| `CAMI_medium.yaml` | Larger CAMI for real runs |

Fields: `name`, `path`, `num_bams`, `signals`.

## `features/`

| File | Produces | Dim |
|------|----------|-----|
| `canonical_kmer.yaml` | 136 canonical 4-mers | 136 |
| `canonical_kmer_abundance.yaml` | 136 canonical 4-mers + 2×50 BAM | 236 |

Params: `k`, `canonical`, `alphabet`, `pseudocount`, `min_length`, `split_min_length`.

## `encoder/`

| File | Arch | Dim | Params |
|------|------|-----|--------|
| `semibin_encoder.yaml` | 3-layer MLP, BN, LeakyReLU | 100 | ~300K |
| `uncertain_gen.yaml` | Dual-head MLP (mean + cov) | 100 (+100 log-σ) | 526K |

Shared: `input_dim`, `embedding_dim`, `dropout`. UncertainGen-only: `hidden_dim`, `include_std`.

## `loss/`

| File | Encoder | Distance | Key param |
|------|---------|----------|-----------|
| `hinge.yaml` | SemiBin | L2 | `margin: 1.0` |
| `mahalanobis_bce.yaml` | UncertainGen | Mahalanobis (learned cov) | `clamp_threshold`, `include_std` |

`include_std` mirrors the encoder flag and is flipped per phase by `TwoPhaseTrainer`.

## `binner/`

| File | Algorithm | Use |
|------|-----------|-----|
| `infomap.yaml` | Dual k-NN graph + Infomap | Short reads, default |
| `dbscan_ensemble.yaml` | 12-eps DBSCAN + F1 selection | Long reads, finer grain |

Infomap params: `k_neighbours: 200`, `n_trials: 10`, `max_bin_size: null`.
DBSCAN params: `eps_values: [0.01..0.55]` (12 values), `min_samples: 5`, `min_bin_size: 1`.

## `pair_sampler/`

| File | Pair source | Works with |
|------|-------------|------------|
| `semibin.yaml` | Split-half must-link + k-mer cannot-link | SemiBin, UncertainGen |
| `uncertain_gen.yaml` | Random sampling, GT label | UncertainGen |
| `hybrid.yaml` | Mix of must/cannot/random/taxonomy | UncertainGen (primary) |

## `trainer/`

| File | Class | When |
|------|-------|------|
| `single_phase.yaml` | `SinglePhaseTrainer` | One optimizer |
| `two_phase.yaml` | `TwoPhaseTrainer` | UncertainGen mean→cov; any N-phase |

`single_phase` keys: `epochs`, `batch_size`, `grad_clip`, `params`, `checkpoint_path`, `checkpoint_every`.
`two_phase` keys: `phases` (list), `checkpoint_path`, `checkpoint_per_phase`.

See [Two-phase training](../04-tutorials/two-phase-training.md) for the phase structure.

## `optimizer/` (all `_partial_: true`)

| File | Class | Default lr |
|------|-------|------------|
| `adam.yaml` | `torch.optim.Adam` | 1e-3 |
| `adamw.yaml` | `torch.optim.AdamW` | 1e-3 |
| `sgd.yaml` | `torch.optim.SGD` | 1e-2 |

## `scheduler/` (all `_partial_: true`)

| File | Class |
|------|-------|
| `constant.yaml` | `ConstantLR` |
| `step_lr.yaml` | `StepLR` |
| `cosine.yaml` | `CosineAnnealingLR` |

`scheduler: null` disables.

## `logger/`

| File | Class | Behaviour |
|------|-------|-----------|
| `tensorboard.yaml` | `TensorBoardLogger` | Events + checkpoints + run_meta.json |
| `none.yaml` | `NoOpLogger` | Silent |

TensorBoard keys: `logdir`, `name`, `flush_secs`.

## `evaluator/`

| File | Class | Backs |
|------|-------|-------|
| `checkm2.yaml` | `CheckM2Evaluator` | `checkm2 predict` |

Keys: `threads: 8`, `model: null`.

## `experiment/`

**Headline** (override-friendly):

| File | Encoder | Sampler | Trainer | Notes |
|------|---------|---------|---------|-------|
| `hybrid_uncertain_gen.yaml` | UncertainGen | hybrid | two_phase | Primary baseline |
| `semibin_pairs_only.yaml` | UncertainGen | semibin | two_phase | SemiBin pairs, UG encoder |
| `random_pairs_only.yaml` | UncertainGen | uncertain_gen | two_phase | Ablation |

**Pinned reproducible** (`experiment/training/`):

| File | Notes |
|------|-------|
| `uncertain_gen_cami_toy.yaml` | First-run tutorial target |
| `semibin_cami_toy.yaml` | SemiBin baseline |

## Interpolations

- `${hydra:runtime.output_dir}` — current run dir
- `${seed}` — top-level seed
- `${binner.k_neighbours}` — any path

Escape literal `$`: `\${...}`.

## CLI cheat sheet

```bash
encoder=semibin_encoder
binner.k_neighbours=50
'binner.eps_values=[0.1,0.2,0.3]'
encoder=semibin_encoder loss=hinge trainer=single_phase pair_sampler=semibin
-m seed=1,2,3 binner.k_neighbours=50,100,200
hydra.run.dir=outputs/ablation_k_neighbours/k\${binner.k_neighbours}_seed\${seed}
logger=none
resume_from=outputs/2026-04-23/14-35-02/encoder.pt
```

## Sources

- [configs/](https://github.com/Tinnifo/Metagenomic-Binning/tree/main/configs)
- [configs/experiment/](https://github.com/Tinnifo/Metagenomic-Binning/tree/main/configs/experiment)
- [configs/trainer/two_phase.yaml](https://github.com/Tinnifo/Metagenomic-Binning/blob/main/configs/trainer/two_phase.yaml)
