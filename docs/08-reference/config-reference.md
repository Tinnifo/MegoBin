# Config reference

Every component in MegoBin is selected and parameterized via a YAML config. This chapter catalogues every config in the repo: its purpose, its parameters, and when to reach for it. Use it as a lookup when you're writing a CLI override and can't remember whether the binner key is `k_neighbours` or `n_neighbors`.

For how configs *compose* (defaults, `_target_`, `_partial_`, CLI overrides), see [Hydra configs](../03-architecture/hydra-configs.md).

## Top-level layout

```
configs/
├── dataset/          # What data exists, which signals are on disk
├── features/         # How to compute feature vectors from that data
├── encoder/          # Encoder architectures
├── loss/             # Training losses
├── binner/           # Clustering algorithms
├── pair_sampler/     # Pair sampling strategies for contrastive training
├── trainer/          # Training loops
├── optimizer/        # torch.optim factories
├── scheduler/        # lr scheduler factories
├── logger/           # Experiment tracker configs
├── evaluator/        # Bin quality scorers
└── experiment/       # Composed experiments (inherit from the above)
```

Each subdirectory is a Hydra "group". CLI override: `<group>=<config_name>` selects one file from the group.

## `dataset/` — data capability descriptors

| File | Purpose |
|------|---------|
| `CAMI_toy.yaml` | Small (50-BAM) CAMI benchmark, declares `[kmers, abundance, taxonomy]` signals. Use for smoke tests and the first-run tutorial. |
| `CAMI_medium.yaml` | Larger CAMI benchmark for real training runs. |

Every dataset config declares:
- `name` — directory name under `data/`
- `path` — relative path
- `num_bams` — expected count
- `signals` — what's available; used for the fail-fast dataset compatibility check

## `features/` — feature extractors

| File | Produces | Dim |
|------|----------|-----|
| `canonical_kmer.yaml` | 136 canonical 4-mers | 136 |
| `canonical_kmer_abundance.yaml` | 136 canonical 4-mers + 2×50 BAM coverage | 236 |

Key params:
- `k: 4` — mer size
- `canonical: true` — collapse reverse complements
- `alphabet: ATGC` — skip contigs containing N
- `pseudocount: 1e-5` — Laplace smoothing
- `min_length: 1000` — filter short contigs
- `split_min_length: 2000` — threshold for generating two half-profiles per contig (must-link pair)

## `encoder/` — encoders

| File | Arch | Embedding dim | Params |
|------|------|---------------|--------|
| `semibin_encoder.yaml` | 3-layer MLP (input → 512 → 512 → 100), BN, LeakyReLU | 100 | ~300K |
| `uncertain_gen.yaml` | Dual-head MLP: shared trunk → (mean head, cov head) | 100 (+100 log-σ) | 526K |

Key shared params (both):
- `input_dim: 236` — default feature dim
- `embedding_dim: 100`
- `dropout: 0.1`

UncertainGen-specific:
- `hidden_dim: 512`
- `include_std: false` — controls whether forward emits `[μ | σ]` concat (true) or just μ (false). Toggled at phase boundary in two-phase training.

## `loss/` — contrastive losses

| File | Target encoder | Distance | Key param |
|------|----------------|----------|-----------|
| `hinge.yaml` | SemiBin | L2 on raw embeddings | `margin: 1.0` |
| `mahalanobis_bce.yaml` | UncertainGen | Mahalanobis (learned cov) | `clamp_threshold: 1.0`, `include_std: false` |

`include_std` on `MahalanobisBCELoss` mirrors the encoder flag — in phase 1 both are `false`, in phase 2 both are `true`. The `TwoPhaseTrainer` flips them at phase boundaries via `loss_attrs`.

## `binner/` — clustering algorithms

| File | Algorithm | Typical use |
|------|-----------|-------------|
| `infomap.yaml` | Dual k-NN graph + Infomap community detection | Short reads, general default |
| `dbscan_ensemble.yaml` | 12-eps DBSCAN sweep + per-bin F1 selection | Long reads; more bins, finer granularity |

`infomap.yaml` params:
- `k_neighbours: 200` — k-NN graph connectivity
- `n_trials: 10` — Infomap restart count
- `max_bin_size: null` — cap on a single bin's size

`dbscan_ensemble.yaml` params:
- `eps_values: [0.01, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.55]`
- `min_samples: 5`
- `min_bin_size: 1`

## `pair_sampler/` — contrastive pair strategies

| File | Pair source | Works with |
|------|-------------|------------|
| `semibin.yaml` | Must-link from split halves + cannot-link from k-mer distance | SemiBin, UncertainGen |
| `uncertain_gen.yaml` | Random sampling, label = ground-truth match | UncertainGen |
| `hybrid.yaml` | Mix of must-link, cannot-link, random, and taxonomy-informed | UncertainGen (primary) |

## `trainer/` — training loops

| File | Class | When |
|------|-------|------|
| `single_phase.yaml` | `SinglePhaseTrainer` | SemiBin and anything with one optimizer |
| `two_phase.yaml` | `TwoPhaseTrainer` | UncertainGen's mean→cov schedule; any N-phase schedule |

`single_phase.yaml` key params:
- `epochs: 10`, `batch_size: 2048`, `grad_clip: null`
- `params: all` — which parameter group to train
- `checkpoint_path: ${hydra:runtime.output_dir}/encoder.pt`
- `checkpoint_every: null` — or an integer epoch stride

`two_phase.yaml` key params:
- `phases: [ { ... }, { ... } ]` — a list, each phase a dict of `params`, `epochs`, `optimizer`, `scheduler`, `encoder_attrs`, `loss_attrs`.
- `checkpoint_per_phase: false` — flip to true to keep `encoder_phase_1.pt`, `encoder_phase_2.pt`.

See [Two-phase training](../04-tutorials/two-phase-training.md) for the full phase structure.

## `optimizer/` — `_partial_` factories

| File | Class | Default lr |
|------|-------|------------|
| `adam.yaml` | `torch.optim.Adam` | 1e-3 |
| `adamw.yaml` | `torch.optim.AdamW` | 1e-3 |
| `sgd.yaml` | `torch.optim.SGD` | 1e-2 |

All three set `_partial_: true`, so Hydra instantiates a factory. The trainer binds parameters at the right moment (per-phase for `TwoPhaseTrainer`, once for `SinglePhaseTrainer`).

## `scheduler/` — `_partial_` factories

| File | Class |
|------|-------|
| `constant.yaml` | `torch.optim.lr_scheduler.ConstantLR` (effectively no-op) |
| `step_lr.yaml` | `torch.optim.lr_scheduler.StepLR` |
| `cosine.yaml` | `torch.optim.lr_scheduler.CosineAnnealingLR` |

Also `_partial_: true`. Set `scheduler: null` in a trainer config to disable scheduling.

## `logger/` — experiment trackers

| File | Class | Behaviour |
|------|-------|-----------|
| `tensorboard.yaml` | `TensorBoardLogger` | Writes events + checkpoints + run_meta.json |
| `none.yaml` | `NoOpLogger` | Writes nothing; Protocol satisfied, every call a no-op |

`tensorboard.yaml` key params:
- `logdir: ${hydra:runtime.output_dir}/tb`
- `name: null` — or a descriptive string for TB UI
- `flush_secs: 30`

## `evaluator/` — bin quality scorers

| File | Class | Backs |
|------|-------|-------|
| `checkm2.yaml` | `CheckM2Evaluator` | `checkm2 predict` subprocess |

Key params:
- `threads: 8` — passed to CheckM2
- `model: null` — CheckM2 model file (default model otherwise)

## `experiment/` — composed runs

Top-level experiment configs that compose all of the above. These are what you point `--config-name` at.

### Headline experiments

| File | Encoder | Pair sampler | Trainer | Notes |
|------|----------------|--------------|---------|-------|
| `experiment/hybrid_uncertain_gen.yaml` | UncertainGen | hybrid | two_phase | Primary baseline |
| `experiment/semibin_pairs_only.yaml` | UncertainGen | semibin | two_phase | SemiBin-style pairs, UncertainGen encoder |
| `experiment/random_pairs_only.yaml` | UncertainGen | uncertain_gen (random) | two_phase | Ablation: remove must-link structure |

### Pinned reproducible runs

Under `experiment/training/` — each locks every hyperparameter so the run is reproducible to the commit.

| File | Notes |
|------|-------|
| `experiment/training/uncertain_gen_cami_toy.yaml` | UncertainGen on CAMI toy; first-run tutorial targets this |
| `experiment/training/semibin_cami_toy.yaml` | SemiBin baseline on CAMI toy |

## Interpolations that appear throughout

Hydra's `${...}` syntax resolves at instantiation time:

- `${hydra:runtime.output_dir}` — absolute path to the current run's output directory (e.g. `outputs/2026-04-23/14-35-02/`)
- `${seed}` — whatever the top-level `seed` config key resolves to
- `${binner.k_neighbours}` — any path through the config tree

These get used in `checkpoint_path`, `logger.logdir`, and ablation-sweep directory names. Literal dollar signs in strings need escaping: `\${...}`.

## CLI override cheat sheet

```bash
# Swap a group
encoder=semibin_encoder

# Override a single parameter within a group
binner.k_neighbours=50

# Lists
'binner.eps_values=[0.1,0.2,0.3]'

# Multiple groups
encoder=semibin_encoder loss=hinge trainer=single_phase pair_sampler=semibin

# Sweep (multirun) — three seeds × three k values
-m seed=1,2,3 binner.k_neighbours=50,100,200

# Group into a named ablation directory
hydra.run.dir=outputs/ablation_k_neighbours/k\${binner.k_neighbours}_seed\${seed}

# Quiet logging for CI
logger=none

# Skip training, reuse a saved encoder
resume_from=outputs/2026-04-23/14-35-02/encoder.pt
```

## Sources

- [configs/](https://github.com/Tinnifo/Metagenomic-Binning/tree/main/configs) — full tree
- [configs/experiment/](https://github.com/Tinnifo/Metagenomic-Binning/tree/main/configs/experiment) — composed runs
- [configs/trainer/two_phase.yaml](https://github.com/Tinnifo/Metagenomic-Binning/blob/main/configs/trainer/two_phase.yaml) — phase structure reference
