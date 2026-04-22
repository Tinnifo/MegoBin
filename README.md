# MegoBin

MegoBin is a modular research pipeline for **metagenomic binning** — the problem of clustering assembled contigs into bins that each correspond to a single microbial genome. It exists to support rapid iteration on new deep-learning approaches, particularly for recovering low-abundance and microbial-dark-matter organisms that established tools like VAMB and SemiBin2 struggle with. Every part of the pipeline — encoders, losses, pair samplers, trainers, binners, evaluators, loggers — is a swappable slot defined by a Python `Protocol` and composed via [Hydra](https://hydra.cc/) YAML configs, so adding a new method means writing one file and dropping in a config.

```
Dataset → Features (shared) → Representation → Trainer → Binner → Evaluator
                                    ↑            ↑
                                Loss, Sampler   Optimizer, Scheduler, Logger
```

## What's in the box

**Encoders** ([src/representations/](src/representations/))

| Encoder | Architecture | Loss | Distance |
|---------|--------------|------|----------|
| **Poisson** | Embedding table (65K params) | Poisson NLL on k-mer co-occurrence | L1 |
| **Contrastive MLP** | 2-layer Siamese MLP (262K params) | BCE on exp(-d²) | L2 |
| **UncertainGen** | Dual-head MLP: mean + covariance (526K params) | Mahalanobis BCE | Mahalanobis |
| **SemiBin** | 3-layer MLP (100-d output) | Hinge contrastive | L2 |

**Binners** ([src/binners/](src/binners/)): K-Medoids (greedy), Infomap (dual k-NN graph + community detection), DBSCAN ensemble.

**Trainers** ([src/trainers/](src/trainers/)): `SinglePhaseTrainer`, `TwoPhaseTrainer` (UncertainGen's mean→cov schedule and any future N-phase regime). Optimizers and schedulers are Hydra `_partial_` factories, so any `torch.optim` / `torch.optim.lr_scheduler` works.

**Samplers** ([src/data/](src/data/)): `UncertainGenPairSampler`, `SemiBinPairSampler`, `HybridPairSampler`, `CooccurrencePairSampler`.

**Evaluator** ([src/evaluators/checkm2.py](src/evaluators/checkm2.py)): CheckM2 subprocess wrapper returning a DataFrame of completeness + contamination per bin.

## Install

### Option A — Conda (local development, BioCloud)

```bash
mamba env create -f environment.yml
mamba activate megobin
```

On BioCloud use `mamba` (pre-installed, faster than conda). On any other machine either `mamba` or `conda` works.

### Option B — Singularity / Apptainer (HPC, reproducibility)

```bash
# Build once (on a machine with root/fakeroot, or via Sylabs Cloud)
singularity build metagenomic-binning.sif environment.def

# DEIS-MCC (GPU):
singularity exec --nv metagenomic-binning.sif python src/pipeline.py ...

# BioCloud (GPU — note different flag):
apptainer run --nvccli metagenomic-binning.sif python src/pipeline.py ...
```

See [environment.def](environment.def) for the container spec.

## Run an experiment

```bash
# Primary baselines (pre-pinned hyperparameters under configs/experiment/)
python src/pipeline.py --config-name experiment/baseline_rk              # Poisson
python src/pipeline.py --config-name experiment/hybrid_uncertain_gen     # UncertainGen (primary)
python src/pipeline.py --config-name experiment/semibin_pairs_only       # UncertainGen + SemiBin pairs
python src/pipeline.py --config-name experiment/random_pairs_only        # UncertainGen + random pairs

# Fully-reproducible per-(encoder, dataset) runs (configs/experiment/training/)
python src/pipeline.py --config-name experiment/training/poisson_cami_toy
python src/pipeline.py --config-name experiment/training/contrastive_mlp_cami_toy
python src/pipeline.py --config-name experiment/training/uncertain_gen_cami_toy
python src/pipeline.py --config-name experiment/training/semibin_cami_toy

# Override any slot without editing YAML
python src/pipeline.py --config-name experiment/baseline_rk representation=contrastive_mlp loss=bce

# Resume from a saved checkpoint (skips training)
python src/pipeline.py --config-name experiment/baseline_rk resume_from=outputs/2026-04-22/12-00-00/encoder.pt
```

Every run writes TensorBoard event files, a `run_meta.json`, and a checkpoint into a fresh `outputs/<date>/<time>/` directory.

## Add a new encoder

The pipeline discovers components by Protocol conformance — no registration step. To add e.g. a `DNABERT-S` encoder:

1. **Implement the Protocol.** Create [src/representations/dnabert_s.py](src/representations/) subclassing `nn.Module` and satisfying the `Representation` contract in [src/representations/base.py](src/representations/base.py):
   - `encode(features: np.ndarray) -> np.ndarray` — inference path
   - `training_step(batch, loss_fn) -> Tensor` — forward + loss for one batch
   - `parameter_groups() -> dict[str, list[nn.Parameter]]` — named groups for phase-based trainers (at minimum `{"all": [...]}`)
   - `embedding_dim: int` property

2. **Add a config.** Create `configs/representation/dnabert_s.yaml` with `_target_: src.representations.dnabert_s.DNABertS` plus any constructor kwargs.

3. **Run it.**
   ```bash
   python src/pipeline.py --config-name experiment/baseline_rk representation=dnabert_s
   ```

4. **Update tests** — extend [tests/test_interfaces.py](tests/test_interfaces.py) with Protocol-compliance and [tests/test_overfit_batch.py](tests/test_overfit_batch.py) with a smoke test that loss decreases on 100 contigs.

Zero changes to [src/pipeline.py](src/pipeline.py) required. The same recipe works for new losses ([src/losses/](src/losses/)), binners ([src/binners/](src/binners/)), samplers ([src/data/](src/data/)), trainers ([src/trainers/](src/trainers/)), loggers ([src/utils/](src/utils/)) — find the Protocol in the matching `base.py`, implement, add a YAML, done.

## Project layout

```
configs/
  dataset/          Capability descriptors (which signals live on disk)
  features/         K-mer + abundance feature configs
  representation/   Encoder configs
  loss/             Loss configs
  binner/           Binner configs
  pair_sampler/     Sampler configs
  trainer/          Trainer configs
  optimizer/        Optimizer partials (Adam, AdamW, SGD)
  scheduler/        LR scheduler partials (step, cosine, constant)
  logger/           TensorBoard / no-op logger configs
  evaluator/        CheckM2 config
  experiment/       Composed experiments; training/ holds fully-pinned runs

src/
  representations/  Encoder implementations + Protocol
  losses/           Loss functions + Protocol
  binners/          Clustering implementations + Protocol
  evaluators/       CheckM2 wrapper + Protocol
  features/         Shared k-mer profiles + abundance computation
  data/             Pair samplers + Protocol
  trainers/         SinglePhase + TwoPhase trainers + Protocol
  pipeline.py       Main entry point (Hydra)
  utils/            Logger Protocol, TensorBoard / no-op implementations,
                    checkpoint save/load

tests/              Protocol compliance, overfit smoke, invariance, directional,
                    feature validation, pair samplers, trainers, checkpoints,
                    logger, dataset compatibility, end-to-end integration

hpc/                Snakefile + SLURM scripts for BioCloud / DEIS-MCC
```

## Experiment tracking (TensorBoard)

Each Hydra run writes a `SummaryWriter` event file, a `run_meta.json` (git SHA, environment hash, DVC version, checkpoint paths), and a `checkpoints/` directory under `outputs/<date>/<time>/tb/`.

```bash
# Locally
tensorboard --logdir outputs/

# On a cluster, tunnel from your workstation
ssh -L 6006:localhost:6006 <cluster>
# then on the cluster:
tensorboard --logdir outputs/ --port 6006 --bind_all
```

Swap loggers via config: `python src/pipeline.py logger=tensorboard` or `logger=none`. Push the `checkpoints/` tree through DVC if you need versioned artifact lineage.

## HPC

- **BioCloud** — feature computation, Snakemake pipeline, CheckM2 evaluation
- **DEIS-MCC** — GPU training (turing nodes: 6× T4; ada nodes: 2× L4)

```bash
# BioCloud: full pipeline
sbatch hpc/slurm/biocloud_pipeline.sh CAMI_medium

# DEIS-MCC: GPU training
sbatch hpc/slurm/mcc_train.sh contrastive_mlp CAMI_medium

# Either cluster: smoke test
sbatch hpc/slurm/smoke_test.sh
```

Features (small `.npy`) flow BioCloud → DEIS-MCC via `rsync`; trained `.pt` checkpoints flow back for CheckM2 evaluation.

## Tests

```bash
pytest tests/                                  # full suite
pytest tests/test_interfaces.py                # Protocol compliance
pytest tests/test_overfit_batch.py             # encoder smoke test
pytest tests/test_invariance.py                # reverse complement invariance
pytest tests/test_directional.py               # binner sanity (ARI > 0.95)
pytest tests/test_feature_validation.py        # k-mer profile data checks
pytest tests/test_pair_samplers.py             # pair sampler shapes
pytest tests/test_trainers.py                  # trainer + phase isolation
pytest tests/test_checkpoints.py               # save/load round-trip
pytest tests/test_logger.py                    # TensorBoard + NoOp logger
pytest tests/test_dataset_compatibility.py     # fail-fast signal check
pytest tests/test_end_to_end.py                # full pipeline on synthetic data
```

## Further reading

- [CLAUDE-codebase.md](CLAUDE-codebase.md) — full architecture spec (encoder details, Protocols, hyperparameters, HPC conventions, build order)
- [CLAUDE.md](CLAUDE.md) — internal notes for Claude Code when working in this repo
