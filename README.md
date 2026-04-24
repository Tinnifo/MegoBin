# MegoBin

MegoBin is a modular research pipeline for **metagenomic binning** — the problem of clustering assembled contigs into bins that each correspond to a single microbial genome. It exists to support rapid iteration on new deep-learning approaches, particularly for recovering low-abundance and microbial-dark-matter organisms that established tools like VAMB and SemiBin2 struggle with. Every part of the pipeline — encoders, losses, pair samplers, trainers, binners, evaluators, loggers — is a swappable slot defined by a Python `Protocol` and composed via [Hydra](https://hydra.cc/) YAML configs, so adding a new method means writing one file and dropping in a config.

```
Dataset → Features (shared) → Encoder → Trainer → Binner → Evaluator
                                 ↑         ↑
                          Loss, Sampler   Optimizer, Scheduler, Logger
```

## What's in the box

**Encoders** ([megobin/encoders/](megobin/encoders/))

| Encoder | Architecture | Loss | Distance |
|---------|--------------|------|----------|
| **UncertainGen** | Dual-head MLP: mean + covariance (526K params) | Mahalanobis BCE | Mahalanobis |
| **SemiBin** | 3-layer MLP (100-d output) | Hinge contrastive | L2 |

**Binners** ([megobin/binners/](megobin/binners/)): Infomap (dual k-NN graph + community detection, SemiBin short reads), DBSCAN ensemble (SemiBin long reads).

**Trainers** ([megobin/trainers/](megobin/trainers/)): `SinglePhaseTrainer` (SemiBin), `TwoPhaseTrainer` (UncertainGen's mean→cov schedule and any future N-phase regime). Optimizers and schedulers are Hydra `_partial_` factories, so any `torch.optim` / `torch.optim.lr_scheduler` works.

**Samplers** ([megobin/data/](megobin/data/)): `UncertainGenPairSampler`, `SemiBinPairSampler`, `HybridPairSampler`.

**Evaluator** ([megobin/evaluators/checkm2.py](megobin/evaluators/checkm2.py)): CheckM2 subprocess wrapper returning a DataFrame of completeness + contamination per bin.

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
singularity exec --nv metagenomic-binning.sif python megobin/pipeline.py ...

# BioCloud (GPU — note different flag):
apptainer run --nvccli metagenomic-binning.sif python megobin/pipeline.py ...
```

See [environment.def](environment.def) for the container spec.

## Run an experiment

```bash
# Primary baselines (pre-pinned hyperparameters under configs/experiment/)
python megobin/pipeline.py --config-name experiment/hybrid_uncertain_gen     # UncertainGen (primary)
python megobin/pipeline.py --config-name experiment/semibin_pairs_only       # UncertainGen + SemiBin pairs
python megobin/pipeline.py --config-name experiment/random_pairs_only        # UncertainGen + random pairs

# Fully-reproducible per-(encoder, dataset) runs (configs/experiment/training/)
python megobin/pipeline.py --config-name experiment/training/uncertain_gen_cami_toy
python megobin/pipeline.py --config-name experiment/training/semibin_cami_toy

# Override any slot without editing YAML
python megobin/pipeline.py --config-name experiment/hybrid_uncertain_gen encoder=semibin_encoder loss=hinge binner=dbscan_ensemble

# Resume from a saved checkpoint (skips training)
python megobin/pipeline.py --config-name experiment/hybrid_uncertain_gen resume_from=outputs/2026-04-22/12-00-00/encoder.pt
```

Every run writes TensorBoard event files, a `run_meta.json`, and a checkpoint into a fresh `outputs/<date>/<time>/` directory.

## Add a new encoder

The pipeline discovers components by Protocol conformance — no registration step. To add e.g. a `DNABERT-S` encoder:

1. **Implement the Protocol.** Create [megobin/encoders/dnabert_s.py](megobin/encoders/) subclassing `nn.Module` and satisfying the `Encoder` contract in [megobin/encoders/base.py](megobin/encoders/base.py):
   - `encode(features: np.ndarray) -> np.ndarray` — inference path
   - `training_step(batch, loss_fn) -> Tensor` — forward + loss for one batch
   - `parameter_groups() -> dict[str, list[nn.Parameter]]` — named groups for phase-based trainers (at minimum `{"all": [...]}`)
   - `embedding_dim: int` property

2. **Add a config.** Create `configs/encoder/dnabert_s.yaml` with `_target_: megobin.encoders.dnabert_s.DNABertS` plus any constructor kwargs.

3. **Run it.**
   ```bash
   python megobin/pipeline.py --config-name experiment/hybrid_uncertain_gen encoder=dnabert_s
   ```

4. **Update tests** — extend [tests/test_interfaces.py](tests/test_interfaces.py) with a Protocol-compliance case, and optionally add an end-to-end case to [tests/test_end_to_end.py](tests/test_end_to_end.py) (ARI > 0.3 on synthetic Dirichlet genomes).

Zero changes to [megobin/pipeline.py](megobin/pipeline.py) required. The same recipe works for new losses ([megobin/losses/](megobin/losses/)), binners ([megobin/binners/](megobin/binners/)), samplers ([megobin/data/](megobin/data/)), trainers ([megobin/trainers/](megobin/trainers/)), loggers ([megobin/utils/](megobin/utils/)) — find the Protocol in the matching `base.py`, implement, add a YAML, done.

## Project layout

```
configs/
  dataset/          Capability descriptors (which signals live on disk)
  features/         K-mer + abundance feature configs
  encoder/          Encoder configs
  loss/             Loss configs
  binner/           Binner configs
  pair_sampler/     Sampler configs
  trainer/          Trainer configs
  optimizer/        Optimizer partials (Adam, AdamW, SGD)
  scheduler/        LR scheduler partials (step, cosine, constant)
  logger/           TensorBoard / no-op logger configs
  evaluator/        CheckM2 config
  experiment/       Composed experiments; training/ holds fully-pinned runs

megobin/
  encoders/         Encoder implementations + Protocol
  losses/           Loss functions + Protocol
  binners/          Clustering implementations + Protocol
  evaluators/       CheckM2 wrapper + Protocol
  features/         Shared k-mer profiles + abundance computation
  data/             Pair samplers + Protocol
  trainers/         SinglePhase + TwoPhase trainers + Protocol
  pipeline.py       Main entry point (Hydra)
  utils/            Logger Protocol, TensorBoard / no-op implementations,
                    checkpoint save/load

tests/              Protocol compliance, dataset compatibility, end-to-end integration

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

Swap loggers via config: `python megobin/pipeline.py logger=tensorboard` or `logger=none`. Push the `checkpoints/` tree through DVC if you need versioned artifact lineage.

## HPC

- **BioCloud** — feature computation, Snakemake pipeline, CheckM2 evaluation
- **DEIS-MCC** — GPU training (turing nodes: 6× T4; ada nodes: 2× L4)

```bash
# BioCloud: full pipeline
sbatch hpc/slurm/biocloud_pipeline.sh CAMI_medium

# DEIS-MCC: GPU training
sbatch hpc/slurm/deis_mcc_train.sh uncertain_gen CAMI_medium

# Either cluster: smoke test
sbatch hpc/slurm/smoke_test.sh
```

Features (small `.npy`) flow BioCloud → DEIS-MCC via `rsync`; trained `.pt` checkpoints flow back for CheckM2 evaluation.

## Tests

```bash
pytest tests/                                  # full suite
pytest tests/test_interfaces.py                # Protocol compliance for every component
pytest tests/test_dataset_compatibility.py     # fail-fast signal check + required-slot coverage
pytest tests/test_end_to_end.py                # full pipeline on synthetic Dirichlet genomes
```