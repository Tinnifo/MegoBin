# MegoBin

Modular research pipeline for metagenomic binning. Three swappable slots — Representation, Binner, Evaluator — connected by Protocol interfaces and composed via [Hydra](https://hydra.cc/) YAML configs.

```
Features (shared) --> Representation (encoder) --> Binner (clusterer) --> Evaluator (scorer)
```

## Encoders

| Encoder | Architecture | Loss | Distance |
|---------|-------------|------|----------|
| **Poisson** | Embedding table (65K params) | Poisson NLL on k-mer co-occurrence | L1 |
| **Contrastive MLP** | 2-layer Siamese MLP (262K params) | BCE on exp(-d^2) | L2 |
| **UncertainGen** | Dual-head MLP: mean + covariance (526K params) | Mahalanobis BCE | Mahalanobis |
| **SemiBin** | 3-layer MLP (100-d output) | Hinge contrastive | L2 |

## Binners

| Binner | Method |
|--------|--------|
| **K-Medoids** | Greedy medoid selection with similarity threshold |
| **Infomap** | Dual k-NN graph + community detection |
| **DBSCAN Ensemble** | Multi-eps sweep with best-bin selection |

## Quick Start

```bash
# Setup
mamba env create -f environment.yml
mamba activate megobin

# Run an experiment
python src/pipeline.py --config-name experiment/baseline_rk

# Override components
python src/pipeline.py --config-name experiment/baseline_rk \
    representation=contrastive_mlp loss=bce

# Run tests
pytest tests/
```

## Adding a New Component

1. Implement the Protocol in `src/{slot}/new_component.py`
2. Add `configs/{slot}/new_component.yaml`
3. Run: `python src/pipeline.py {slot}=new_component`

Zero changes to pipeline code.

## Project Structure

```
configs/             Hydra YAML configs (one per component)
src/
  representations/   Encoder implementations + Protocol
  losses/            Loss functions + Protocol
  binners/           Clustering implementations + Protocol
  evaluators/        CheckM2 wrapper + Protocol
  features/          Shared k-mer profiles + abundance computation
  pipeline.py        Main entry point (Hydra)
  utils/             W&B logger
tests/               Protocol compliance, overfit, invariance, directional, feature validation
hpc/                 Snakefile + SLURM scripts for BioCloud / DEIS-MCC
```

## HPC

- **BioCloud** -- feature computation, Snakemake pipeline, CheckM2 evaluation
- **DEIS-MCC** -- GPU training (turing: 6x T4, ada: 2x L4)

```bash
# BioCloud: full pipeline
sbatch hpc/slurm/biocloud_pipeline.sh CAMI_medium

# DEIS-MCC: GPU training
sbatch hpc/slurm/mcc_train.sh contrastive_mlp CAMI_medium

# Either cluster: smoke test
sbatch hpc/slurm/smoke_test.sh
```

## Tests

```bash
pytest tests/                        # all tests
pytest tests/test_interfaces.py      # protocol compliance
pytest tests/test_overfit_batch.py   # encoder smoke test
pytest tests/test_invariance.py      # reverse complement invariance
pytest tests/test_directional.py     # binner sanity (ARI > 0.95)
pytest tests/test_feature_validation.py  # k-mer profile data checks
```
