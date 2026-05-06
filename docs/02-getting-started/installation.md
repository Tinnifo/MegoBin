# Installation

Three environments: laptop (edits + smoke tests), BioCloud (pipeline + CheckM2), DEIS-MCC (GPU training).

## Prerequisites

`git`, `mamba` (preferred over `conda`), SSH keys for any cluster you'll use. ~15 GB for the env, plus dataset (CAMI_toy <5 GB, CAMI_medium ~40 GB).

## Clone

```bash
git clone https://github.com/Tinnifo/Metagenomic-Binning.git
cd Metagenomic-Binning
```

## Option A — Conda (laptop, BioCloud)

```bash
mamba env create -f environment.yml
mamba activate megobin
pip install -e .
```

Verify:

```bash
python -c "import megobin; print(megobin.__file__)"
pytest tests/test_interfaces.py -v
```

## Option B — Singularity (HPC)

Build once:

```bash
singularity build megobin.sif environment.def
```

Run:

```bash
# DEIS-MCC
singularity exec --nv megobin.sif python megobin/pipeline.py ...

# BioCloud
apptainer run --nvccli megobin.sif python megobin/pipeline.py ...
```

DEIS-MCC gotcha: `export SINGULARITY_TMPDIR=/scratch/$(whoami)` before building.

## Option C — Read-only

```bash
pip install hydra-core omegaconf pyyaml
pip install -e .
```

Enough to import `megobin` and compose configs. No training.

## Cluster notes

- **BioCloud** — `bash -l` shebang in SLURM scripts; profile at `--profile biocloud`.
- **DEIS-MCC** — request GPUs via `--gres=gpu:N --partition=turing` (6× T4) or `--partition=ada` (2× L4). `/scratch` (~60 GB) is non-persistent — copy outputs before job ends.

See Chapter 6 for full HPC setup.
