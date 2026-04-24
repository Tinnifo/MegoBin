# Installation

You have three realistic environments where MegoBin runs: a laptop (usually macOS) for editing and smoke tests, BioCloud (CMC-AAU) for the full feature-computation-plus-evaluation pipeline, and DEIS-MCC for GPU training. The install story is slightly different on each. This chapter walks through all three.

## Prerequisites

You will need `git`, access to a conda-compatible Python distribution (`mamba` is strongly preferred over `conda` — it resolves environments roughly ten times faster), and an SSH key registered with both AAU clusters if you intend to use them. Storage requirements: roughly 15 GB for the conda environment, plus whatever your dataset needs (CAMI_toy is under 5 GB, CAMI_medium is around 40 GB).

## Clone the repository

```bash
git clone https://github.com/Tinnifo/Metagenomic-Binning.git
cd Metagenomic-Binning
```

From here every command assumes you are inside the repo root. The top-level layout is small enough to list in full:

```
CLAUDE-codebase.md       Deep internal documentation (the spec this guide summarizes)
CLAUDE.md                Short top-level project card
README.md                The README this guide complements
configs/                 YAML configs for every swappable slot
environment.yml          Conda environment definition
environment.def          Singularity container definition
hpc/                     Snakefile + SLURM scripts for BioCloud and DEIS-MCC
megobin/                 The Python package
notebooks/               Onboarding walkthrough + example notebook
pyproject.toml           Package metadata (editable install)
tests/                   14 test files, grouped by concern
```

## Option A — Conda (local development, BioCloud)

This is the path you want on your laptop and on BioCloud. BioCloud has `mamba` pre-installed; on a laptop install it via [miniforge](https://github.com/conda-forge/miniforge) or use whatever conda distribution you already have.

```bash
mamba env create -f environment.yml
mamba activate megobin
pip install -e .
```

The `pip install -e .` at the end makes `megobin` importable while still reading from your working copy — edits show up without a reinstall. `pyproject.toml` defines the package name as `megobin` and the package root as the `megobin/` folder.

Verify the install:

```bash
python -c "import megobin; print(megobin.__file__)"
pytest tests/test_interfaces.py -v
```

The first command prints the path to the installed package. The second runs the Protocol-compliance suite — if any encoder, binner, or evaluator fails to satisfy its Protocol, this test tells you so, with a message pointing at the offending method. All Protocol checks should pass on a clean checkout.

## Option B — Singularity / Apptainer (HPC, reproducibility)

On HPC — especially DEIS-MCC, which has a different CUDA version and fewer conveniences than BioCloud — you almost always want to run inside a container instead of fighting conda. The repo ships `environment.def`, a Singularity definition file built from `nvcr.io/nvidia/pytorch:23.04-py3` with the pipeline dependencies layered on top.

Build the image **once**, on a machine with root or fakeroot, or via Sylabs Cloud:

```bash
singularity build megobin.sif environment.def
```

Then on each cluster, the invocation differs only in the GPU flag:

```bash
# DEIS-MCC (CUDA 13.2, Singularity v4.4.0)
singularity exec --nv megobin.sif python megobin/pipeline.py ...

# BioCloud (Apptainer, different GPU flag name)
apptainer run --nvccli megobin.sif python megobin/pipeline.py ...
```

Both clusters run Ubuntu 22.04 as the container base OS, so one `.sif` works everywhere. Copy it between clusters with `rsync` or rebuild locally.

One gotcha on DEIS-MCC: set `SINGULARITY_TMPDIR=/scratch/$(whoami)` before building or pulling images. The default `/tmp` is small and fills up on large builds.

## Option C — Just enough to edit

If you only want to browse the code, read configs, and make YAML-only changes without ever running anything, a minimal install is enough:

```bash
pip install hydra-core omegaconf pyyaml
pip install -e .
```

This gives you `hydra` for config resolution and `megobin` for import. You will not be able to train or evaluate anything, but `python -c "from megobin.representations import UncertainGenRepresentation"` will work, and you can experiment with Hydra composition:

```bash
python -c "
from hydra import compose, initialize
with initialize(config_path='configs', version_base=None):
    cfg = compose(config_name='experiment/hybrid_uncertain_gen')
    print(cfg)
"
```

This is surprisingly useful for "does my YAML override even parse" type questions without waiting on a real run.

## Cluster-specific setup

On **BioCloud**, use `bash -l` as the shebang in any SLURM script (the login shells initialize conda differently). The conda profile lives at `/etc/xdg/snakemake/biocloud/config.yaml`, and Snakemake maps its resource directives (`threads`, `mem_mb`, `gpus`, `runtime`) onto SLURM's `cpus-per-task`, `mem`, `gpus`, and `time`. The pre-built Snakemake profile is invoked via `--profile biocloud`.

On **DEIS-MCC**, request GPUs explicitly with `--gres=gpu:N` and `--partition=turing` (6× T4 per node) or `--partition=ada` (2× L4 per node). `/scratch` is ~60 GB of shared local SSD and is **not** persistent between jobs — always copy your outputs back to NFS home before the job ends. Contact Morten (`mksc@cs.aau.dk`) before running multi-day or multi-node jobs; the cluster is small and contested.

Chapter 6 walks through both clusters in full.
