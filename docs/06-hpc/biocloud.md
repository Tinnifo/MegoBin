# BioCloud (CMC-AAU)

Where data lives, where Snakemake runs the full pipeline, where CheckM2 evaluation runs.

**Rule:** BioCloud runs the pipeline; DEIS-MCC trains the encoder.

## Access

```bash
ssh tinni@biocloud.cmc.aau.dk
```

(Ask Sebastian if you don't have an account.)

Paths:
- Home: `/home/tinni/` — small, not for data
- Project: `/projects/microbial-dark-matter/metagenomic-binning/` — datasets, outputs
- Containers: `/projects/microbial-dark-matter/containers/` — drop `megobin.sif` here

## Setup

```bash
cd /projects/microbial-dark-matter/metagenomic-binning/
git clone git@github.com:Tinnifo/Metagenomic-Binning.git
cd Metagenomic-Binning
mamba env create -f environment.yml
mamba activate megobin
pip install -e .
```

Singularity flag is `--nvccli` (not `--nv`):

```bash
apptainer run --nvccli megobin.sif python megobin/pipeline.py --config-name <cfg>
```

## Full pipeline (Snakemake)

```bash
sbatch hpc/slurm/biocloud_pipeline.sh CAMI_medium
```

Script runs:

```bash
snakemake --profile biocloud -s hpc/Snakefile --config datasets="[CAMI_medium]" --jobs 50 --printshellcmds
```

Rules per `(dataset, encoder, binner)`:

1. `compute_features` — 4-mer profiles + splits + contig names → `.npy`
2. `train` → `model.pt`
3. `encode` → `embeddings.npy`
4. `bin` → `labels.npy`
5. `evaluate` → `quality_report.tsv`

Each rule is independently cacheable.

## Single training run (no Snakemake)

```bash
sbatch --time=02:00:00 --cpus-per-task=8 --mem=32G --gres=gpu:1 --wrap "\
  cd /projects/microbial-dark-matter/metagenomic-binning/Metagenomic-Binning && \
  mamba activate megobin && \
  python megobin/pipeline.py --config-name experiment/training/uncertain_gen_cami_toy \
    logger.name=H1-seed1-uncertain_gen-biocloud"
```

Always set `logger.name` — otherwise every run shows up as `tb` in TensorBoard.

## SLURM defaults (`hpc/slurm/biocloud_pipeline.sh`)

```bash
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --mail-type=FAIL
```

24-hour ceiling is appropriate — exceed it and you should be on DEIS-MCC.

## Results

```
results/CAMI_medium/uncertain_gen/infomap/
├── labels.npy
└── checkm2/
    ├── bins/
    └── quality_report.tsv
```

Snakemake produces raw artifacts. `pipeline.py` produces TensorBoard-ready logs.

## Common issues

| Symptom | Fix |
|---------|-----|
| CUDA OOM on feature compute | Features should run on CPU — check Snakefile resource section |
| `unknown flag: --nv` | Use `--nvccli` on BioCloud |
| `CheckM2 error: database not found` | `export CHECKM2DB=/projects/microbial-dark-matter/databases/checkm2/uniref100.KO.1.dmnd` |
| Snakemake lockfile | `snakemake --unlock -s hpc/Snakefile` |
