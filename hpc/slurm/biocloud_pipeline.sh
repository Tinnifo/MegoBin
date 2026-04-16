#!/bin/bash -l
#SBATCH --job-name=megobin-pipeline
#SBATCH --output=megobin-pipeline_%j.out
#SBATCH --error=megobin-pipeline_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --mail-type=FAIL

# BioCloud full pipeline: feature computation → binning → CheckM2 evaluation
# Usage: sbatch hpc/slurm/biocloud_pipeline.sh [DATASET]

set -euo pipefail

DATASET="${1:-CAMI_medium}"
PROJECT_DIR="${SLURM_SUBMIT_DIR:-.}"

cd "$PROJECT_DIR"

mamba activate megobin

echo "=== BioCloud pipeline: dataset=${DATASET} ==="
echo "Node: $(hostname), CPUs: ${SLURM_CPUS_PER_TASK}, Job: ${SLURM_JOB_ID}"

snakemake \
    --profile biocloud \
    -s hpc/Snakefile \
    --config datasets="[${DATASET}]" \
    --jobs 50 \
    --printshellcmds

echo "=== Pipeline complete ==="
