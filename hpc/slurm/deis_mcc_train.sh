#!/bin/bash -l
#SBATCH --job-name=megobin-train
#SBATCH --output=megobin-train_%j.out
#SBATCH --error=megobin-train_%j.err
#SBATCH --time=48:00:00  #  time limit in dd:hh:mm:ss format. 
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=turing
#SBATCH --mail-type=FAIL

# DEIS-MCC GPU training via Singularity container
# Usage: sbatch hpc/slurm/deis_mcc_train.sh [ENCODER] [DATASET]

set -euo pipefail

ENCODER="${1:-uncertain_gen}"
DATASET="${2:-CAMI_medium}"
PROJECT_DIR="${SLURM_SUBMIT_DIR:-.}"
SCRATCH="/scratch/$(whoami)/megobin_${SLURM_JOB_ID}"
SIF="${PROJECT_DIR}/megobin.sif"

echo "=== MCC training: encoder=${ENCODER}, dataset=${DATASET} ==="
echo "Node: $(hostname), GPUs: ${SLURM_GPUS_ON_NODE:-1}, Job: ${SLURM_JOB_ID}"

# Set up scratch space
mkdir -p "$SCRATCH"
export SINGULARITY_TMPDIR="$SCRATCH"

# Copy data to scratch for faster I/O
if [ -d "${PROJECT_DIR}/data/${DATASET}" ]; then
    cp -r "${PROJECT_DIR}/data/${DATASET}" "${SCRATCH}/"
fi

singularity exec --nv "$SIF" python megobin/pipeline.py \
    --config-name experiment/hybrid_uncertain_gen \
    representation="$ENCODER" \
    dataset="$DATASET" \
    data_dir="$SCRATCH"

# Copy results back
cp -r "${SCRATCH}/"*.pt "${PROJECT_DIR}/results/${DATASET}/${ENCODER}/" 2>/dev/null || true

# Clean up scratch
rm -rf "$SCRATCH"

echo "=== Training complete ==="
