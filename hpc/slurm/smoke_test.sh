#!/bin/bash -l
#SBATCH --job-name=megobin-smoke
#SBATCH --output=megobin-smoke_%j.out
#SBATCH --error=megobin-smoke_%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --mail-type=FAIL

# Quick smoke test: run pytest + verify pipeline imports
# Works on either cluster. Submit: sbatch hpc/slurm/smoke_test.sh

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-.}"
cd "$PROJECT_DIR"

mamba activate megobin

echo "=== Smoke test on $(hostname), Job: ${SLURM_JOB_ID} ==="

echo "--- pytest ---"
python -m pytest tests/ -v --tb=short

echo "--- Hydra config composition ---"
python -c "
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
import os

with initialize_config_dir(version_base=None, config_dir=os.path.abspath('configs')):
    cfg = compose(config_name='experiment/hybrid_uncertain_gen')
    enc = instantiate(cfg.encoder)
    loss = instantiate(cfg.loss)
    binner = instantiate(cfg.binner)
    evaluator = instantiate(cfg.evaluator)
    print(f'OK: {type(enc).__name__}, {type(loss).__name__}, {type(binner).__name__}, {type(evaluator).__name__}')
"

echo "=== Smoke test passed ==="
