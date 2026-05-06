# DEIS-MCC

Primary GPU cluster (Turing 6× T4, Ada 2× L4). No CAMI data, no CheckM2. Workflow: train here → rsync → evaluate on BioCloud.

## Access

```bash
ssh DB56HW@student.aau.dk@ailab-fe01.srv.aau.dk
```

Don't compute on the frontend — submit via SLURM.

Paths:
- Home: `/home/DB56HW@student.aau.dk/`
- Scratch (NFS): `/work/DB56HW@student.aau.dk/metagenomic-binning/`
- Per-job local SSD: `/scratch/$(whoami)/megobin_${SLURM_JOB_ID}/`

## Singularity, always

CUDA 13.2 on the nodes; `environment.yml` pins 12.1 for BioCloud. Container handles it.

```bash
singularity build megobin.sif environment.def
rsync -avz megobin.sif DB56HW@student.aau.dk@ailab-fe01.srv.aau.dk:/work/.../metagenomic-binning/
```

```bash
singularity exec --nv megobin.sif python megobin/pipeline.py ...
```

`--nv` (not `--nvccli` — that's BioCloud).

## Training

```bash
sbatch hpc/slurm/deis_mcc_train.sh uncertain_gen CAMI_medium
```

Defaults:

```bash
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=turing
```

Script copies `data/<dataset>/` to local `/scratch/.../` first — saves 30–40% wall time vs reading NFS.

## Turing vs Ada

| Partition | GPUs | VRAM | When |
|-----------|------|------|------|
| `turing` (default) | 6× T4/node | 16 GB | UncertainGen, SemiBin — plenty |
| `ada`     | 2× L4/node | 24 GB | DNABERT-S or anything VRAM-bound |

Don't preempt Ada users with jobs that didn't need it.

## Get the checkpoint back

```bash
rsync -avz --progress \
  DB56HW@student.aau.dk@ailab-fe01.srv.aau.dk:/work/.../metagenomic-binning/results/ \
  tinni@biocloud.cmc.aau.dk:/projects/microbial-dark-matter/metagenomic-binning/results/
```

Then on BioCloud:

```bash
sbatch --wrap "\
  cd /projects/microbial-dark-matter/metagenomic-binning/Metagenomic-Binning && \
  mamba activate megobin && \
  python megobin/pipeline.py \
    --config-name experiment/training/uncertain_gen_cami_toy \
    resume_from=results/CAMI_medium/uncertain_gen/model.pt"
```

See [data-flow.md](data-flow.md) for the full round-trip.

## Smoke test before long jobs

```bash
sbatch hpc/slurm/smoke_test.sh
```

Runs `pytest tests/` and Hydra compose+instantiate of the canonical config. 30-min ceiling.

## Common issues

| Symptom | Fix |
|---------|-----|
| Job pending on turing | `squeue -p turing`; switch to `--partition=ada` if memory allows |
| `ModuleNotFoundError: megobin` in container | `pip install -e .` missing from `environment.def`, or wrong working dir |
| Mysteriously slow training | NFS I/O — confirm `cp -r data/.../ $SCRATCH/` ran |
| GPU not detected | Add `nvidia-smi` to job; check `--gres=gpu:1` |
| CUDA mismatch | Use the container, not conda |

## Don't use DEIS-MCC for

- Feature computation (no GPU; data is on BioCloud)
- CheckM2 evaluation (CheckM2 DB on BioCloud)
- Anything <30 min (queue overhead ~5 min)
