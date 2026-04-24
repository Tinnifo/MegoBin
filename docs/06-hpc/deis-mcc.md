# DEIS-MCC

DEIS-MCC is the Department of Computer Science cluster at AAU (Multi-cluster Compute). It's our primary GPU cluster — Turing nodes (6× T4) and Ada nodes (2× L4) — and it's where long training runs belong. It does *not* host the CAMI data or CheckM2 database; that lives on BioCloud. So the typical workflow is: train on DEIS-MCC, rsync the checkpoint to BioCloud, evaluate on BioCloud.

## Access and layout

```
ssh DB56HW@student.aau.dk@ailab-fe01.srv.aau.dk
```

Frontend is `ailab-fe01` (frontend node 1). Don't compute on the frontend — submit via SLURM. The frontend is for editing files, submitting jobs, and running `squeue`.

Key paths:

- Home: `/home/DB56HW@student.aau.dk/` — SLURM-visible, fine for code.
- Scratch: `/work/DB56HW@student.aau.dk/metagenomic-binning/` — the scratch mount. Fast, large. Datasets and outputs go here.
- Per-job scratch: `/scratch/$(whoami)/megobin_${SLURM_JOB_ID}/` — local to the compute node. The SLURM script copies data there before training for faster I/O.

## Environment — Singularity is strongly preferred

Unlike BioCloud, on DEIS-MCC you should use a Singularity container, not a conda environment. The reason is that compute nodes have CUDA 13.2 drivers while `environment.yml` pins `pytorch-cuda=12.1` for BioCloud compatibility; the container shim handles that cleanly, conda doesn't.

Build the container once (on a machine with fakeroot or via Sylabs Cloud):

```bash
singularity build megobin.sif environment.def
```

Copy the resulting `megobin.sif` to your DEIS-MCC scratch:

```bash
rsync -avz megobin.sif DB56HW@student.aau.dk@ailab-fe01.srv.aau.dk:/work/DB56HW@student.aau.dk/metagenomic-binning/
```

Now every training invocation looks like:

```bash
singularity exec --nv megobin.sif python megobin/pipeline.py ...
```

Note `--nv` (not `--nvccli` — that's BioCloud only).

## Submitting training jobs

The canonical script is `hpc/slurm/deis_mcc_train.sh`. It:

```bash
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=turing
```

Submit with encoder and dataset as positional args:

```bash
sbatch hpc/slurm/deis_mcc_train.sh uncertain_gen CAMI_medium
```

This resolves to:

```bash
singularity exec --nv megobin.sif python megobin/pipeline.py \
  --config-name experiment/hybrid_uncertain_gen \
  encoder=uncertain_gen \
  dataset=CAMI_medium \
  data_dir=/scratch/DB56HW@student.aau.dk/megobin_<jobid>
```

The script copies `data/<dataset>/` to `/scratch/.../` before training, so training reads from local SSD not NFS. This matters — NFS access patterns for large k-mer profile arrays are brutal. Copy-to-scratch shaves 30–40% off training wall time for the CAMI datasets.

## Turing vs. Ada — which partition?

The script defaults to `--partition=turing`. Pick per run:

**`--partition=turing`** — 6× T4 per node, 16GB VRAM each. Default choice. Plenty for UncertainGen (526K params) or SemiBin (small MLP). T4 is old (2018) but reliable and rarely congested.

**`--partition=ada`** — 2× L4 per node, 24GB VRAM each. Faster per-GPU but queue-limited. Pick when:
- You're training DNABERT-S (or anything with pretrained transformer weights) and need the VRAM.
- You're doing batched hyperparameter sweeps and the extra throughput matters.

Submit to ada by overriding:

```bash
sbatch --partition=ada hpc/slurm/deis_mcc_train.sh uncertain_gen CAMI_medium
```

For the current encoders (UncertainGen, SemiBin), turing is enough. Don't preempt ada users with jobs that didn't need it.

## Getting the checkpoint back to BioCloud

After the SLURM job finishes, the script copies `*.pt` files back to `${PROJECT_DIR}/results/${DATASET}/${ENCODER}/`:

```bash
cp -r "${SCRATCH}/"*.pt "${PROJECT_DIR}/results/${DATASET}/${ENCODER}/" 2>/dev/null || true
```

That puts them on your DEIS-MCC home mount. Now rsync to BioCloud for CheckM2 evaluation:

```bash
# From your laptop (relay), or from DEIS-MCC directly if SSH keys are set up
rsync -avz --progress \
  DB56HW@student.aau.dk@ailab-fe01.srv.aau.dk:/work/DB56HW@student.aau.dk/metagenomic-binning/results/ \
  tinni@biocloud.cmc.aau.dk:/projects/microbial-dark-matter/metagenomic-binning/results/
```

Then on BioCloud, evaluate the trained encoder with `resume_from`:

```bash
sbatch --wrap "\
  cd /projects/microbial-dark-matter/metagenomic-binning/Metagenomic-Binning && \
  mamba activate megobin && \
  python megobin/pipeline.py \
    --config-name experiment/training/uncertain_gen_cami_toy \
    resume_from=results/CAMI_medium/uncertain_gen/model.pt"
```

See the [cross-cluster data flow](data-flow.md) page for the rsync patterns laid out as a coherent workflow.

## Smoke test — before every real run

Before submitting a 48-hour training job, submit a smoke test:

```bash
sbatch hpc/slurm/smoke_test.sh
```

That runs `pytest tests/` and then does a Hydra compose + instantiate of the canonical experiment config. If this fails, your real training job will also fail — just slower. 30-minute wall-time ceiling on this job so it can't hang.

The relevant part of `smoke_test.sh`:

```python
with initialize_config_dir(version_base=None, config_dir=os.path.abspath('configs')):
    cfg = compose(config_name='experiment/hybrid_uncertain_gen')
    enc = instantiate(cfg.encoder)
    loss = instantiate(cfg.loss)
    binner = instantiate(cfg.binner)
    evaluator = instantiate(cfg.evaluator)
```

If any of those `instantiate` calls raise, your config has a `_target_` typo or a missing required kwarg.

## Common DEIS-MCC issues

**Job pending indefinitely on turing.** Check `squeue -p turing` — the cluster is probably full. Either wait, or switch to `--partition=ada` if your memory profile allows.

**`ModuleNotFoundError: No module named 'megobin'` inside container.** You forgot `pip install -e .` during the container build, or the container's working directory doesn't include the repo. Check `environment.def` has the `pip install -e /app/Metagenomic-Binning` step and that the container mounts your code.

**Mysteriously slow training (~10× expected time).** Almost always NFS I/O. Confirm the `cp -r data/.../ $SCRATCH/` step ran — grep the job's `.out` file for the feature-loading log lines. If data is being read from `/work/.../` instead of `/scratch/.../`, the copy step silently failed.

**GPU not detected.** Check `nvidia-smi` ran in your job (add `nvidia-smi` as the first command in the script). If it shows no GPU, you missed `--gres=gpu:1`, or you ran on a CPU-only node by scheduling accident. `--partition=turing` always has GPUs; `--partition=cpu` does not.

**CUDA version mismatch.** The container ships CUDA 12.1; if you skipped the container and used conda instead, the host driver may be mismatched. Use the container. This is the reason the container exists.

## When NOT to use DEIS-MCC

Don't use DEIS-MCC for:

- Feature computation (no GPU needed, better on BioCloud where the data already lives).
- CheckM2 evaluation (no GPU needed, CheckM2 database is on BioCloud).
- Anything shorter than 30 minutes (SLURM queue overhead is about 5 min — short runs are better done locally or on BioCloud).

Rule: if the run needs a GPU for more than 30 minutes, DEIS-MCC. Otherwise BioCloud or local.
