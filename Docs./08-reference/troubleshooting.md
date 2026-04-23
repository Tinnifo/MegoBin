# Troubleshooting

The errors you'll actually run into, and what to do about each. Organised by where in the stack they come from.

## Config-time errors (before the pipeline runs)

### `omegaconf.errors.MissingMandatoryValue`

Hydra couldn't resolve a `???` value in the composed config. Something upstream is missing.

Fix: check the error trace for the specific key, then either:
- Provide it on the CLI: `key.subkey=value`.
- Inherit from a group that sets it via `defaults:`.

Common culprit: running `pipeline.py` without `--config-name experiment/...` — the top-level `config.yaml` is a stub.

### `InstantiationException` with `Error locating target 'megobin.representations.foo.FooEncoder'`

Your `_target_` string points to a class that doesn't exist at that import path.

Fix: check the YAML's `_target_` value against the actual file/class name. Case-sensitive. This is typo-and-typo-only; there's no class registry.

### `KeyError: 'some_param'` inside `instantiate()`

The target class doesn't accept a kwarg your YAML is passing.

Fix: look at the `__init__` signature of the target class. The YAML keys (other than `_target_` and `_partial_`) must match the `__init__` parameters exactly.

### `Dataset compatibility: signal 'taxonomy' required but not in dataset signals: [kmers, abundance]`

The fail-fast dataset compatibility check fired. Your dataset doesn't have the signals your feature extractor / sampler wants.

Fix: pick a dataset that has those signals, or a feature/sampler config that doesn't require them. See the table in [Config reference](config-reference.md).

## Training-time errors

### `RuntimeError: CUDA out of memory`

GPU ran out of VRAM. Either the batch is too large or the model has grown.

Fix, in order of cheapness:
1. Drop batch size: `trainer.batch_size=2048`.
2. Drop `phases[*].batch_size` for `TwoPhaseTrainer`.
3. If feature dim is huge, you're probably running DNABERT-S on too-long sequences — check `input_dim`.
4. Move to `--partition=ada` (24 GB VRAM vs T4's 16 GB) on DEIS-MCC.

### `RuntimeError: Expected all tensors to be on the same device`

Some tensor stayed on CPU while the model is on GPU.

Fix: this usually means your custom encoder's `training_step` didn't call `.to(device)` on the batch. Pattern:

```python
def training_step(self, batch, loss_fn):
    feat_i, feat_j, label = batch
    device = next(self.parameters()).device
    z_i = self._encode_raw(feat_i.to(device))  # <-- the .to(device) is load-bearing
    z_j = self._encode_raw(feat_j.to(device))
    return loss_fn(z_i, z_j, label.to(device))
```

### Training loss is NaN after a few epochs

Classic: a division by zero in the loss, or a degenerate pair batch (all same label).

Fix:
1. Check your loss handles empty masks — MegoBin's `InfoNCELoss` tutorial shows the pattern (`if pos_mask.any():` guard).
2. Check `clamp_threshold` on Mahalanobis BCE. If covariance entries are getting very small, the Mahalanobis distance blows up.
3. Add a `assert torch.isfinite(loss)` at the end of your loss, and run with `trainer.epochs=2 trainer.batch_size=32` to reproduce quickly.

### Training loss decreases but never learns the pattern

The end-to-end synthetic test is the discriminator: can your encoder cluster 3 well-separated synthetic genomes?

```bash
pytest tests/test_end_to_end.py -v -k <your-encoder>
```

If that fails (ARI stays near 0), there's a modelling bug — small, clean synthetic data should be easy. If it passes, the bug is in how you're scaling to the real dataset (wrong features, wrong pair sampler, wrong loss weight, wrong hyperparameters for the larger regime).

### `two_phase.yaml` phase 2 loss never decreases

Either:
- Your phase 1 mean head was *too good* — there's no residual variance for covariance to explain. Drop `phases[0].epochs` and retry.
- `loss_attrs.include_std: true` wasn't applied. Check `pipeline.log` for "phase 2" markers — they should include "include_std=True".
- The `cov` parameter group is empty — check `encoder.parameter_groups()["cov"]`.

### `nvidia-smi` shows 0% GPU utilisation during training

GPU is idle; you're CPU-bottlenecked somewhere.

Likely causes:
1. `num_workers: 0` in the DataLoader — single-threaded pair sampling is the bottleneck. Raise to 4–8.
2. Pair sampler doing slow work per batch (e.g. recomputing k-NN every epoch). Cache upstream.
3. Small batch size relative to GPU capacity. Increase `batch_size`.

### Process killed with exit code 137

OOM on CPU (not GPU). Usually feature loading — you tried to `np.load` a 10GB `.npy` file on an 8GB RAM node.

Fix: request more memory in SLURM (`--mem=64G`). Or stream features via `mmap_mode='r'` on `np.load`, though no current feature loader does this.

## Data / path errors

### `FileNotFoundError: encoder.pt`

Training crashed before saving, or `checkpoint_path` was set to a directory that doesn't exist and couldn't be created.

Fix: check that the parent of `checkpoint_path` exists. `save_checkpoint` creates it with `parents=True` but only if the interpolation resolves correctly. `${hydra:runtime.output_dir}` always works; a hand-coded absolute path might not.

### `checkm2 predict: database file not found`

CheckM2's reference database isn't where it expects.

Fix: set the `CHECKM2DB` environment variable:

```bash
export CHECKM2DB=/projects/microbial-dark-matter/databases/checkm2/uniref100.KO.1.dmnd
```

Put it in your shell rc. On DEIS-MCC the database isn't there — this is why CheckM2 runs on BioCloud.

### `rsync: permission denied (publickey)`

SSH keys aren't set up between clusters. You're typing BioCloud password or DEIS-MCC password every time.

Fix: generate keys once, push public key to each cluster's `~/.ssh/authorized_keys`:

```bash
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519_cluster -C "your-email@example.com"
ssh-copy-id -i ~/.ssh/id_ed25519_cluster.pub tinni@biocloud.cmc.aau.dk
ssh-copy-id -i ~/.ssh/id_ed25519_cluster.pub DB56HW@student.aau.dk@ailab-fe01.srv.aau.dk
```

Add to `~/.ssh/config`:

```
Host biocloud
  HostName biocloud.cmc.aau.dk
  User tinni
  IdentityFile ~/.ssh/id_ed25519_cluster

Host mcc
  HostName ailab-fe01.srv.aau.dk
  User DB56HW@student.aau.dk
  IdentityFile ~/.ssh/id_ed25519_cluster
```

Now `ssh biocloud` and `ssh mcc` Just Work.

## SLURM / HPC errors

### Job pending indefinitely

Check `squeue -u $USER` and `sinfo -p <partition>`. Usual causes:

- Partition is full. Wait, or switch (e.g. `--partition=ada` on DEIS-MCC).
- You requested more than the partition provides (`--gres=gpu:4` on a 2-GPU node).
- Your account doesn't have permission on that partition.

`scontrol show job <job-id>` shows `Reason=` with details.

### Job immediately fails with exit code 1

Check the `.err` file first. Common: `mamba activate megobin` not in the SLURM script, so the python binary is system Python without PyTorch.

Fix: the SLURM scripts in `hpc/slurm/` all start with `mamba activate megobin` — copy that pattern.

### `singularity: command not found` on a compute node

Some DEIS-MCC partitions have `singularity`; some have `apptainer`. They're drop-in compatible but the binary name differs.

Fix: detect at runtime:

```bash
SINGULARITY=$(command -v singularity || command -v apptainer)
"$SINGULARITY" exec --nv megobin.sif python megobin/pipeline.py ...
```

### Snakemake complains about a locked working directory

Previous Snakemake run crashed without cleaning up.

Fix:

```bash
snakemake --unlock -s hpc/Snakefile
```

Then re-submit.

## Import / environment errors

### `ModuleNotFoundError: No module named 'megobin'`

Package isn't installed.

Fix: from the repo root, `pip install -e .`. This registers the `megobin` package pointing at `megobin/` in the repo. Editable mode means code changes take effect immediately.

Inside Singularity: the container's `environment.def` should include `pip install -e /app/Metagenomic-Binning` (or whatever the mount path is). If it doesn't, rebuild.

### `ImportError: No module named 'tabulate'`

`TensorBoardLogger.log_dataframe` uses `df.to_markdown()`, which requires `tabulate`.

Fix: it's already in `environment.yml`. If you hand-rolled the environment, `pip install tabulate`.

### `torch.cuda.is_available() is False` on a GPU node

The PyTorch build doesn't match the CUDA driver on that node.

Fix, in order:
1. Check `nvidia-smi` shows a GPU (if not, you're on a CPU node — `--gres=gpu:1` missing).
2. Check `echo $CUDA_VISIBLE_DEVICES` is non-empty.
3. Use the Singularity container — `environment.def` pins a PyTorch build that matches the cluster's CUDA.
4. If using conda directly, `environment.yml` pins `pytorch-cuda=12.1` for BioCloud (CUDA 12.x). DEIS-MCC has CUDA 13.2, which is backward-compatible; if you get a mismatch, the container is the fix.

## Git / DVC

### `git pull` on the cluster pulls your laptop changes but `python megobin/pipeline.py` shows the old behaviour

You're running from an old checkout in a different directory. Check `pwd` and the `git log` of `megobin/pipeline.py`.

### `dvc pull` returns "no remote configured"

The DVC remote isn't set on this clone.

Fix:

```bash
dvc remote add -d biocloud-s3 s3://tinnifo-metagenomic-binning/
# or whatever remote was configured in the original clone
```

Credentials come from `~/.aws/credentials` or whatever auth mechanism the remote uses.

### `dvc add` says "unable to acquire lock"

Another DVC process is running. Common if Snakemake is mid-execution and has pinned the cache.

Fix: wait for it to finish, or force-unlock only if you're sure nothing else is writing: `rm .dvc/tmp/lock` (don't do this on a shared filesystem if others are running).

## TensorBoard

### `tensorboard --logdir outputs/` shows no runs

Either: event files aren't there (training crashed before the first scalar log), or TB version can't read them.

Check: `find outputs -name 'events.out.tfevents.*'` should list files. If so, your TB is too old.

Fix: `pip install --upgrade tensorboard`.

### Can't see my run in the browser after port-forwarding

Ensure you bound TensorBoard to all interfaces:

```bash
tensorboard --logdir outputs/ --port 6006 --bind_all
```

Without `--bind_all`, TB listens only on `localhost` *on the compute node*, which your SSH forward can't reach because your forward targets the login node's localhost.

## When none of the above applies

1. Re-run the smoke test: `sbatch hpc/slurm/smoke_test.sh` or `pytest tests/`.
2. If smoke tests pass, your environment is fine — the bug is in your new code.
3. If smoke tests fail, your environment is broken — rebuild the conda env or container.
4. Check the resolved config: `python megobin/pipeline.py --config-name ... --cfg job`. What you *think* you configured is often not what Hydra composed.
5. Ask Thomas or Kadir if it's a modelling question; Sebastian if it's a data or cluster question.
