# Tutorial: How to run experiments from CLI

A practical [Hydra](https://hydra.cc/docs/intro/) CLI cheatsheet shaped around MegoBin's slot structure. Grouped by what you're *trying to do* rather than by Hydra feature name.

## The mental model

Every CLI argument is one of four things:

1. **Group override** — `binner=dbscan_ensemble` → "for the `binner` slot, load `configs/binner/dbscan_ensemble.yaml`". See [config groups](https://hydra.cc/docs/tutorials/basic/your_first_app/config_groups/).
2. **Value override** — `encoder.latent_dim=128` → "set this leaf field". Use dots to walk into nested keys. See [override grammar](https://hydra.cc/docs/advanced/override_grammar/basic/).
3. **Add a new key** — `+encoder.warmup_steps=500` → leading `+` means "this key doesn't exist yet, create it".
4. **Force-add even if it exists** — `++encoder.latent_dim=128` → overrides without erroring if the key is already there or missing.

Knowing which prefix applies covers ~90% of confusion.

## Single runs

```bash
# Run the default experiment as-is
python megobin/pipeline.py --config-name experiment/uncertain_gen_dbscan

# Same thing, but swap one slot
python megobin/pipeline.py --config-name experiment/uncertain_gen_dbscan \
  binner=dbscan_ensemble

# Swap several slots at once
python megobin/pipeline.py --config-name experiment/uncertain_gen_dbscan \
  binner=dbscan_ensemble loss=hinge trainer=single_phase

# Tweak hyperparameters of the currently-selected encoder
python megobin/pipeline.py --config-name experiment/uncertain_gen_dbscan \
  encoder.latent_dim=64 encoder.dropout=0.2 trainer.lr=3e-4

# Combine slot swaps and value overrides
python megobin/pipeline.py --config-name experiment/uncertain_gen_dbscan \
  encoder=vae encoder.latent_dim=64 loss=ntxent loss.temperature=0.1
```

## Lists, dicts, and weird values

```bash
# Override a list 
python megobin/pipeline.py 'evaluator.contamination_thresholds=[5,10,15]'

# Append to a list
python megobin/pipeline.py '+evaluator.contamination_thresholds=[20]'

# null / None
python megobin/pipeline.py trainer.scheduler=null

# String with spaces
python megobin/pipeline.py "run_name='vae baseline, seed 1'"

# Disable / remove a slot entirely
python megobin/pipeline.py ~filter
```

That last one — `~key` — is the *delete* prefix. Useful when an experiment config sets a filter but you want to run without one.

## Multirun: sweeps via `-m` / `--multirun`

This is the workhorse. Comma-separated values become a Cartesian product. See [multi-run](https://hydra.cc/docs/tutorials/basic/running_your_app/multi-run/).

```bash
# 1D sweep: 3 seeds
python megobin/pipeline.py -m --config-name experiment/uncertain_gen_dbscan \
  seed=1,2,3

# 2D ablation: 3 encoders × 2 binners = 6 runs
python megobin/pipeline.py -m --config-name experiment/uncertain_gen_dbscan \
  encoder=vae,dnaberts,kmer_count binner=dbscan_ensemble,hdbscan

# Mix slot swaps with hyperparameter sweeps
python megobin/pipeline.py -m --config-name experiment/uncertain_gen_dbscan \
  loss=hinge,ntxent loss.temperature=0.05,0.1,0.2
```

Hydra has built-in [sweep helpers](https://hydra.cc/docs/advanced/override_grammar/extended/) for the common shapes:

```bash
# range(start, stop[, step]) — exclusive on stop
python megobin/pipeline.py -m 'trainer.lr=range(1e-4, 1e-3, 1e-4)'

# choice — same as comma list, but explicit
python megobin/pipeline.py -m 'encoder.latent_dim=choice(32, 64, 128, 256)'

# glob — match config files in a group by pattern
python megobin/pipeline.py -m 'encoder=glob(vae*)'   # all encoders starting with "vae"
python megobin/pipeline.py -m 'encoder=glob(*)'       # every encoder
```

For real hyperparameter optimization (not just grid), install a sweeper plugin: [`hydra-optuna-sweeper`](https://hydra.cc/docs/plugins/optuna_sweeper/) for Bayesian, [`hydra-ax-sweeper`](https://hydra.cc/docs/plugins/ax_sweeper/) for Ax. You add `hydra/sweeper=optuna` to the CLI and Hydra calls Optuna under the hood. Worth doing once `lr` and `latent_dim` start mattering.

## Common research patterns

**Baseline vs. proposed method, $n$ seeds each.** This is the "is my method actually better" run.

```bash
python megobin/pipeline.py -m --config-name experiment/uncertain_gen_dbscan \
  encoder=vae,uncertain_vae seed=1,2,3,4,5
```

That's 10 runs. After they finish, point TensorBoard at `outputs/` and you have your mean ± std plot.

**Ablation table.** Drop one component at a time. Easiest way is to predefine "off" variants in each config group (a `loss/no_contrastive.yaml`, a `filter/none.yaml`, etc.), then:

```bash
python megobin/pipeline.py -m --config-name experiment/uncertain_gen_dbscan \
  loss=full,no_contrastive,no_recon filter=marker,none
```

**Hyperparameter scan around a known-good setting.**

```bash
python megobin/pipeline.py -m --config-name experiment/uncertain_gen_dbscan \
  'trainer.lr=range(1e-5, 1e-3, 1e-5)' \
  'encoder.latent_dim=choice(64, 128, 256)' \
  seed=1,2,3
```

**One-off "what if I do X" probes.** Don't bother adding to an experiment file — override on the CLI, let Hydra dump the resolved config into `outputs/`, and grep for it later if it worked.

## Naming runs and controlling output dirs

By default Hydra writes to `outputs/<date>/<time>/`. For sweeps, it switches to `multirun/<date>/<time>/<job_num>/`. You override either pattern with the [`hydra.*` namespace](https://hydra.cc/docs/configure_hydra/workdir/):

```bash
# Custom single-run output dir
python megobin/pipeline.py \
  hydra.run.dir=outputs/h1_baseline_seed1

# Custom multirun root + per-job subdir
python megobin/pipeline.py -m \
  hydra.sweep.dir=outputs/h1_ablation \
  'hydra.sweep.subdir=${encoder}_${seed}' \
  encoder=vae,uncertain_vae seed=1,2,3
```

That last pattern is the recommended run-naming convention (`{hypothesis-id}-{seed}-{config-name}`). Pin it once at the top of each experiment config:

```yaml
# configs/experiment/h1_uncertainty.yaml
hydra:
  sweep:
    dir: outputs/${hypothesis_id}
    subdir: ${hypothesis_id}-seed${seed}-${encoder}
hypothesis_id: H1
seed: 1
```

…and then `-m seed=1,2,3 encoder=vae,uncertain_vae` gives you nicely named directories without typing `hydra.sweep.subdir` every time.

## Inspecting and debugging configs

The two flags worth memorizing see [Hydra command-line flags](https://hydra.cc/docs/advanced/hydra-command-line-flags/) for \--cfg``.

```bash
# Print the fully resolved config without running anything
python megobin/pipeline.py --config-name experiment/uncertain_gen_dbscan --cfg job

# Print the config including all the Hydra plumbing
python megobin/pipeline.py --config-name experiment/uncertain_gen_dbscan --cfg hydra

# List all config groups and their options
python megobin/pipeline.py --help

# Check what a multirun would run, without launching jobs
python megobin/pipeline.py -m encoder=vae,dnaberts seed=1,2,3 --cfg job --resolve
```

When a sweep launches the wrong thing, `--cfg job --resolve` will save you an hour.

## Launching on the cluster

Hydra has [launcher plugins](https://hydra.cc/docs/plugins/submitit_launcher/). The one you want for AAU AI Lab is the Submitit launcher (SLURM-aware):

```bash
pip install hydra-submitit-launcher

python megobin/pipeline.py -m \
  hydra/launcher=submitit_slurm \
  hydra.launcher.partition=gpu \
  hydra.launcher.timeout_min=720 \
  hydra.launcher.gpus_per_node=1 \
  encoder=vae,uncertain_vae seed=1,2,3,4,5
```

That submits 10 individual SLURM jobs from one command. Each gets its own GPU and its own output dir. Pin the launcher settings into a YAML at `configs/hydra/launcher/aau_gpu.yaml` so you just write `hydra/launcher=aau_gpu` in CLI / experiment files.

## Gotchas

A few things that bite people once and then never again:

The shell will mangle some characters. `'`-quote anything with `()`, `[]`, `*`, `,`, or spaces. If a sweep value looks like an expression, quote the whole `key='expr(...)'` argument.

`+key=value` vs. `key=value`. If the key doesn't exist in your config yet, plain `key=value` errors out. Use `+`. If it does exist and you want to be defensive in scripts, use `++`.

Hydra changes the working directory at runtime. Inside your code, use [`hydra.utils.get_original_cwd()`](https://hydra.cc/docs/tutorials/basic/running_your_app/working_directory/) to refer to repo-root paths (configs do this for you with `${oc.env:...}` or `${hydra:runtime.cwd}`). Otherwise relative paths break.

Multirun does NOT parallelize automatically on a single machine. With no launcher specified you get sequential runs (basic launcher) or local-parallel ([joblib launcher](https://hydra.cc/docs/plugins/joblib_launcher/)). Use `hydra/launcher=joblib` and `hydra.launcher.n_jobs=4` to run four at a time locally.

## Further reading

- [Hydra intro](https://hydra.cc/docs/intro/)
- [Override grammar reference](https://hydra.cc/docs/advanced/override_grammar/basic/)
- [Extended override grammar (range, choice, glob)](https://hydra.cc/docs/advanced/override_grammar/extended/)
- [Multi-run sweeps](https://hydra.cc/docs/tutorials/basic/running_your_app/multi-run/)
- [Configuring `hydra.run.dir` / `hydra.sweep.dir`](https://hydra.cc/docs/configure_hydra/workdir/)
- [Object instantiation (`_target_`)](https://hydra.cc/docs/advanced/instantiate_objects/overview/)
- [Submitit / SLURM launcher](https://hydra.cc/docs/plugins/submitit_launcher/)
- [Optuna sweeper plugin](https://hydra.cc/docs/plugins/optuna_sweeper/)
