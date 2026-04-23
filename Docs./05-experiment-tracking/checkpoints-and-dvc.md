# Checkpoints and DVC

A run produces three artifacts you want to keep: the trained encoder, the exact config that produced it, and enough provenance metadata that someone a year from now can tell what Git SHA, what environment, and what data version were in play. MegoBin splits this across three files — `encoder.pt`, the Hydra config snapshot, and `run_meta.json` — and uses DVC when the artifacts are large enough that shipping them through Git would be painful.

## What gets saved

After a successful run, the output directory looks like this:

```
outputs/2026-04-23/14-35-02/
├── .hydra/
│   ├── config.yaml          # resolved config (inputs to instantiate())
│   ├── hydra.yaml           # Hydra's own config
│   └── overrides.yaml       # CLI overrides that landed
├── pipeline.log             # run log
├── encoder.pt               # trained weights
├── bins/                    # output FASTA bins
│   ├── bin_0000.fasta
│   └── ...
├── checkm2/                 # CheckM2 evaluator output
│   └── quality_report.tsv
└── tb/
    ├── events.out.tfevents.*
    ├── run_meta.json        # provenance
    └── checkpoints/
        └── checkpoint.pt    # logger's copy
```

The `encoder.pt` at the run root is the authoritative checkpoint. The `tb/checkpoints/checkpoint.pt` is a copy the `TensorBoardLogger` makes so the event file and the checkpoint travel together — useful if you archive just `tb/` and forget everything else.

## How checkpointing works in the trainer

From `configs/trainer/single_phase.yaml`:

```yaml
checkpoint_path: ${hydra:runtime.output_dir}/encoder.pt
checkpoint_every: null
```

`checkpoint_path` interpolates to the run directory. Set it to `null` to disable saving entirely. `checkpoint_every: N` saves an intermediate snapshot every N epochs; the final checkpoint always saves at end of `fit()`.

`TwoPhaseTrainer` uses the same `checkpoint_path` but trades `checkpoint_every` for `checkpoint_per_phase`:

```yaml
# configs/trainer/two_phase.yaml
checkpoint_path: ${hydra:runtime.output_dir}/encoder.pt
checkpoint_per_phase: false
```

With `checkpoint_per_phase: true` you get `encoder_phase_1.pt`, `encoder_phase_2.pt`, ..., plus the final `encoder.pt`. Useful when you want to diagnose phase-1 convergence without re-running.

Both trainers call `megobin.utils.checkpoints.save_checkpoint(encoder, path)`, which saves just the `state_dict()` — not the module object. Loading requires an encoder instance with the same architecture already in memory, which is what `resume_from` does.

## Resuming and "train-once, bin-many"

The pipeline's main entry point handles `resume_from`:

```python
# megobin/pipeline.py (around L155)
resume_from = cfg.get("resume_from")
if resume_from:
    log.info("resume_from set — skipping training, loading %s", resume_from)
    load_checkpoint(representation, resume_from)
else:
    # ...train normally...
```

`resume_from` is a top-level config key. It's null by default; set it on the CLI to reuse a trained encoder:

```bash
python megobin/pipeline.py \
  --config-name experiment/training/uncertain_gen_cami_toy \
  resume_from=outputs/2026-04-23/14-35-02/encoder.pt \
  binner=dbscan_ensemble
```

This is how the binner-swap tutorial works — train once, run the same embeddings through Infomap, then DBSCAN, then any other binner you want. The key observation is that `resume_from` doesn't change anything about the representation, features, or evaluator pipeline; it only short-circuits the training phase.

**Important caveat:** `load_checkpoint` loads weights in place against an encoder instantiated from the *current* config. If you change `representation.input_dim` or `representation.embedding_dim` between save and load, you will get a cryptic shape-mismatch error from `load_state_dict`. Keep the representation config pinned, or use a pinned experiment YAML (e.g. `experiment/training/uncertain_gen_cami_toy.yaml`) on both the training run and the resume run.

## run_meta.json — the provenance record

When `TensorBoardLogger.log_config` runs (once per run, at the top of the pipeline), it writes `run_meta.json` containing:

```json
{
  "name": "2026-04-23_14-35-02",
  "logdir": "outputs/2026-04-23/14-35-02/tb",
  "checkpoints": [
    {
      "name": "checkpoint",
      "path": "outputs/.../tb/checkpoints/checkpoint.pt",
      "source": "outputs/.../encoder.pt"
    }
  ],
  "config": { ... full resolved config ... },
  "git_sha": "a1b2c3d4e5f6...",
  "env_hash": "sha256 of environment.yml",
  "dvc_version": "3.x.y"
}
```

Four fields to know about:

**git_sha** is the output of `git rev-parse HEAD` at run-time. If you have uncommitted changes, this SHA is misleading — it points to the last committed state, not what was actually running. Commit before long runs. `unknown` appears when the run is outside a Git checkout (e.g. in a Singularity container with the code baked in).

**env_hash** is the sha256 of `environment.yml`. Any change to the pinned dependency versions produces a new hash. Two runs with the same `env_hash` used compatible environments; two runs with different hashes may not reproduce.

**dvc_version** is the output of `dvc version`. It records the DVC tool version, not the data version — that lives elsewhere (see below).

**checkpoints** is a list because `log_checkpoint` can be called more than once. `TwoPhaseTrainer` with `checkpoint_per_phase: true` produces multiple entries.

## DVC for versioned artifacts

Git is fine for code and small configs. It's not fine for 50-row-GB CAMI datasets or 300MB pretrained DNABERT-S weights. DVC solves this with content-addressed storage plus Git-tracked `.dvc` pointer files.

The repo's `environment.yml` includes `dvc` and `dvc-s3`. The `.gitignore` ignores `data/`, `outputs/`, and `.dvc/cache/` so those large artifacts never leak into Git.

Typical workflow for versioning a dataset:

```bash
# One-time setup
dvc init
dvc remote add -d biocloud-s3 s3://tinnifo-metagenomic-binning/

# Version a dataset
dvc add data/CAMI_toy
git add data/CAMI_toy.dvc .gitignore
git commit -m "Track CAMI_toy with DVC"
dvc push
```

Now `data/CAMI_toy.dvc` is a 200-byte pointer file in Git, and the actual 12GB of CAMI toy data lives in S3 under a content hash. Colleagues run `dvc pull` to materialise it.

Typical workflow for versioning a checkpoint:

```bash
# After a successful run
dvc add outputs/2026-04-23/14-35-02/encoder.pt
git add outputs/2026-04-23/14-35-02/encoder.pt.dvc
git commit -m "Pin H1-seed1-uncertain_gen encoder"
dvc push
```

Someone else can now reproduce that exact run with `dvc pull` and `resume_from` — no need to re-train.

## When to DVC-track vs leave local

Rule of thumb:

- **Always DVC-track:** datasets that take nontrivial wall-clock time to regenerate (CAMI, real metagenomic assemblies, any external downloads), and checkpoints that back a published hypothesis or ablation result.
- **Leave local:** intermediate `outputs/<date>/<time>/` directories from exploratory runs. These are noise — you have hundreds of them, they weren't informative, you don't need them in the cache.
- **Always commit to Git:** the config that produced an interesting run, and a short note in the hypothesis folder explaining what the run showed. The checkpoint is the artifact; the config is the recipe.

The anti-pattern is DVC-tracking every single `outputs/` directory. That bloats your remote with runs that proved nothing.

## Reproducing a run from scratch

Given someone else's `run_meta.json`:

```bash
# 1. Check out the exact code state
git checkout $(jq -r '.git_sha' run_meta.json)

# 2. Recreate the environment (or check env_hash matches)
mamba env create -f environment.yml
mamba activate megobin

# 3. Pull the data they used
dvc pull data/CAMI_toy

# 4. Re-run with the same config
python megobin/pipeline.py --config-name experiment/training/uncertain_gen_cami_toy seed=1

# 5. (Optional) verify the checkpoint matches bit-for-bit
sha256sum outputs/.../encoder.pt
# compare with the sha256 of the DVC-tracked encoder they pushed
```

Step 5 is rarely bit-for-bit identical (floating-point non-determinism on GPUs), but the eval metrics should land within a reasonable tolerance. If they don't, walk back through steps 1–4 and find the divergence. The env hash is usually the culprit.

## Summary

`encoder.pt` is the checkpoint. `.hydra/config.yaml` is the recipe. `run_meta.json` is the provenance. Together they define a run. DVC handles artifacts too big for Git. `resume_from` is how you separate training from everything else, and it's the load-bearing mechanism behind every binner-swap and ablation in this repo.
