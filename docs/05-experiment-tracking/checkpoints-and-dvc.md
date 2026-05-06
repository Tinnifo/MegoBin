# Checkpoints and DVC

Three artifacts per run: `encoder.pt`, `.hydra/config.yaml`, `run_meta.json`. DVC handles the large stuff.

## What's saved

```
outputs/2026-04-23/14-35-02/
├── .hydra/
│   ├── config.yaml          resolved config
│   ├── hydra.yaml
│   └── overrides.yaml       CLI overrides
├── pipeline.log
├── encoder.pt               trained weights
├── bins/                    output FASTAs
├── checkm2/quality_report.tsv
└── tb/
    ├── events.out.tfevents.*
    ├── run_meta.json        provenance
    └── checkpoints/checkpoint.pt   (logger's copy)
```

`encoder.pt` at run root is authoritative; the `tb/checkpoints/` copy travels with the event file.

## Trainer config

```yaml
# single_phase
checkpoint_path: ${hydra:runtime.output_dir}/encoder.pt
checkpoint_every: null     # or int (epoch stride)

# two_phase
checkpoint_path: ${hydra:runtime.output_dir}/encoder.pt
checkpoint_per_phase: false  # true → encoder_phase_1.pt, encoder_phase_2.pt, ...
```

Set `checkpoint_path: null` to disable. Both trainers use `save_checkpoint(encoder, path)` — saves only the `state_dict`.

## Resume / "train once, bin many"

```bash
python megobin/pipeline.py \
  --config-name experiment/uncertain_gen_dbscan \
  resume_from=outputs/2026-04-23/14-35-02/encoder.pt \
  binner=dbscan_ensemble
```

`resume_from` skips training and loads weights into the encoder built from the **current** config. Changing `encoder.input_dim` or `embedding_dim` between save and load = shape-mismatch error. Use a pinned config for both runs.

## `run_meta.json`

Written by `TensorBoardLogger.log_config`:

```json
{
  "name": "2026-04-23_14-35-02",
  "logdir": "outputs/.../tb",
  "checkpoints": [{"name": "checkpoint", "path": "...", "source": "..."}],
  "config": { ... },
  "git_sha": "a1b2c3...",
  "env_hash": "sha256 of environment.yml",
  "dvc_version": "3.x.y"
}
```

- `git_sha` — `git rev-parse HEAD` at run time. Misleading if uncommitted changes; commit before long runs. `unknown` outside a git checkout.
- `env_hash` — sha256 of `environment.yml`. Different hash → environments may not reproduce.
- `dvc_version` — DVC tool version, not data version.
- `checkpoints` — list (multi-entry for `checkpoint_per_phase: true`).

## DVC

For datasets and pinned checkpoints. `.gitignore` already excludes `data/`, `outputs/`, `.dvc/cache/`.

Setup:

```bash
dvc init
dvc remote add -d biocloud-s3 s3://tinnifo-metagenomic-binning/
```

Version a dataset:

```bash
dvc add data/<your_dataset>
git add data/<your_dataset>.dvc .gitignore
git commit -m "Track <your_dataset> with DVC"
dvc push
```

Version a checkpoint:

```bash
dvc add outputs/2026-04-23/14-35-02/encoder.pt
git add outputs/2026-04-23/14-35-02/encoder.pt.dvc
git commit -m "Pin H1-seed1-uncertain_gen encoder"
dvc push
```

## When to track

| Artifact | Action |
|----------|--------|
| Datasets, published-result checkpoints | DVC-track |
| Exploratory `outputs/<date>/<time>/` | Leave local |
| Configs, hypothesis notes | Commit to Git |

Anti-pattern: DVC-tracking every `outputs/` directory.

## Reproducing a run

```bash
git checkout $(jq -r '.git_sha' run_meta.json)
mamba env create -f environment.yml && mamba activate megobin
dvc pull data/<your_dataset>
python megobin/pipeline.py --config-name experiment/uncertain_gen_dbscan seed=1
```

Bit-for-bit identical is rare on GPUs. Eval metrics should land within tolerance.
