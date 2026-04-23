# Your first run

This chapter takes about ten minutes. At the end of it you will have trained the UncertainGen encoder on the CAMI toy dataset, clustered the embeddings with Infomap, evaluated the bins with CheckM2, and looked at loss curves in TensorBoard. Nothing in this chapter is stubbed out — every command runs real code against real data.

## Goal

Run `experiment/training/uncertain_gen_cami_toy` end-to-end and verify that:

1. The pipeline reports loaded features of shape `(N, 236)`.
2. The two-phase trainer prints two phases — `mean` (50 epochs) then `cov` (25 epochs) — with training loss trending down.
3. The binner reports some number of bins.
4. CheckM2 writes a completeness/contamination DataFrame to the log.
5. TensorBoard shows a `train/loss` curve under `outputs/<date>/<time>/tb/`.

## Prerequisites

You have followed the install steps and `mamba activate megobin` (or equivalent) is active. `pytest tests/test_interfaces.py` passes. You have the CAMI toy dataset at `data/CAMI_toy/` containing `kmer_profiles.npy`, `abundance.npy`, and `contigs.fasta` at minimum. If you do not — ask Sebastian or pull it from BioCloud `/projects/microbial-dark-matter/metagenomic-binning/data/CAMI_toy/`.

## Step 1 — Smoke-test on synthetic data

Before running on CAMI toy, confirm the pipeline itself works on synthetic data. The end-to-end test constructs a small synthetic dataset on the fly and walks through the full pipeline:

```bash
pytest tests/test_end_to_end.py -v
```

This should complete in under a minute. If it fails, your install is broken — stop here and fix the install before continuing. Common causes: missing `hydra-core`, missing `checkm2` in the environment, a Python version mismatch (the repo targets 3.10).

## Step 2 — Dry-run the config

Hydra's `--cfg job` mode prints the fully-resolved config without running anything. This is the single most useful command for diagnosing config issues:

```bash
python megobin/pipeline.py \
  --config-name experiment/training/uncertain_gen_cami_toy \
  --cfg job
```

You should see the composed YAML printed to stdout. Verify the `dataset.path`, `features.k`, `representation._target_`, and `trainer.phases` look right. If any of these are missing or surprising, Hydra's composition is misbehaving — usually a typo in a `defaults:` entry.

## Step 3 — Launch the real run

```bash
python megobin/pipeline.py --config-name experiment/training/uncertain_gen_cami_toy
```

On a recent MacBook Pro this takes ten to fifteen minutes. On a T4 GPU it finishes in two to three. The console output walks through the pipeline in the same order as [`megobin/pipeline.py`](https://github.com/Tinnifo/Metagenomic-Binning/blob/main/megobin/pipeline.py):

```
[megobin.pipeline] Config:
  seed: 42
  dataset: {name: CAMI_toy, path: data/CAMI_toy, signals: [kmers, abundance, taxonomy], num_bams: 50}
  ...
[megobin.pipeline] Representation: UncertainGenRepresentation
[megobin.pipeline] Loss:           MahalanobisBCELoss
[megobin.pipeline] Binner:         InfomapBinner
[megobin.pipeline] Evaluator:      CheckM2Evaluator
[megobin.pipeline] Dataset:        CAMI_toy (data/CAMI_toy)
[megobin.pipeline] Loaded features: k-mer + abundance → (N, 236)
[megobin.pipeline] Logger:         TensorBoardLogger
[megobin.pipeline] Trainer:        TwoPhaseTrainer
[megobin.pipeline] Sampler:        HybridPairSampler (size=...)
[megobin.trainers.two_phase] Phase 1/2 (params=mean, epochs=50) ...
[megobin.trainers.two_phase] Phase 2/2 (params=cov, epochs=25)  ...
[megobin.pipeline] Embeddings: (N, 256)
[megobin.pipeline] Bins: K
[megobin.pipeline] Wrote K bin FASTA files to bins
[megobin.pipeline] CheckM2 results:
                     completeness  contamination
bin_0000.fasta          92.1            1.4
bin_0001.fasta          78.4            3.8
...
```

The key checks:

- `Loaded features: ... → (N, 236)` — 136 canonical k-mers + 2 × 50 BAMs (mean and variance per BAM). If this says `(N, 136)`, abundance failed to load; check `data/CAMI_toy/abundance.npy` exists.
- `Phase 1/2` and `Phase 2/2` both print. If you only see Phase 1, the trainer crashed mid-training — inspect `outputs/<date>/<time>/pipeline.log` for the traceback.
- `Bins: K` where K is tens to hundreds. A single bin means the clustering collapsed; thousands means it shattered. Either case means the encoder did not produce useful embeddings — that's useful information, but not what we want on a working day.
- The CheckM2 block is present. If CheckM2 is not installed the pipeline logs `CheckM2 not available — skipping evaluation.` and exits cleanly; the evaluation step is best-effort.

## Step 4 — Look at the run directory

Hydra creates a fresh output directory per run. After the run, inspect it:

```bash
RUN_DIR=$(ls -dt outputs/*/* | head -1)
ls "$RUN_DIR"
```

You should see:

```
.hydra/                Resolved config, Hydra metadata
pipeline.log          Full stdout+stderr from the run
encoder.pt            Saved trainer checkpoint
bins/                 Per-bin FASTA files
tb/                   TensorBoard event files
```

Open the log:

```bash
less "$RUN_DIR/pipeline.log"
```

## Step 5 — Launch TensorBoard

```bash
tensorboard --logdir outputs/
```

Navigate to [localhost:6006](http://localhost:6006). You should see:

- **Scalars** → `train/loss`, `train/epoch_loss`, `train/lr`, plus `phase1/...` and `phase2/...` sub-scopes.
- **Scalars** → `eval/mean_completeness`, `eval/mean_contamination`, `eval/n_bins`.
- **Histograms** → per-column histograms over CheckM2 scores.
- **Text** → the full resolved Hydra config.

The `outputs/` directory is the canonical logdir. If you run more experiments later, pointing TensorBoard at `outputs/` (not a specific run) lets you overlay curves across runs.

## What you just did

You composed an experiment from eight YAML files (dataset, features, representation, loss, binner, evaluator, pair_sampler, trainer, logger), Hydra resolved them into a single config, `pipeline.py` instantiated every component, the trainer ran two phases of training, Infomap clustered the resulting embeddings, and CheckM2 scored the bins. Every step was a call through a Protocol — `encoder.training_step(...)`, `binner.cluster(...)`, `evaluator.score(...)` — and no two components imported from each other.

The next chapter explains where the output files live and what each one means.
