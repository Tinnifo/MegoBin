# FAQ

Short answers to questions people actually ask. If the question isn't here and it probably should be, add it.

## What is MegoBin for?

Metagenomic binning — clustering assembled contigs into bins that each correspond to a single microbial genome — specifically for the kinds of organisms (low-abundance, CPR, microbial dark matter) that VAMB and SemiBin2 struggle with. It's a research codebase, not a production tool: the priority is rapid iteration on new deep-learning approaches, not stability.

## How is this different from SemiBin?

MegoBin treats SemiBin as *one configuration* of a general three-slot pipeline (Representation → Binner → Evaluator). You can still run a SemiBin-equivalent — `experiment/training/semibin_cami_toy.yaml` does exactly that — but you can also swap in UncertainGen, DNABERT-S, or a future encoder without touching the binner or evaluator. The architecture is the point.

See [Design philosophy](../01-introduction/design-philosophy.md) for the full argument.

## Which encoder should I use?

Default: **UncertainGen** (`experiment/hybrid_uncertain_gen.yaml`). The Mahalanobis distance plus two-phase training is the current best baseline on CAMI toy.

If you're debugging something and want a simpler encoder with no phased training, **SemiBin** (`experiment/training/semibin_cami_toy.yaml`).

If you're ablating the hybrid sampler's contribution, `experiment/semibin_pairs_only.yaml` or `experiment/random_pairs_only.yaml`.

## Infomap or DBSCAN ensemble?

Infomap is the default. SemiBin2's paper uses Infomap for short reads, DBSCAN ensemble for long reads. Our CAMI benchmarks are short-read, so Infomap.

When you care: running both and comparing is the typical ablation ("does my new encoder help both binners, or just one?"). See [Swap the binner](../04-tutorials/swap-the-binner.md).

## Why two-phase training for UncertainGen?

The mean head and covariance head fight each other if trained simultaneously — the covariance finds a low-energy solution where it inflates to absorb loss the mean head should have been responsible for. Train the mean first to convergence, freeze it, then train covariance on the residual. See [Two-phase training](../04-tutorials/two-phase-training.md).

## Can I add a new encoder without modifying pipeline.py?

Yes. One Python file, one YAML, one test case. Zero changes to `pipeline.py`, `binners/`, `evaluators/`, or any other slot. See [Add a new encoder](../04-tutorials/add-a-new-encoder.md).

## How does pair sampling interact with the loss?

The pair sampler produces `(feature_i, feature_j, label)` triples. The encoder's `training_step` forwards both features through the model to get `(z_i, z_j)`, then calls `loss_fn(z_i, z_j, label)`. The loss interprets `label` — `HingeContrastiveLoss` treats it as `1 = must-link, 0 = cannot-link`; `MahalanobisBCELoss` does the same but computes distance in Mahalanobis space.

Changing the label semantics (e.g. taxonomy IDs instead of 0/1) breaks this contract and requires a new loss *and* a new sampler. The `ContrastiveLoss` Protocol signature is designed around the 0/1 case.

## How do I run an ablation sweep?

Hydra's multirun (`-m` flag):

```bash
python megobin/pipeline.py -m \
  --config-name experiment/training/uncertain_gen_cami_toy \
  seed=1,2,3 \
  binner.k_neighbours=50,100,200
```

Nine runs, each in its own output directory. For TensorBoard-friendly grouping, set `hydra.run.dir` to a parent directory name that includes the ablation variable:

```bash
hydra.run.dir=outputs/ablation_k_neighbours/k\${binner.k_neighbours}_seed\${seed}
```

## Where are my results?

`outputs/<date>/<time>/` for each run:

- `encoder.pt` — trained weights
- `bins/bin_*.fasta` — cluster FASTAs
- `checkm2/quality_report.tsv` — CheckM2 output
- `tb/` — TensorBoard event files + checkpoints + run_meta.json
- `pipeline.log` — run log

See [Reading the output](../02-getting-started/reading-the-output.md).

## My run crashed halfway through. What do I lose?

If it crashed during training: nothing persisted unless `checkpoint_every: N` was set. Set it for long runs.

If it crashed during binning or evaluation (after training): `encoder.pt` is already saved. Resume with `resume_from=outputs/.../encoder.pt` and a different binner/evaluator. See [Checkpoints and DVC](../05-experiment-tracking/checkpoints-and-dvc.md).

## Why does TensorBoard show my run as just `tb`?

`logger.name` defaults to the logdir basename, which is `tb` for every run. Override:

```bash
logger.name=H1-seed1-uncertain_gen
```

The naming convention for sweeps: `{hypothesis-id}-{seed}-{config-name}`.

## Can I use Weights & Biases instead of TensorBoard?

Yes, by writing a `WandbLogger` class that satisfies the six-method `Logger` Protocol (see `megobin/utils/logger.py`). Also add a `configs/logger/wandb.yaml`. No pipeline changes needed.

The `NoOpLogger` (`megobin/utils/no_op_logger.py`) is a one-file template showing the minimum bar.

## BioCloud or DEIS-MCC for my training run?

**DEIS-MCC if:**
- The run needs a GPU for more than 30 minutes.
- You're sweeping multiple configurations and need throughput.
- The dataset is already on DEIS-MCC scratch.

**BioCloud if:**
- You need CheckM2 evaluation (the database lives there).
- You're debugging feature computation.
- The run is short (<30 min).
- The dataset isn't on DEIS-MCC yet.

See [BioCloud](../06-hpc/biocloud.md) and [DEIS-MCC](../06-hpc/deis-mcc.md).

## Why is my feature computation running on GPU?

It shouldn't be. `compute_kmer_profiles_with_splits` is CPU-only. If `nvidia-smi` shows GPU utilisation during feature computation, something weird is happening — check you're not accidentally running the training pipeline when you meant to compute features.

## Why does the pipeline fail with "dataset compatibility"?

The dataset config declares `signals: [kmers, abundance]` but your encoder or feature config requires `taxonomy`. The pipeline does a fail-fast compatibility check before training. Either:

- Use a dataset with taxonomy (e.g. `CAMI_toy` has `[kmers, abundance, taxonomy]`).
- Switch to a features config that doesn't require taxonomy (`canonical_kmer_abundance` is fine).

## How much VRAM does UncertainGen need?

With `batch_size: 10000` and the default `input_dim: 236`, `hidden_dim: 512`, `embedding_dim: 100`: about 2–3 GB during training. Fits comfortably on a T4 (16GB). Much smaller models than vision backbones.

## How long does a full run take?

On CAMI toy, DEIS-MCC turing partition, UncertainGen hybrid two-phase training: roughly 15–30 minutes end-to-end (train + encode + bin + evaluate). CAMI medium is a few hours. Larger datasets scale roughly linearly in the feature-computation and binning steps; training is fixed-size.

## How do I reproduce someone else's run?

Checkout the Git SHA from their `run_meta.json`, recreate the conda env (or verify `env_hash` matches), `dvc pull` the data, rerun with their config:

```bash
git checkout $(jq -r '.git_sha' run_meta.json)
mamba env create -f environment.yml
dvc pull
python megobin/pipeline.py --config-name experiment/training/uncertain_gen_cami_toy seed=1
```

See [Checkpoints and DVC](../05-experiment-tracking/checkpoints-and-dvc.md).

## What's the difference between `configs/experiment/` and `configs/experiment/training/`?

`experiment/*.yaml` are "headline" configs — they pin the representation, loss, binner, and sampler but leave hyperparameters at sensible defaults. Good for iteration.

`experiment/training/*.yaml` are "fully pinned" configs — every hyperparameter is fixed, cited to the paper it came from. Good for reproducing a result months later. Always use these for runs that back a hypothesis.

## How do I cite the hyperparameters I used?

Put the citation as a YAML comment in the experiment config. Example (from an ideal `experiment/training/uncertain_gen_cami_toy.yaml`):

```yaml
# UncertainGen on CAMI toy.
# Hyperparameters from Çelikkanat et al. 2024 (https://arxiv.org/abs/xxxx.yyyy)
#   phase 1 epochs=50, phase 2 epochs=25 — their Table 3
#   hidden_dim=512, embedding_dim=100 — their §4.2
```

Commit the config. Run `git log configs/experiment/training/` to see when each pinned config landed.

## Is there a Windows version?

No. MegoBin has been tested only on Linux (BioCloud, DEIS-MCC, an Ubuntu laptop). It might run on macOS with `conda` but that's untested. WSL2 probably works but nobody's tried.
