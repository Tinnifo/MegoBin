# TensorBoard

MegoBin uses TensorBoard as the default experiment tracker. It is not the only option — the `Logger` Protocol would let you add Weights & Biases, MLflow, or a custom backend — but TensorBoard is what ships and what every shared config uses. This chapter covers what gets logged, how to compare runs, and how cross-cluster visibility works.

## What's logged per run

The `TensorBoardLogger` in `megobin/utils/tensorboard_logger.py` implements the six methods of the `Logger` Protocol. Every run produces:

**Resolved Hydra config.** Logged as a text artifact at the start of the run. Inspectable in the TensorBoard "Text" tab as a readable YAML dump.

**Training loss scalars.** Per-step `train/loss` (frequency governed by `trainer.log_every`), per-epoch `train/epoch_loss`, and `train/lr`. Two-phase runs additionally emit `phase1/loss`, `phase1/epoch_loss`, `phase2/loss`, `phase2/epoch_loss` so you can overlay phases without having to filter by step range.

**Evaluation scalars.** `eval/mean_completeness`, `eval/mean_contamination`, `eval/n_bins` at step 0 after evaluation completes.

**Evaluation DataFrame.** The CheckM2 per-bin DataFrame rendered as a markdown table in the "Text" tab under `checkm2_results`.

**Per-column histograms.** One histogram per numeric column of the CheckM2 DataFrame. Useful for seeing the distribution of completeness scores across bins, not just the mean.

**Checkpoints.** The trainer calls `logger.log_checkpoint(path)` after saving. `TensorBoardLogger` copies the checkpoint into a `checkpoints/` subdirectory of the logdir so the event file and the checkpoint travel together.

None of this is magic — every one of these calls is visible in the trainer and pipeline source. `TensorBoardLogger.log_dataframe` specifically renders via `df.to_markdown()`, which requires the `tabulate` package; `environment.yml` includes it.

## Where the logdir lives

Default:

```yaml
# configs/logger/tensorboard.yaml
_target_: megobin.utils.tensorboard_logger.TensorBoardLogger
logdir: ${hydra:runtime.output_dir}/tb
name: null         # defaults to the logdir's basename
flush_secs: 30
```

`${hydra:runtime.output_dir}` resolves to `outputs/<date>/<time>/`, so each run gets its own event file at `outputs/<date>/<time>/tb/`. Point TensorBoard at the `outputs/` root — not an individual run — and you can overlay every run you've ever done:

```bash
tensorboard --logdir outputs/
```

`name` is used as the run's display name in TensorBoard. Null means "use the logdir's basename", which gives you `tb` for every run — not helpful for comparison. Set it to something distinguishing when running sweeps:

```bash
python megobin/pipeline.py \
  --config-name experiment/training/uncertain_gen_cami_toy \
  logger.name="uncertain_gen_seed42"
```

Convention for sweep runs: `{hypothesis-id}-{seed}-{config-name}`, e.g. `H1-seed1-vae-baseline`. This plays nicely with the Linear hypothesis-board workflow — run names round-trip to hypothesis IDs.

## Comparing runs

TensorBoard's run filter (the box at the top-left of Scalars) accepts regex. Useful patterns:

- `H1-` — all runs for hypothesis H1
- `seed1` — all seed-1 runs for ablations
- `uncertain_gen` — all UncertainGen runs regardless of hypothesis

The smoothing slider on the right-hand side of Scalars plots is worth knowing about — MegoBin logs every step, so raw curves are noisy. Smoothing 0.6 is a good default for reading epoch-level trends.

To hide runs from the plot, use the checkboxes in the runs list. Clicking a run's color swatch lets you change its color — useful when you have six similar-looking curves and need to tell them apart.

## Cross-cluster visibility

TensorBoard has no cloud backend. You have two realistic workflows:

**Shared NFS.** Set `logger.logdir` to a path on a shared filesystem (`/projects/microbial-dark-matter/metagenomic-binning/tb/` on BioCloud) so both clusters write to the same tree. Run `tensorboard --logdir` once from a login node with SSH port forwarding:

```bash
ssh -L 6006:localhost:6006 tinni@biocloud
tensorboard --logdir /projects/microbial-dark-matter/metagenomic-binning/tb/ --port 6006 --bind_all
```

Open [localhost:6006](http://localhost:6006) in your browser on the laptop.

**Pull-to-local.** Leave the logdir at its default and rsync the tree to your workstation periodically:

```bash
rsync -avz tinni@biocloud:/home/tinni/Metagenomic-Binning/outputs/ ~/local-outputs/
tensorboard --logdir ~/local-outputs/
```

Either works. The rule is that event files must be visible to one process that runs `tensorboard`. Most people use the pull-to-local pattern while iterating, and the shared NFS pattern for long sweeps everyone on the team needs to see.

## Ablation runs

Ablation sweeps benefit from a parent-directory convention. Instead of letting every run default to `outputs/<date>/<time>/tb/`, group them under a shared ablation name:

```bash
python megobin/pipeline.py -m \
  --config-name experiment/training/uncertain_gen_cami_toy \
  seed=1,2,3 \
  hydra.run.dir=outputs/ablation_k_neighbours/k\${binner.k_neighbours}_seed\${seed} \
  binner.k_neighbours=50,100,200
```

Now `outputs/ablation_k_neighbours/` holds nine runs, each with a descriptive name. Point TensorBoard at `outputs/ablation_k_neighbours/` and you get a clean view of just the ablation.

## Writing a different Logger

The `Logger` Protocol is six methods. If you want to add W&B, you write one class that satisfies those six methods and one YAML. No pipeline change. The relevant methods:

```python
def log_scalars(self, values: dict[str, float], step: int) -> None: ...
def log_config(self, config: dict[str, Any]) -> None: ...
def log_text(self, key: str, text: str) -> None: ...
def log_dataframe(self, key: str, df: pd.DataFrame) -> None: ...
def log_checkpoint(self, path: Path, name: str = "checkpoint") -> None: ...
def finish(self) -> None: ...
```

The NoOp logger (`megobin/utils/no_op_logger.py`) is the shortest possible reference — it implements all six as no-ops. Use it as a template.

## Quiet runs

For smoke tests and CI, silence the logger:

```bash
python megobin/pipeline.py --config-name experiment/hybrid_uncertain_gen logger=none
```

The `NoOpLogger` satisfies the Protocol but does nothing. Trainers and the pipeline keep calling the same methods; they just no-op. Nothing appears in `outputs/<run>/tb/` because nothing is written. Very useful for the end-to-end test — running it with real TensorBoard events would create log noise on every test invocation.
