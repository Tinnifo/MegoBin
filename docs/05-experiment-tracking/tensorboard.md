# TensorBoard

Default tracker. The `Logger` Protocol is six methods — swap to W&B or MLflow by writing one class.

## What's logged

| Artifact                 | Where in TB              | Frequency                    |
|--------------------------|--------------------------|------------------------------|
| Resolved Hydra config    | Text (`config`)          | Start                        |
| Per-step `train/loss`    | Scalars                  | Every `trainer.log_every`    |
| Per-epoch `train/epoch_loss`, `train/lr` | Scalars  | Each epoch / step            |
| Phase-scoped `phase{n}/...` | Scalars               | Two-phase only               |
| `eval/mean_completeness`, `eval/mean_contamination`, `eval/n_bins` | Scalars | Once at evaluation |
| CheckM2 per-bin DataFrame | Text (markdown)         | Once at evaluation           |
| Per-column histograms    | Histograms               | Once at evaluation           |
| Checkpoints              | `checkpoints/` subdir    | After save                   |

`log_dataframe` uses `df.to_markdown()` — needs `tabulate` (in `environment.yml`).

## Logdir

```yaml
# configs/logger/tensorboard.yaml
_target_: megobin.utils.tensorboard_logger.TensorBoardLogger
logdir: ${hydra:runtime.output_dir}/tb
name: null         # defaults to logdir basename
flush_secs: 30
```

Point at `outputs/` to overlay all runs:

```bash
tensorboard --logdir outputs/
```

Set a meaningful name when comparing:

```bash
python megobin/pipeline.py --config-name <cfg> logger.name="H1-seed1-uncertain_gen"
```

Convention for sweeps: `{hypothesis-id}-{seed}-{config-name}`.

## Comparing runs

Run filter accepts regex: `H1-`, `seed1`, `uncertain_gen`. Smoothing 0.6 is a good default. Click the colour swatch to recolour a run.

## Cross-cluster

**Shared NFS:**

```bash
ssh -L 6006:localhost:6006 tinni@biocloud
tensorboard --logdir /projects/microbial-dark-matter/metagenomic-binning/tb/ --port 6006 --bind_all
```

**Pull to local:**

```bash
rsync -avz tinni@biocloud:/home/tinni/Metagenomic-Binning/outputs/ ~/local-outputs/
tensorboard --logdir ~/local-outputs/
```

## Ablation runs

Group under a named directory:

```bash
python megobin/pipeline.py -m \
  --config-name experiment/uncertain_gen_dbscan \
  seed=1,2,3 \
  hydra.run.dir=outputs/ablation_k_neighbours/k\${binner.k_neighbours}_seed\${seed} \
  binner.k_neighbours=50,100,200
```

Then `tensorboard --logdir outputs/ablation_k_neighbours/`.

## A different Logger

Implement six methods:

```python
def log_scalars(self, values: dict[str, float], step: int) -> None: ...
def log_config(self, config: dict[str, Any]) -> None: ...
def log_text(self, key: str, text: str) -> None: ...
def log_dataframe(self, key: str, df: pd.DataFrame) -> None: ...
def log_checkpoint(self, path: Path, name: str = "checkpoint") -> None: ...
def finish(self) -> None: ...
```

Reference: `megobin/utils/no_op_logger.py` (all six are no-ops).

## Quiet runs

```bash
python megobin/pipeline.py --config-name <cfg> logger=none
```
