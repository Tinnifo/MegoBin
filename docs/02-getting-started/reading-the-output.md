# Reading the output

Every `python megobin/pipeline.py ...` invocation writes everything into a fresh `outputs/<date>/<time>/` directory. Hydra manages that; nothing in the code hardcodes a path. This chapter is a tour of what lives in that directory, what each file is for, and how to read it when something has gone wrong.

## Top-level layout

A completed run looks like this:

```
outputs/2026-04-23/14-27-51/
├── .hydra/
│   ├── config.yaml         Fully-resolved config used for this run
│   ├── hydra.yaml          Hydra's own config (rarely matters)
│   └── overrides.yaml      Just the CLI overrides you passed
├── pipeline.log            Full stdout+stderr
├── encoder.pt              Final encoder checkpoint
├── bins/                   Per-bin FASTA files
│   ├── bin_0000.fasta
│   ├── bin_0001.fasta
│   └── ...
└── tb/                     TensorBoard event files
    └── events.out.tfevents...
```

The canonical reference is `pipeline.py` — the `bins/` directory is written at lines ~189–208, the checkpoint path comes from the trainer config (`checkpoint_path: ${hydra:runtime.output_dir}/encoder.pt` in `configs/trainer/*.yaml`), and the TensorBoard logdir likewise comes from `configs/logger/tensorboard.yaml`.

## `.hydra/config.yaml` — the exact config that ran

This is the single most useful file in a run directory. It captures every default, every override, every interpolation resolved. When a result is surprising six months from now and you want to know **what was the lr again?**, open this file.

```bash
cat outputs/2026-04-23/14-27-51/.hydra/config.yaml
```

If you ever want to **rerun** an old experiment with exactly the same config, the simplest path is:

```bash
python megobin/pipeline.py --config-path outputs/2026-04-23/14-27-51/.hydra --config-name config
```

Hydra will happily consume its own output.

## `pipeline.log` — the run transcript

This is the full stdout+stderr from the run, captured by Hydra's logging. It has the same content you saw on the terminal during `python megobin/pipeline.py ...`, plus anything that scrolled off screen.

Most useful patterns:

```bash
# Grep for errors and warnings only
grep -E "WARNING|ERROR" pipeline.log

# Extract the per-epoch training loss
grep "epoch_loss" pipeline.log

# Find the final CheckM2 table
grep -A 50 "CheckM2 results" pipeline.log
```

If a run failed, this file has the Python traceback. If a run hung, this file has the last thing it said before freezing — often a data-loading step.

## `encoder.pt` — the trained weights

A standard PyTorch state dict. Reload it via the `load_checkpoint` helper in [`megobin/utils/checkpoints.py`](https://github.com/Tinnifo/Metagenomic-Binning/blob/main/megobin/utils/checkpoints.py), or via Hydra's `resume_from` override, which skips training entirely:

```bash
python megobin/pipeline.py \
  --config-name experiment/hybrid_uncertain_gen \
  resume_from=outputs/2026-04-23/14-27-51/encoder.pt
```

That rerun loads the checkpoint, encodes, bins, and evaluates — useful for re-running evaluation against a different binner or evaluator without retraining.

Two useful details. First, the checkpoint is saved at the **end** of `trainer.fit()`. If you need intermediate snapshots, set `checkpoint_every: N` in `configs/trainer/single_phase.yaml` or `checkpoint_per_phase: true` in `configs/trainer/two_phase.yaml`. Second, setting `checkpoint_path: null` in the trainer config disables saving entirely — useful for smoke tests where you do not care about the output.

## `bins/` — FASTA files, one per bin

Each file is a standard FASTA with the contigs assigned to that bin. File names are zero-padded four-digit bin IDs (`bin_0000.fasta`, `bin_0001.fasta`, ...). Contigs are pulled by name from the input `contigs.fasta` living under the dataset directory; a bin is skipped for a contig only when the sequence lookup fails, which should never happen in a healthy dataset.

The `bins/` directory is the input to the evaluator. CheckM2 ingests it, runs Prodigal → DIAMOND blastp → KEGG feature engineering → dual ML models, and produces a `quality_report.tsv`. The wrapper in `megobin/evaluators/checkm2.py` parses that TSV into a DataFrame with `completeness` and `contamination` columns.

## `tb/` — TensorBoard event files

One event file per run, written by the `TensorBoardLogger` in [`megobin/utils/tensorboard_logger.py`](https://github.com/Tinnifo/Metagenomic-Binning/blob/main/megobin/utils/tensorboard_logger.py). Point TensorBoard at the `outputs/` root rather than this specific directory so you can overlay multiple runs:

```bash
tensorboard --logdir outputs/
```

The Logger Protocol guarantees the following artifacts are written on every run:

| What                       | Where in TB              | When written                              |
|----------------------------|--------------------------|-------------------------------------------|
| Full resolved config       | Text (`config`)          | Start of run                              |
| Per-step training loss     | Scalars (`train/loss`)   | Every `log_every` steps during training   |
| Per-epoch training loss    | Scalars (`train/epoch_loss`) | Once per epoch                        |
| Learning rate              | Scalars (`train/lr`)     | Every step                                |
| Phase-scoped variants      | Scalars (`phase{n}/...`) | Two-phase trainer only                    |
| CheckM2 DataFrame          | Text (markdown table)    | Once at evaluation                        |
| CheckM2 per-col histograms | Histograms               | Once at evaluation                        |
| CheckM2 mean scalars       | Scalars (`eval/...`)     | Once at evaluation                        |

If you want to compare two runs, start them under distinguishable run directories — the `TensorBoardLogger` uses the logdir's basename as the run name by default.

Chapter 8's troubleshooting page expands this list.
