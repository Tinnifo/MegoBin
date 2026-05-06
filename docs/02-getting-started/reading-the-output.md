# Reading the output

Every run writes to `outputs/<date>/<time>/`.

## Layout

```
outputs/2026-04-23/14-27-51/
├── .hydra/
│   ├── config.yaml         Fully-resolved config
│   ├── hydra.yaml          Hydra's own config
│   └── overrides.yaml      CLI overrides
├── pipeline.log            stdout+stderr
├── encoder.pt              Final checkpoint
├── bins/                   bin_XXXX.fasta files
└── tb/                     TensorBoard events
```

## `.hydra/config.yaml`

The exact config that ran. Most useful file in the directory.

Re-run an old experiment:

```bash
python megobin/pipeline.py --config-path outputs/2026-04-23/14-27-51/.hydra --config-name config
```

## `pipeline.log`

Full stdout+stderr.

```bash
grep -E "WARNING|ERROR" pipeline.log
grep "epoch_loss" pipeline.log
grep -A 50 "CheckM2 results" pipeline.log
```

## `encoder.pt`

PyTorch state dict. Reuse without retraining:

```bash
python megobin/pipeline.py \
  --config-name experiment/hybrid_uncertain_gen \
  resume_from=outputs/2026-04-23/14-27-51/encoder.pt
```

Saved at end of `trainer.fit()`. Disable saving with `checkpoint_path: null`. Intermediate snapshots: `checkpoint_every: N` (single phase) or `checkpoint_per_phase: true` (two phase).

## `bins/`

One FASTA per bin, zero-padded IDs (`bin_0000.fasta`). Input to the evaluator.

## `tb/`

Per-run TensorBoard event file. Point at `outputs/` to overlay runs:

```bash
tensorboard --logdir outputs/
```

| Logged                     | Where                        | When                      |
|----------------------------|------------------------------|---------------------------|
| Resolved config            | Text (`config`)              | Start                     |
| Per-step loss              | Scalars (`train/loss`)       | Every `log_every` steps   |
| Per-epoch loss             | Scalars (`train/epoch_loss`) | Each epoch                |
| Learning rate              | Scalars (`train/lr`)         | Each step                 |
| Phase-scoped variants      | Scalars (`phase{n}/...`)     | Two-phase trainer only    |
| CheckM2 DataFrame          | Text (markdown)              | Once at evaluation        |
| CheckM2 histograms         | Histograms                   | Once at evaluation        |
| CheckM2 mean scalars       | Scalars (`eval/...`)         | Once at evaluation        |
