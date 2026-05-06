# Tutorial: Swap the binner

Train once, cluster many. The binner is independent of the encoder.

## Why

- **Infomap** — assumes dense, noise-robust space. Tends to over-merge.
- **DBSCAN ensemble** — sweeps 12 eps, picks per-bin via marker-gene F1. Tends to over-split.

A useful encoder helps both. Helping only one is suspicious.

## Step 1 — Train, save checkpoint path

```bash
python megobin/pipeline.py --config-name experiment/training/uncertain_gen_cami_toy
CHECKPOINT=$(ls -t outputs/*/*/encoder.pt | head -1)
```

## Step 2 — Re-bin with Infomap

```bash
python megobin/pipeline.py \
  --config-name experiment/training/uncertain_gen_cami_toy \
  resume_from="$CHECKPOINT" \
  binner=infomap
```

## Step 3 — Re-bin with DBSCAN ensemble

```bash
python megobin/pipeline.py \
  --config-name experiment/training/uncertain_gen_cami_toy \
  resume_from="$CHECKPOINT" \
  binner=dbscan_ensemble
```

## Step 4 — Compare

```bash
tensorboard --logdir outputs/
```

Watch `eval/mean_completeness`, `eval/mean_contamination`, `eval/n_bins`. More bins ≠ better.

## Tuning

**Infomap:**

```bash
binner.k_neighbours=50      # smaller k → sparser graph → more bins
binner.n_trials=10          # restart count
binner.max_bin_size=null    # cap (null = no cap)
```

**DBSCAN ensemble:**

```bash
'binner.eps_values=[0.1,0.2,0.3]'
binner.min_samples=3        # lower = more, smaller bins
binner.min_bin_size=1
```
