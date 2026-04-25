# Tutorial: Swap the binner

The binner is one of the three core slots, and it is completely independent from the encoder. An encoder produces `(N, d)` embeddings; any binner that satisfies the `Binner` Protocol can consume them. That means you can train once and cluster many times.

## Goal

Train UncertainGen on CAMI toy, then run the same embeddings through both Infomap and DBSCAN ensemble, and compare the resulting bins.

## Why you'd do this

Binners make very different assumptions about the embedding space. Infomap assumes the space is dense and noise-robust — it builds a k-NN graph, intersects it with the raw k-mer graph, and runs community detection. DBSCAN ensemble assumes nothing about density structure and sweeps 12 `eps` values, picking per-bin using a marker-gene F1 score. On the same embeddings, these two will produce very different bin counts, and their error modes differ — Infomap tends to over-merge, DBSCAN to over-split.

When a new encoder claims to be better, one of the first ablations is: does it help Infomap, DBSCAN, or both? A useful encoder helps both. An encoder that only helps one is suspicious — usually it's encoding something density-specific.

## Step 1 — Train once, save the checkpoint

```bash
python megobin/pipeline.py --config-name experiment/training/uncertain_gen_cami_toy
```

Grab the checkpoint path for reuse:

```bash
CHECKPOINT=$(ls -t outputs/*/*/encoder.pt | head -1)
echo "$CHECKPOINT"
```

## Step 2 — Rerun with Infomap (the default), using the checkpoint

```bash
python megobin/pipeline.py \
  --config-name experiment/training/uncertain_gen_cami_toy \
  resume_from="$CHECKPOINT" \
  binner=infomap
```

`resume_from` skips training. The pipeline loads the checkpoint, encodes, clusters with Infomap, writes FASTAs, and evaluates. Look at the output directory — no `tb/` events for training loss because the trainer didn't run, but `eval/` scalars are logged.

## Step 3 — Rerun with DBSCAN ensemble, same checkpoint

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

Look at `eval/n_bins` across the two runs. Typical numbers on CAMI toy with UncertainGen embeddings:

- Infomap (default `k_neighbours: 200`): tens of bins.
- DBSCAN ensemble (12 eps values, `min_samples: 5`): significantly more bins per eps value, with the final count determined by per-bin F1 selection.

Compare `eval/mean_completeness` and `eval/mean_contamination` — these are the only numbers you actually care about. More bins is not better if half of them are low-completeness fragments.

## Tuning each binner

Infomap exposes `k_neighbours`, `n_trials`, and `max_bin_size`:

```bash
python megobin/pipeline.py \
  --config-name experiment/training/uncertain_gen_cami_toy \
  resume_from="$CHECKPOINT" \
  binner=infomap \
  binner.k_neighbours=50
```

A smaller `k` produces a sparser graph and typically more bins. `n_trials` is Infomap's internal restart count — 10 is usually plenty. `max_bin_size` caps the size of any single bin (null means no cap).

DBSCAN ensemble exposes `eps_values`, `min_samples`, and `min_bin_size`:

```bash
python megobin/pipeline.py \
  --config-name experiment/training/uncertain_gen_cami_toy \
  resume_from="$CHECKPOINT" \
  binner=dbscan_ensemble \
  'binner.eps_values=[0.1,0.2,0.3]' \
  binner.min_samples=3
```

A shorter `eps_values` list means fewer DBSCAN runs (fewer candidate bins). `min_samples` is DBSCAN's core-point threshold; lower values produce more, smaller bins.
