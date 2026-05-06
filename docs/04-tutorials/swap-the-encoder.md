# Tutorial: Swap the encoder

Run UncertainGen and SemiBin on CAMI toy, compare in TensorBoard. ~5 min.

## Step 1 — UncertainGen

```bash
python megobin/pipeline.py --config-name experiment/training/uncertain_gen_cami_toy
UNCERTAIN_RUN=$(ls -dt outputs/*/* | head -1)
```

## Step 2 — SemiBin

The pinned config is the easy path:

```bash
python megobin/pipeline.py --config-name experiment/training/semibin_cami_toy
```

The manual override path (for components without a pinned config):

```bash
python megobin/pipeline.py \
  --config-name experiment/training/uncertain_gen_cami_toy \
  encoder=semibin_encoder \
  loss=hinge \
  trainer=single_phase \
  pair_sampler=semibin \
  'encoder.input_dim=236' \
  'encoder.embedding_dim=100' \
  '~encoder.hidden_dim' \
  '~encoder.dropout=null' \
  +encoder.dropout=0.2
```

`~` deletes a key (SemiBin has no `hidden_dim`).

## Step 3 — Compare

```bash
tensorboard --logdir outputs/
```

Watch:
- `train/loss` — different scales (different losses); both should descend.
- `eval/mean_completeness` / `eval/mean_contamination` — the metrics that matter.
- `eval/n_bins` — large differences usually point to a binning issue, not encoder issue.

## What changed

```bash
diff <(python megobin/pipeline.py --config-name experiment/training/uncertain_gen_cami_toy --cfg job 2>/dev/null) \
     <(python megobin/pipeline.py --config-name experiment/training/semibin_cami_toy --cfg job 2>/dev/null)
```

Four blocks: `encoder`, `loss`, `trainer`, `pair_sampler`. Zero Python changes.
