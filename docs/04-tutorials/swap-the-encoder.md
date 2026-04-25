# Tutorial: Swap the encoder

Five minutes. You will run the same experiment twice, once with UncertainGen and once with SemiBin, and compare the results in TensorBoard. This is the simplest possible demonstration of the slot-based architecture: no code changes, two CLI overrides.

## Goal

Produce two runs of the `experiment/hybrid_uncertain_gen` pipeline on CAMI toy — one with the default UncertainGen encoder, one with the SemiBin encoder — and open both in TensorBoard to compare loss curves and bin counts.

## Prerequisites

Installation done (Chapter 2), `data/CAMI_toy/` present with features, and `pytest tests/test_interfaces.py` passes.

## Step 1 — Run UncertainGen (the default)

```bash
python megobin/pipeline.py --config-name experiment/training/uncertain_gen_cami_toy
```

This is the same command as Chapter 2's first-run tutorial. When it finishes, note the output directory — we'll need it in Step 3.

```bash
UNCERTAIN_RUN=$(ls -dt outputs/*/* | head -1)
echo "$UNCERTAIN_RUN"
```

## Step 2 — Run SemiBin by swapping three fields

You could use `configs/experiment/training/semibin_cami_toy.yaml` which already pins the right combination, but for the point of this tutorial, do it the manual way — that's how you'd swap in an encoder that does not yet have its own training config.

SemiBin and UncertainGen disagree on three things: the encoder, the loss, and the trainer. UncertainGen uses Mahalanobis BCE over a two-phase schedule; SemiBin uses hinge contrastive over a single-phase schedule. So the command is:

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

A few notes on why those extra overrides are there. The UncertainGen YAML has `hidden_dim` and `dropout` fields; SemiBin does not have `hidden_dim` (it's a fixed 3-layer MLP with 512-wide hidden layers) — the `~` syntax deletes a key. SemiBin's embedding dim is 100, not 256.

A cleaner shortcut exists: just use the pre-pinned SemiBin config.

```bash
python megobin/pipeline.py --config-name experiment/training/semibin_cami_toy
```

This one is preferred in practice — that's what `configs/experiment/training/` is for. The long-form above is here to show you that the CLI override mechanism is fully general.

After the run:

```bash
SEMIBIN_RUN=$(ls -dt outputs/*/* | head -1)
echo "$SEMIBIN_RUN"
```

## Step 3 — Compare in TensorBoard

```bash
tensorboard --logdir outputs/
```

Open [localhost:6006](http://localhost:6006). In the Scalars tab you should see two runs (identified by the timestamped paths). Compare:

- `train/loss` — different scales because the loss functions are different, but both should trend down.
- `train/epoch_loss` — same.
- `eval/mean_completeness` and `eval/mean_contamination` — these are the interpretable ones. A higher completeness at a similar contamination is a better encoder on this dataset.
- `eval/n_bins` — dramatically different bin counts often indicate a binning issue rather than an encoder issue.

You can use TensorBoard's regex filter to scope to just `eval/` or `train/`. The checkbox on the left lets you hide runs; handy if you have a lot of old ones cluttering the view.

## What changed and what didn't

Compare the two resolved configs:

```bash
diff <(python megobin/pipeline.py --config-name experiment/training/uncertain_gen_cami_toy --cfg job 2>/dev/null) \
     <(python megobin/pipeline.py --config-name experiment/training/semibin_cami_toy --cfg job 2>/dev/null)
```

You will see four blocks of differences: the `encoder` block (class and its kwargs), the `loss` block (class and margin vs clamp_threshold), the `trainer` block (two_phase vs single_phase plus all the nested phase structure), and the `pair_sampler` block (hybrid vs semibin).

That's the whole change surface: four YAML files of configuration, zero lines of Python. `megobin/pipeline.py` did not change. None of the other components changed. The binner, evaluator, logger, feature loader, and FASTA writer are all shared.
