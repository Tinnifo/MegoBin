# Your first run

Train UncertainGen on CAMI toy, cluster with Infomap, evaluate with CheckM2. ~10 min on laptop, 2–3 min on a T4.

## Goal

Run `experiment/training/uncertain_gen_cami_toy` end-to-end. Verify:

1. Features loaded with shape `(N, 236)`.
2. Two phases printed: `mean` (50 epochs) → `cov` (25 epochs).
3. Bin count is reasonable (tens to hundreds).
4. CheckM2 DataFrame in the log.
5. `train/loss` curve in TensorBoard.

## Prerequisites

- Install done, `mamba activate megobin`.
- `pytest tests/test_interfaces.py` passes.
- `data/CAMI_toy/` has `kmer_profiles.npy`, `abundance.npy`, `contigs.fasta`.

## Step 1 — Smoke test

```bash
pytest tests/test_end_to_end.py -v
```

Should pass in <1 min. If it fails, fix the install before continuing.

## Step 2 — Dry-run the config

```bash
python megobin/pipeline.py \
  --config-name experiment/training/uncertain_gen_cami_toy \
  --cfg job
```

Prints the resolved config. Best diagnostic for Hydra issues.

## Step 3 — Run

```bash
python megobin/pipeline.py --config-name experiment/training/uncertain_gen_cami_toy
```

Expect:

```
[megobin.pipeline] Encoder:        UncertainGenEncoder
[megobin.pipeline] Loaded features: k-mer + abundance → (N, 236)
[megobin.trainers.two_phase] Phase 1/2 (params=mean, epochs=50) ...
[megobin.trainers.two_phase] Phase 2/2 (params=cov, epochs=25)  ...
[megobin.pipeline] Bins: K
[megobin.pipeline] CheckM2 results: ...
```

Checks:
- `(N, 236)` — 136 k-mers + 2×50 BAMs. `(N, 136)` means abundance failed to load.
- Both phases run. Only Phase 1 = trainer crashed; check `pipeline.log`.
- `Bins: K` — 1 bin = collapsed; thousands = shattered.
- CheckM2 missing → pipeline logs `CheckM2 not available — skipping evaluation.` and exits cleanly.

## Step 4 — Inspect output

```bash
RUN_DIR=$(ls -dt outputs/*/* | head -1)
ls "$RUN_DIR"
```

```
.hydra/        Resolved config + Hydra metadata
pipeline.log   stdout+stderr
encoder.pt     Trainer checkpoint
bins/          Per-bin FASTA files
tb/            TensorBoard event files
```

## Step 5 — TensorBoard

```bash
tensorboard --logdir outputs/
```

Open [localhost:6006](http://localhost:6006). You should see `train/loss`, `train/epoch_loss`, `train/lr`, `phase1/...`, `phase2/...`, `eval/mean_completeness`, `eval/mean_contamination`, `eval/n_bins`, plus the resolved config under Text.

Point at `outputs/` (not a single run) to overlay multiple runs.
