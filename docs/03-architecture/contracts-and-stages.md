# Contracts and stages

A run is a small DAG of **stages**. Every stage says what it **needs** and what it **provides**, by *port* and by *content label*. The order is resolved backwards from the requested target, and every contract is checked before any compute.

```
features ──► embedder ──► filter ──► binner ──► [refiner]
 (features)    (z)        (z_kept)    (bins)    (bins_refined)
```

| Stage | Class | Needs | Provides |
|---|---|---|---|
| `features` | `megobin.stages.features.PrecomputedFeatures` | `feature_files.data`, `signals.<required_signals>` | `features: {kmer, depth?, split?}` |
| `embedder` | `megobin.stages.embedder.TrainedEmbedder` | `features: <encoder.needs> (+ split if it trains)` | `z: <encoder.provides>` |
| `embedder` (no model) | `megobin.stages.embedder.PassThroughEmbedder` | `features: {kmer}` | `z: {raw_features}` |
| `filter` | `megobin.stages.filter.FilterStage` | `z: <filter.needs>` | `z_kept: {mu}` |
| `binner` | `megobin.stages.binner.BinnerStage` | `<binner.needs>` (`z_kept`, `contigs`, `graph`, ...) | `bins: {one_label_per_contig}` |
| `refiner` | `megobin.stages.refiner.RefinerStage` | `<refiner.needs>` | `bins_refined` |

The refiner is **optional**: it only enters the run when `pipeline.target` is `bins_refined`. With `target: bins` it is configured but never resolved, never run.

## Ports and labels

A *port* is a named piece of data. Its *labels* say what is in it. Checking labels (not file names) is what catches a pairing that type-checks but cannot work, e.g. a loss that reads a variance from an encoder that sends none.

| Port | Labels | Set by |
|---|---|---|
| `signals` | `kmers`, `abundance`, ... | the dataset config (`signals: [...]`) |
| `feature_files` | `data`, `split` | `data.csv` / `data_split.csv` exist |
| `contigs` | `fasta` | `contigs.fasta` exists |
| `graph` | `assembly_graph` | `assembly_graph.gfa` exists |
| `features` | `kmer`, `depth`, `split` | features stage (from the `data.csv` header and the files) |
| `z` | `mu`, `var`, `raw_features` | embedder |
| `z_kept` | `mu` | filter |
| `bins`, `bins_refined` | `one_label_per_contig` | binner, refiner |

The first four are the **manifest** of the dataset (`megobin/manifest.py`): the ports the prepared data offers. Binners no longer receive a fixed kwargs dict that `pipeline.py` filtered by reading `__init__` signatures; a binner class declares `needs` (ports) and `bindings` (constructor kwarg -> named runtime value) and the stage hands those over.

`_check_signal_compatibility` (the old `features.required_signals` ⊆ `dataset.signals` check) is now one case of this: the features stage *needs* `signals.<required_signals>`, and the dataset offers `signals.<signals>`. Either side absent is still a no-op.

## Inner contracts: the embedder

`TrainedEmbedder` hides a training loop (sampler, encoder, loss, trainer). Its `check_inner()` checks:

- every `loss.needs["batch"]` label is offered by the sampler (`provides`, e.g. `views=2`, `label`);
- every `loss.needs["z"]` label can be sent by the encoder (`provides`), **for every phase** of the trainer (`trainer.phase_losses(default)`);
- the sampler's feature needs (`split`) join the stage's needs when it trains.

### Labelled `z`

`megobin.z.Z(mu, var=None, extra={})` replaces the two independent `include_std` flags (one on the encoder, one on the loss, flipped by `setattr` between phases).

- The **encoder** owns *what it can send*: `Encoder.provides` (`{"mu"}` for `SemiBin2Encoder`, `{"mu", "var"}` for `UncertainGenEncoder`).
- The **loss** owns *what it needs*: `loss.needs["z"]`. `HingeContrastiveLoss` needs `{"mu"}`; `MahalanobisBCELoss(include_std=...)` needs `{"mu"}` (unit covariance) or `{"mu", "var"}`.
- Hinge behind a variance-sending encoder is fine: the extra field is ignored. Mahalanobis with `include_std=true` behind an encoder that sends no variance is a **loud error at pipeline start**:

```
embedder/MahalanobisBCELoss: needs z.['var']; offered z.['mu'] by encoder SemiBin2Encoder
```

During training the encoder computes exactly the fields in `loss.needs["z"]` (`UncertainGenEncoder.z(x, want)`); at inference it returns everything it has (`encode_z`). Heads nobody asked for are not run during training: running them would draw dropout random numbers and update BatchNorm statistics, which changes the numbers.

Two-phase training: each phase declares its loss (`phases[i].loss`), so phase 1 trains the mean with a loss that needs only `mu`, and phase 2 trains the covariance with a loss that needs `var`. `encoder_attrs` / `loss_attrs` are rejected with a message pointing here.

### Feature views

The SemiBin-specific preprocessing that used to sit in `pipeline.py` lives in `megobin/features/views.py` and is chosen by the encoder (`Encoder.feature_view()`):

| View | Encoder | Does |
|---|---|---|
| `FullMatrixView` | `SemiBin2Encoder` (single-sample) | passes k-mer + depth on; the encoder reads depth itself |
| `CombinedNormView` | `SemiBin2Encoder(is_combined=True)` | column-sum + L1 normalisation when `norm_abundance` passes; same sums reused for `data_split.csv` |
| `KmerGateView` | `UncertainGenEncoder` | `norm_abundance` gate: drops abundance when too few BAMs |

## Resolution and checking

```python
from megobin.contracts import resolve, check

order = resolve(stages, "bins", manifest.available)   # backward from the target
problems = check(stages, order, manifest.available)   # labels, incl. check_inner()
```

`megobin.pipeline.run_pipeline(cfg)` does: seed -> build stages -> bind to the manifest -> validate the interventions ledger -> resolve -> check -> run in order -> write `bins/` and `status.json`. A contract failure writes `status.json` with `CONTRACT_FAIL`, names every problem, and stops before any stage runs. The command line exits with code 2.

Every message names the component, the port and the missing labels:

```
binner: needs port 'contigs' with labels ['fasta'], but no stage provides it and the input data does not offer it
refiner: needs port 'graph' with labels ['assembly_graph'], but no stage provides it and the input data does not offer it
filter: needs z.['var']; offered z.['mu'] by embedder
```

## Config layout

```
configs/pipeline/embed_and_bin.yaml      features -> embedder -> filter -> binner (target: bins)
configs/pipeline/bin_raw_features.yaml   features -> pass-through embedder -> binner (no model)
configs/experiment/<name>.yaml           picks /pipeline: <shape>, components from the usual groups,
                                         and sets pipeline.name / pipeline.interventions
```

Component groups (`encoder`, `loss`, `pair_sampler`, `trainer`, `filter`, `binner`, `features`) are unchanged; the pipeline shape wires them into stages by interpolation. Any component can be swapped on the command line as before (`loss=hinge`, `filter=uncertainty`).

## `status.json`

Written next to `.hydra/` at the start (after the contract check) and rewritten at the end:

```json
{
 "pipeline": "semibin2_longread",
 "target": "bins",
 "status": "OK",                       // STARTED | CONTRACT_FAIL | ERROR | OK
 "order": ["features", "embedder", "filter", "binner"],
 "contract": {"status": "OK", "problems": []},
 "config_sha256": "…",                 // composed config, interpolations unresolved
 "seed": 42,
 "interventions": [],                  // [{name, level: L0..L3, note}]
 "dataset": {"name": "…", "path": "…"},
 "summary": {"n_contigs": 40, "n_bins": 0, "n_unbinned": 40, "n_dropped_by_filter": 0}
}
```

`interventions` is the ledger of how far a run bends the reference method; entries need a `name` and a `level` in `L0`..`L3`, and a malformed entry is a contract failure.

## The judge is not a stage

Scoring is a separate entry point so every method's bins are judged identically:

```bash
python -m megobin.evaluate bins_dir=outputs/<run>/bins           # CheckM2 by default
python -m megobin.evaluate bins_dir=... evaluator.threads=8
```

It writes `quality.tsv` and `evaluation.json` into its own Hydra output directory.

## Adding a component under contracts

| Component | Declare |
|---|---|
| Encoder | `needs = {"features": {...}}`, `provides = {...}`, `feature_view()`, `encode_z()` |
| Loss | `needs = {"batch": {...}, "z": {...}}` (instance property if it depends on a constructor argument) |
| Pair sampler | `provides` (batch labels), `needs = {"features": {"split"}}`, `bindings = {kwarg: name}` |
| Filter | `needs = {"z": {...}}` |
| Binner | `needs`, `provides`, `embedding = (port, label)`, `bindings` |
| Refiner | `needs`, `provides = {"bins_refined": ...}`, `bindings`, `refine(bins, **inputs)` |
| New stage | subclass `megobin.contracts.Stage`; set `needs` / `provides`; implement `run(data, ctx)` |

## The `hpc/Snakefile`

`hpc/Snakefile` is **not touched by this branch** and already referred to paths that do not exist at 655b52e (`src/pipeline.py`, `src.features.kmer_profiles`). What the contract refactor additionally changes for anyone still driving it:

- the method run no longer writes `checkm2/` or scores bins; the `evaluate` rule's job is `python -m megobin.evaluate bins_dir=<bins>`;
- `bins/` is written inside the Hydra run directory (`<run>/bins`), not the current working directory;
- `dataset=<name>` on the command line selects a dataset config as before.
