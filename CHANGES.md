# v2-contracts: what changed relative to 655b52e

Scope: the **method** and **training** layers. A run is now a DAG of contracted stages; stages declare what they need and provide by port and content label; the order is resolved backwards from the target; every contract is checked before any compute. The structure changes, the numbers do not (see "Numbers" below). The benchmark layer (Snakemake / cluster workflow) is not part of this branch.

Design write-up: [docs/03-architecture/contracts-and-stages.md](docs/03-architecture/contracts-and-stages.md).

## What moved where

| Before (655b52e) | After (v2-contracts) |
|---|---|
| `pipeline.py` ran a fixed sequence and read hidden flags with `getattr` / `hasattr` | `megobin/pipeline.py` composes stages from `cfg.pipeline`, calls `resolve()` and `check()` (`megobin/contracts.py`), then runs them in order |
| `_check_signal_compatibility(cfg)` (features `required_signals` vs dataset `signals`) | one case of the contract check: features stage *needs* `signals.<required_signals>`; the function remains as a thin wrapper over the same check |
| `consumes_depth` (class attr), `is_combined` read by `pipeline.py`; `norm_abundance` gate and combined-mode normalisation inline in `pipeline.py` | `Encoder.feature_view()` returns a view from `megobin/features/views.py` (`FullMatrixView`, `CombinedNormView`, `KmerGateView`); the embedder stage applies it |
| `encoder.include_std` + `loss.include_std`, flipped by `setattr` from `encoder_attrs` / `loss_attrs` in `configs/trainer/two_phase.yaml` | labelled `Z(mu, var=None, extra)` (`megobin/z.py`): encoder `provides`, loss `needs["z"]`; each phase of the two-phase trainer declares its own loss |
| `encode_with_uncertainty` found by `hasattr`; filter fed `side_outputs` | `Encoder.encode_z()` returns a labelled `Z`; `UncertaintyFilter.needs` asks for `var` |
| `_instantiate_sampler` / `_instantiate_binner` filtered kwargs by reading `__init__` signatures | `bindings` (constructor kwarg -> named value) declared on the component class; values come from the dataset **manifest** (`megobin/manifest.py`) and the stage |
| `CheckM2Evaluator` instantiated and called inside `main()` | not part of the run; separate entry point `python -m megobin.evaluate` (`megobin/evaluate.py`, `configs/evaluate.yaml`) |
| `.hydra/` only | plus `status.json` (`megobin/status.py`): resolved order, contract result, config sha256, seed, interventions ledger L0-L3, summary |
| no refiner slot, encoder always required | optional `refiner` stage (`megobin/refiners/`), resolved only for `pipeline.target: bins_refined`; `PassThroughEmbedder` for tools with no model |
| flat experiment configs | experiments pick `/pipeline: embed_and_bin` (`configs/pipeline/`) and set `pipeline.name` / `pipeline.interventions` |

New modules: `megobin/contracts.py`, `z.py`, `ports.py`, `manifest.py`, `status.py`, `evaluate.py`, `stages/{features,embedder,filter,binner,refiner}.py`, `refiners/{base,no_op}.py`, `features/views.py`. New configs: `configs/pipeline/{embed_and_bin,bin_raw_features}.yaml`, `configs/evaluate.yaml`. Unchanged: `megobin/features/*` (other than the new `views.py`), `megobin/utils/*`, `megobin/evaluators/*`, loggers, `hpc/Snakefile`.

## Signatures that changed

Import paths and class names of existing components are unchanged. Changes:

| Component | Change |
|---|---|
| `megobin.pipeline` | removed private helpers `_load_data_csv`, `_load_data_split_csv`, `_load_sampler_inputs`, `_instantiate_sampler`, `_instantiate_binner` (loading moved to `Features` in `ports.py` / `stages/features.py`; introspection replaced by `bindings`). `_check_signal_compatibility(cfg)` kept; now raises `ContractError` (a `ValueError` subclass). `seed_everything`, `main` kept. New: `run_pipeline`, `build_stages`, `make_plan`, `RunContext`, `Plan`, `RunResult` |
| `Encoder` protocol | + `needs`, `provides`, `feature_view()`, `encode_z()` |
| `SemiBin2Encoder` | constructor unchanged. Removed class attribute `consumes_depth`. `training_step` now passes `Z` objects to the loss. + `needs = {features: {kmer, depth}}`, `provides = {mu}`, `feature_view()`, `encode_z()` |
| `UncertainGenEncoder` | constructor parameter `include_std` still accepted but **ignored with a `DeprecationWarning`** (default `False` -> `None`); instance attribute `include_std` removed. `training_step` computes only the heads in `loss_fn.needs["z"]` and passes `Z`. + `needs`, `provides = {mu, var}`, `z(x, want)`, `feature_view()`, `encode_z()`. `encode()` and `encode_with_uncertainty()` unchanged |
| `ContrastiveLoss` protocol | + `needs`; `__call__(z_i, z_j, label)` takes `Z` objects |
| `HingeContrastiveLoss` | `forward` accepts `Z` or a bare tensor (read as `Z(mu=tensor)`); + `needs` |
| `MahalanobisBCELoss` | `forward` takes `Z`; a bare tensor is read as `Z(mu=...)`. The old `include_std=True` input of concatenated `[mu \| var]` tensors is **no longer accepted** (it raises a `ContractError` naming the missing `var`). Constructor unchanged; `include_std` is now the loss's own declaration of what it needs (`needs` property) |
| `PairSampler` protocol | + `provides`, `needs`, `bindings` (class attributes on `SemiBin2PairSampler`, `UncertainGenPairSampler`; constructors unchanged) |
| `Trainer` protocol | + `phase_losses(default)`. `TwoPhaseTrainer`: phase keys `encoder_attrs` / `loss_attrs` now raise `ValueError` pointing to `phase["loss"]`; `phase["loss"]` is new. `fit(encoder, sampler, loss_fn)` unchanged |
| `Filter` protocol | + `needs` (`NoOpFilter`: `z.mu`; `UncertaintyFilter`: `z.mu`, `z.var`); `fit_transform` unchanged |
| `Binner` protocol | + `needs`, `provides`, `embedding`, `bindings`. `DBSCANEnsembleBinner`: constructor and `cluster` unchanged, class attributes added |
| `configs/trainer/two_phase.yaml` | `encoder_attrs` / `loss_attrs` replaced by `phases[i].loss` (full `MahalanobisBCELoss` config per phase) |
| `configs/experiment/*.yaml` | `/evaluator` removed from `defaults`; `/pipeline: embed_and_bin` added; `pipeline:` block added |

Behaviour changes that are not numerical:

- `data.csv` missing: the old run logged a warning and exited 0; now a `ContractError` (exit code 2).
- `SemiBin2Encoder` needs the `depth` label: a k-mer-only dataset now fails at start instead of a warning at encode time.
- The binner needs the `contigs.fasta` port: a dataset without it fails at start (it used to fail later, inside the binner).
- `bins/` and `markers/` are written under the Hydra run directory, not the current working directory (the old `markers/` path was the repository's tracked `markers/` directory when run from the repo root).
- The run no longer scores bins: no CheckM2 DataFrame / `eval/*` scalars in the experiment logger.
- A contract failure writes `status.json` (`CONTRACT_FAIL`), logs every problem and exits with code 2.

## How to run

```bash
# 1. features (unchanged): data.csv, data_split.csv, contigs.fasta in one directory
python -m megobin.features.run_features --contigs contigs.fasta --bams reads.sorted.bam --out data/mydata

# 2. SemiBin2 single-sample long-read path
python megobin/pipeline.py --config-name experiment/semibin2_longread dataset.path=data/mydata

# 3. UncertainGen (default binner.num_process is 0, which fails when no marker cache exists; use >= 1)
python megobin/pipeline.py --config-name experiment/uncertain_gen_dbscan dataset.path=data/mydata binner.num_process=1

# 4. standalone judge on a run's bins (needs CheckM2 on PATH)
python -m megobin.evaluate bins_dir=outputs/<date>/<time>/bins
```

Each run directory holds `.hydra/`, `status.json`, `bins/`, `markers/`, `encoder.pt`, `pipeline.log`.

## Numbers

Same-seed comparison of this tree against 655b52e, run outside the repository (script and tables are delivered separately, not part of the branch). Data: SemiBin v2.6.0 `test/single_sample_data` (`input.fasta`, 40 contigs x 20,000 bp = 800,000 bp, plus `input.sorted.bam`), CPU, one torch thread, each tree's own `run_features`. 116 old-vs-new rows in the summary table (166 in the detail table, which adds one row per epoch for the cannot-link pairs) covering feature matrices and file bytes, sampler input, must-link pairs, cannot-link pairs per epoch, per-batch and per-epoch loss, encoder initial and final weights, encoder inputs/outputs, final embeddings, contig -> bin labels, RNG state after the run, for `semibin2_longread` (seeds 42 and 7, plus a `binner.minfasta=50000` variant so that bins exist), `uncertain_gen_dbscan` (seeds 42 and 7, `binner.num_process=1`) and all branches of the feature views on synthetic matrices: **all bit-identical** (max abs diff 0). A control (655b52e against itself) is also bit-identical, 132 comparisons. CheckM2 was not run.

Why it holds: random-number consumption order is unchanged (encoder built right after seeding; samplers keep their own generators; the DataLoader shuffle and dropout use the global torch generator in the same sequence). One consequence of that requirement: during training the encoder computes only the heads the loss asks for (see open decisions).

## Open and provisional decisions

1. **Encoder provides what it can; training computes what the loss asks for.** The brief was "each phase's loss declares what it needs, the encoder always provides what it has". Computing the covariance head in phase 1 would run its dropout (extra RNG draws) and update its BatchNorm statistics, changing the numbers. So `provides` is the capability, `loss.needs["z"]` selects what is computed in training, and inference (`encode_z`) returns everything. Provisional.
2. **Per-phase loss lives in the trainer config** as a full loss config (`phases[i].loss`). `clamp_threshold` is therefore repeated there, and `loss=<other>` on the command line does not change the two-phase losses. Alternative: phases carry only a delta applied to the `loss` group by the embedder stage. Provisional.
3. **Pass-through embedder** declares `z: {raw_features}` by default; a binner has to ask for exactly that. Declaring `mu` instead (`labels: [mu]`) lets raw features stand in for an embedding and should be recorded as an intervention.
4. **Filter is always a stage** in `embed_and_bin` (default no-op) and forwards only `mu` as `z_kept`; `var` is not forwarded.
5. **`SemiBin2Encoder` strictly needs `depth`** (it used to warn). `UncertainGenEncoder` needs only `kmer`; whether the abundance columns survive is still decided by `norm_abundance` at run time (`KmerGateView`), so the contract cannot yet say whether `encoder.input_dim` matches.
6. **Features stage reads the precomputed CSVs only**; computing them stays in `megobin.features.run_features`. Label `depth` is derived from the `data.csv` header, `split` from the existence of `data_split.csv`.
7. **`Bins` holds one integer label per contig.** Multi-label output of a refiner (a contig in two bins) is not writable to `bins/` yet.
8. **`check_inner` does not yet check** that a trainer's `params` group exists in `encoder.parameter_groups()`.
9. **`hpc/Snakefile` is untouched.** It already pointed at paths absent at 655b52e (`src/pipeline.py`, `src.features.kmer_profiles`); see the last section of `docs/03-architecture/contracts-and-stages.md` for what else no longer applies.
10. **Commit order.** The commits are ordered for review (contracts, labelled z, stages, pipeline and judge, configs, tests, docs). The test suite passes on every commit tree against the tests present at that commit (31, 33, 33, 33, 33, 56, 56 passed). Between the second and fourth commit the old `pipeline.py` is still in place next to the new components and is not exercised by any test, so only the later commits should be run end to end.

## Observations about the SemiBin2 path (not changed here)

Read from the code of this repository and SemiBin 2.6.0 (file:line pairs are in the audit table delivered with this work). Numerical behaviour was deliberately left as it is.

- SemiBin draws cannot-link pairs from the N **whole-contig** feature rows and offsets the second index within N contigs (`self_supervised_model.py:135-138`); `SemiBin2PairSampler` draws from the 2N **half-contig** rows (`semibin2_sampler.py:44-49`), so a cannot-link pair can be the two halves of one contig, and the features differ.
- SemiBin computes the hinge loss in float64 with `d = ||z1 - z2||` (`self_supervised_model.py:18,164`); `HingeContrastiveLoss` uses float32 and `sqrt(sum + 1e-8)`.
- SemiBin's `DataLoader` uses `drop_last=True` (`:154`); `SinglePhaseTrainer` keeps the last partial batch.
- SemiBin's model also builds a decoder (`semi_supervised_model.py:102`), whose initialisation draws from the torch RNG; `SemiBin2Encoder` has none. Same seed therefore does not give the same draws as SemiBin, and the cannot-link generator differs (global numpy RNG vs `default_rng([seed, epoch])`).
- On the SemiBin test data (800 kb in total) `semibin2_longread` at the default `minfasta` (200,000 bp) outputs 0 bins in both trees; with `binner.minfasta=50000` it outputs 8 bins (29 of 40 contigs). This is a property of the tiny test set, not evidence about real data.
- `uncertain_gen_dbscan`: the default `binner.num_process: 0` raises `ValueError: Number of processes must be at least 1` when no marker cache exists (observed on 655b52e from an empty working directory). In phase 2 the loss is constant at 0.4614 from the first epoch on this data: `log_q` is clamped at -1 for every pair (`mahalanobis_bce.py` clamp), which gives exactly that value for 40 positives and 8000 negatives, and the clamp passes no gradient, so the covariance head does not learn here (provisional: 40 contigs).
- `norm_abundance` hard-codes 136 as the k-mer width in both code bases.
