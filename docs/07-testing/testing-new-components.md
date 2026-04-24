# Testing new components

MegoBin's test suite is deliberately small — three files, all under `tests/`. The whole thing runs in well under a minute on CPU and is meant to catch the broad-stroke regressions: a new component that doesn't satisfy its Protocol, a config that doesn't compose, a pipeline change that breaks the end-to-end integration. Fine-grained per-component testing (overfit smoke, save/load round trips, sampler shape checks, and so on) used to live in separate files; those have been consolidated into the three that remain, or removed when they stopped earning their keep.

When you add a new encoder, loss, binner, trainer, sampler, or evaluator, you extend the relevant existing file — usually `test_interfaces.py`.

## The three files

```
tests/
├── test_interfaces.py            # Protocol compliance for every component
├── test_dataset_compatibility.py # Config / signal compatibility checks
└── test_end_to_end.py            # Two integration runs + checkpoint resume
```

That's it. No hidden magic, no separate smoke/unit/integration split. Line count is about 570 for the full suite.

## `tests/test_interfaces.py` — Protocol conformance

The workhorse file. Every component added to the repo must have a test case here. The existing structure groups tests by Protocol, one `TestXxx` class per component. Pattern (from the real file):

```python
class TestSemiBinEncoderTrainingContract:
    def setup_method(self):
        self.model = SemiBinEncoder(input_dim=32, embedding_dim=8)

    def test_parameter_groups(self):
        assert "all" in self.model.parameter_groups()

    def test_encode_shape(self):
        features = np.random.randn(20, 32).astype("float32")
        z = self.model.encode(features)
        assert z.shape == (20, 8)

    def test_embedding_dim(self):
        assert self.model.embedding_dim == 8

    def test_training_step_returns_scalar_with_grad(self):
        x_i = torch.randn(16, 32)
        x_j = torch.randn(16, 32)
        label = torch.rand(16)
        loss = self.model.training_step((x_i, x_j, label), HingeContrastiveLoss())
        assert loss.shape == ()
        loss.backward()
```

Four assertions. Copy this shape for every new component:

- **Encoder** — four methods (`encode`, `embedding_dim`, `training_step`, `parameter_groups`).
- **ContrastiveLoss** — `__call__` returns scalar tensor, `loss.backward()` populates `.grad` on inputs.
- **Binner** — `cluster(X)` returns a 1-D integer array of length N, all ≥ 0.
- **Evaluator** — `score(bins_dir)` returns a DataFrame with `completeness` and `contamination` columns; missing columns raise `KeyError`.

The evaluator case has an instructive pattern — it mocks `subprocess.run` so the test doesn't need a real CheckM2 installation:

```python
def fake_run(cmd, **kwargs):
    out_dir = Path(cmd[cmd.index("-o") + 1])
    (out_dir / "quality_report.tsv").write_text(tsv_content)

with patch("subprocess.run", side_effect=fake_run):
    df = evaluator.score(Path("/fake/bins"))
```

Copy this for any future evaluator that shells out.

For `UncertainGenEncoder`, notice the `parameter_groups` assertion is stricter — it checks for `{"mean", "cov", "all"}` rather than just `"all"`, because the phased trainer depends on those specific group names:

```python
def test_parameter_groups_has_mean_and_cov(self):
    groups = self.model.parameter_groups()
    assert set(groups) == {"mean", "cov", "all"}
    assert len(groups["mean"]) > 0
    assert len(groups["cov"]) > 0
```

And the training step has two tests — one per phase — exercising `include_std=False` and `include_std=True` separately:

```python
def test_training_step_phase1(self):
    self.model.include_std = False
    # ... loss.backward(), assert scalar
```

If you add a phased encoder, copy both.

## `tests/test_dataset_compatibility.py` — config composition

The fail-fast signal check. Every dataset config declares `signals: [kmers, abundance, ...]`, every feature config declares `required_signals: [...]`, and the pipeline verifies they line up before training. This test verifies:

- Every shipped dataset config parses and has the expected keys.
- Every feature config declares `required_signals`.
- All shipped experiment configs pass the compatibility check.
- A deliberately mismatched `(dataset, features)` pair raises with a clear error.
- Absent `required_signals` or `signals` is a no-op (backwards compatibility).

If you add a new dataset, feature extractor, or experiment config, extend the parametrized lists here:

```python
@pytest.mark.parametrize(
    "name,expected_signals",
    [
        ("canonical_kmer", ["kmers"]),
        ("canonical_kmer_abundance", ["kmers", "abundance"]),
        # ... add yours
    ],
)
def test_feature_has_required_signals(self, name, expected_signals):
    ...
```

This is the test that catches "I added a new dataset config but forgot the `signals` field" and "I added a feature extractor that depends on taxonomy but forgot to declare it" — both before the pipeline spends 20 minutes setting up a run that was never going to work.

## `tests/test_end_to_end.py` — integration on synthetic data

The slowest file, but still CPU-only and under a minute per test. Three classes:

**`TestUncertainGenEndToEnd`** — generates 3 synthetic "genomes" as Dirichlet-drawn profiles, builds 90 contigs with low noise, trains UncertainGen with `TwoPhaseTrainer` for 3+2 epochs, encodes, clusters with Infomap, and asserts:

```python
ari = adjusted_rand_score(labels, predicted)
assert ari > 0.3, f"ARI {ari:.3f} is not above the random baseline"

elapsed = time.perf_counter() - start
assert elapsed < 60, f"took {elapsed:.1f}s, needs to stay under 60s"
```

ARI > 0.3 is a very loose bar — well-separated synthetic clusters should give you 0.8+. The bar is set low so the test doesn't flake on unlucky seeds; it's a regression guard, not a quality gate.

**`TestSemiBinEndToEnd`** — same shape, but with `SemiBinEncoder` + `SinglePhaseTrainer` + `HingeContrastiveLoss`. Acts as a regression guard for the "simple" path.

**`TestCheckpointResume`** — trains an encoder, saves, loads into a fresh instance, and asserts the two produce bit-for-bit equal embeddings on the same input:

```python
assert np.allclose(z_a, z_b, atol=1e-6)
```

This is the only save/load test in the suite since the dedicated `test_checkpoints.py` was retired — the `np.allclose` check here exercises the full `save_checkpoint` / `load_checkpoint` round trip on a realistic encoder.

If you add a new encoder that you'd like integration coverage for, add a third class to this file following the `TestSemiBinEndToEnd` shape. Keep the wall time under 60s (the file enforces this explicitly).

## Running the suite

Standard cadence:

```bash
# Fast (~5s) — before every commit
pytest tests/test_interfaces.py

# Full (~90s) — before opening a PR
pytest tests/

# On HPC
sbatch hpc/slurm/smoke_test.sh
```

`hpc/slurm/smoke_test.sh` runs `pytest tests/` plus a Hydra compose + instantiate check for the primary experiment config. It's under a 30-minute SLURM wall-time ceiling and is the canonical "is the environment working" probe on either cluster.

## What to add when you add a new component

Minimum bar for merging:

| Component type    | Where to add a test                                                                          |
|-------------------|----------------------------------------------------------------------------------------------|
| Encoder           | `test_interfaces.py` (Protocol class). Optionally a new `TestXxxEndToEnd` in `test_end_to_end.py` if it's a primary baseline. |
| Loss              | `test_interfaces.py` (Protocol class).                                                       |
| Binner            | `test_interfaces.py` (Protocol class).                                                       |
| Pair sampler      | `test_interfaces.py` (construct + sample one batch, check shapes/dtypes).                    |
| Trainer           | `test_interfaces.py` (construct + single `fit()` call), or a fixture in `test_end_to_end.py` if it's a new training regime. |
| Feature extractor | `test_dataset_compatibility.py` (add to the parametrized lists).                             |
| Evaluator         | `test_interfaces.py` (mock `subprocess.run` if it shells out).                               |
| Dataset           | `test_dataset_compatibility.py` (add to `parametrize("name", [...])`).                       |
| Logger            | `test_interfaces.py` (construct, call all six Protocol methods, check no exceptions).        |

If this feels light, note that each case is 5–20 lines. The *thinking* part — "what would be a useful failure mode to catch?" — is the only slow bit.

## Test philosophy in two sentences

Protocol conformance is cheap and catches everything that's structurally wrong. End-to-end on synthetic data catches everything that's wrong with how slots compose. Anything finer-grained is either caught by those two layers or isn't worth the maintenance cost of a dedicated test file.

## When tests should be behavioural, not structural

A common anti-pattern: over-assert structure.

```python
# Bad
assert model.hidden_dim == 512
assert len(model.layers) == 3
```

Structure-level assertions break when someone refactors the internals. Prefer behavioural assertions:

```python
# Good
assert model.encode(np.random.randn(10, 32).astype("float32")).shape == (10, 8)
loss.backward()  # succeeds or raises
```

The exception is `parameter_groups` — that's contract-level, not structure-level. Other code depends on the group names being `"mean"` and `"cov"` for phased training, so asserting on the keys is correct.

## Sources

- [tests/test_interfaces.py](https://github.com/Tinnifo/Metagenomic-Binning/blob/main/tests/test_interfaces.py) — canonical "how to add a Protocol-compliance case"
- [tests/test_dataset_compatibility.py](https://github.com/Tinnifo/Metagenomic-Binning/blob/main/tests/test_dataset_compatibility.py) — signal compatibility
- [tests/test_end_to_end.py](https://github.com/Tinnifo/Metagenomic-Binning/blob/main/tests/test_end_to_end.py) — integration on synthetic data
