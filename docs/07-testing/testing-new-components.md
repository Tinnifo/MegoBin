# Testing new components

Three test files. <1 min on CPU.

```
tests/
├── test_interfaces.py            Protocol compliance
├── test_dataset_compatibility.py Config / signal checks
└── test_end_to_end.py            Checkpoint resume guard
```

## `test_interfaces.py` — Protocol conformance

The workhorse. Pattern (from real file):

```python
class TestUncertainGenTrainingContract:
    def setup_method(self):
        self.model = UncertainGenEncoder(
            input_dim=32, hidden_dim=16, embedding_dim=8
        )

    def test_parameter_groups_has_mean_and_cov(self):
        groups = self.model.parameter_groups()
        assert set(groups) == {"mean", "cov", "all"}

    def test_encode_shape(self):
        z = self.model.encode(np.random.randn(20, 32).astype("float32"))
        assert z.shape == (20, 8)

    def test_training_step_phase1(self):
        self.model.include_std = False
        x_i, x_j = torch.randn(8, 32), torch.randn(8, 32)
        label = torch.rand(8)
        loss = self.model.training_step(
            (x_i, x_j, label), MahalanobisBCELoss(include_std=False)
        )
        assert loss.shape == ()
        loss.backward()
```

Per-Protocol checks:

| Protocol | Check |
|----------|-------|
| Encoder | four methods (`encode`, `embedding_dim`, `training_step`, `parameter_groups`) |
| ContrastiveLoss | `__call__` returns scalar; `loss.backward()` populates `.grad` |
| Binner | `cluster(X)` returns 1-D int array, length N, all ≥ 0 |
| Evaluator | `score(bins_dir)` returns DataFrame with `completeness`, `contamination` |

**Mock CheckM2** in evaluator tests:

```python
def fake_run(cmd, **kwargs):
    out_dir = Path(cmd[cmd.index("-o") + 1])
    (out_dir / "quality_report.tsv").write_text(tsv_content)

with patch("subprocess.run", side_effect=fake_run):
    df = evaluator.score(Path("/fake/bins"))
```

**Phased encoders** assert specific group names:

```python
def test_parameter_groups_has_mean_and_cov(self):
    groups = self.model.parameter_groups()
    assert set(groups) == {"mean", "cov", "all"}
```

Plus a `training_step` test per phase (`include_std=False`, then `True`).

## `test_dataset_compatibility.py` — fail-fast signal checks

Verifies:
- Every dataset config parses with expected keys.
- Every feature config declares `required_signals`.
- All experiment configs pass compatibility.
- Mismatched (dataset, features) raises with a clear error.
- Missing `required_signals` / `signals` is a no-op.

Add new datasets/features to the parametrized lists:

```python
@pytest.mark.parametrize(
    "name,expected_signals",
    [
        ("canonical_kmer", ["kmers"]),
        ("canonical_kmer_abundance", ["kmers", "abundance"]),
    ],
)
def test_feature_has_required_signals(self, name, expected_signals):
    ...
```

## `test_end_to_end.py` — checkpoint resume guard

| Class | What |
|-------|------|
| `TestCheckpointResume` | Train → save → load fresh → assert `np.allclose(z_a, z_b, atol=1e-6)` |

For a new encoder you want integration coverage on, add a `TestXxxEndToEnd` here that trains briefly, encodes synthetic features, and asserts cluster recovery via your binner of choice. Keep wall time <60s.

## Running

```bash
pytest tests/test_interfaces.py    # ~5s — before commit
pytest tests/                      # ~90s — before PR
```

## Where to add a test

| Component | File |
|-----------|------|
| Encoder | `test_interfaces.py` (+ optional `TestXxxEndToEnd`) |
| Loss | `test_interfaces.py` |
| Binner | `test_interfaces.py` |
| Pair sampler | `test_interfaces.py` (construct + sample one batch) |
| Trainer | `test_interfaces.py` (construct + one `fit()`) |
| Feature extractor | `test_dataset_compatibility.py` |
| Evaluator | `test_interfaces.py` (mock subprocess) |
| Dataset | `test_dataset_compatibility.py` |
| Logger | `test_interfaces.py` (call all six methods) |

## Behavioural over structural

Bad:
```python
assert model.hidden_dim == 512
assert len(model.layers) == 3
```

Good:
```python
assert model.encode(np.random.randn(10, 32).astype("float32")).shape == (10, 8)
loss.backward()
```

Exception: `parameter_groups` keys are contract-level — assert on them.

## Sources

- [tests/test_interfaces.py](https://github.com/Tinnifo/Metagenomic-Binning/blob/main/tests/test_interfaces.py)
- [tests/test_dataset_compatibility.py](https://github.com/Tinnifo/Metagenomic-Binning/blob/main/tests/test_dataset_compatibility.py)
- [tests/test_end_to_end.py](https://github.com/Tinnifo/Metagenomic-Binning/blob/main/tests/test_end_to_end.py)
