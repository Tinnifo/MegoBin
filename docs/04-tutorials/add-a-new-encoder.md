# Tutorial: Add a new encoder

This is the tutorial that justifies the whole architecture. You will add a new encoder — we'll use DNABERT-S as the worked example — and the pipeline will accept it with zero changes to pipeline code, other components, or tests. Three files is all it takes: one Python file, one YAML, and one test addition.

The worked example here is intentionally **a stub**, not a real DNABERT-S implementation. DNABERT-S has its own tokenizer, pretrained weights, and input format that are out of scope for this tutorial. The scaffold below demonstrates the integration mechanics; filling in the real model is a separate task.

## Goal

Create `megobin/encoders/dnabert_s.py` satisfying the `Encoder` Protocol, add `configs/encoder/dnabert_s.yaml`, run the pipeline with it via a CLI override, and extend `tests/test_interfaces.py` to cover the new encoder.

## Step 1 — Know the Protocol

From `megobin/encoders/base.py`:

```python
class Encoder(Protocol):
    def encode(self, features: np.ndarray) -> np.ndarray: ...

    @property
    def embedding_dim(self) -> int: ...

    def training_step(
        self, batch: tuple[torch.Tensor, ...], loss_fn: nn.Module
    ) -> torch.Tensor: ...

    def parameter_groups(self) -> dict[str, list[nn.Parameter]]: ...
```

Four surfaces: `encode` (NumPy → NumPy, inference), `embedding_dim` property, `training_step` (one batch → scalar loss), `parameter_groups` (named parameter sets for phase-based trainers). Minimum viable: `parameter_groups` returns `{"all": [...]}`.

## Step 2 — Write the encoder

Create `megobin/encoders/dnabert_s.py`:

```python
"""Stub DNABERT-S encoder for MegoBin.

This is scaffolding — the real implementation will load DNABERT-S weights
from HuggingFace and replace the dummy projection below with the pretrained
transformer. Until then, this file exists to demonstrate the Encoder
Protocol integration path.
"""

import numpy as np
import torch
import torch.nn as nn


class DNABertS(nn.Module):
    """Stub encoder. Replace ``_encode_raw`` with a real DNABERT-S forward."""

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int = 768,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self._embedding_dim = embedding_dim

        # Placeholder projection — swap for a pretrained DNABERT-S here.
        self.projection = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, embedding_dim),
        )

    # ------------------------------------------------------------------
    # Protocol surface
    # ------------------------------------------------------------------

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def _encode_raw(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass, gradient-enabled. Replace with real DNABERT-S."""
        return self.projection(features)

    def encode(self, features: np.ndarray) -> np.ndarray:
        """Inference path. NumPy in, NumPy out, no gradients."""
        self.eval()
        device = next(self.parameters()).device
        x = torch.from_numpy(features).float().to(device)
        with torch.no_grad():
            z = self._encode_raw(x)
        return z.cpu().numpy()

    def training_step(
        self,
        batch: tuple[torch.Tensor, ...],
        loss_fn: nn.Module,
    ) -> torch.Tensor:
        """One batch → scalar loss.

        Batch layout matches PairSampler's (feature_i, feature_j, label).
        """
        feat_i, feat_j, label = batch
        device = next(self.parameters()).device
        z_i = self._encode_raw(feat_i.to(device))
        z_j = self._encode_raw(feat_j.to(device))
        return loss_fn(z_i, z_j, label.to(device))

    def parameter_groups(self) -> dict[str, list[nn.Parameter]]:
        """Single-phase encoder. One group covers all parameters."""
        return {"all": list(self.parameters())}
```

Three implementation choices worth calling out. **First**, the encoder inherits from `nn.Module` but from no base class of ours. Protocol conformance is structural, not nominal — if it has the right methods, it qualifies. **Second**, `encode` always runs under `torch.no_grad()` and returns NumPy. **Third**, `training_step` dispatches to `loss_fn(z_i, z_j, label)` which exactly matches the `ContrastiveLoss` Protocol; that's why you can pair this encoder with any existing loss.

## Step 3 — Add the config

Create `configs/encoder/dnabert_s.yaml`:

```yaml
_target_: megobin.encoders.dnabert_s.DNABertS
input_dim: 236
embedding_dim: 768
dropout: 0.1
```

`input_dim: 236` matches the default feature dimensionality (136 canonical k-mers + 2 × 50 BAM coverage features for CAMI toy). `embedding_dim: 768` matches real DNABERT-S; adjust if you've changed the head.

## Step 4 — Run it

Zero changes to `megobin/pipeline.py` required:

```bash
python megobin/pipeline.py \
  --config-name experiment/hybrid_uncertain_gen \
  encoder=dnabert_s \
  'encoder.input_dim=236' \
  loss=hinge \
  trainer=single_phase \
  pair_sampler=semibin
```

Hinge loss, single-phase trainer, and SemiBin pair sampler are chosen because the stub encoder is single-head. If you later upgrade DNABERT-S to have a covariance head, switch the loss, trainer, and sampler accordingly.

Check the log for:

```
[megobin.pipeline] Encoder:        DNABertS
[megobin.pipeline] Loss:           HingeContrastiveLoss
[megobin.pipeline] Trainer:        SinglePhaseTrainer
```

If any of those are wrong, your CLI override didn't land — recheck with `--cfg job`.

## Step 5 — Add a Protocol-compliance test

The pipeline will run even if your encoder has a subtle bug in one method — until that method is called. `test_interfaces.py` catches this early via `isinstance(instance, Encoder)`, which works because `Encoder` is `@runtime_checkable`. Add a case to `tests/test_interfaces.py`:

```python
def test_dnabert_s_satisfies_encoder():
    from megobin.encoders.base import Encoder
    from megobin.encoders.dnabert_s import DNABertS

    enc = DNABertS(input_dim=236, embedding_dim=768)
    assert isinstance(enc, Encoder)
```

Expand the case with the four standard contract assertions (`encode` shape, `embedding_dim`, `parameter_groups`, `training_step` returns a scalar with grad). Pattern-match from `TestSemiBinEncoderTrainingContract` in the same file — see [Testing new components](../07-testing/testing-new-components.md) for the full template.

If you want integration coverage (recommended for any encoder you plan to run real experiments against), add a `TestDNABertSEndToEnd` class to `tests/test_end_to_end.py` in the shape of `TestSemiBinEndToEnd`: generate synthetic Dirichlet-drawn genomes, train for a handful of epochs, cluster with Infomap, assert `ARI > 0.3` and wall time under 60 s. That's the current replacement for the retired `test_overfit_batch.py` — one class guards both "the encoder learns" and "the encoder plays nicely with the full pipeline".

Run the tests:

```bash
pytest tests/test_interfaces.py -v -k DNABertS
pytest tests/test_end_to_end.py::TestDNABertSEndToEnd -v  # if you added it
```

If the interface test passes but the end-to-end test fails, your encoder satisfies the contract but doesn't actually learn (or doesn't compose with the binner). Drop to a smaller batch and `trainer.epochs=2` and look at the loss curve.

## Step 6 — Pin a reproducible training config (optional but recommended)

Once the encoder works, copy `configs/experiment/training/semibin_cami_toy.yaml` to `configs/experiment/training/dnabert_s_cami_toy.yaml`, swap `/encoder: semibin_encoder` for `/encoder: dnabert_s`, adjust `encoder.input_dim` and `encoder.embedding_dim` if needed, and cite your hyperparameter sources in inline comments. Now this encoder has a pinned reproducible run:

```bash
python megobin/pipeline.py --config-name experiment/training/dnabert_s_cami_toy
```

## Checklist for a new encoder

A complete addition is six things:

1. `megobin/encoders/{name}.py` satisfying the Protocol.
2. `configs/encoder/{name}.yaml` with `_target_` and kwargs.
3. `tests/test_interfaces.py` case: a `TestXxxTrainingContract` class with the four standard assertions.
4. Optionally, `tests/test_end_to_end.py` case: a `TestXxxEndToEnd` class (ARI > 0.3, <60s) for integration coverage.
5. Optionally, `configs/experiment/training/{name}_{dataset}.yaml`.
6. A paragraph in this guide — specifically [`what-is-megobin.md`](../01-introduction/what-is-megobin.md) under "What ships in the repo today", and the encoder should appear in `configs/encoder/` tables in the config reference.

Zero changes to `megobin/pipeline.py`, `megobin/binners/`, `megobin/evaluators/`, or any other existing component. That's the deal.
