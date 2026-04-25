# Tutorial: Add a new loss

Shorter than the encoder tutorial because the surface is smaller. A loss is one `__call__` method. Two files: one Python, one YAML. Plus one test case.

## Goal

Add an InfoNCE-style loss and run it against the existing SemiBin encoder. InfoNCE is the standard contrastive loss from the SimCLR / Wav2Vec family; it's a useful alternative to the hinge loss SemiBin originally uses.

## Step 1 — Know the Protocol

From `megobin/losses/base.py`:

```python
class ContrastiveLoss(Protocol):
    def __call__(
        self, z_i: torch.Tensor, z_j: torch.Tensor, label: torch.Tensor
    ) -> torch.Tensor:
        """Pair of embeddings + same/different label → scalar loss"""
        ...
```

Two embedding tensors `(B, d)`, a label tensor `(B,)` with 1 for must-link and 0 for cannot-link, scalar out.

## Step 2 — Write the loss

Create `megobin/losses/info_nce.py`:

```python
"""InfoNCE contrastive loss for MegoBin.

Adapts the standard InfoNCE form to the (z_i, z_j, label) triple layout
that the ContrastiveLoss Protocol requires. For each must-link pair
(label=1), the "positive" is z_j and the "negatives" are every other
example in the batch. Cannot-link pairs (label=0) are pulled apart via
an auxiliary margin term on the raw pair distance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    def __init__(self, temperature: float = 0.07, margin: float = 1.0) -> None:
        super().__init__()
        self.temperature = temperature
        self.margin = margin

    def forward(
        self,
        z_i: torch.Tensor,
        z_j: torch.Tensor,
        label: torch.Tensor,
    ) -> torch.Tensor:
        # Normalise to unit sphere — cosine-space InfoNCE is the standard
        # formulation, and it stabilises temperature scaling.
        z_i = F.normalize(z_i, dim=-1)
        z_j = F.normalize(z_j, dim=-1)

        # Must-link mask and cannot-link mask
        pos_mask = label.bool()
        neg_mask = ~pos_mask

        # --- Must-link InfoNCE term ---
        # For each must-link pair, cross-entropy over similarities against
        # the rest of z_j as in-batch negatives.
        info_nce = torch.tensor(0.0, device=z_i.device)
        if pos_mask.any():
            sim = z_i[pos_mask] @ z_j[pos_mask].T / self.temperature
            targets = torch.arange(sim.size(0), device=sim.device)
            info_nce = F.cross_entropy(sim, targets)

        # --- Cannot-link margin term ---
        # Push negative pairs at least ``margin`` apart in cosine distance.
        neg_margin = torch.tensor(0.0, device=z_i.device)
        if neg_mask.any():
            neg_dist = 1.0 - (z_i[neg_mask] * z_j[neg_mask]).sum(dim=-1)
            neg_margin = F.relu(self.margin - neg_dist).pow(2).mean()

        return info_nce + neg_margin
```

A few choices worth defending. **First**, the signature matches the Protocol exactly — this is important; the encoder's `training_step` will call `loss_fn(z_i, z_j, label)` and pass whatever label tensor the pair sampler produced. **Second**, the loss inherits from `nn.Module` so it can be instantiated via Hydra just like an encoder, and so any learnable parameters (e.g. a future learned temperature) are automatically registered. **Third**, the InfoNCE term only runs when there are must-link pairs in the batch, and the margin term only when there are cannot-link pairs — otherwise the loss is 0, not NaN.

## Step 3 — Add the config

Create `configs/loss/info_nce.yaml`:

```yaml
_target_: megobin.losses.info_nce.InfoNCELoss
temperature: 0.07
margin: 1.0
```

## Step 4 — Run it

```bash
python megobin/pipeline.py \
  --config-name experiment/training/semibin_cami_toy \
  loss=info_nce
```

Check the log:

```
[megobin.pipeline] Loss: InfoNCELoss
```

Compare the resulting loss curves in TensorBoard against a hinge-loss run. InfoNCE curves will look very different — the scale is different, and the early epochs tend to be noisier.

## Step 5 — Add a Protocol-compliance test

In `tests/test_interfaces.py`:

```python
def test_info_nce_satisfies_contrastive_loss():
    import torch
    from megobin.losses.base import ContrastiveLoss
    from megobin.losses.info_nce import InfoNCELoss

    loss = InfoNCELoss()
    assert isinstance(loss, ContrastiveLoss)

    # Smoke test — a mixed batch produces a finite scalar.
    z_i = torch.randn(8, 100, requires_grad=True)
    z_j = torch.randn(8, 100)
    label = torch.tensor([1, 1, 1, 1, 0, 0, 0, 0])
    out = loss(z_i, z_j, label)
    assert out.ndim == 0
    assert torch.isfinite(out)
    out.backward()  # gradient path exists
```

Run:

```bash
pytest tests/test_interfaces.py::test_info_nce_satisfies_contrastive_loss -v
```

## When a loss is not swappable

Not every encoder-loss pairing works. UncertainGen's `training_step` passes `(μ, cov)` concatenations to the loss in its second training phase; `MahalanobisBCELoss.include_std=True` knows how to split that concatenation. Giving UncertainGen an InfoNCE loss would work mathematically on the mean head but would silently drop the covariance signal — a mis-matched loss is not a runtime error, it's a research bug.

When you introduce a loss that depends on encoder-specific structure (like UncertainGen's covariance), document the compatibility in the loss file's docstring and add an assertion or warning in `__init__` if practical. The Protocol cannot catch this for you.

