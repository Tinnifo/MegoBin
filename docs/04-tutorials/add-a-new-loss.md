# Tutorial: Add a new loss

Two files + one test. Worked example: InfoNCE.

## Step 1 — The Protocol

```python
class ContrastiveLoss(Protocol):
    def __call__(self, z_i, z_j, label) -> torch.Tensor: ...
```

`(B, d), (B, d), (B,)` → scalar. `label`: 1=must-link, 0=cannot-link.

## Step 2 — Write `megobin/losses/info_nce.py`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    def __init__(self, temperature: float = 0.07, margin: float = 1.0):
        super().__init__()
        self.temperature = temperature
        self.margin = margin

    def forward(self, z_i, z_j, label):
        z_i = F.normalize(z_i, dim=-1)
        z_j = F.normalize(z_j, dim=-1)

        pos_mask = label.bool()
        neg_mask = ~pos_mask

        info_nce = torch.tensor(0.0, device=z_i.device)
        if pos_mask.any():
            sim = z_i[pos_mask] @ z_j[pos_mask].T / self.temperature
            targets = torch.arange(sim.size(0), device=sim.device)
            info_nce = F.cross_entropy(sim, targets)

        neg_margin = torch.tensor(0.0, device=z_i.device)
        if neg_mask.any():
            neg_dist = 1.0 - (z_i[neg_mask] * z_j[neg_mask]).sum(dim=-1)
            neg_margin = F.relu(self.margin - neg_dist).pow(2).mean()

        return info_nce + neg_margin
```

Inheriting from `nn.Module` lets Hydra instantiate it and registers any future learnable params.

## Step 3 — Add `configs/loss/info_nce.yaml`

```yaml
_target_: megobin.losses.info_nce.InfoNCELoss
temperature: 0.07
margin: 1.0
```

## Step 4 — Run

```bash
python megobin/pipeline.py \
  --config-name experiment/training/semibin_cami_toy \
  loss=info_nce
```

## Step 5 — Add a test

```python
def test_info_nce_satisfies_contrastive_loss():
    import torch
    from megobin.losses.base import ContrastiveLoss
    from megobin.losses.info_nce import InfoNCELoss

    loss = InfoNCELoss()
    assert isinstance(loss, ContrastiveLoss)

    z_i = torch.randn(8, 100, requires_grad=True)
    z_j = torch.randn(8, 100)
    label = torch.tensor([1, 1, 1, 1, 0, 0, 0, 0])
    out = loss(z_i, z_j, label)
    assert out.ndim == 0
    assert torch.isfinite(out)
    out.backward()
```

## When a loss is not swappable

A loss that depends on encoder-specific structure (e.g. UncertainGen's covariance head) is not freely interchangeable. Document the compatibility in the loss docstring; the Protocol can't catch it.
