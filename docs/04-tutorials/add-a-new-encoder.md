# Tutorial: Add a new encoder

Three files. No changes to `pipeline.py`. Worked example: a DNABERT-S **stub**.

## Step 1 — The Protocol

```python
class Encoder(Protocol):
    def encode(self, features: np.ndarray) -> np.ndarray: ...
    @property
    def embedding_dim(self) -> int: ...
    def training_step(self, batch, loss_fn) -> torch.Tensor: ...
    def parameter_groups(self) -> dict[str, list[nn.Parameter]]: ...
```

Minimum `parameter_groups`: `{"all": list(self.parameters())}`.

## Step 2 — Write `megobin/encoders/dnabert_s.py`

```python
import numpy as np
import torch
import torch.nn as nn


class DNABertS(nn.Module):
    def __init__(self, input_dim: int, embedding_dim: int = 768, dropout: float = 0.1):
        super().__init__()
        self._embedding_dim = embedding_dim
        self.projection = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, embedding_dim),
        )

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def _encode_raw(self, features: torch.Tensor) -> torch.Tensor:
        return self.projection(features)

    def encode(self, features: np.ndarray) -> np.ndarray:
        self.eval()
        device = next(self.parameters()).device
        x = torch.from_numpy(features).float().to(device)
        with torch.no_grad():
            z = self._encode_raw(x)
        return z.cpu().numpy()

    def training_step(self, batch, loss_fn):
        feat_i, feat_j, label = batch
        device = next(self.parameters()).device
        z_i = self._encode_raw(feat_i.to(device))
        z_j = self._encode_raw(feat_j.to(device))
        return loss_fn(z_i, z_j, label.to(device))

    def parameter_groups(self):
        return {"all": list(self.parameters())}
```

`encode` always under `torch.no_grad()` and returns NumPy. `training_step` calls `loss_fn(z_i, z_j, label)` — Protocol-shaped.

## Step 3 — Add `configs/encoder/dnabert_s.yaml`

```yaml
_target_: megobin.encoders.dnabert_s.DNABertS
input_dim: 136
embedding_dim: 768
dropout: 0.1
```

[`_target_`](https://hydra.cc/docs/advanced/instantiate_objects/overview/) is the import path Hydra calls with the remaining keys as kwargs.

## Step 4 — Run

```bash
python megobin/pipeline.py \
  --config-name experiment/uncertain_gen_dbscan \
  encoder=dnabert_s \
  loss=hinge \
  trainer=single_phase \
  pair_sampler=uncertain_gen
```

(Single-head encoder → single-phase trainer + hinge loss.) Each `key=value` is a [Hydra config-group override](https://hydra.cc/docs/advanced/override_grammar/basic/).

## Step 5 — Add a test

In `tests/test_interfaces.py`:

```python
def test_dnabert_s_satisfies_encoder():
    from megobin.encoders.base import Encoder
    from megobin.encoders.dnabert_s import DNABertS

    enc = DNABertS(input_dim=136, embedding_dim=768)
    assert isinstance(enc, Encoder)
```

Pattern-match `TestUncertainGenTrainingContract` for the four standard assertions.

```bash
pytest tests/test_interfaces.py -v -k DNABertS
```

## Step 6 — Pin a reproducible config (optional)

Copy `configs/experiment/uncertain_gen_dbscan.yaml` → `dnabert_s_dbscan.yaml`, swap `/encoder` and dims.

## Checklist

1. `megobin/encoders/{name}.py`
2. `configs/encoder/{name}.yaml`
3. `tests/test_interfaces.py` case
4. *(optional)* `tests/test_end_to_end.py` case
5. *(optional)* `configs/experiment/training/{name}_{dataset}.yaml`
