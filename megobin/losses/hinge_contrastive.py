import torch
import torch.nn as nn

from megobin.z import Z, as_z


class HingeContrastiveLoss(nn.Module):
    #: Needs only ``z.mu``; any extra field an encoder sends (e.g. ``var``) is ignored.
    needs = {
        "batch": frozenset({"views=2", "label"}),
        "z": frozenset({"mu"}),
    }

    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(
        self, z_i: "Z | torch.Tensor", z_j: "Z | torch.Tensor", label: torch.Tensor
    ) -> torch.Tensor:
        a, b = as_z(z_i).array("mu"), as_z(z_j).array("mu")
        d = torch.sqrt(((a - b) ** 2).sum(dim=-1) + 1e-8)
        pos = label * d**2
        neg = (1 - label) * torch.clamp(self.margin - d, min=0) ** 2
        return (pos + neg).mean()
