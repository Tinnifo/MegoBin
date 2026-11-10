import torch
import torch.nn as nn


class HingeContrastiveLoss(nn.Module):
    """SemiBin's self-supervised contrastive loss.

    ``mean(label * d**2 + (1 - label) * relu(margin - d)**2)`` with
    ``d = ||z_i - z_j||_2``. Must-link pairs (label 1) are pulled together,
    cannot-link pairs (label 0) are pushed at least ``margin`` apart.

    SemiBin evaluates this loss in float64 (it casts the embeddings and labels
    with ``.double()``), so it is evaluated in float64 here as well. Devices
    without float64 support (MPS) fall back to the input precision.
    """

    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(
        self, z_i: torch.Tensor, z_j: torch.Tensor, label: torch.Tensor
    ) -> torch.Tensor:
        if z_i.device.type != "mps":
            z_i, z_j, label = z_i.double(), z_j.double(), label.double()
        d = torch.norm(z_i - z_j, p=2, dim=-1)
        pos = label * torch.square(d)
        neg = (1 - label) * torch.square(torch.relu(self.margin - d))
        return (pos + neg).mean()
