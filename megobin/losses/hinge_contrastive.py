import torch
import torch.nn as nn


class HingeContrastiveLoss(nn.Module):
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(
        self, z_i: torch.Tensor, z_j: torch.Tensor, label: torch.Tensor
    ) -> torch.Tensor:
        d = torch.sqrt(((z_i - z_j) ** 2).sum(dim=-1) + 1e-8)
        pos = label * d**2
        neg = (1 - label) * torch.clamp(self.margin - d, min=0) ** 2
        return (pos + neg).mean()
