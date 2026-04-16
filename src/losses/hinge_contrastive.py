import torch
import torch.nn as nn


class HingeContrastiveLoss(nn.Module):
    """Hinge contrastive loss (SemiBin).

    y · d² + (1 − y) · max(0, margin − d)²

    where d = L2 distance between embeddings, y = 1 for must-link
    (same genome) and y = 0 for cannot-link (different genome).
    """

    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(
        self, z_i: torch.Tensor, z_j: torch.Tensor, label: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            z_i: (batch, d) embeddings from branch 1.
            z_j: (batch, d) embeddings from branch 2.
            label: (batch,) 1 = must-link, 0 = cannot-link.

        Returns:
            Scalar mean hinge contrastive loss.
        """
        d = torch.sqrt(((z_i - z_j) ** 2).sum(dim=-1) + 1e-8)
        pos = label * d ** 2
        neg = (1 - label) * torch.clamp(self.margin - d, min=0) ** 2
        return (pos + neg).mean()
