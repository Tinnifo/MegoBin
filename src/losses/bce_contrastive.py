import torch
import torch.nn as nn
import torch.nn.functional as F


class BCEContrastiveLoss(nn.Module):
    """BCE contrastive loss (RevisitingKmers).

    Predicted similarity:  p = exp(-d²)  where d = L2 distance.
    Loss:  binary_cross_entropy(p, label)

    label = 1 for same-genome pairs, 0 for different-genome pairs.
    """

    def forward(
        self, z_i: torch.Tensor, z_j: torch.Tensor, label: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            z_i: (batch, d) embeddings from branch 1.
            z_j: (batch, d) embeddings from branch 2.
            label: (batch,) binary same/different labels.

        Returns:
            Scalar mean BCE loss.
        """
        d_sq = ((z_i - z_j) ** 2).sum(dim=-1)  # squared L2 distance
        p = torch.exp(-d_sq)                     # predicted similarity ∈ (0, 1]
        return F.binary_cross_entropy(p, label)
