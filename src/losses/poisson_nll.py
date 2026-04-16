import torch
import torch.nn as nn


class PoissonNLLLoss(nn.Module):
    """Poisson negative log-likelihood on k-mer co-occurrence counts.

    Given k-mer embedding pairs (z_i, z_j) and their co-occurrence count c:
        λ = exp(z_i · z_j)
        NLL = λ − c · log(λ) = exp(dot) − c · dot

    Fits the ContrastiveLoss protocol where *label* carries co-occurrence
    counts instead of binary same/different labels.
    """

    def forward(
        self, z_i: torch.Tensor, z_j: torch.Tensor, label: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            z_i: (batch, d) embeddings for first k-mer in each pair.
            z_j: (batch, d) embeddings for second k-mer in each pair.
            label: (batch,) co-occurrence counts.

        Returns:
            Scalar mean Poisson NLL.
        """
        log_lambda = (z_i * z_j).sum(dim=-1)  # dot product per pair
        loss = torch.exp(log_lambda) - label * log_lambda
        return loss.mean()
