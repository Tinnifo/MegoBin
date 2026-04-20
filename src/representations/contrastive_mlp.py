import numpy as np
import torch
import torch.nn as nn


class ContrastiveMLP(nn.Module):
    """RevisitingKmers 2-layer Siamese MLP.

    Architecture:
        Linear(input_dim → hidden_dim) → BatchNorm1d → Sigmoid
        → Dropout(dropout) → Linear(hidden_dim → embedding_dim)

    Both branches of the Siamese network share these weights.
    """

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 512,
        embedding_dim: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        self._embedding_dim = embedding_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Sigmoid(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Single-branch forward pass. (batch, input_dim) → (batch, embedding_dim)."""
        return self.net(x)

    def encode(self, features: np.ndarray) -> np.ndarray:
        """(N, input_dim) → (N, embedding_dim).  Inference in eval mode."""
        self.eval()
        with torch.no_grad():
            x = torch.from_numpy(features).float()
            z = self.forward(x)
        return z.cpu().numpy()

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim
