import numpy as np
import torch
import torch.nn as nn


class SemiBinEncoder(nn.Module):
    """SemiBin 3-layer Siamese encoder.

    Architecture:
        Linear(input_dim → 512) → BN → LeakyReLU(0.01) → Dropout(0.2)
        Linear(512 → 512)       → BN → LeakyReLU(0.01) → Dropout(0.2)
        Linear(512 → embedding_dim)

    Input is canonical 4-mers (136 dims) concatenated with abundance
    features, so input_dim = 136 + 2 * num_bams.
    Embedding dim is fixed at 100.
    """

    def __init__(
        self,
        input_dim: int = 136,
        embedding_dim: int = 100,
        dropout: float = 0.2,
    ):
        super().__init__()
        self._embedding_dim = embedding_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout),
            nn.Linear(512, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, input_dim) → (batch, embedding_dim)."""
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
