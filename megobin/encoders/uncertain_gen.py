import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from megobin.losses.base import ContrastiveLoss


class UncertainGenEncoder(nn.Module):
    """UncertainGen variational encoder.

    Two heads with identical architecture:

        Linear(input_dim → hidden_dim)
        BatchNorm1d
        Sigmoid
        Dropout
        Linear(hidden_dim → embedding_dim)

    The mean head emits μ; the cov head emits log_cov, exponentiated to
    produce a strictly-positive diagonal covariance.

    Inference:
      - ``encode``                  → μ only,    (N, embedding_dim)
      - ``encode_with_uncertainty`` → (μ, cov),  both (N, embedding_dim)

    ``output_normalize=True`` L2-normalises μ on the inference path so
    the binner's ε sweep stays in a known scale; the cov head output is
    not normalised. Training-time forwards are NOT normalised — the
    Mahalanobis loss expects unbounded μ to learn meaningful scale.

    Training is two-phase via ``TwoPhaseTrainer``:
      - ``include_std=False`` → ``loss(μ_i, μ_j, label)``
      - ``include_std=True``  → ``loss(cat[μ_i, cov_i], cat[μ_j, cov_j], label)``
    """

    def __init__(
        self,
        input_dim: int = 136,
        hidden_dim: int = 512,
        embedding_dim: int = 256,
        dropout: float = 0.2,
        include_std: bool = False,
        output_normalize: bool = True,
    ):
        super().__init__()
        self._embedding_dim = embedding_dim
        self.include_std = include_std 
        self.output_normalize = output_normalize

        self.mean = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Sigmoid(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
        )
        self.cov = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Sigmoid(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(batch, input_dim) → (μ, cov) with cov = exp(log_cov)."""
        mu = self.mean(x)
        cov = torch.exp(self.cov(x))
        return mu, cov

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _maybe_normalize(self, mu: torch.Tensor) -> torch.Tensor:
        if self.output_normalize:
            return F.normalize(mu, p=2, dim=-1, eps=1e-12)
        return mu

    def encode(self, features: np.ndarray) -> np.ndarray:
        """(N, input_dim) → (N, embedding_dim)."""
        self.eval()
        with torch.no_grad():
            x = torch.from_numpy(features).float()
            mu = self.mean(x)
            mu = self._maybe_normalize(mu)
        return mu.cpu().numpy()

    def encode_with_uncertainty(
        self, features: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """(N, input_dim) → (μ, cov). μ obeys ``output_normalize``; cov is raw."""
        self.eval()
        with torch.no_grad():
            x = torch.from_numpy(features).float()
            mu, cov = self.forward(x)
            mu = self._maybe_normalize(mu)
        return mu.cpu().numpy(), cov.cpu().numpy()

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def training_step(
        self,
        batch: tuple[torch.Tensor, ...],
        loss_fn: ContrastiveLoss,
    ) -> torch.Tensor:
        """Forward both branches → ``loss_fn(z_i, z_j, label)``.

        Phase 1 (``include_std=False``): ``z`` is just μ, width = ``embedding_dim``.
        Phase 2 (``include_std=True``):  ``z`` is ``cat([μ, cov])``,  width = ``2 * embedding_dim``.
        """
        x_i, x_j, label = batch
        if self.include_std:
            mu_i, cov_i = self.forward(x_i)
            mu_j, cov_j = self.forward(x_j)
            z_i = torch.cat([mu_i, cov_i], dim=-1)
            z_j = torch.cat([mu_j, cov_j], dim=-1)
        else:
            z_i = self.mean(x_i)
            z_j = self.mean(x_j)
        return loss_fn(z_i, z_j, label.float())

    def parameter_groups(self) -> dict[str, list[nn.Parameter]]:
        return {
            "mean": list(self.mean.parameters()),
            "cov": list(self.cov.parameters()),
            "all": list(self.parameters()),
        }
