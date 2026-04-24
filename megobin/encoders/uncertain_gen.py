import numpy as np
import torch
import torch.nn as nn


def _make_head(input_dim: int, hidden_dim: int, output_dim: int, dropout: float) -> nn.Sequential:
    """Build one 2-layer MLP head used by both mean and covariance branches."""
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.Sigmoid(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, output_dim),
    )


class UncertainGenEncoder(nn.Module):
    """UncertainGen dual-head Siamese encoder.

    Two heads with identical topology:
      - **mean head** → μ  (deterministic embedding)
      - **covariance head** → exp(log_σ²) ensuring positivity
        → diagonal Gaussian per sequence

    Training is sequential:
      Phase 1 — train mean head only (covariance frozen, include_std=False)
      Phase 2 — freeze mean head, train covariance head (include_std=True)

    Inference (encode) returns the mean embedding μ.
    """

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 512,
        embedding_dim: int = 256,
        dropout: float = 0.2,
        include_std: bool = False,
    ):
        super().__init__()
        self._embedding_dim = embedding_dim
        self.include_std = include_std

        self.mean_head = _make_head(input_dim, hidden_dim, embedding_dim, dropout)
        self.cov_head = _make_head(input_dim, hidden_dim, embedding_dim, dropout)

    def forward(
        self, x: torch.Tensor, include_std: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Single-branch forward.

        Args:
            x: (batch, input_dim)
            include_std: If True return (mean, cov) tuple; otherwise mean only.

        Returns:
            mean (batch, embedding_dim) or (mean, cov) where cov = exp(log_cov).
        """
        mu = self.mean_head(x)
        if not include_std:
            return mu
        log_cov = self.cov_head(x)
        cov = torch.exp(log_cov)
        return mu, cov

    # ------------------------------------------------------------------
    # Phase helpers
    # ------------------------------------------------------------------

    def freeze_cov(self) -> None:
        """Phase 1: train mean only."""
        for p in self.cov_head.parameters():
            p.requires_grad = False

    def freeze_mean(self) -> None:
        """Phase 2: train cov only."""
        for p in self.mean_head.parameters():
            p.requires_grad = False
        for p in self.cov_head.parameters():
            p.requires_grad = True

    def unfreeze_all(self) -> None:
        for p in self.parameters():
            p.requires_grad = True

    # ------------------------------------------------------------------
    # Encoder Protocol
    # ------------------------------------------------------------------

    def encode(self, features: np.ndarray) -> np.ndarray:
        """(N, input_dim) → (N, embedding_dim).  Returns mean embeddings.

        `features` may be a pure k-mer profile or a concatenation of
        k-mer and abundance vectors — the model is agnostic as long as
        the width matches `input_dim`.
        """
        self.eval()
        with torch.no_grad():
            x = torch.from_numpy(features).float()
            mu = self.mean_head(x)
        return mu.cpu().numpy()

    def encode_with_uncertainty(
        self, features: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """(N, input_dim) → (mean, cov) both (N, embedding_dim)."""
        self.eval()
        with torch.no_grad():
            x = torch.from_numpy(features).float()
            mu = self.mean_head(x)
            cov = torch.exp(self.cov_head(x))
        return mu.cpu().numpy(), cov.cpu().numpy()

    def training_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        loss_fn: nn.Module,
    ) -> torch.Tensor:
        """Siamese forward → ``loss_fn(z_i, z_j, label)``.

        In Phase 1 (``include_std=False``) ``z`` is just the mean μ.
        In Phase 2 (``include_std=True``) ``z`` is ``cat([μ, cov])`` so
        ``MahalanobisBCELoss`` can split it back into its two parts.
        The loss's ``include_std`` flag must match the encoder's.
        """
        x_i, x_j, label = batch
        if self.include_std:
            mu_i, cov_i = self.forward(x_i, include_std=True)
            mu_j, cov_j = self.forward(x_j, include_std=True)
            z_i = torch.cat([mu_i, cov_i], dim=-1)
            z_j = torch.cat([mu_j, cov_j], dim=-1)
        else:
            z_i = self.forward(x_i, include_std=False)
            z_j = self.forward(x_j, include_std=False)
        return loss_fn(z_i, z_j, label.float())

    def parameter_groups(self) -> dict[str, list[nn.Parameter]]:
        return {
            "mean": list(self.mean_head.parameters()),
            "cov": list(self.cov_head.parameters()),
            "all": list(self.parameters()),
        }

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim
