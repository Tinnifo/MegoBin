import warnings
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from megobin.contracts import ContractError, Problem
from megobin.features.views import FeatureView, KmerGateView
from megobin.losses.base import ContrastiveLoss
from megobin.z import Z


class UncertainGenEncoder(nn.Module):
    #: Input contract: k-mer block only (the ``norm_abundance`` gate in
    #: :class:`KmerGateView` decides whether depth columns are kept).
    needs = {"features": frozenset({"kmer"})}
    #: The encoder can send both heads. Which ones are *computed* during
    #: training is decided by the loss (``loss.needs["z"]``).
    provides = frozenset({"mu", "var"})

    def __init__(
        self,
        input_dim: int = 136,
        hidden_dim: int = 512,
        embedding_dim: int = 256,
        dropout: float = 0.2,
        include_std: bool | None = None,
        output_normalize: bool = True,
    ):
        super().__init__()
        if include_std is not None:
            warnings.warn(
                "UncertainGenEncoder(include_std=...) is ignored: the loss declares "
                "whether it needs z.var (MahalanobisBCELoss.include_std).",
                DeprecationWarning,
                stacklevel=2,
            )
        self._embedding_dim = embedding_dim
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

    def feature_view(self) -> FeatureView:
        return KmerGateView()

    def z(self, x: torch.Tensor, want: Iterable[str]) -> Z[torch.Tensor]:
        """Compute exactly the labelled fields in ``want``.

        Heads that nobody asked for are not run: their dropout would consume
        RNG draws and their BatchNorm would update running statistics.
        """
        want = frozenset(want)
        unknown = want - self.provides
        if unknown:
            raise ContractError(
                [
                    Problem(
                        type(self).__name__,
                        "z",
                        tuple(sorted(unknown)),
                        tuple(sorted(self.provides)),
                    )
                ]
            )
        if "var" in want:
            mu, cov = self.forward(x)
            return Z(mu=mu, var=cov)
        return Z(mu=self.mean(x))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(batch, input_dim) → (μ, cov) with cov = exp(log_cov)."""
        mu = self.mean(x)
        cov = torch.exp(self.cov(x))
        return mu, cov

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

    def encode_z(self, features: np.ndarray) -> Z[np.ndarray]:
        mu, cov = self.encode_with_uncertainty(features)
        return Z(mu=mu, var=cov)

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def training_step(
        self,
        batch: tuple[torch.Tensor, ...],
        loss_fn: ContrastiveLoss,
    ) -> torch.Tensor:
        x_i, x_j, label = batch
        want = loss_fn.needs["z"]
        z_i = self.z(x_i, want)
        z_j = self.z(x_j, want)
        return loss_fn(z_i, z_j, label.float())

    def parameter_groups(self) -> dict[str, list[nn.Parameter]]:
        return {
            "mean": list(self.mean.parameters()),
            "cov": list(self.cov.parameters()),
            "all": list(self.parameters()),
        }
