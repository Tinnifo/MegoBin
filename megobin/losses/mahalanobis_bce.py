import torch
import torch.nn as nn
import torch.nn.functional as F


class MahalanobisBCELoss(nn.Module):
    def __init__(self, clamp_threshold: float = 1.0, include_std: bool = False):
        super().__init__()
        self.clamp_threshold = clamp_threshold
        self.include_std = include_std

    def forward(
        self, z_i: torch.Tensor, z_j: torch.Tensor, label: torch.Tensor
    ) -> torch.Tensor:
        if self.include_std:
            half = z_i.shape[-1] // 2
            mu_i, s_i = z_i[:, :half], z_i[:, half:]
            mu_j, s_j = z_j[:, :half], z_j[:, half:]
        else:
            mu_i, mu_j = z_i, z_j
            s_i = torch.ones_like(z_i)
            s_j = torch.ones_like(z_j)

        log_q = -((mu_i - mu_j) ** 2 / (s_i + s_j)).sum(dim=-1)

        # Clamp only in Phase 2: learned covariances can be tiny, causing
        # extreme log_q.  In Phase 1 (unit cov) the denominator is 2 per
        # dim, so log_q = -d²/2 which is naturally bounded.
        if self.include_std:
            log_q = torch.clamp(log_q, -self.clamp_threshold, self.clamp_threshold)

        p = torch.exp(log_q)
        return F.binary_cross_entropy(p, label)
