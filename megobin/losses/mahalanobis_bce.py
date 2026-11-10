import torch
import torch.nn as nn
import torch.nn.functional as F

from megobin.z import Z, as_z


class MahalanobisBCELoss(nn.Module):
    """UncertainGen's Mahalanobis BCE loss.

    ``include_std`` is the loss's own declaration of what it needs: ``True``
    means it reads ``z.var`` (learned covariance, with clamping); ``False``
    means unit covariance and ``z.mu`` only. The encoder no longer carries a
    matching flag; a mismatch fails at pipeline start (see ``needs``).
    """

    def __init__(self, clamp_threshold: float = 1.0, include_std: bool = False):
        super().__init__()
        self.clamp_threshold = clamp_threshold
        self.include_std = include_std

    @property
    def needs(self) -> dict[str, frozenset[str]]:
        z = {"mu", "var"} if self.include_std else {"mu"}
        return {"batch": frozenset({"views=2", "label"}), "z": frozenset(z)}

    def forward(
        self, z_i: "Z | torch.Tensor", z_j: "Z | torch.Tensor", label: torch.Tensor
    ) -> torch.Tensor:
        zi, zj = as_z(z_i), as_z(z_j)
        zi.require(self.needs["z"], type(self).__name__)
        zj.require(self.needs["z"], type(self).__name__)
        mu_i, mu_j = zi.array("mu"), zj.array("mu")
        if self.include_std:
            s_i, s_j = zi.array("var"), zj.array("var")
        else:
            s_i = torch.ones_like(mu_i)
            s_j = torch.ones_like(mu_j)

        log_q = -((mu_i - mu_j) ** 2 / (s_i + s_j)).sum(dim=-1)

        # Clamp only in Phase 2: learned covariances can be tiny, causing
        # extreme log_q.  In Phase 1 (unit cov) the denominator is 2 per
        # dim, so log_q = -d²/2 which is naturally bounded.
        if self.include_std:
            log_q = torch.clamp(log_q, -self.clamp_threshold, self.clamp_threshold)

        p = torch.exp(log_q)
        return F.binary_cross_entropy(p, label)
