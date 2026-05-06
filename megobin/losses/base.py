from typing import Any, Protocol, runtime_checkable

import torch


@runtime_checkable
class ContrastiveLoss(Protocol):
    def __call__(
        self, z_i: torch.Tensor, z_j: torch.Tensor, label: torch.Tensor
    ) -> torch.Tensor:
        """Pair of embeddings + same/different label → scalar loss"""
        ...

    # ``Any`` so concrete losses (which subclass ``nn.Module``) can
    # return ``Self`` without tripping covariance checks.
    def to(self, device: torch.device | str) -> Any:
        """Move loss buffers/parameters to ``device``. Satisfied by ``nn.Module``."""
        ...
