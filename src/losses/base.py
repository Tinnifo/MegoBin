from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class ContrastiveLoss(Protocol):
    def __call__(
        self, z_i: torch.Tensor, z_j: torch.Tensor, label: torch.Tensor
    ) -> torch.Tensor:
        """Pair of embeddings + same/different label → scalar loss"""
        ...
