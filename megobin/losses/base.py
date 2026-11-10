from typing import Any, Mapping, Protocol, runtime_checkable

import torch

from megobin.z import Z


@runtime_checkable
class ContrastiveLoss(Protocol):
    @property
    def needs(self) -> Mapping[str, frozenset[str]]:
        """What the loss needs, by port: ``{"batch": {...}, "z": {...}}``.

        ``batch`` labels come from the pair sampler (``views=2``, ``label``);
        ``z`` labels come from the encoder (``mu``, ``var``).
        """
        ...

    def __call__(
        self, z_i: Z[torch.Tensor], z_j: Z[torch.Tensor], label: torch.Tensor
    ) -> torch.Tensor: ...

    # ``Any`` so concrete losses (which subclass ``nn.Module``) can
    # return ``Self`` without tripping covariance checks.
    def to(self, device: torch.device | str) -> Any: ...
