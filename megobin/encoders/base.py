from typing import Any, Iterator, Protocol, runtime_checkable

import numpy as np
import torch
import torch.nn as nn

from megobin.losses.base import ContrastiveLoss


@runtime_checkable
class Encoder(Protocol):
    def encode(self, features: np.ndarray) -> np.ndarray: ...

    @property
    def embedding_dim(self) -> int: ...

    def training_step(
        self,
        batch: tuple[torch.Tensor, ...],
        loss_fn: ContrastiveLoss,
    ) -> torch.Tensor: ...

    def parameter_groups(self) -> dict[str, list[nn.Parameter]]: ...

    # nn.Module surface used by trainers — every concrete encoder is an
    # nn.Module subclass, so these are satisfied automatically. Declared
    # here so trainer signatures can accept ``Encoder`` without losing
    # access to the methods they actually call.

    # Return ``Any`` so concrete encoders can return ``Self`` (the
    # nn.Module convention) without tripping covariance checks.
    def to(self, device: torch.device | str) -> Any: ...

    def train(self, mode: bool = True) -> Any: ...

    def eval(self) -> Any: ...

    def parameters(self) -> Iterator[nn.Parameter]: ...

    def state_dict(self) -> dict[str, torch.Tensor]: ...

    def load_state_dict(self, state_dict: dict[str, torch.Tensor]) -> Any: ...
