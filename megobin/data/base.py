from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class PairSampler(Protocol):
    def __len__(self) -> int: ...

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...
