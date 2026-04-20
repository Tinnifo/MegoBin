from typing import Protocol

import torch


class PairSampler(Protocol):
    """Produces (feature_i, feature_j, label) triples for contrastive training.

    label ∈ {0, 1}:  1 = must-link (same genome), 0 = cannot-link (different).

    Implementations are expected to be torch-Dataset-compatible so they can
    be wrapped in a standard DataLoader. Feature arrays are supplied at
    construction time — samplers never compute features themselves.
    """

    def __len__(self) -> int: ...

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...
