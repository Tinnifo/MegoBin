from typing import Mapping, Protocol, runtime_checkable

import torch


@runtime_checkable
class PairSampler(Protocol):
    @property
    def provides(self) -> frozenset[str]:
        """Batch labels the sampler provides (``views=2``, ``label``, ...); a
        loss lists the ones it requires in ``loss.needs["batch"]``."""
        ...

    @property
    def needs(self) -> Mapping[str, frozenset[str]]:
        """Labels required on the ``features`` port (e.g. ``split`` for
        ``data_split.csv``)."""
        ...

    @property
    def bindings(self) -> Mapping[str, str]:
        """Constructor kwargs -> named values the embedder stage hands over
        (``features_split``). Replaces ``__init__`` signature introspection."""
        ...

    def __len__(self) -> int: ...

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...
