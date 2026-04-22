from typing import Protocol, runtime_checkable

import numpy as np
import torch
import torch.nn as nn


@runtime_checkable
class Representation(Protocol):
    """Encoder contract.

    An encoder must support two modes:

    - **Inference** via ``encode`` — the Numpy-in/Numpy-out method used by
      the binner after training is complete.
    - **Training** via ``training_step`` — a single forward pass that
      consumes one batch produced by a ``PairSampler`` and returns a
      scalar loss. The encoder decides what intermediates to feed into
      ``loss_fn`` (e.g. plain embeddings, ``(μ, cov)`` concatenations, or
      k-mer embedding lookups), so losses stay fully swappable.

    Encoders additionally expose named ``parameter_groups`` so that
    phase-based trainers can freeze / unfreeze subsets of the network
    (e.g. UncertainGen's ``mean`` and ``cov`` heads).
    """

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def encode(self, features: np.ndarray) -> np.ndarray:
        """(N, input_dim) → (N, embedding_dim)."""
        ...

    @property
    def embedding_dim(self) -> int: ...

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def training_step(
        self,
        batch: tuple[torch.Tensor, ...],
        loss_fn: nn.Module,
    ) -> torch.Tensor:
        """Forward pass over one batch → scalar loss.

        The batch layout is encoder-specific and must match what the
        configured ``PairSampler`` produces. Implementations should pass
        the right arguments to ``loss_fn`` (which is a ``nn.Module`` with
        a ``__call__`` whose signature is loss-specific).
        """
        ...

    def parameter_groups(self) -> dict[str, list[nn.Parameter]]:
        """Named parameter groups for phase-based training.

        Every encoder exposes at least ``{"all": [...]}``. Encoders with
        multiple heads (e.g. UncertainGen) expose per-head groups so a
        two-phase trainer can freeze / unfreeze them by name.
        """
        ...
