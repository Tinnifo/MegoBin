from typing import Protocol, runtime_checkable

import torch.nn as nn
from torch.utils.data import Dataset


@runtime_checkable
class Trainer(Protocol):
    """Training-strategy contract.

    A trainer owns the optimization loop — epochs, batching, optimizer
    and scheduler stepping, gradient clipping, phase scheduling, logger
    logging — and mutates ``encoder`` in place. Encoders expose
    ``training_step(batch, loss_fn) -> scalar`` and named
    ``parameter_groups`` so the trainer can treat them uniformly while
    still supporting multi-phase regimes.

    ``sampler`` is a torch ``Dataset``-compatible ``PairSampler`` whose
    batch layout matches what the encoder expects. ``loss_fn`` is an
    ``nn.Module`` whose call signature matches what the encoder passes.
    Compatibility between (encoder, sampler, loss) is the user's
    responsibility at config time.
    """

    def fit(
        self,
        encoder: nn.Module,
        sampler: Dataset,
        loss_fn: nn.Module,
    ) -> None: ...
