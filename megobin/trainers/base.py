from typing import Protocol, runtime_checkable

from megobin.data.base import PairSampler
from megobin.encoders.base import Encoder
from megobin.losses.base import ContrastiveLoss


@runtime_checkable
class Trainer(Protocol):
    """Training-strategy contract.

    A trainer owns the optimization loop — epochs, batching, optimizer
    and scheduler stepping, gradient clipping, phase scheduling, logger
    logging — and mutates ``encoder`` in place. Encoders expose
    ``training_step(batch, loss_fn) -> scalar`` and named
    ``parameter_groups`` so the trainer can treat them uniformly while
    still supporting multi-phase regimes.

    Compatibility between (encoder, sampler, loss) is the user's
    responsibility at config time — the Protocols guarantee shape, not
    that the batch layouts agree.
    """

    def fit(
        self,
        encoder: Encoder,
        sampler: PairSampler,
        loss_fn: ContrastiveLoss,
    ) -> None: ...
