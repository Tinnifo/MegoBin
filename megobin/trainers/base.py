from typing import Protocol, Sequence, runtime_checkable

from megobin.data.base import PairSampler
from megobin.encoders.base import Encoder
from megobin.losses.base import ContrastiveLoss


@runtime_checkable
class Trainer(Protocol):
    def phase_losses(self, default: ContrastiveLoss) -> Sequence[ContrastiveLoss]:
        """The loss used in each phase, in order. ``default`` is the loss passed
        to :meth:`fit`; a phase that declares none of its own uses it. The
        embedder stage checks each one against the encoder and sampler at
        pipeline start."""
        ...

    def fit(
        self,
        encoder: Encoder,
        sampler: PairSampler,
        loss_fn: ContrastiveLoss,
    ) -> None: ...
