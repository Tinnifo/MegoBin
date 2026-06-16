from typing import Protocol, runtime_checkable

from megobin.data.base import PairSampler
from megobin.encoders.base import Encoder
from megobin.losses.base import ContrastiveLoss


@runtime_checkable
class Trainer(Protocol):
    def fit(
        self,
        encoder: Encoder,
        sampler: PairSampler,
        loss_fn: ContrastiveLoss,
    ) -> None: ...
