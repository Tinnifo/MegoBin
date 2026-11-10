from __future__ import annotations

import logging
from typing import Any

import hydra
from omegaconf import DictConfig

from megobin.contracts import Stage
from megobin.filters.base import Filter
from megobin.filters.no_op import NoOpFilter
from megobin.ports import Embedding
from megobin.z import Z

log = logging.getLogger(__name__)


class FilterStage(Stage):
    """Port ``z`` -> ``z_kept``. ``config=None`` means keep everything (NoOpFilter).

    The filter declares which labels of ``z`` it reads (``filter.needs``); a
    filter that needs ``var`` behind an encoder that sends none fails at start.
    """

    def __init__(self, config: DictConfig | None = None):
        self.filter: Filter = (
            hydra.utils.instantiate(config) if config is not None else NoOpFilter()
        )
        super().__init__(
            needs={"z": self.filter.needs["z"]},
            provides={"z_kept": {"mu"}},
        )
        log.info("Filter:         %s", type(self.filter).__name__)

    def run(self, data: dict[str, Any], ctx: Any) -> dict[str, Any]:
        emb: Embedding = data["z"]
        z = emb.z
        side_outputs = {"covariance": z.var} if z.var is not None else None
        kept, kept_names, dropped = self.filter.fit_transform(
            z.array("mu"), emb.names, side_outputs
        )
        if len(dropped) > 0:
            log.info(
                "Filter dropped %d/%d contigs (%s)",
                len(dropped),
                len(emb.names),
                type(self.filter).__name__,
            )
        return {"z_kept": Embedding(z=Z(mu=kept), names=kept_names, dropped=dropped)}
