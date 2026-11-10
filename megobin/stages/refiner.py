from __future__ import annotations

import logging
from typing import Any

import hydra
from hydra.utils import get_class
from omegaconf import DictConfig

from megobin.contracts import Stage, bind_kwargs

log = logging.getLogger(__name__)


class RefinerStage(Stage):
    """Optional stage: ``bins`` (+ ports the refiner names) -> ``bins_refined``.

    It is only resolved into the run when the pipeline target is
    ``bins_refined``; with target ``bins`` it is never built into the order.
    """

    def __init__(self, config: DictConfig):
        self.config = config
        cls: Any = get_class(config._target_)
        self.cls = cls
        super().__init__(needs=cls.needs, provides=cls.provides)

    def run(self, data: dict[str, Any], ctx: Any) -> dict[str, Any]:
        kwargs = bind_kwargs(
            self.cls.bindings, data, f"refiner/{self.cls.__name__}"
        )
        refiner = hydra.utils.instantiate(self.config, **kwargs)
        log.info("Refiner:        %s", type(refiner).__name__)
        inputs = {port: data[port] for port in self.cls.needs if port != "bins"}
        return {"bins_refined": refiner.refine(data["bins"], **inputs)}
