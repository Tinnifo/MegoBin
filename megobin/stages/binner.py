from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import hydra
import numpy as np
from hydra.utils import get_class
from omegaconf import DictConfig

from megobin.contracts import Stage, bind_kwargs
from megobin.ports import Bins, ContigSet, Embedding

log = logging.getLogger(__name__)


class BinnerStage(Stage):
    """Ports ``z_kept`` (+ ``contigs``, ``graph`` ...) -> ``bins``.

    The binner class declares ``needs`` / ``provides`` / ``bindings``. The
    binner is instantiated at run time, after the filter, so contig names align
    with the rows it clusters. Constructor kwargs come from the declared
    ``bindings`` and the manifest ports, not from signature introspection.
    """

    def __init__(self, config: DictConfig):
        self.config = config
        cls: Any = get_class(config._target_)
        self.cls = cls
        super().__init__(needs=cls.needs, provides=cls.provides)

    def run(self, data: dict[str, Any], ctx: Any) -> dict[str, Any]:
        port, label = self.cls.embedding
        emb: Embedding = data[port]
        contigs: ContigSet | None = data.get("contigs")
        values: dict[str, Any] = {
            k: v for k, v in data.items() if k not in ("features", "z", "z_kept")
        }
        values.update(
            contig_fasta=str(contigs.fasta) if contigs is not None else None,
            contig_names=emb.names,
            contig_lengths=contigs.lengths_for(emb.names) if contigs is not None else None,
            markers_dir=str(Path(ctx.run_dir) / "markers"),
        )
        kwargs = bind_kwargs(self.cls.bindings, values, f"binner/{self.cls.__name__}")
        binner = hydra.utils.instantiate(self.config, **kwargs)
        log.info("Binner:         %s", type(binner).__name__)

        labels = binner.cluster(emb.z.array(label))
        log.info("Bins: %d", len(np.unique(labels[labels >= 0])))
        return {"bins": Bins(labels=labels, names=emb.names)}
