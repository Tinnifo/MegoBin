"""Manifest of the ports a dataset offers to the pipeline.

Replaces the fixed kwargs dict that ``pipeline.py`` used to hand to binners
(and the ``__init__`` introspection that filtered it). A dataset directory is
inspected once; stages ask for ports by name and labels, and the contract
check reports what is missing before any compute.

Ports offered:

=================  ===================  ====================================
port               labels               when
=================  ===================  ====================================
``signals``        ``dataset.signals``  the dataset config declares them
``feature_files``  ``data``, ``split``  ``data.csv`` / ``data_split.csv``
``contigs``        ``fasta``            ``contigs.fasta`` exists
``graph``          ``assembly_graph``   ``assembly_graph.gfa`` exists
=================  ===================  ====================================
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from megobin.ports import ContigSet, GraphFile

log = logging.getLogger(__name__)


@dataclass
class Manifest:
    name: str
    root: Path
    ports: dict[str, frozenset[str]] = field(default_factory=dict)
    values: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dataset(
        cls,
        root: Path,
        name: str,
        signals: list[str] | None = None,
    ) -> "Manifest":
        m = cls(name=name, root=Path(root))
        if signals is not None:
            m.ports["signals"] = frozenset(str(s) for s in signals)
        files: set[str] = set()
        if (m.root / "data.csv").exists():
            files.add("data")
        if (m.root / "data_split.csv").exists():
            files.add("split")
        if files:
            m.ports["feature_files"] = frozenset(files)
        fasta = m.root / "contigs.fasta"
        if fasta.exists():
            m.ports["contigs"] = frozenset({"fasta"})
            m.values["contigs"] = ContigSet(fasta=fasta)
        gfa = m.root / "assembly_graph.gfa"
        if gfa.exists():
            m.ports["graph"] = frozenset({"assembly_graph"})
            m.values["graph"] = GraphFile(path=gfa)
        return m

    @classmethod
    def from_cfg(cls, cfg: DictConfig) -> "Manifest":
        """``cfg.dataset`` is a capability descriptor dict (see configs/dataset/).

        The legacy scalar form ``dataset: <name>`` still resolves via ``data_dir``.
        """
        ds = cfg.get("dataset")
        if ds is not None and hasattr(ds, "path"):
            root = Path(ds.path)
            name = ds.get("name", root.name)
            signals = ds.get("signals")
            signals = None if signals is None else [str(s) for s in signals]
        else:
            data_dir = Path(cfg.get("data_dir", "data"))
            root = data_dir / str(ds)
            name = str(ds)
            signals = None
        return cls.from_dataset(root, name, signals)

    @property
    def available(self) -> dict[str, frozenset[str]]:
        return dict(self.ports)

    def initial_data(self) -> dict[str, Any]:
        return dict(self.values)
