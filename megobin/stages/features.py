"""Features stage: reads the feature matrix that ``megobin.features`` wrote.

Also owns the signal check that used to be ``_check_signal_compatibility`` in
``pipeline.py``: the feature config's ``required_signals`` are a *need* on the
``signals`` port, the dataset's ``signals`` are what that port offers.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
from omegaconf import DictConfig

from megobin.contracts import Problem, Stage, check
from megobin.manifest import Manifest
from megobin.ports import Features

log = logging.getLogger(__name__)

DEFAULT_KMER_DIM = 136

# dataset signal -> content label on the ``features`` port
SIGNAL_LABELS = {"kmers": "kmer", "abundance": "depth"}


def required_signals(features_cfg: Any) -> list[str] | None:
    if features_cfg is None:
        return None
    req = features_cfg.get("required_signals")
    return None if req is None else [str(s) for s in req]


def signal_problems(features_cfg: Any, dataset_cfg: Any) -> list[Problem]:
    """Signal mismatch between a feature config and a dataset config.

    Both sides opt in: a feature config gains ``required_signals``, a dataset
    config gains ``signals``. Either side missing is a no-op (configs that
    predate the check keep working).
    """
    required = required_signals(features_cfg)
    provided = dataset_cfg.get("signals") if dataset_cfg is not None else None
    if required is None or provided is None:
        return []
    stage = PrecomputedFeatures(features_cfg)
    stage.needs = {"signals": frozenset(required)}
    return check({"features": stage}, ["features"], {"signals": frozenset(provided)})


class PrecomputedFeatures(Stage):
    """Port ``features`` from ``data.csv`` / ``data_split.csv`` in the dataset directory.

    ``config`` is the ``features`` config group (``dims`` is the k-mer block
    width). Labels provided: ``kmer``; ``depth`` when ``data.csv`` has columns
    beyond the k-mer block; ``split`` when ``data_split.csv`` exists. The label
    set is derived from the file headers at :meth:`bind`, i.e. before compute.
    """

    def __init__(self, config: DictConfig | None = None):
        super().__init__()
        self.config = config
        self.kmer_dim = int(config.get("dims", DEFAULT_KMER_DIM)) if config is not None else DEFAULT_KMER_DIM
        self.provides = {"features": frozenset({"kmer"})}
        self._root: Path | None = None

    def bind(self, manifest: Manifest) -> None:
        self._root = manifest.root
        needs: dict[str, frozenset[str]] = {"feature_files": frozenset({"data"})}
        req = required_signals(self.config)
        if req is not None and "signals" in manifest.ports:
            needs["signals"] = frozenset(req)
        self.needs = needs

        labels = {"kmer"}
        data_csv = manifest.root / "data.csv"
        if data_csv.exists():
            width = len(pd.read_csv(data_csv, nrows=0).columns) - 1  # minus index col
            if width - self.kmer_dim > 0:
                labels.add("depth")
        if "split" in manifest.ports.get("feature_files", ()):
            labels.add("split")
        self.provides = {"features": frozenset(labels)}

    def run(self, data: dict[str, Any], ctx: Any) -> dict[str, Any]:
        assert self._root is not None, "bind() must run before run()"
        path = self._root / "data.csv"
        df = pd.read_csv(path, index_col=0)
        x = df.values.astype("float32")
        names = df.index.to_numpy()
        log.info("Loaded features (kmer + abundance): %s", x.shape)
        split = self._root / "data_split.csv"
        return {
            "features": Features(
                x=x,
                names=names,
                kmer_dim=self.kmer_dim,
                split_path=split if split.exists() else None,
            )
        }
