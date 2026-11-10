"""Shared fixtures: a tiny synthetic dataset directory in the layout
``megobin.features.run_features`` writes (data.csv, data_split.csv, contigs.fasta)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

CONFIG_DIR = str((Path(__file__).parent.parent / "configs").resolve())
N_CONTIGS = 24
KMER_DIM = 136


def make_dataset(root: Path, n: int = N_CONTIGS, with_fasta: bool = True) -> Path:
    rng = np.random.default_rng(0)
    root.mkdir(parents=True, exist_ok=True)
    names = [f"c{i}" for i in range(n)]
    kmer = rng.dirichlet(np.ones(KMER_DIM), size=n)
    depth = np.stack([rng.uniform(1, 5, n), rng.uniform(0.1, 1, n)], axis=1)
    pd.DataFrame(np.hstack([kmer, depth]), index=names).to_csv(root / "data.csv")

    split_names, rows = [], []
    for i, nm in enumerate(names):
        for half in (1, 2):
            split_names.append(f"{nm}_{half}")
            rows.append(rng.dirichlet(np.ones(KMER_DIM)))
    pd.DataFrame(np.array(rows), index=split_names).to_csv(root / "data_split.csv")

    if with_fasta:
        with open(root / "contigs.fasta", "w") as f:
            for nm in names:
                seq = "".join(rng.choice(list("ACGT"), size=3000))
                f.write(f">{nm}\n{seq}\n")
    return root


@pytest.fixture
def dataset(tmp_path) -> Path:
    return make_dataset(tmp_path / "ds")


@pytest.fixture
def dataset_no_fasta(tmp_path) -> Path:
    return make_dataset(tmp_path / "ds_nofasta", with_fasta=False)


def compose_experiment(name: str, dataset: Path, *overrides: str):
    """Compose an experiment config pointed at ``dataset``, quick to run."""
    markers = {f"c{i}": [f"g{i % 3}"] for i in range(N_CONTIGS)}
    ov = [
        f"dataset.path={dataset}",
        "logger=none",
        "trainer.checkpoint_path=null",
        "binner.num_process=1",
        *overrides,
    ]
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        cfg = compose(config_name=f"experiment/{name}", overrides=ov)
    OmegaConf.set_struct(cfg, False)
    # Marker calls come from a dict, so the test needs no hmmsearch.
    cfg.binner.contig_to_marker = markers
    return cfg
