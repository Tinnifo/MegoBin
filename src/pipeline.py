"""Main entry point — run experiments via Hydra config composition.

Usage:
    python src/pipeline.py --config-name baseline_rk
    python src/pipeline.py --config-name baseline_rk representation=contrastive_mlp loss=hinge
"""

import logging
import random
import sys
from pathlib import Path

# Ensure project root is importable even when Hydra changes the CWD.
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@hydra.main(version_base=None, config_path="../configs", config_name="experiment/baseline_rk")
def main(cfg: DictConfig) -> None:
    log.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    seed_everything(cfg.seed)

    # ---- Instantiate components from config ----
    representation = hydra.utils.instantiate(cfg.representation)
    loss_fn = hydra.utils.instantiate(cfg.loss)
    binner = hydra.utils.instantiate(cfg.binner)
    evaluator = hydra.utils.instantiate(cfg.evaluator)

    log.info("Representation: %s", type(representation).__name__)
    log.info("Loss:           %s", type(loss_fn).__name__)
    log.info("Binner:         %s", type(binner).__name__)
    log.info("Evaluator:      %s", type(evaluator).__name__)

    # ---- Feature loading ----
    data_dir = Path(cfg.get("data_dir", "data"))
    dataset = cfg.dataset

    kmer_path = data_dir / dataset / "kmer_profiles.npy"
    names_path = data_dir / dataset / "contig_names.npy"

    if kmer_path.exists():
        kmer_profiles = np.load(kmer_path)
        log.info("Loaded k-mer profiles: %s", kmer_profiles.shape)
    else:
        log.warning("K-mer profiles not found at %s — skipping pipeline run.", kmer_path)
        return

    contig_names = np.load(names_path, allow_pickle=True) if names_path.exists() else None

    # ---- Train (encoder-specific) ----
    # Training logic is encoder-dependent.  For the Poisson model the
    # co-occurrence matrix must be pre-computed and stored alongside the
    # k-mer profiles.  Contrastive encoders need pair samplers.
    # This section will be extended per-encoder; the pipeline keeps the
    # shared orchestration: load → train → encode → cluster → evaluate.

    log.info("Training not yet wired for %s — using current weights.", type(representation).__name__)

    # ---- Encode ----
    embeddings = representation.encode(kmer_profiles)
    log.info("Embeddings: %s", embeddings.shape)

    # ---- Cluster ----
    labels = binner.cluster(embeddings)
    n_bins = len(np.unique(labels))
    log.info("Bins: %d", n_bins)

    # ---- Write bins as FASTA for evaluator ----
    bins_dir = Path("bins")
    bins_dir.mkdir(exist_ok=True)

    if contig_names is not None:
        fasta_path = data_dir / dataset / "contigs.fasta"
        if fasta_path.exists():
            from src.features.kmer_profiles import read_fasta

            all_names, all_seqs = read_fasta(fasta_path)
            name_to_seq = dict(zip(all_names, all_seqs))

            for bin_id in np.unique(labels):
                members = contig_names[labels == bin_id]
                with open(bins_dir / f"bin_{bin_id:04d}.fasta", "w") as f:
                    for name in members:
                        seq = name_to_seq.get(str(name), "")
                        if seq:
                            f.write(f">{name}\n{seq}\n")

            log.info("Wrote %d bin FASTA files to %s", n_bins, bins_dir)

    # ---- Evaluate ----
    try:
        scores = evaluator.score(bins_dir)
        log.info("CheckM2 results:\n%s", scores.to_string())
    except FileNotFoundError:
        log.warning("CheckM2 not available — skipping evaluation.")


if __name__ == "__main__":
    main()
