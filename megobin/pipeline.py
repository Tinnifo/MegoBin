"""Main entry point — run experiments via Hydra config composition.

Usage:
    python src/pipeline.py --config-name experiment/hybrid_uncertain_gen
    python src/pipeline.py --config-name experiment/training/semibin_cami_toy
"""

import inspect
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
from hydra.utils import get_class
from omegaconf import DictConfig, OmegaConf

from megobin.utils.checkpoints import load_checkpoint

log = logging.getLogger(__name__)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_sampler_inputs(dataset_path: Path) -> dict[str, np.ndarray]:
    """Load any on-disk arrays that samplers might consume.

    Keyed by the kwarg name samplers expect. Missing files are skipped;
    sampler instantiation will fail with a clear error if a required
    input isn't available.
    """
    candidates = {
        "features_whole": dataset_path / "features_whole.npy",
        "features_split": dataset_path / "features_split.npy",
        "cannot_link_pairs": dataset_path / "cannot_link_pairs.npy",
    }
    loaded: dict[str, np.ndarray] = {}
    for name, path in candidates.items():
        if path.exists():
            loaded[name] = np.load(path)
            log.info("Loaded sampler input '%s' from %s", name, path)
    return loaded


def _check_signal_compatibility(cfg: DictConfig) -> None:
    """Fail fast if the feature config needs signals the dataset lacks.

    Both sides opt in: a feature config gains `required_signals: [...]`,
    a dataset config gains `signals: [...]`. Missing either side is a
    no-op (so configs that predate this check keep working). When both
    are present, mismatch aborts with a clear error before any file I/O.
    """
    features = cfg.get("features")
    dataset = cfg.get("dataset")
    if features is None or dataset is None:
        return
    required = features.get("required_signals")
    provided = dataset.get("signals") if hasattr(dataset, "get") else None
    if required is None or provided is None:
        return

    missing = [s for s in required if s not in provided]
    if missing:
        dataset_name = dataset.get("name", "<unnamed>")
        raise ValueError(
            f"Dataset '{dataset_name}' provides signals {list(provided)} "
            f"but the feature config requires {list(required)}. "
            f"Missing: {missing}. Either switch dataset or pick a feature "
            f"config without those signals."
        )


def _instantiate_sampler(cfg_sampler: DictConfig, data: dict[str, np.ndarray]):
    """Instantiate a sampler, passing only the data kwargs it declares.

    Introspects the target class' ``__init__`` signature so pipeline.py
    doesn't need to know which sampler wants which arrays.
    """
    sampler_cls = get_class(cfg_sampler._target_)
    sig = inspect.signature(sampler_cls.__init__)
    kwargs = {k: v for k, v in data.items() if k in sig.parameters}
    return hydra.utils.instantiate(cfg_sampler, **kwargs)


@hydra.main(version_base=None, config_path="../configs", config_name="experiment/hybrid_uncertain_gen")
def main(cfg: DictConfig) -> None:
    log.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    seed_everything(cfg.seed)

    # ---- Capability check (runs before any I/O) ----
    _check_signal_compatibility(cfg)

    # ---- Instantiate components from config ----
    encoder = hydra.utils.instantiate(cfg.encoder)
    loss_fn = hydra.utils.instantiate(cfg.loss)
    binner = hydra.utils.instantiate(cfg.binner)
    evaluator = hydra.utils.instantiate(cfg.evaluator)

    log.info("Encoder:        %s", type(encoder).__name__)
    log.info("Loss:           %s", type(loss_fn).__name__)
    log.info("Binner:         %s", type(binner).__name__)
    log.info("Evaluator:      %s", type(evaluator).__name__)

    # ---- Feature loading ----
    # `cfg.dataset` is a capability descriptor dict (see configs/dataset/).
    # Legacy scalar `dataset: <name>` is still supported via `data_dir`.
    if hasattr(cfg.dataset, "path"):
        dataset_path = Path(cfg.dataset.path)
        dataset_name = cfg.dataset.get("name", dataset_path.name)
    else:
        data_dir = Path(cfg.get("data_dir", "data"))
        dataset_path = data_dir / str(cfg.dataset)
        dataset_name = str(cfg.dataset)
    log.info("Dataset:        %s (%s)", dataset_name, dataset_path)

    kmer_path = dataset_path / "kmer_profiles.npy"
    abundance_path = dataset_path / "abundance.npy"
    names_path = dataset_path / "contig_names.npy"

    if not kmer_path.exists():
        log.warning("K-mer profiles not found at %s — skipping pipeline run.", kmer_path)
        return

    features = np.load(kmer_path)
    if cfg.get("use_abundance", False) and abundance_path.exists():
        abundance = np.load(abundance_path)
        features = np.concatenate([features, abundance], axis=1)
        log.info("Loaded features: k-mer + abundance → %s", features.shape)
    else:
        log.info("Loaded features (k-mer only): %s", features.shape)

    contig_names = np.load(names_path, allow_pickle=True) if names_path.exists() else None

    # ---- Logger ----
    logger_cfg = cfg.get("logger")
    experiment_logger = hydra.utils.instantiate(logger_cfg) if logger_cfg is not None else None
    if experiment_logger is not None:
        log.info("Logger:         %s", type(experiment_logger).__name__)
        experiment_logger.log_config(OmegaConf.to_container(cfg, resolve=True))

    # ---- Load checkpoint OR train ----
    resume_from = cfg.get("resume_from")
    if resume_from:
        log.info("resume_from set — skipping training, loading %s", resume_from)
        load_checkpoint(encoder, resume_from)
    else:
        trainer_cfg = cfg.get("trainer")
        sampler_cfg = cfg.get("pair_sampler")

        if trainer_cfg is not None and sampler_cfg is not None:
            trainer = hydra.utils.instantiate(trainer_cfg, logger=experiment_logger)
            log.info("Trainer:        %s", type(trainer).__name__)

            sampler_inputs = _load_sampler_inputs(dataset_path)
            sampler = _instantiate_sampler(sampler_cfg, sampler_inputs)
            log.info("Sampler:        %s (size=%d)", type(sampler).__name__, len(sampler))

            trainer.fit(encoder=encoder, sampler=sampler, loss_fn=loss_fn)
        else:
            log.warning(
                "No trainer/pair_sampler configured — skipping training "
                "(encoder will run with its current weights)."
            )

    # ---- Encode ----
    embeddings = encoder.encode(features)
    log.info("Embeddings: %s", embeddings.shape)

    # ---- Cluster ----
    labels = binner.cluster(embeddings)
    n_bins = len(np.unique(labels))
    log.info("Bins: %d", n_bins)

    # ---- Write bins as FASTA for evaluator ----
    bins_dir = Path("bins")
    bins_dir.mkdir(exist_ok=True)

    if contig_names is not None:
        fasta_path = dataset_path / "contigs.fasta"
        if fasta_path.exists():
            from megobin.features.kmer_profiles import read_fasta

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
        if experiment_logger is not None:
            experiment_logger.log_dataframe("checkm2_results", scores)
            experiment_logger.log_scalars(
                {
                    "eval/mean_completeness": float(scores["completeness"].mean()),
                    "eval/mean_contamination": float(scores["contamination"].mean()),
                    "eval/n_bins": float(len(scores)),
                },
                step=0,
            )
    except FileNotFoundError:
        log.warning("CheckM2 not available — skipping evaluation.")

    if experiment_logger is not None:
        experiment_logger.finish()


if __name__ == "__main__":
    main()
