"""Main entry point — run experiments via Hydra config composition.

Usage:
    python megobin/pipeline.py --config-name experiment/uncertain_gen_dbscan
"""

import inspect
import logging
import random
import sys
from pathlib import Path
from typing import cast

# Ensure project root is importable even when Hydra changes the CWD.
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import hydra
import numpy as np
import pandas as pd
import torch
from hydra.utils import get_class
from omegaconf import DictConfig, OmegaConf

from megobin.binners.base import Binner
from megobin.data.base import PairSampler
from megobin.encoders.base import Encoder
from megobin.evaluators.base import Evaluator
from megobin.filters.base import Filter
from megobin.losses.base import ContrastiveLoss
from megobin.trainers.base import Trainer
from megobin.utils.checkpoints import load_checkpoint
from megobin.utils.logger import Logger

log = logging.getLogger(__name__)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_data_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a feature_merge ``data.csv`` (kmer + abundance, indexed by contig).

    Returns ``(features, contig_names)``.
    """
    df = pd.read_csv(path, index_col=0)
    return df.values.astype(np.float32), df.index.to_numpy()


def _load_data_split_csv(path: Path) -> np.ndarray:
    """Read a feature_merge ``data_split.csv`` and return the
    ``[left_halves; right_halves]`` layout the samplers expect.

    feature_merge writes split rows interleaved as ``[c1_1, c1_2, c2_1, c2_2, ...]``
    (contig name suffixed ``_1`` / ``_2`` for the two halves). Samplers expect
    all left halves first, then all right halves.
    """
    df = pd.read_csv(path, index_col=0)
    idx = df.index.astype(str)
    left = df.loc[idx.str.endswith("_1")].values
    right = df.loc[idx.str.endswith("_2")].values
    if left.shape[0] != right.shape[0]:
        raise ValueError(
            f"data_split.csv at {path} has mismatched halves: "
            f"{left.shape[0]} '_1' rows vs {right.shape[0]} '_2' rows"
        )
    return np.concatenate([left, right], axis=0).astype(np.float32)


def _load_sampler_inputs(
    dataset_path: Path, features_whole: np.ndarray
) -> dict[str, np.ndarray]:
    """Load any on-disk arrays that samplers might consume.

    ``features_whole`` is reused from the main feature load (data.csv).
    ``features_split`` comes from data_split.csv. Missing files are
    skipped; sampler instantiation will fail with a clear error if a
    required input isn't available.
    """
    loaded: dict[str, np.ndarray] = {"features_whole": features_whole}

    split_csv = dataset_path / "data_split.csv"
    if split_csv.exists():
        loaded["features_split"] = _load_data_split_csv(split_csv)
        log.info("Loaded sampler input 'features_split' from %s", split_csv)

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


def _instantiate_sampler(
    cfg_sampler: DictConfig, data: dict[str, np.ndarray]
) -> PairSampler:
    """Instantiate a sampler, passing only the data kwargs it declares.

    Introspects the target class' ``__init__`` signature so pipeline.py
    doesn't need to know which sampler wants which arrays.
    """
    sampler_cls = get_class(cfg_sampler._target_)
    sig = inspect.signature(sampler_cls.__init__)  # type: ignore[misc]
    kwargs = {k: v for k, v in data.items() if k in sig.parameters}
    return hydra.utils.instantiate(cfg_sampler, **kwargs)


def _instantiate_binner(cfg_binner: DictConfig, data: dict) -> Binner:
    """Instantiate a binner, passing only the runtime kwargs it declares.

    Mirrors `_instantiate_sampler` so binners that depend on the contig
    FASTA / names / lengths (e.g. DBSCANEnsembleBinner's marker-F1 path)
    can pull them from the active dataset without having to be hard-coded
    in YAML.
    """
    binner_cls = get_class(cfg_binner._target_)
    sig = inspect.signature(binner_cls.__init__)  # type: ignore[misc]
    kwargs = {k: v for k, v in data.items() if k in sig.parameters}
    return hydra.utils.instantiate(cfg_binner, **kwargs)


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="experiment/uncertain_gen_dbscan",
)
def main(cfg: DictConfig) -> None:
    log.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    seed_everything(cfg.seed)

    # ---- Capability check (runs before any I/O) ----
    _check_signal_compatibility(cfg)

    # ---- Instantiate components from config ----
    # Binner is instantiated later (after encode + filter) so marker-aware
    # binners like DBSCANEnsembleBinner receive the post-filter contig
    # names that align with the embeddings handed to ``cluster``.
    encoder: Encoder = hydra.utils.instantiate(cfg.encoder)
    loss_fn: ContrastiveLoss = hydra.utils.instantiate(cfg.loss)
    evaluator: Evaluator = hydra.utils.instantiate(cfg.evaluator)

    filter_cfg = cfg.get("filter")
    filter_obj: Filter
    if filter_cfg is not None:
        filter_obj = hydra.utils.instantiate(filter_cfg)
    else:
        from megobin.filters.no_op import NoOpFilter

        filter_obj = NoOpFilter()

    log.info("Encoder:        %s", type(encoder).__name__)
    log.info("Loss:           %s", type(loss_fn).__name__)
    log.info("Evaluator:      %s", type(evaluator).__name__)
    log.info("Filter:         %s", type(filter_obj).__name__)

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

    data_csv = dataset_path / "data.csv"
    if not data_csv.exists():
        log.warning("data.csv not found at %s — skipping pipeline run.", data_csv)
        return

    features, contig_names = _load_data_csv(data_csv)
    log.info("Loaded features (kmer + abundance): %s", features.shape)
    if cfg.get("use_abundance") is not None:
        log.info(
            "use_abundance is no longer honored — abundance is baked into data.csv "
            "via feature_merge."
        )

    # Mirror SemiBin's `norm_abundance` gate: when too few BAMs make abundance
    # statistically uninformative, drop it so encoder inference dims match
    # training dims (data_split.csv, which is kmer-only in single-sample mode).
    kmer_dim = int(cfg.features.dims) if cfg.get("features") is not None else 136
    n_abund = features.shape[1] - kmer_dim
    if n_abund > 0:
        from megobin.utils.SemiBin_utils import norm_abundance

        if norm_abundance(features):
            log.info("norm_abundance gate: keeping %d abundance cols", n_abund)
        else:
            log.warning(
                "norm_abundance gate: %d abundance cols below threshold — "
                "dropping abundance to match data_split.csv (kmer-only). "
                "Re-run feature_merge in multi-sample mode (≥5 BAMs) to use abundance.",
                n_abund,
            )
            features = features[:, :kmer_dim]

    # ---- Logger ----
    logger_cfg = cfg.get("logger")
    experiment_logger: Logger | None = (
        hydra.utils.instantiate(logger_cfg) if logger_cfg is not None else None
    )
    if experiment_logger is not None:
        log.info("Logger:         %s", type(experiment_logger).__name__)
        config_dict = cast(dict, OmegaConf.to_container(cfg, resolve=True))
        experiment_logger.log_config(config_dict)

    # ---- Load checkpoint OR train ----
    resume_from = cfg.get("resume_from")
    if resume_from:
        log.info("resume_from set — skipping training, loading %s", resume_from)
        load_checkpoint(encoder, resume_from)
    else:
        trainer_cfg = cfg.get("trainer")
        sampler_cfg = cfg.get("pair_sampler")

        if trainer_cfg is not None and sampler_cfg is not None:
            trainer: Trainer = hydra.utils.instantiate(
                trainer_cfg, logger=experiment_logger
            )
            log.info("Trainer:        %s", type(trainer).__name__)

            sampler_inputs = _load_sampler_inputs(dataset_path, features)
            sampler = _instantiate_sampler(sampler_cfg, sampler_inputs)
            log.info(
                "Sampler:        %s (size=%d)", type(sampler).__name__, len(sampler)
            )

            trainer.fit(encoder=encoder, sampler=sampler, loss_fn=loss_fn)
        else:
            log.warning(
                "No trainer/pair_sampler configured — skipping training "
                "(encoder will run with its current weights)."
            )

    # ---- Encode (+ optional uncertainty side-output) ----
    side_outputs: dict[str, np.ndarray] | None = None
    if hasattr(encoder, "encode_with_uncertainty"):
        embeddings, covariance = encoder.encode_with_uncertainty(features)
        side_outputs = {"covariance": covariance}
        log.info(
            "Embeddings: %s (with covariance %s)", embeddings.shape, covariance.shape
        )
    else:
        embeddings = encoder.encode(features)
        log.info("Embeddings: %s", embeddings.shape)

    # ---- Filter ----
    embeddings, kept_names, dropped_names = filter_obj.fit_transform(
        embeddings, contig_names, side_outputs
    )
    if len(dropped_names) > 0:
        log.info(
            "Filter dropped %d/%d contigs (%s)",
            len(dropped_names),
            len(contig_names),
            type(filter_obj).__name__,
        )

    # ---- Binner (instantiated post-filter so contig_names aligns) ----
    binner = _instantiate_binner(
        cfg.binner,
        {
            "contig_fasta": str(dataset_path / "contigs.fasta"),
            "contig_names": kept_names,
            "output_dir": str(Path.cwd() / "markers"),
        },
    )
    log.info("Binner:         %s", type(binner).__name__)

    # ---- Cluster ----
    labels = binner.cluster(embeddings)
    n_bins = len(np.unique(labels))
    log.info("Bins: %d", n_bins)

    # ---- Write bins as FASTA for evaluator ----
    bins_dir = Path("bins")
    bins_dir.mkdir(exist_ok=True)

    if kept_names is not None:
        fasta_path = dataset_path / "contigs.fasta"
        if fasta_path.exists():
            from megobin.utils.fasta import fasta_iter

            name_to_seq = dict(fasta_iter(str(fasta_path)))

            for bin_id in np.unique(labels):
                members = kept_names[labels == bin_id]
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
