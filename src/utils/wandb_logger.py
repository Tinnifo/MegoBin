"""W&B experiment tracking helper.

Entity:  Metagenomic-Binning
Project: metagenomic-binning
Run naming: {hypothesis_id}-{seed}-{config_name}

Logs Huyen's 8 artifact types per run:
  1. Full Hydra config (serialized)
  2. Model checkpoint (.pt) as W&B Artifact
  3. Git commit SHA
  4. environment.yml hash
  5. DVC data version
  6. Loss curves, learning rate schedule
  7. CheckM2 results (completeness, contamination per bin)
  8. Tags: hypothesis-id, encoder type, dataset, seed
"""

import hashlib
import subprocess
from pathlib import Path

import wandb
from omegaconf import DictConfig, OmegaConf

ENTITY = "Metagenomic-Binning"
PROJECT = "metagenomic-binning"


def _git_sha() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _file_hash(path: Path) -> str:
    if not path.exists():
        return "missing"
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def _dvc_version() -> str:
    try:
        return (
            subprocess.check_output(
                ["dvc", "version"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
            .split("\n")[0]
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


# ---------------------------------------------------------------------------
# Init / finish
# ---------------------------------------------------------------------------


def init_run(
    cfg: DictConfig,
    hypothesis_id: str | None = None,
    tags: list[str] | None = None,
    project_root: Path = Path("."),
) -> wandb.sdk.wandb_run.Run:
    """Initialise a W&B run and log static metadata (artifacts 1, 3-5, 8).

    Args:
        cfg: Full Hydra DictConfig for this experiment.
        hypothesis_id: Optional hypothesis prefix for run name.
        tags: Extra tags to attach.
        project_root: Repo root (for environment.yml lookup).

    Returns:
        The active wandb Run.
    """
    seed = cfg.get("seed", 0)
    config_name = cfg.get("config_name", "experiment")
    encoder_type = OmegaConf.select(cfg, "representation._target_", default="unknown")
    dataset = cfg.get("dataset", "unknown")

    # Run name
    parts = []
    if hypothesis_id:
        parts.append(hypothesis_id)
    parts.extend([f"seed{seed}", config_name])
    run_name = "-".join(parts)

    # Tags
    run_tags = [encoder_type.rsplit(".", 1)[-1], dataset, f"seed{seed}"]
    if hypothesis_id:
        run_tags.append(hypothesis_id)
    if tags:
        run_tags.extend(tags)

    run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        name=run_name,
        config=OmegaConf.to_container(cfg, resolve=True),
        tags=run_tags,
    )

    # Artifact 3: git SHA
    run.config["git_sha"] = _git_sha()

    # Artifact 4: environment.yml hash
    run.config["env_hash"] = _file_hash(project_root / "environment.yml")

    # Artifact 5: DVC version
    run.config["dvc_version"] = _dvc_version()

    return run


# ---------------------------------------------------------------------------
# Per-step logging
# ---------------------------------------------------------------------------


def log_loss(step: int, loss: float, lr: float | None = None) -> None:
    """Artifact 6: loss curve + learning rate schedule."""
    payload: dict = {"loss": loss, "step": step}
    if lr is not None:
        payload["lr"] = lr
    wandb.log(payload, step=step)


def log_checkm2(df: "pandas.DataFrame") -> None:  # noqa: F821
    """Artifact 7: CheckM2 completeness & contamination per bin."""
    table = wandb.Table(dataframe=df.reset_index())
    wandb.log({"checkm2_results": table})

    wandb.run.summary["mean_completeness"] = float(df["completeness"].mean())
    wandb.run.summary["mean_contamination"] = float(df["contamination"].mean())
    wandb.run.summary["n_bins"] = len(df)


# ---------------------------------------------------------------------------
# Checkpoint artifact
# ---------------------------------------------------------------------------


def log_checkpoint(path: Path, name: str = "model-checkpoint") -> None:
    """Artifact 2: upload model .pt as a versioned W&B Artifact."""
    artifact = wandb.Artifact(name, type="model")
    artifact.add_file(str(path))
    wandb.log_artifact(artifact)


def finish_run() -> None:
    wandb.finish()
