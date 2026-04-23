import hashlib
import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd
from torch.utils.tensorboard import SummaryWriter

log = logging.getLogger(__name__)


class TensorBoardLogger:
    """TensorBoard-backed experiment tracker.

    One ``SummaryWriter`` per run, rooted at ``logdir``. Checkpoints are
    copied into ``logdir/checkpoints/`` and their paths recorded in
    ``run_meta.json`` so downstream tools can pick them up without a
    full artifact store.

    Scalars, config/text, and hyperparameters map cleanly to the
    TensorBoard equivalents. ``log_dataframe`` renders as a markdown
    table via ``add_text`` — TB has no native table widget, so this is
    an intentional downgrade.

    The ``name`` is advisory: we use it to tag run_meta.json and as a
    prefix when logging, but the on-disk location is controlled by
    ``logdir``.
    """

    def __init__(
        self,
        logdir: str | Path,
        name: str | None = None,
        flush_secs: int = 30,
    ):
        self.logdir = Path(logdir)
        self.logdir.mkdir(parents=True, exist_ok=True)
        self.name = name or self.logdir.name
        self.writer = SummaryWriter(log_dir=str(self.logdir), flush_secs=flush_secs)
        self._meta: dict[str, Any] = {
            "name": self.name,
            "logdir": str(self.logdir),
            "checkpoints": [],
        }

    # ------------------------------------------------------------------
    # Scalars
    # ------------------------------------------------------------------

    def log_scalars(self, values: dict[str, float], step: int) -> None:
        for k, v in values.items():
            self.writer.add_scalar(k, v, global_step=step)

    # ------------------------------------------------------------------
    # Static metadata
    # ------------------------------------------------------------------

    def log_config(self, config: dict[str, Any]) -> None:
        """Record the full config + repo provenance.

        Full config goes in as a ``text`` blob (YAML dump) because
        ``add_hparams`` can't represent nested dicts. A flattened
        scalar hparams view is added alongside so the HParams tab is
        usable for cross-run comparison.
        """
        import yaml

        self.writer.add_text("config", f"```yaml\n{yaml.safe_dump(config)}\n```")

        flat = _flatten(config)
        scalar_hparams = {
            k: v for k, v in flat.items() if isinstance(v, (int, float, str, bool))
        }
        if scalar_hparams:
            self.writer.add_hparams(scalar_hparams, metric_dict={})

        self._meta["config"] = config
        self._meta["git_sha"] = _git_sha()
        self._meta["env_hash"] = _file_hash(Path("environment.yml"))
        self._meta["dvc_version"] = _dvc_version()
        self._write_meta()

    def log_text(self, key: str, text: str) -> None:
        self.writer.add_text(key, text)

    # ------------------------------------------------------------------
    # Dataframes (best-effort: markdown table)
    # ------------------------------------------------------------------

    def log_dataframe(self, key: str, df: pd.DataFrame) -> None:
        self.writer.add_text(key, df.to_markdown(index=False))
        for col in df.select_dtypes(include="number").columns:
            self.writer.add_histogram(f"{key}/{col}", df[col].to_numpy(), global_step=0)

    # ------------------------------------------------------------------
    # Checkpoints (local copy + run_meta.json pointer)
    # ------------------------------------------------------------------

    def log_checkpoint(self, path: Path, name: str = "checkpoint") -> None:
        ckpt_dir = self.logdir / "checkpoints"
        ckpt_dir.mkdir(exist_ok=True)
        dest = ckpt_dir / f"{name}{path.suffix}"
        shutil.copy2(path, dest)
        self._meta["checkpoints"].append(
            {"name": name, "path": str(dest), "source": str(path)}
        )
        self._write_meta()
        log.info("checkpoint '%s' → %s", name, dest)

    def finish(self) -> None:
        self.writer.flush()
        self.writer.close()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _write_meta(self) -> None:
        (self.logdir / "run_meta.json").write_text(json.dumps(self._meta, indent=2))


# ---------------------------------------------------------------------------
# Repo-provenance helpers
# ---------------------------------------------------------------------------


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
            subprocess.check_output(["dvc", "version"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
            .split("\n")[0]
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _flatten(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        elif isinstance(v, list):
            out[key] = str(v)
        else:
            out[key] = v
    return out
