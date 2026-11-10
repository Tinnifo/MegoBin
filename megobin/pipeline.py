"""Main entry point — run experiments via Hydra config composition.

Usage:
    python megobin/pipeline.py --config-name experiment/uncertain_gen_dbscan

A run is a DAG of contracted stages (features -> embedder -> filter -> binner
[-> refiner]). The stage order is resolved backwards from ``pipeline.target``
and every contract is checked before any compute. See docs/03-architecture/.
Scoring bins is a separate entry point: ``python -m megobin.evaluate``.
"""

import logging
import random
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Ensure project root is importable even when Hydra changes the CWD.
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from megobin.contracts import ContractError, Problem, Stage, check, resolve
from megobin.manifest import Manifest
from megobin.ports import Bins, ContigSet, Embedding, write_bins
from megobin.stages.features import signal_problems
from megobin.status import config_sha256, intervention_problems, write_status
from megobin.utils.logger import Logger

log = logging.getLogger(__name__)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class RunContext:
    """What a stage may read at run time (besides the data on its input ports)."""

    cfg: DictConfig
    run_dir: Path
    logger: Logger | None
    manifest: Manifest


@dataclass
class Plan:
    stages: dict[str, Stage]
    order: list[str]
    problems: list[Problem] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems


@dataclass
class RunResult:
    status: dict[str, Any]
    data: dict[str, Any]
    run_dir: Path


def _check_signal_compatibility(cfg: DictConfig) -> None:
    """Fail fast if the feature config needs signals the dataset lacks.

    Thin wrapper over the contract check (the features stage's need on the
    ``signals`` port); kept so callers that only want this check can have it.
    """
    problems = signal_problems(cfg.get("features"), cfg.get("dataset"))
    if problems:
        name = cfg.dataset.get("name", "<unnamed>")
        raise ContractError(
            [
                Problem(
                    f"dataset '{name}' / features", q.port, q.missing, q.offered, q.provider
                )
                for q in problems
            ]
        )


def build_stages(cfg: DictConfig) -> dict[str, Stage]:
    """Instantiate ``cfg.pipeline.stages`` in config order.

    Order matters for reproducibility: the embedder builds the encoder, whose
    initial weights draw from the global torch RNG seeded by ``cfg.seed``.
    """
    stages: dict[str, Stage] = {}
    for name, spec in cfg.pipeline.stages.items():
        stage = hydra.utils.instantiate(spec)
        stage.name = name
        stages[name] = stage
    return stages


def make_plan(
    stages: dict[str, Stage], target: str, manifest: Manifest, interventions: list[Any]
) -> Plan:
    """Resolve the order backwards from ``target`` and check every contract.

    Nothing is computed. ``plan.problems`` lists every violation found.
    """
    for st in stages.values():
        st.bind(manifest)
    problems = intervention_problems(interventions)
    try:
        order = resolve(stages, target, manifest.available)
    except ContractError as e:
        return Plan(stages, [], problems + e.problems)
    problems += check(stages, order, manifest.available)
    return Plan(stages, order, problems)


def _run_dir(explicit: Path | None) -> Path:
    if explicit is not None:
        return Path(explicit)
    if HydraConfig.initialized():
        return Path(HydraConfig.get().runtime.output_dir)
    return Path.cwd()


def run_pipeline(cfg: DictConfig, run_dir: Path | None = None) -> RunResult:
    """Compose -> resolve -> check -> run. Writes ``status.json`` in ``run_dir``.

    Raises :class:`ContractError` (after writing the status file) if any
    contract fails; nothing has been computed at that point.
    """
    run_dir = _run_dir(run_dir)
    pcfg = cfg.pipeline
    interventions = OmegaConf.to_container(pcfg.get("interventions", []), resolve=True)
    assert isinstance(interventions, list)
    status: dict[str, Any] = {
        "pipeline": pcfg.get("name"),
        "target": pcfg.target,
        "status": "STARTED",
        "order": [],
        "contract": {"status": "UNCHECKED", "problems": []},
        "config_sha256": config_sha256(cfg),
        "seed": cfg.get("seed"),
        "interventions": interventions,
    }

    log.info("Config:\n%s", OmegaConf.to_yaml(cfg))
    seed_everything(cfg.seed)
    if cfg.get("use_abundance") is not None:
        log.info(
            "use_abundance is no longer honored — abundance is baked into data.csv "
            "via feature_merge."
        )

    manifest = Manifest.from_cfg(cfg)
    status["dataset"] = {"name": manifest.name, "path": str(manifest.root)}
    log.info("Dataset:        %s (%s)", manifest.name, manifest.root)

    try:
        stages = build_stages(cfg)
        plan = make_plan(stages, pcfg.target, manifest, interventions)
    except Exception as e:  # construction errors are config errors; keep the record
        status.update(status="ERROR", error=f"{type(e).__name__}: {e}")
        write_status(run_dir, status)
        raise

    status["order"] = plan.order
    if not plan.ok:
        status["status"] = "CONTRACT_FAIL"
        status["contract"] = {"status": "FAIL", "problems": [str(p) for p in plan.problems]}
        write_status(run_dir, status)
        raise ContractError(plan.problems)
    status["contract"] = {"status": "OK", "problems": []}
    write_status(run_dir, status)
    log.info("Contract OK. Stage order: %s", " -> ".join(plan.order))

    experiment_logger: Logger | None = None
    logger_cfg = cfg.get("logger")
    if logger_cfg is not None:
        experiment_logger = hydra.utils.instantiate(logger_cfg)
        log.info("Logger:         %s", type(experiment_logger).__name__)
        assert experiment_logger is not None
        experiment_logger.log_config(OmegaConf.to_container(cfg, resolve=True))  # type: ignore[arg-type]

    ctx = RunContext(cfg=cfg, run_dir=run_dir, logger=experiment_logger, manifest=manifest)
    data: dict[str, Any] = manifest.initial_data()
    try:
        for name in plan.order:
            data.update(plan.stages[name].run(data, ctx))

        out = data.get(pcfg.target)
        summary: dict[str, Any] = {}
        if "features" in data:
            summary["n_contigs"] = int(len(data["features"].names))
        if isinstance(out, Bins):
            contigs: ContigSet | None = data.get("contigs")
            n_bins = write_bins(out, contigs, run_dir / "bins")
            summary.update(
                n_bins=int(n_bins), n_unbinned=int((out.labels < 0).sum())
            )
            log.info("Wrote %d bin FASTA files to %s", n_bins, run_dir / "bins")
        if isinstance(data.get("z_kept"), Embedding):
            dropped = data["z_kept"].dropped
            summary["n_dropped_by_filter"] = 0 if dropped is None else int(len(dropped))
        status.update(status="OK", summary=summary)
        write_status(run_dir, status)
    except Exception as e:
        status.update(status="ERROR", error=f"{type(e).__name__}: {e}")
        write_status(run_dir, status)
        log.error("run failed:\n%s", traceback.format_exc())
        raise
    finally:
        if experiment_logger is not None:
            experiment_logger.finish()
    return RunResult(status=status, data=data, run_dir=run_dir)


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="experiment/uncertain_gen_dbscan",
)
def main(cfg: DictConfig) -> None:
    try:
        run_pipeline(cfg)
    except ContractError as e:
        log.error("Contract check failed before any compute:\n  - %s", "\n  - ".join(e.messages))
        raise SystemExit(2)


if __name__ == "__main__":
    main()
