"""``status.json``: the record a run leaves next to ``.hydra/``.

Fields: pipeline name, target, resolved stage order, contract result (with
every problem), config hash, seed, dataset, interventions ledger (levels
L0-L3) and a small run summary.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from megobin.contracts import Problem

LEVELS = ("L0", "L1", "L2", "L3")


def config_sha256(cfg: DictConfig) -> str:
    """Hash of the composed config, interpolations left unresolved.

    Unresolved so run-specific values (Hydra's timestamped output dir) do not
    change the hash; two runs with the same config get the same hash.
    """
    return hashlib.sha256(OmegaConf.to_yaml(cfg, resolve=False).encode()).hexdigest()


def intervention_problems(entries: list[Any]) -> list[Problem]:
    """Validate the interventions ledger: each entry has ``name`` and a level L0-L3."""
    problems: list[Problem] = []
    for i, e in enumerate(entries):
        if not isinstance(e, dict) or "name" not in e or "level" not in e:
            problems.append(
                Problem("interventions", "", detail=f"entry {i} needs 'name' and 'level': {e!r}")
            )
        elif e["level"] not in LEVELS:
            problems.append(
                Problem(
                    "interventions",
                    "",
                    detail=f"entry '{e['name']}' has level {e['level']!r}; use one of {list(LEVELS)}",
                )
            )
    return problems


def write_status(run_dir: Path, status: dict[str, Any]) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "status.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(status, indent=1, default=str))
    os.replace(tmp, path)
    return path
