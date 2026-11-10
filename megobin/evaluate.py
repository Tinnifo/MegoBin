"""Standalone judge: score a directory of bin FASTAs.

Usage:
    python -m megobin.evaluate bins_dir=outputs/<run>/bins
    python -m megobin.evaluate bins_dir=... evaluator.threads=8

The judge is deliberately not part of the method pipeline: every method's
bins must be scored by the same evaluator, so it is not a swappable stage.
"""

import json
import logging
from pathlib import Path

import hydra
import pandas as pd
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig

from megobin.evaluators.base import Evaluator

log = logging.getLogger(__name__)


def evaluate(bins_dir: Path, evaluator: Evaluator) -> pd.DataFrame:
    bins_dir = Path(bins_dir)
    if not bins_dir.is_dir():
        raise FileNotFoundError(f"bins_dir not found: {bins_dir}")
    return evaluator.score(bins_dir)


@hydra.main(version_base=None, config_path="../configs", config_name="evaluate")
def main(cfg: DictConfig) -> None:
    evaluator: Evaluator = instantiate(cfg.evaluator)
    out_dir = Path(HydraConfig.get().runtime.output_dir)
    try:
        scores = evaluate(Path(cfg.bins_dir), evaluator)
    except FileNotFoundError as e:
        log.error("Evaluation not possible: %s", e)
        raise SystemExit(2)
    log.info("%s results:\n%s", type(evaluator).__name__, scores.to_string())
    scores.to_csv(out_dir / "quality.tsv", sep="\t")
    summary = {
        "bins_dir": str(cfg.bins_dir),
        "evaluator": type(evaluator).__name__,
        "n_bins": int(len(scores)),
        "mean_completeness": float(scores["completeness"].mean()) if len(scores) else None,
        "mean_contamination": float(scores["contamination"].mean()) if len(scores) else None,
    }
    (out_dir / "evaluation.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
