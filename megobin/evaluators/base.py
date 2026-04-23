from pathlib import Path
from typing import Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class Evaluator(Protocol):
    def score(self, bins_dir: Path) -> pd.DataFrame:
        """Path to bin FASTAs → DataFrame with 'completeness' and 'contamination' columns"""
        ...
