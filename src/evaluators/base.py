from pathlib import Path
from typing import Protocol

import pandas as pd


class Evaluator(Protocol):
    def score(self, bins_dir: Path) -> pd.DataFrame:
        """Path to bin FASTAs → DataFrame with 'completeness' and 'contamination' columns"""
        ...
