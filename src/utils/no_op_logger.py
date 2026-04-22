from pathlib import Path
from typing import Any

import pandas as pd


class NoOpLogger:
    """Logger that silently discards everything.

    Useful for tests and for runs where you don't want an event file on
    disk. Satisfies the ``Logger`` Protocol.
    """

    def log_scalars(self, values: dict[str, float], step: int) -> None:
        pass

    def log_config(self, config: dict[str, Any]) -> None:
        pass

    def log_text(self, key: str, text: str) -> None:
        pass

    def log_dataframe(self, key: str, df: pd.DataFrame) -> None:
        pass

    def log_checkpoint(self, path: Path, name: str = "checkpoint") -> None:
        pass

    def finish(self) -> None:
        pass
