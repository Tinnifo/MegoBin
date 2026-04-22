from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class Logger(Protocol):
    """Experiment-tracking contract.

    Trainers call ``log_scalars`` every few steps; the pipeline calls
    the static-metadata methods once per run. Implementations are free
    to downgrade calls they can't represent (e.g. TensorBoard has no
    rich table widget, so ``log_dataframe`` renders as markdown text).
    """

    def log_scalars(self, values: dict[str, float], step: int) -> None: ...

    def log_config(self, config: dict[str, Any]) -> None: ...

    def log_text(self, key: str, text: str) -> None: ...

    def log_dataframe(self, key: str, df: pd.DataFrame) -> None: ...

    def log_checkpoint(self, path: Path, name: str = "checkpoint") -> None: ...

    def finish(self) -> None: ...
