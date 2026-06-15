import subprocess
import tempfile
from pathlib import Path

import pandas as pd


class CheckM2Evaluator:
    """Wrapper around the CheckM2 CLI.

    Calls ``checkm2 predict`` on a directory of bin FASTA files and parses the
    resulting ``quality_report.tsv`` into a DataFrame.
    """

    def __init__(self, threads: int = 1, extra_args: list[str] | None = None):
        self.threads = threads
        self.extra_args = extra_args or []

    def score(self, bins_dir: Path) -> pd.DataFrame:
        """Run CheckM2 on *bins_dir* and return completeness / contamination.

        Args:
            bins_dir: Directory containing one FASTA file per bin.

        Returns:
            DataFrame indexed by bin name with at least ``completeness``
            and ``contamination`` columns.
        """
        bins_dir = Path(bins_dir)

        with tempfile.TemporaryDirectory(prefix="checkm2_") as tmp:
            output_dir = Path(tmp)

            cmd = [
                "checkm2", "predict",
                "-i", str(bins_dir),
                "-o", str(output_dir),
                "--threads", str(self.threads),
                *self.extra_args,
            ]

            subprocess.run(cmd, check=True)

            report = output_dir / "quality_report.tsv"
            df = pd.read_csv(report, sep="\t")

        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

        for col in ("completeness", "contamination"):
            if col not in df.columns:
                raise KeyError(
                    f"CheckM2 report missing '{col}' column. "
                    f"Found: {list(df.columns)}"
                )

        if "name" in df.columns:
            df = df.set_index("name")

        return df[["completeness", "contamination"]]
