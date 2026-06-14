"""Regenerate the committed reference-oracle cache from a live oracle.

Developer-only, run offline on a machine with SemiBin installed; never invoked
by the test suite. It materialises the canonical fixture deterministically,
runs the genuine in-process SemiBin2 oracle, and writes ``run_*.csv`` (text,
git-committable) plus a ``provenance.json`` recording the source, SemiBin
version, seeds, and the fixture manifest hash (so a stale cache is detected).

    python -m megobin.fidelity.refresh_cache \
        --out tests/fidelity/data/cache/genome_set_v1 --seeds 10
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
from importlib import metadata
from pathlib import Path

from megobin.fidelity.fixtures import canonical_genome_fixture
from megobin.fidelity.oracles_semibin2 import (
    SEMIBIN_N_TOTAL_MARKERS,
    SemiBin2GetBestBinOracle,
)


def _semibin_version() -> str:
    try:
        return metadata.version("SemiBin")
    except metadata.PackageNotFoundError:  # pragma: no cover
        return "unknown"


def refresh(out_dir: Path, *, n_seeds: int, minfasta: int) -> Path:
    fixture = canonical_genome_fixture()
    oracle = SemiBin2GetBestBinOracle(minfasta=minfasta, seeds=range(n_seeds))
    runs = oracle.runs(fixture)

    out_dir.mkdir(parents=True, exist_ok=True)
    for existing in out_dir.glob("run_*.csv"):
        existing.unlink()
    for k, labels in enumerate(runs):
        with open(out_dir / f"run_{k}.csv", "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["contig_name", "label"])
            for name, label in zip(fixture.contig_names, labels):
                writer.writerow([str(name), int(label)])

    provenance = {
        "source": "SemiBin2 get_best_bin (in-process)",
        "semibin_version": _semibin_version(),
        "fixture_id": fixture.fixture_id,
        "fixture_manifest_sha256": fixture.manifest_sha256,
        "seeds": list(range(n_seeds)),
        "minfasta": minfasta,
        "n_total_markers": SEMIBIN_N_TOTAL_MARKERS,
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    with open(out_dir / "provenance.json", "w") as fh:
        json.dump(provenance, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("tests/fidelity/data/cache/genome_set_v1"),
    )
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--minfasta", type=int, default=200000)
    args = parser.parse_args()
    out = refresh(args.out, n_seeds=args.seeds, minfasta=args.minfasta)
    print(f"wrote {args.seeds} cached runs + provenance to {out}")


if __name__ == "__main__":
    main()
