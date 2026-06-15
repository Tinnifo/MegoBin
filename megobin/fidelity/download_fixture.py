"""Download the real-genome fidelity fixture from NCBI (developer-only, offline).

Reads an accessions manifest (one ``ACCESSION  description`` per line, ``#``
comments) and fetches each genome via NCBI efetch into a single multi-FASTA. The
FASTA is git-ignored (regenerate on demand); only the small text manifest is
committed. Used by the ``real_data``-gated tests, which skip when the FASTA is
absent.

    python -m megobin.fidelity.download_fixture \
        --manifest tests/fidelity/data/accessions.txt \
        --out tests/fidelity/data/real/genomes.fasta
"""

from __future__ import annotations

import argparse
import time
import urllib.request
from pathlib import Path

_EFETCH = (
    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    "?db=nuccore&id={acc}&rettype=fasta&retmode=text"
)


def read_accessions(manifest: Path) -> list[str]:
    accessions = []
    for line in manifest.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        accessions.append(line.split()[0])
    return accessions


def download(manifest: Path, out: Path, *, timeout: float = 60.0) -> Path:
    accessions = read_accessions(manifest)
    if not accessions:
        raise ValueError(f"no accessions in {manifest}")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        for i, acc in enumerate(accessions):
            if i:
                time.sleep(0.5)  # be gentle with NCBI
            with urllib.request.urlopen(_EFETCH.format(acc=acc), timeout=timeout) as r:
                text = r.read().decode()
            if not text.startswith(">"):
                raise RuntimeError(f"efetch for {acc} did not return FASTA")
            fh.write(text if text.endswith("\n") else text + "\n")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=Path("tests/fidelity/data/accessions.txt")
    )
    parser.add_argument(
        "--out", type=Path, default=Path("tests/fidelity/data/real/genomes.fasta")
    )
    args = parser.parse_args()
    out = download(args.manifest, args.out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
