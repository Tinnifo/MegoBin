# Installation

## Prerequisites

`git`, `mamba` (preferred over `conda`). ~15 GB for the env plus disk for whichever dataset you point ``configs/dataset/example.yaml`` at.

## Clone

```bash
git clone https://github.com/Tinnifo/MegoBin.git
cd MegoBin
```

## Option A — Conda (laptop, BioCloud)

```bash
mamba env create -f environment.yml
mamba activate megobin
pip install -e .
```

Verify:

```bash
python -c "import megobin; print(megobin.__file__)"
pytest tests/test_interfaces.py -v
```

## Option B — Singularity (HPC)

Build once:

```bash
singularity build megobin.sif environment.def
```

Run:

```bash
singularity exec --nv megobin.sif python megobin/pipeline.py ...
```

## Option C — Read-only

```bash
pip install hydra-core omegaconf pyyaml
pip install -e .
```

Enough to import `megobin` and compose configs. No training.

## External tools

The pipeline core (encoder, training, embedding) runs from the conda env alone. The marker-aware DBSCAN binner ([megobin/binners/dbscan_ensemble.py](../../megobin/binners/dbscan_ensemble.py)) calls out to two external tools:

| Tool | Used by | Install |
|------|---------|---------|
| [`hmmsearch`](http://hmmer.org/documentation.html) (HMMER ≥3) | `_call_markers_from_fasta` | `brew install hmmer` (macOS), `mamba install -c bioconda hmmer` |
| [`prodigal`](https://github.com/hyattpd/Prodigal) (optional, only if `orf_finder=prodigal`) | `run_prodigal` | `mamba install -c bioconda prodigal` |
| [`FragGeneScan`](https://sourceforge.net/projects/fraggenescan/) (optional, only if `orf_finder=fraggenescan`) | `run_fraggenescan` | `mamba install -c bioconda fraggenescan` |

The default `orf_finder=fast-naive` is pure-Python ([megobin/utils/naive_orffinder.py](../../megobin/utils/naive_orffinder.py)) and needs no external binary.

The marker HMM database (107 single-copy marker genes) lives at [megobin/utils/marker.hmm](../../megobin/utils/marker.hmm) — fetched verbatim from [SemiBin](https://github.com/BigDataBiology/SemiBin/blob/main/SemiBin/marker.hmm). If you cloned without it:

```bash
curl -sL https://raw.githubusercontent.com/BigDataBiology/SemiBin/main/SemiBin/marker.hmm \
  -o megobin/utils/marker.hmm
```

To skip the marker pathway entirely (e.g. on a machine without HMMER), pass `+binner.contig_to_marker={}` on the CLI — DBSCAN then runs without marker-F1 bin selection and every contig becomes its own singleton, so this is for plumbing tests, not real runs.

[`CheckM2`](https://github.com/chklovski/CheckM2) is also optional — `pipeline.py` skips evaluation with a warning if its CLI isn't on PATH. Install via `mamba install -c bioconda checkm2` and download the diamond DB per the [CheckM2 quick start](https://github.com/chklovski/CheckM2#installation).
