# BioCloud (CMC-AAU)

BioCloud is the biology-hosted cluster at CMC-AAU. It's where data lives, where Snakemake orchestrates the full binning pipeline, and where CheckM2 evaluation runs. It's not our primary GPU cluster — that's DEIS-MCC — but it has enough GPU capacity for short training runs and for debugging.

Rule of thumb: **BioCloud runs the pipeline; DEIS-MCC trains the encoder**. If you're working on feature engineering, sampler logic, evaluator behaviour, or anything that needs the real CheckM2 database, BioCloud. If you're training for 8+ hours on a GPU, DEIS-MCC.

## Access and layout

```
ssh tinni@biocloud.cmc.aau.dk
```

(Ask Sebastian for the exact hostname and your account if you don't already have one.)

Key paths:

- Your home: `/home/tinni/` — small, not for datasets.
- Project scratch: `/projects/microbial-dark-matter/metagenomic-binning/` — shared with the bio team. Put datasets, checkpoints, and long-running outputs here.
- Singularity cache: `/projects/microbial-dark-matter/containers/` — drop `megobin.sif` here so other lab members can reuse it.

## Environment

BioCloud has `mamba` pre-installed (faster than `conda`). One-time environment setup:

```bash
cd /projects/microbial-dark-matter/metagenomic-binning/
git clone git@github.com:Tinnifo/Metagenomic-Binning.git
cd Metagenomic-Binning
mamba env create -f environment.yml
mamba activate megobin
pip install -e .
```

For Singularity / Apptainer runs, use `apptainer` with the `--nvccli` flag (different from DEIS-MCC's `--nv`):

```bash
apptainer run --nvccli megobin.sif python megobin/pipeline.py --config-name experiment/training/uncertain_gen_cami_toy
```

This is one of the small but annoying BioCloud-vs-DEIS differences — the GPU runtime flag is not portable. Keep both in your notes or bake a `make biocloud-run` target in.

## Running the full pipeline (Snakemake)

The end-to-end orchestration lives in `hpc/Snakefile` and is run via the `biocloud` Snakemake profile (which sets SLURM defaults). Submit the whole thing as one SLURM job:

```bash
sbatch hpc/slurm/biocloud_pipeline.sh CAMI_medium
```

That script is thin — it just activates the environment and calls:

```bash
snakemake --profile biocloud -s hpc/Snakefile --config datasets="[CAMI_medium]" --jobs 50 --printshellcmds
```

What that produces, for each `(dataset, encoder, binner)` tuple in the config:

1. **`compute_features`** — reads `data/CAMI_medium/contigs.fasta`, computes canonical 4-mer profiles for the whole contigs and for random halves (SemiBin's must-link trick), writes `kmer_profiles.npy`, `kmer_profiles_split.npy`, and `contig_names.npy`.
2. **`train`** — trains the encoder on those profiles, writes `model.pt` to `results/<dataset>/<encoder>/model.pt`.
3. **`encode`** — loads the trained model, runs `model.encode(profiles)`, writes `embeddings.npy`.
4. **`bin`** — loads the embeddings, runs the configured binner, writes `labels.npy`.
5. **`evaluate`** — rehydrates the bins back into FASTA files using `labels.npy` + `contig_names.npy` + the original FASTA, runs `checkm2 predict`, writes `quality_report.tsv`.

Each rule is independently cacheable. If you re-run and only the binner config changed, steps 1–3 get skipped (Snakemake sees the upstream files are unchanged). This is the whole reason Snakemake exists for pipelines like this — it lets you swap binners cheaply.

## Running a single encoder training run

If you just want to train and don't need the full `features → embed → bin → evaluate` pipeline, skip Snakemake and run `pipeline.py` directly via SLURM:

```bash
sbatch --time=02:00:00 --cpus-per-task=8 --mem=32G --gres=gpu:1 --wrap "\
  cd /projects/microbial-dark-matter/metagenomic-binning/Metagenomic-Binning && \
  mamba activate megobin && \
  python megobin/pipeline.py --config-name experiment/training/uncertain_gen_cami_toy \
    logger.name=H1-seed1-uncertain_gen-biocloud"
```

The `logger.name` override is worth always setting on HPC — otherwise every run ends up as `tb` in TensorBoard and you can't tell them apart.

## SLURM notes for BioCloud

The script at `hpc/slurm/biocloud_pipeline.sh` uses these defaults:

```bash
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --mail-type=FAIL
```

Notable about BioCloud vs. DEIS-MCC:

- **No `--partition` in the default script.** BioCloud's default partition is sensible for mixed CPU/GPU work; DEIS-MCC requires you to specify `turing` or `ada`.
- **24-hour wall-time ceiling** is conservative but appropriate — if a run exceeds 24 hours on BioCloud, something's wrong and you should be on DEIS-MCC anyway.
- **8 CPUs is a lot** if you're not doing feature computation. Drop to 4 for pure training.
- `--mail-type=FAIL` — BioCloud's mail relay works; you'll actually get the email. (DEIS-MCC's is flakier.)

## Reading results

Snakemake writes to `results/<dataset>/<encoder>/<binner>/`:

```
results/CAMI_medium/uncertain_gen/infomap/
├── labels.npy
├── checkm2/
│   ├── bins/
│   │   ├── bin_0000.fasta
│   │   └── ...
│   └── quality_report.tsv     # the thing you actually care about
```

`quality_report.tsv` has one row per bin with columns `Name`, `Completeness`, `Contamination`, `Contig_N50`, etc. This is the output of CheckM2's `predict` subcommand, ingested by `megobin/evaluators/checkm2.py` when running via `pipeline.py`. Via Snakemake, the evaluator isn't used — the raw CheckM2 report is your ground truth.

To pull into TensorBoard-style comparison, use `pipeline.py` instead; Snakemake is the "produce the raw artifact for the paper" path, `pipeline.py` is the "log everything to TensorBoard for iteration" path.

## Common BioCloud issues

**"CUDA out of memory" on feature computation.** You're somehow computing features on GPU — the Snakefile routes feature computation to CPU workers. Check you haven't overridden the `compute_features` resource section.

**Singularity flag error (`unknown flag: --nv`).** You're using the DEIS-MCC command on BioCloud. Swap `--nv` for `--nvccli` when calling `apptainer` / `singularity` on BioCloud.

**CheckM2 database missing.** CheckM2 needs a pre-downloaded database; on BioCloud it lives at `/projects/microbial-dark-matter/databases/checkm2/`. Set `CHECKM2DB` in your `.bashrc`:

```bash
export CHECKM2DB=/projects/microbial-dark-matter/databases/checkm2/uniref100.KO.1.dmnd
```

If you see `CheckM2 error: database not found`, this is why.

**Snakemake lockfile error.** Only one Snakemake invocation can touch a given working directory at a time. If a previous run crashed mid-way, run:

```bash
snakemake --unlock -s hpc/Snakefile
```

before resubmitting.
