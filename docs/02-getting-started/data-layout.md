# Data layout

A dataset on disk is three files in one directory, pointed at by `configs/dataset/<name>.yaml`:

```
data/example/
├── data.csv          (N_contigs, kmer + abundance)        whole-contig features
├── data_split.csv    (2 * N_contigs, kmer [+ abundance])  per-half features for pair sampling
└── contigs.fasta                                          one record per contig
```

`pipeline.py` loads `data.csv` for inference, `data_split.csv` for training, and `contigs.fasta` for the marker-aware binner.

## `data.csv`

Indexed by contig name. Columns: 136 canonical 4-mer frequencies followed by `2 * num_bams` abundance columns (mean and var per BAM). Produced by [megobin/features/feature_merge.py](../../megobin/features/feature_merge.py).

## `data_split.csv`

Indexed by `<contig>_1` / `<contig>_2` (the two halves of each contig). 136 kmer columns. **Single-sample mode appends no abundance here** — see [feature_merge.py:233-248](../../megobin/features/feature_merge.py#L233-L248), abundance is only joined into splits when `is_combined=True` (multi-sample, ≥2 BAMs).

`pipeline._load_data_split_csv` re-orders this from feature_merge's interleaved `[c1_1, c1_2, c2_1, c2_2, …]` into the `[left_halves; right_halves]` layout the pair samplers expect.

## The `norm_abundance` gate

Whole-contig features can be 138 dims (kmer + 1 BAM × 2) while splits are 136 dims (kmer only). The encoder needs the same `input_dim` for training and inference, so `pipeline.py` mirrors SemiBin's [`norm_abundance`](../../megobin/utils/SemiBin_utils.py#L650) heuristic:

| n_abund_cols | mean(sum) | Keep abundance? |
|--------------|-----------|-----------------|
| ≥ 20         | any       | yes             |
| 5..19        | > 2       | yes             |
| 5..19        | ≤ 2       | no              |
| < 5          | any       | no              |

When the gate returns False, `pipeline.py` trims `data.csv` to the kmer-only width and logs:

```
norm_abundance gate: 2 abundance cols below threshold — dropping abundance
to match data_split.csv (kmer-only). Re-run feature_merge in multi-sample mode
(≥5 BAMs) to use abundance.
```

Want abundance to actually contribute? See [Linear OPE2-109](https://linear.app/tinnifo/issue/OPE2-109/multi-bam-abundance-pipeline-matching-semibin2-setup) — generate features with ≥5 BAMs and `feature_merge --bams ...` so both files end up at matching `136 + 2 * n_bams` width.

## Dataset config

```yaml
# configs/dataset/example.yaml
name: example
path: data/example/
signals: [kmers, abundance]
num_bams: 1
```

`signals` is a capability descriptor — the feature config declares `required_signals: [kmers, abundance]` and `_check_signal_compatibility` aborts before any I/O if the dataset can't supply them.

`pipeline.py` reads `cfg.dataset.path` directly, so any directory with the three files above will work.
