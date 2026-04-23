from pathlib import Path

import numpy as np


def compute_abundance(
    bam_paths: list[Path],
    contig_names: list[str],
) -> np.ndarray:
    """Compute mean and variance of per-base coverage depth from BAM files.

    Args:
        bam_paths: Sorted, indexed BAM files.
        contig_names: Contigs to compute coverage for (must exist in BAM headers).

    Returns:
        (N, 2 * num_bams) float64 array — columns are
        [mean_1, var_1, mean_2, var_2, ...] for each BAM.
    """
    import pysam  # deferred: only needed on HPC where pysam is installed

    num_contigs = len(contig_names)
    num_bams = len(bam_paths)
    abundance = np.zeros((num_contigs, 2 * num_bams), dtype=np.float64)

    for bam_idx, bam_path in enumerate(bam_paths):
        with pysam.AlignmentFile(str(bam_path), "rb") as bam:
            for contig_idx, contig in enumerate(contig_names):
                contig_length = bam.get_reference_length(contig)
                if contig_length == 0:
                    continue

                # count_coverage returns four arrays (A, C, G, T) of per-base counts
                a, c, g, t = bam.count_coverage(
                    contig, start=0, stop=contig_length
                )
                depths = (
                    np.array(a, dtype=np.float64)
                    + np.array(c, dtype=np.float64)
                    + np.array(g, dtype=np.float64)
                    + np.array(t, dtype=np.float64)
                )

                abundance[contig_idx, 2 * bam_idx] = depths.mean()
                abundance[contig_idx, 2 * bam_idx + 1] = depths.var()

    return abundance
