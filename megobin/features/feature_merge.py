# This script has been adapted from the file available at the following address:
# https://github.com/BigDataBiology/SemiBin/blob/main/SemiBin/main.py

import sys
import os
from os import path
from itertools import groupby
import contextlib
import multiprocessing as mp

from megobin.features.generate_kmers import generate_kmer_features_from_fasta
from megobin.features.generate_abundance import (
    combine_cov,
    generate_cov,
    generate_cov_from_abundances,
)
from megobin.utils.SemiBin_utils import (
    get_must_link_threshold,
    load_fasta,
    split_data,
)
from megobin.utils.atomicwrite import atomic_write
from megobin.utils.fasta import fasta_iter


# SemiBin switches to combined (multi-sample) mode once at least this many
# samples are present: coverage is embedded directly and must-link split
# coverage is produced. The strobealign-aemb abundance input also requires
# this many samples.
COMBINED_MODE_MIN_SAMPLES = 5


# Helper functions from: https://github.com/BigDataBiology/SemiBin/blob/main/SemiBin/utils.py
@contextlib.contextmanager
def possibly_compressed_write(filename):
    if filename.endswith(".gz"):
        import gzip

        mode = "wb"
        transf = lambda f: gzip.open(f, mode="wt")
    elif filename.endswith(".bz2"):
        import bz2

        mode = "wb"
        transf = lambda f: bz2.open(f, mode="wt")
    elif filename.endswith(".xz"):
        import lzma

        mode = "wb"
        transf = lambda f: lzma.open(f, mode="wt")
    else:
        mode = "wt"
        transf = lambda f: f
    with atomic_write(filename, mode=mode, overwrite=True) as f:
        g = transf(f)
        yield g
        if g is not f:
            g.close()


def maybe_compute_min_length(min_length, fafile, ratio):
    if min_length is not None:
        return min_length
    c_min_len, _, _ = load_fasta(fafile, ratio)
    return c_min_len


Pool = mp.get_context("spawn").Pool


def generate_sequence_features_single(
    logger,
    contig_fasta,
    bams,
    binned_length,
    must_link_threshold,
    num_process,
    output,
    abundances=None,
    only_kmer=False,
):
    """Generate data.csv and data_split.csv for single-sample binning.

    Ported from SemiBin's ``generate_sequence_features_single``
    (https://github.com/BigDataBiology/SemiBin/blob/main/SemiBin/main.py).
    A single assembly with bare contig names (no sample separator); BAM
    references match those bare names (``sep=None``). ``is_combined`` is
    ``len(bams) >= COMBINED_MODE_MIN_SAMPLES`` — combined mode writes one
    coverage column per BAM,
    single-sample mode writes ``[mean, var]`` per BAM.

    data.csv has the (kmer + abundance) features for the original contigs;
    data_split.csv has them for the must-link split halves.
    """
    import pandas as pd

    if bams is None and abundances is None and not only_kmer:
        raise ValueError(
            "Need BAM files or abundance files to calculate coverage features."
        )

    logger.debug("Generating kmer features from fasta file.")
    kmer_whole = generate_kmer_features_from_fasta(contig_fasta, binned_length, 4)
    if only_kmer:
        with atomic_write(os.path.join(output, "data.csv"), overwrite=True) as ofile:
            kmer_whole.to_csv(ofile)
        return

    kmer_split = generate_kmer_features_from_fasta(
        contig_fasta, 1000, 4, split=True, split_threshold=must_link_threshold
    )

    is_combined = False
    data_cov = None
    data_split_cov = None

    if bams:
        is_combined = len(bams) >= COMBINED_MODE_MIN_SAMPLES
        logger.info("Calculating coverage for every BAM.")
        with Pool(min(max(num_process, 1), len(bams))) as pool:
            results = [
                pool.apply_async(
                    generate_cov,
                    args=(
                        bam_file,
                        bam_index,
                        output,
                        must_link_threshold,
                        is_combined,
                        binned_length,
                        logger,
                        None,
                    ),
                )
                for bam_index, bam_file in enumerate(bams)
            ]
            for r in results:
                logger.info(f"Processed: {r.get()}")
        data_cov, data_split_cov = combine_cov(output, bams, is_combined)

    if abundances:
        if len(abundances) < COMBINED_MODE_MIN_SAMPLES:
            raise ValueError(
                f"abundances (strobealign-aemb) require at least "
                f"{COMBINED_MODE_MIN_SAMPLES} samples."
            )
        logger.info("Reading abundance information from abundance files.")
        data_cov, data_split_cov = generate_cov_from_abundances(
            abundances, output, contig_fasta, binned_length
        )
        is_combined = True

    if is_combined:
        data_split = pd.merge(
            kmer_split,
            data_split_cov,
            how="inner",
            on=None,
            left_index=True,
            right_index=True,
            sort=False,
            copy=True,
        )
    else:
        data_split = kmer_split

    kmer_whole.index = kmer_whole.index.astype(str)
    data = pd.merge(
        kmer_whole,
        data_cov,
        how="inner",
        on=None,
        left_index=True,
        right_index=True,
        sort=False,
        copy=True,
    )

    with atomic_write(os.path.join(output, "data.csv"), overwrite=True) as ofile:
        data.to_csv(ofile)
    with atomic_write(os.path.join(output, "data_split.csv"), overwrite=True) as ofile:
        data_split.to_csv(ofile)


def generate_sequence_features_multi(logger, args):
    """
    Generate data.csv and data_split.csv for every sample of multi-sample binning mode.
    data.csv has the (kmer and abundance) features for the original contigs.
    data_split.csv has them for the must-link split halves.
    """
    import pandas as pd

    if not args.bams and not args.abundances:
        logger.error(
            f"Error: You need to specify input BAM files or abundance files.\n"
        )
        sys.exit(1)

    n_sample = len(args.bams) if args.bams else len(args.abundances)
    if args.abundances and n_sample < COMBINED_MODE_MIN_SAMPLES:
        logger.error(
            f"Error: abundances from strobealign-aemb can only be used when at least {COMBINED_MODE_MIN_SAMPLES} samples are used.\n"
        )
        sys.exit(1)

    is_combined = n_sample >= COMBINED_MODE_MIN_SAMPLES

    sample_list = []
    contig_lengths = []

    os.makedirs(os.path.join(args.output, "samples"), exist_ok=True)

    def fasta_sample_iter(fn):
        for h, seq in fasta_iter(fn):
            if args.separator not in h:
                raise ValueError(
                    f"Expected contigs to contain separator character ({args.separator}), found {h}"
                )
            sample_name, contig_name = h.split(args.separator, 1)
            yield sample_name, contig_name, seq

    for sample_name, contigs in groupby(
        fasta_sample_iter(args.contig_fasta), lambda sn_cn_seq: sn_cn_seq[0]
    ):
        with possibly_compressed_write(
            os.path.join(args.output, "samples", f"{sample_name}.fa")
        ) as out:
            for _, contig_name, seq in contigs:
                out.write(f">{contig_name}\n{seq}\n")
                contig_lengths.append(len(seq))
        sample_list.append(sample_name)
    if len(sample_list) != len(set(sample_list)):
        logger.error(
            f"Concatenated FASTA file {args.contig_fasta} not in expected format. Samples should follow each other."
        )
        sys.exit(1)

    must_link_threshold = (
        get_must_link_threshold(contig_lengths)
        if args.ml_threshold is None
        else args.ml_threshold
    )
    binning_threshold = {}
    for sample in sample_list:
        binning_threshold[sample] = maybe_compute_min_length(
            args.min_len, os.path.join(args.output, f"samples/{sample}.fa"), args.ratio
        )

    if args.bams:
        logger.info("Calculating coverage for every sample.")
        with Pool(min(args.num_process, len(args.bams))) as pool:
            results = [
                pool.apply_async(
                    generate_cov,
                    args=(
                        bam_file,
                        bam_index,
                        os.path.join(args.output, "samples"),
                        must_link_threshold,
                        is_combined,
                        binning_threshold,
                        logger,
                        args.separator,
                    ),
                )
                for bam_index, bam_file in enumerate(args.bams)
            ]
            for r in results:
                s = r.get()
                logger.info(f"Processed: {s}")

        for bam_index, bam_file in enumerate(args.bams):
            if not path.exists(
                path.join(
                    args.output,
                    "samples",
                    f"{path.split(bam_file)[-1]}_{bam_index}_data_cov.csv",
                )
            ):
                sys.stderr.write(
                    f"Error: Generating coverage file failed (for BAM file {bam_file})\n"
                )
                sys.exit(1)
            if is_combined:
                if not path.exists(
                    path.join(
                        args.output,
                        "samples",
                        f"{path.split(bam_file)[-1]}_{bam_index}_data_split_cov.csv",
                    )
                ):
                    sys.stderr.write(
                        f"Error: Generating split coverage file failed (for BAM file {bam_file})\n"
                    )
                    sys.exit(1)

        data_cov, data_split_cov = combine_cov(
            os.path.join(args.output, "samples"), args.bams, is_combined
        )
        if is_combined:
            data_split_cov = data_split_cov.reset_index()
            columns_list = list(data_split_cov.columns)
            columns_list[0] = "contig_name"
            data_split_cov.columns = columns_list

        data_cov = data_cov.reset_index()
        columns_list = list(data_cov.columns)
        columns_list[0] = "contig_name"
        data_cov.columns = columns_list

        for sample in sample_list:
            output_path = os.path.join(args.output, "samples", sample)
            os.makedirs(output_path, exist_ok=True)

            part_data = split_data(data_cov, sample, args.separator, is_combined)
            part_data.to_csv(os.path.join(output_path, "data_cov.csv"))

            if is_combined:
                part_data = split_data(
                    data_split_cov, sample, args.separator, is_combined
                )
                part_data.to_csv(os.path.join(output_path, "data_split_cov.csv"))

            sample_contig_fasta = os.path.join(args.output, f"samples/{sample}.fa")
            kmer_whole = generate_kmer_features_from_fasta(
                sample_contig_fasta, binning_threshold[sample], 4
            )
            kmer_split = generate_kmer_features_from_fasta(
                sample_contig_fasta,
                1000,
                4,
                split=True,
                split_threshold=must_link_threshold,
            )

            sample_cov = pd.read_csv(
                os.path.join(output_path, "data_cov.csv"), index_col=0
            )
            kmer_whole.index = kmer_whole.index.astype(str)
            sample_cov.index = sample_cov.index.astype(str)
            data = pd.merge(
                kmer_whole,
                sample_cov,
                how="inner",
                on=None,
                left_index=True,
                right_index=True,
                sort=False,
                copy=True,
            )
            if is_combined:
                sample_cov_split = pd.read_csv(
                    os.path.join(output_path, "data_split_cov.csv"), index_col=0
                )
                data_split = pd.merge(
                    kmer_split,
                    sample_cov_split,
                    how="inner",
                    on=None,
                    left_index=True,
                    right_index=True,
                    sort=False,
                    copy=True,
                )
            else:
                data_split = kmer_split

            with atomic_write(
                os.path.join(output_path, "data.csv"), overwrite=True
            ) as ofile:
                data.to_csv(ofile)

            with atomic_write(
                os.path.join(output_path, "data_split.csv"), overwrite=True
            ) as ofile:
                data_split.to_csv(ofile)

    if args.abundances:
        logger.info("Reading abundance information from abundance files.")
        abun_split = generate_cov_from_abundances(
            args.abundances,
            os.path.join(args.output, "samples"),
            contig_path=args.contig_fasta,
            sep=args.separator,
            sample_contig_threshold=binning_threshold,
        )
        abun_split = abun_split.reset_index()
        columns_list = list(abun_split.columns)
        columns_list[0] = "contig_name"
        abun_split.columns = columns_list

        for sample in sample_list:
            output_path = os.path.join(args.output, "samples", sample)
            os.makedirs(output_path, exist_ok=True)
            part_data_split = split_data(
                abun_split, sample, args.separator, is_combined=True
            )
            part_data_split.to_csv(os.path.join(output_path, "data_split_cov.csv"))

            index_name = part_data_split.index.tolist()
            data_index_name = []
            for t in range(len(index_name) // 2):
                data_index_name.append(index_name[2 * t][0:-2])
            part_data_split_values = part_data_split.values

            part_data = (part_data_split_values[::2] + part_data_split_values[1::2]) / 2
            columns = [f"{abun}" for abun in args.abundances]
            part_data = pd.DataFrame(part_data, index=data_index_name, columns=columns)
            part_data.to_csv(os.path.join(output_path, "data_cov.csv"))

            sample_contig_fasta = os.path.join(args.output, f"samples/{sample}.fa")
            kmer_whole = generate_kmer_features_from_fasta(
                sample_contig_fasta, binning_threshold[sample], 4
            )
            kmer_split = generate_kmer_features_from_fasta(
                sample_contig_fasta,
                1000,
                4,
                split=True,
                split_threshold=must_link_threshold,
            )
            data = pd.merge(
                kmer_whole,
                part_data,
                how="inner",
                on=None,
                left_index=True,
                right_index=True,
                sort=False,
                copy=True,
            )
            data_split = pd.merge(
                kmer_split,
                part_data_split,
                how="inner",
                on=None,
                left_index=True,
                right_index=True,
                sort=False,
                copy=True,
            )

            with atomic_write(
                os.path.join(output_path, "data.csv"), overwrite=True
            ) as ofile:
                data.to_csv(ofile)

            with atomic_write(
                os.path.join(output_path, "data_split.csv"), overwrite=True
            ) as ofile:
                data_split.to_csv(ofile)

    return sample_list
