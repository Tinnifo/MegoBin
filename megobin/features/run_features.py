"""CLI: generate SemiBin2 features from a single assembly + BAM(s).

Produces ``data.csv``, ``data_split.csv`` and ``contigs.fasta`` in the output
directory — exactly the inputs ``megobin/pipeline.py`` consumes. Mirrors
SemiBin2's ``single_easy_bin`` feature step (single assembly, bare contig
names, BAM references matching those names). ``is_combined`` is decided by the
number of BAMs (``>= 5`` → combined / multi-sample-abundance layout).

Usage::

    python -m megobin.features.run_features \
        --contigs contigs.fasta --bams sample.bam --out data/ds_single

    # combined mode (>=5 BAMs of the same assembly):
    python -m megobin.features.run_features \
        --contigs contigs.fasta --bams s1.bam s2.bam s3.bam s4.bam s5.bam \
        --out data/ds_combined
"""

import argparse
import logging
import os
import shutil

from megobin.features.feature_merge import generate_sequence_features_single
from megobin.utils.SemiBin_utils import load_fasta


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--contigs", required=True, help="Assembly FASTA (bare contig names).")
    p.add_argument(
        "--bams", nargs="+", default=None, help="Sorted/indexed BAM file(s)."
    )
    p.add_argument(
        "--abundances",
        nargs="+",
        default=None,
        help="Pre-computed abundance files (strobealign-aemb; >=5 samples).",
    )
    p.add_argument("--out", required=True, help="Output dataset directory.")
    p.add_argument(
        "--min-len",
        type=int,
        default=None,
        help="Minimum contig length; default auto (load_fasta + ratio).",
    )
    p.add_argument(
        "--ratio",
        type=float,
        default=0.05,
        help="Ratio for the 1000/2500 bp min-length switch (default 0.05).",
    )
    p.add_argument(
        "--ml-threshold",
        type=int,
        default=None,
        help="Must-link split threshold (bp); default auto (2%% cumsum, >=4000).",
    )
    p.add_argument("--num-process", type=int, default=1, help="Worker processes.")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    logger = logging.getLogger("megobin.features")

    os.makedirs(args.out, exist_ok=True)

    computed_min, ml_thr, _ = load_fasta(args.contigs, args.ratio)
    binned_length = args.min_len if args.min_len is not None else computed_min
    must_link_threshold = (
        args.ml_threshold if args.ml_threshold is not None else ml_thr
    )
    logger.info(
        "binned_length=%s, must_link_threshold=%s, n_bams=%s",
        binned_length,
        must_link_threshold,
        len(args.bams) if args.bams else 0,
    )

    generate_sequence_features_single(
        logger,
        args.contigs,
        args.bams,
        binned_length,
        must_link_threshold,
        args.num_process,
        args.out,
        abundances=args.abundances,
    )

    # The pipeline reads contigs.fasta from the dataset dir (binner markers +
    # bin writing). Provide it next to data.csv with the same bare names.
    dst = os.path.join(args.out, "contigs.fasta")
    if os.path.abspath(args.contigs) != os.path.abspath(dst):
        shutil.copyfile(args.contigs, dst)

    logger.info("Wrote data.csv, data_split.csv, contigs.fasta to %s", args.out)


if __name__ == "__main__":
    main()
