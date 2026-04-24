# What is MegoBin?

MegoBin is a modular research pipeline for **metagenomic binning**. Given a pile of assembled contigs from an environmental DNA sample, its job is to cluster those contigs so that each cluster (a "bin") corresponds, as closely as possible, to a single microbial genome. The pipeline is not trying to beat VAMB or SemiBin2 on well-studied organisms — it exists to give us a fast iteration loop for **novel deep-learning approaches**, particularly for low-abundance organisms and microbial dark matter that existing tools still struggle with.

## The problem, in one paragraph

When biologists sequence an environmental sample (soil, gut, wastewater), they get a mess of short or long reads mixed from hundreds of different organisms. An **assembler** stitches those reads into longer contiguous sequences — contigs — but a single contig is still just a fragment of one genome, and contigs from different genomes look a lot like each other at the nucleotide level. **Binning** is the clustering step that groups contigs by their genome of origin. The clustering signal comes from two sources: **k-mer composition** (different organisms use different short-oligonucleotide frequencies) and **abundance profiles across samples** (contigs from the same genome co-vary in coverage across different samples, because they physically travel together). State-of-the-art tools combine both signals using learned representations plus a downstream clustering algorithm.

## What MegoBin actually is

A Python package plus configs plus tests, organized as three swappable slots wired together by a single Hydra-driven entry point.

```
Dataset → Features (shared) → Encoder → Trainer → Binner → Evaluator
                                 ↑         ↑
                          Loss, Sampler   Optimizer, Scheduler, Logger
```

The **Encoder** turns features into embeddings. The **Binner** turns embeddings into cluster labels. The **Evaluator** turns cluster labels into completeness/contamination scores via CheckM2. Everything that feeds those three slots — features, losses, pair samplers, trainers, optimizers, schedulers, loggers — is itself a swappable component backed by a Python `Protocol` and composed via YAML.

A single command runs a full experiment:

```bash
python megobin/pipeline.py --config-name experiment/hybrid_uncertain_gen
```

And a single override swaps any component without editing code:

```bash
python megobin/pipeline.py --config-name experiment/hybrid_uncertain_gen \
  encoder=semibin_encoder loss=hinge binner=dbscan_ensemble
```

## What ships in the repo today

Two encoders live in [megobin/encoders/](https://github.com/Tinnifo/Metagenomic-Binning/tree/main/megobin/encoders):

**UncertainGen** is a dual-head MLP (~526K parameters) that emits both a mean and a diagonal covariance per contig. It is trained with a Mahalanobis BCE loss over a two-phase schedule — train the mean head first for 50 epochs, then freeze it and train the covariance head for 25 more. Embedding dim is 256. This is our primary architecture.

**SemiBin** is a reimplementation of SemiBin2's Siamese encoder: a 3-layer MLP (136→512→512→100) trained with a hinge contrastive loss on must-link / cannot-link pairs. Single-phase training, 15 epochs. Embedding dim is 100. This is our baseline.

Two binners live in [megobin/binners/](https://github.com/Tinnifo/Metagenomic-Binning/tree/main/megobin/binners):

**Infomap** (used for SemiBin short reads) builds a dual k-nearest-neighbour graph — one over embeddings, one over raw k-mer profiles — intersects them, and runs Infomap community detection with 10 trials.

**DBSCAN ensemble** (used for SemiBin long reads) runs DBSCAN at 12 different `eps` values (0.01 through 0.55) and picks the best bins via a marker-gene F1 score.

One evaluator lives in [megobin/evaluators/checkm2.py](https://github.com/Tinnifo/Metagenomic-Binning/blob/main/megobin/evaluators/checkm2.py): a subprocess wrapper around CheckM2 that returns a pandas DataFrame of completeness and contamination per bin.

Plus three pair samplers (UncertainGen, SemiBin, Hybrid), two trainers (SinglePhase and TwoPhase), and a pluggable Logger with TensorBoard and NoOp implementations.

## What MegoBin is not

It is not a production tool. It is not a maintained release. It is not a VAMB/SemiBin2 replacement — those are tuned and battle-tested on specific benchmarks, and MegoBin's implementations of their ideas are research reproductions, not the canonical versions. And it is not a black box: every component is ~100 lines of readable Python, and Chapter 3 of this guide walks through the full pipeline from CLI to evaluation.

The goal is fast iteration. Adding a new encoder or loss or binner should mean **one file, one YAML, one test** — not a refactor of the pipeline. The rest of this guide is about how the architecture delivers on that goal, and how you work inside it.
