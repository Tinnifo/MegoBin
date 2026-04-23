# MegoBin — Collaborator's Guide

Welcome. This guide walks you end-to-end through the **MegoBin** research pipeline: what it is, how to install it, how to run your first experiment, how to swap components, and how to add your own. It is written for collaborators on the Metagenomic-Binning project — Thomas, Kadir, Mads, Sebastian, Morten, and anyone else who clones the repo and needs to be productive in an afternoon.

The guide assumes you know roughly what metagenomic binning is (clustering assembled contigs into bins, one bin per microbial genome) and have touched Python + PyTorch before. You do **not** need prior knowledge of Hydra, Singularity, or the AAU HPC clusters — those are covered here.

## How to use this guide

Every chapter is a tutorial, not a reference dump. It is meant to be read in order the first time through, and dipped into later when you hit a specific question. The chapters are grouped into three layers:

**Layer 1 — get it running** (Chapters 1–2). Understand the problem MegoBin is trying to solve, install it, and execute your first experiment in roughly ten minutes.

**Layer 2 — understand the shape** (Chapter 3). Internalize the slot-based architecture so that every later tutorial is obvious rather than mysterious.

**Layer 3 — make it your own** (Chapters 4–7). Swap components, add new ones, wire up experiment tracking, submit jobs on the cluster, and write tests that prevent regressions.

Chapter 8 is pure reference — a config dictionary, FAQ, and troubleshooting notes. It exists for lookups, not for reading cover-to-cover.

## Where the code lives

The canonical source is [github.com/Tinnifo/Metagenomic-Binning](https://github.com/Tinnifo/Metagenomic-Binning). Every code path in this guide refers to files in that repository; wherever possible, a file path is followed by line numbers so you can verify what the guide claims against the real source.

## A note on scope

MegoBin is an **active research pipeline**, not a production tool. Chapters describe the code as it exists today. When the code changes, this guide changes too — consider the `docs/` folder in the repo the source of truth, and this GitBook a rendered view of it. Pull requests that change code without updating the matching chapter will fail review.
