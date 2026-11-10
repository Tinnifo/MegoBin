"""Encoder-side views of the feature matrix.

These hold the SemiBin-specific preprocessing that used to sit in
``pipeline.py`` (the ``norm_abundance`` gate and the combined-mode
normalisation). Each encoder returns the view it needs from
``Encoder.feature_view()``; the embedder stage applies it.

A view maps the raw ``[k-mer | depth]`` matrix to the matrix the encoder is
trained on and encodes. ``transform_split`` maps ``data_split.csv`` rows the
same way, using the state fitted on the whole matrix.
"""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

import numpy as np

log = logging.getLogger(__name__)


@runtime_checkable
class FeatureView(Protocol):
    def fit_transform(self, x: np.ndarray, kmer_dim: int) -> np.ndarray: ...

    def transform_split(self, x_split: np.ndarray) -> np.ndarray: ...


class FullMatrixView:
    """Pass the whole matrix (k-mer + depth) to the encoder unchanged.

    For encoders that slice the k-mer block and read the depth columns
    themselves (SemiBin2 single-sample long-read mode).
    """

    def fit_transform(self, x: np.ndarray, kmer_dim: int) -> np.ndarray:
        n_abund = x.shape[1] - kmer_dim
        if n_abund > 0:
            log.info(
                "depth-consuming encoder: passing full feature matrix "
                "(%d abundance cols) to encode()",
                n_abund,
            )
        return x

    def transform_split(self, x_split: np.ndarray) -> np.ndarray:
        return x_split


class KmerGateView:
    """Mirror SemiBin's ``norm_abundance`` gate for k-mer-only encoders.

    Drops the abundance columns when too few BAMs make them uninformative, so
    the encoder's inference dimension matches the training dimension
    (``data_split.csv`` is k-mer-only in single-sample mode).
    """

    def fit_transform(self, x: np.ndarray, kmer_dim: int) -> np.ndarray:
        n_abund = x.shape[1] - kmer_dim
        if n_abund <= 0:
            return x
        from megobin.utils.SemiBin_utils import norm_abundance

        if norm_abundance(x):
            log.info("norm_abundance gate: keeping %d abundance cols", n_abund)
            return x
        log.warning(
            "norm_abundance gate: %d abundance cols below threshold — "
            "dropping abundance to match data_split.csv (kmer-only). "
            "Re-run feature_merge in multi-sample mode (≥5 BAMs).",
            n_abund,
        )
        return x[:, :kmer_dim]

    def transform_split(self, x_split: np.ndarray) -> np.ndarray:
        return x_split


class CombinedNormView:
    """SemiBin combined mode (``train_self`` / ``cluster_long_read``).

    Normalise the whole matrix once by column sums, then L1 row-normalise.
    The same column sums are reused for ``data_split.csv`` so train and
    inference scaling stay identical. Applies only when the ``norm_abundance``
    gate passes; otherwise the full matrix is passed unchanged.
    """

    def __init__(self) -> None:
        self.column_sums: np.ndarray | None = None

    def fit_transform(self, x: np.ndarray, kmer_dim: int) -> np.ndarray:
        n_abund = x.shape[1] - kmer_dim
        if n_abund <= 0:
            return x
        from megobin.utils.SemiBin_utils import norm_abundance

        if norm_abundance(x):
            from sklearn.preprocessing import normalize

            self.column_sums = x.sum(axis=0)
            out = normalize(x / self.column_sums, axis=1, norm="l1").astype(np.float32)
            log.info(
                "combined-mode abundance normalisation applied (%d abundance cols)",
                n_abund,
            )
            return out
        log.info(
            "depth-consuming encoder: passing full feature matrix "
            "(%d abundance cols) to encode()",
            n_abund,
        )
        return x

    def transform_split(self, x_split: np.ndarray) -> np.ndarray:
        if self.column_sums is None:
            return x_split
        from sklearn.preprocessing import normalize

        return normalize(x_split / self.column_sums, axis=1, norm="l1").astype(
            np.float32
        )
