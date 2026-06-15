# This module has been adapted from SemiBin's self-supervised encoder and its
# long-read clustering preprocessing:
#   https://github.com/BigDataBiology/SemiBin/blob/main/SemiBin/semi_supervised_model.py
#   https://github.com/BigDataBiology/SemiBin/blob/main/SemiBin/long_read_cluster.py

import logging

import numpy as np
import torch
import torch.nn as nn

from megobin.losses.base import ContrastiveLoss

logger = logging.getLogger(__name__)


class SemiBin2Encoder(nn.Module):
    """SemiBin2 self-supervised encoder for the long-read DBSCAN path.

    The network is SemiBin's ``Semi_encoding_single.encoder1`` (and the
    identical ``Semi_encoding_multiple`` encoder for combined mode)::

        Linear(input_dim -> 512) -> BatchNorm1d -> LeakyReLU -> Dropout(0.2)
        Linear(512 -> 512)       -> BatchNorm1d -> LeakyReLU -> Dropout(0.2)
        Linear(512 -> output_dim)   # output_dim = 100

    ``encode`` reproduces SemiBin's ``cluster_long_read`` preprocessing so
    the representation handed to :class:`DBSCANEnsembleBinner` matches what
    SemiBin feeds to its kNN graph + DBSCAN ensemble:

    - **Single-sample** (``is_combined=False``): the network embeds the
      k-mer block only (``features[:, :kmer_dim]``); the per-sample *mean*
      coverage (even columns of the ``[mean, var, ...]`` depth block) is
      ``log(clip(., 1e-6))``-transformed and concatenated to the
      embedding → ``[emb | log_depth]`` of width ``output_dim + n_sample``.
    - **Combined** (``is_combined=True``, >=5 samples): the network embeds
      the full ``kmer + abundance`` matrix and the embedding is returned
      as-is (no depth concat). The abundance column+L1 normalisation that
      SemiBin applies in this mode is done once, globally, in
      ``pipeline.py`` (over the whole feature matrix, exactly like
      ``train_self``), so it is *not* repeated here.

    The output is intentionally **not** L2-normalised — SemiBin's eps
    sweep is tuned to the raw embedding (+ log-depth) scale.

    ``consumes_depth = True`` tells the pipeline to pass the full
    ``data.csv`` matrix (k-mer + abundance) to ``encode`` instead of
    dropping abundance via the ``norm_abundance`` gate.
    """

    #: Pipeline reads this to skip the abundance-dropping gate (see pipeline.py).
    consumes_depth = True

    def __init__(
        self,
        input_dim: int = 136,
        kmer_dim: int = 136,
        hidden_dim: int = 512,
        output_dim: int = 100,
        dropout: float = 0.2,
        is_combined: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.kmer_dim = kmer_dim
        self._output_dim = output_dim
        self.is_combined = is_combined

        self.encoder1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder1(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder1(x)

    def _net_input(self, x: torch.Tensor) -> torch.Tensor:
        """The slice the network actually consumes.

        Single-sample feeds the k-mer block only; combined feeds the full
        (already-normalised) k-mer + abundance matrix.
        """
        if self.is_combined:
            return x
        return x[:, : self.kmer_dim]

    def encode(self, features: np.ndarray) -> np.ndarray:
        """(N, feature_dim) → (N, output_dim [+ n_sample] ) clustering coords."""
        self.eval()
        with torch.no_grad():
            x = torch.from_numpy(np.asarray(features)).float()
            emb = self.encoder1(self._net_input(x)).cpu().numpy()

        if self.is_combined:
            # Abundance already lives in the embedding; SemiBin does no concat.
            return emb

        n_depth = features.shape[1] - self.kmer_dim
        if n_depth <= 0:
            logger.warning(
                "SemiBin2Encoder.encode received k-mer-only features (%d cols); "
                "skipping long-read log-depth augmentation. Ensure abundance "
                "columns reach the encoder (consumes_depth gate in pipeline.py).",
                features.shape[1],
            )
            return emb

        # SemiBin cluster_long_read: per-sample MEAN coverage = even columns
        # of the [mean, var, ...] depth block; variance columns are dropped.
        depth = np.asarray(features[:, self.kmer_dim :], dtype=np.float32)
        mean_depth = depth[:, ::2]
        log_depth = np.log(np.clip(mean_depth, 1e-6, None))
        return np.concatenate([emb, log_depth], axis=1)

    @property
    def embedding_dim(self) -> int:
        # Network output width. encode() returns this (+ n_sample in
        # single-sample mode after the log-depth concat); nothing downstream
        # reads this property (the binner uses embeddings.shape directly).
        return self._output_dim

    def training_step(
        self,
        batch: tuple[torch.Tensor, ...],
        loss_fn: ContrastiveLoss,
    ) -> torch.Tensor:
        """Forward both branches → ``loss_fn(z_i, z_j, label)``.

        Pairs come from ``data_split.csv`` (k-mer-only in single-sample
        mode; k-mer + abundance, pre-normalised, in combined mode), so the
        per-mode ``_net_input`` slice keeps train and inference dims aligned.
        """
        x_i, x_j, label = batch
        z_i = self.encoder1(self._net_input(x_i))
        z_j = self.encoder1(self._net_input(x_j))
        return loss_fn(z_i, z_j, label.float())

    def parameter_groups(self) -> dict[str, list[nn.Parameter]]:
        return {"all": list(self.parameters())}
