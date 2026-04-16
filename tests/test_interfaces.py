"""Protocol compliance tests for all components.

Every component must satisfy its Protocol:
- Representation.encode() returns (N, embedding_dim) ndarray
- ContrastiveLoss.__call__() returns a scalar tensor with gradients
- Binner.cluster() returns (N,) integer ndarray
- Evaluator.score() returns DataFrame with completeness & contamination
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch

from src.binners.kmedoids import KMedoidsBinner
from src.evaluators.checkm2 import CheckM2Evaluator
from src.losses.poisson_nll import PoissonNLLLoss
from src.representations.poisson import PoissonRepresentation


# ---- Representation --------------------------------------------------------


class TestPoissonRepresentation:
    def setup_method(self):
        self.model = PoissonRepresentation(num_kmers=256, embedding_dim=128)

    def test_encode_shape(self):
        profiles = np.random.dirichlet(np.ones(256), size=20)
        z = self.model.encode(profiles)
        assert z.shape == (20, 128)

    def test_encode_dtype(self):
        profiles = np.random.dirichlet(np.ones(256), size=5)
        z = self.model.encode(profiles)
        assert z.dtype == np.float64 or z.dtype == np.float32

    def test_embedding_dim(self):
        assert self.model.embedding_dim == 128

    def test_single_sample(self):
        profiles = np.random.dirichlet(np.ones(256), size=1)
        z = self.model.encode(profiles)
        assert z.shape == (1, 128)


# ---- ContrastiveLoss -------------------------------------------------------


class TestPoissonNLLLoss:
    def setup_method(self):
        self.loss_fn = PoissonNLLLoss()

    def test_returns_scalar(self):
        z_i = torch.randn(32, 64, requires_grad=True)
        z_j = torch.randn(32, 64, requires_grad=True)
        label = torch.rand(32)
        loss = self.loss_fn(z_i, z_j, label)
        assert loss.shape == ()

    def test_has_grad(self):
        z_i = torch.randn(16, 64, requires_grad=True)
        z_j = torch.randn(16, 64, requires_grad=True)
        label = torch.rand(16)
        loss = self.loss_fn(z_i, z_j, label)
        loss.backward()
        assert z_i.grad is not None
        assert z_j.grad is not None

    def test_callable(self):
        """Loss satisfies Protocol by being callable."""
        assert callable(self.loss_fn)


# ---- Binner ----------------------------------------------------------------


class TestKMedoidsBinner:
    def setup_method(self):
        self.binner = KMedoidsBinner(min_bin_size=5)

    def test_cluster_returns_1d_int(self):
        embeddings = np.random.randn(50, 10)
        labels = self.binner.cluster(embeddings)
        assert labels.ndim == 1
        assert labels.shape[0] == 50
        assert np.issubdtype(labels.dtype, np.integer)

    def test_all_points_assigned(self):
        embeddings = np.random.randn(30, 8)
        labels = self.binner.cluster(embeddings)
        assert len(labels) == 30
        # Every point should have a non-negative label (no -1 remaining)
        assert (labels >= 0).all()


# ---- Evaluator -------------------------------------------------------------


class TestCheckM2Evaluator:
    def test_score_returns_correct_columns(self):
        """Mock subprocess to test TSV parsing path."""
        evaluator = CheckM2Evaluator(threads=1)

        tsv_content = (
            "Name\tCompleteness\tContamination\n"
            "bin_001\t95.0\t1.5\n"
            "bin_002\t80.0\t2.0\n"
        )

        def fake_run(cmd, **kwargs):
            # Write a fake quality_report.tsv into the output dir
            out_dir = Path(cmd[cmd.index("-o") + 1])
            (out_dir / "quality_report.tsv").write_text(tsv_content)

        with patch("subprocess.run", side_effect=fake_run):
            df = evaluator.score(Path("/fake/bins"))

        assert isinstance(df, pd.DataFrame)
        assert "completeness" in df.columns
        assert "contamination" in df.columns
        assert len(df) == 2

    def test_score_missing_column_raises(self):
        evaluator = CheckM2Evaluator()

        bad_tsv = "Name\tSomethingElse\nbin_001\t42\n"

        def fake_run(cmd, **kwargs):
            out_dir = Path(cmd[cmd.index("-o") + 1])
            (out_dir / "quality_report.tsv").write_text(bad_tsv)

        with patch("subprocess.run", side_effect=fake_run):
            with pytest.raises(KeyError, match="completeness"):
                evaluator.score(Path("/fake/bins"))
