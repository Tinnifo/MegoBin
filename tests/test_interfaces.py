"""Protocol compliance tests for all components.

Every component must satisfy its Protocol:
- Encoder.encode() returns (N, embedding_dim) ndarray
- ContrastiveLoss.__call__() returns a scalar tensor with gradients
- Binner.cluster() returns (N,) integer ndarray
- Evaluator.score() returns DataFrame with completeness & contamination
"""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch

from megobin.binners.semibin_2 import DBSCANEnsembleBinner
from megobin.evaluators.checkm2 import CheckM2Evaluator
from megobin.losses.hinge_contrastive import HingeContrastiveLoss
from megobin.losses.mahalanobis_bce import MahalanobisBCELoss
from megobin.encoders.uncertaingen import UncertainGenEncoder


# ---- Encoder ---------------------------------------------------------------


class TestUncertainGenTrainingContract:
    def setup_method(self):
        self.model = UncertainGenEncoder(
            input_dim=32, hidden_dim=16, embedding_dim=8
        )

    def test_parameter_groups_has_mean_and_cov(self):
        groups = self.model.parameter_groups()
        assert set(groups) == {"mean", "cov", "all"}
        assert len(groups["mean"]) > 0
        assert len(groups["cov"]) > 0

    def test_encode_shape(self):
        features = np.random.randn(20, 32).astype("float32")
        z = self.model.encode(features)
        assert z.shape == (20, 8)

    def test_training_step_phase1(self):
        self.model.include_std = False
        x_i = torch.randn(8, 32)
        x_j = torch.randn(8, 32)
        label = torch.rand(8)
        loss = self.model.training_step(
            (x_i, x_j, label), MahalanobisBCELoss(include_std=False)
        )
        assert loss.shape == ()
        loss.backward()

    def test_training_step_phase2(self):
        self.model.include_std = True
        x_i = torch.randn(8, 32)
        x_j = torch.randn(8, 32)
        label = torch.rand(8)
        loss = self.model.training_step(
            (x_i, x_j, label), MahalanobisBCELoss(include_std=True)
        )
        assert loss.shape == ()
        loss.backward()


# ---- ContrastiveLoss -------------------------------------------------------


class TestHingeContrastiveLoss:
    def setup_method(self):
        self.loss_fn = HingeContrastiveLoss()

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
        assert callable(self.loss_fn)


class TestMahalanobisBCELoss:
    def test_phase1_returns_scalar(self):
        loss_fn = MahalanobisBCELoss(include_std=False)
        z_i = torch.randn(32, 64, requires_grad=True)
        z_j = torch.randn(32, 64, requires_grad=True)
        label = torch.rand(32)
        loss = loss_fn(z_i, z_j, label)
        assert loss.shape == ()
        loss.backward()
        assert z_i.grad is not None

    def test_phase2_splits_mean_and_cov(self):
        loss_fn = MahalanobisBCELoss(include_std=True)
        # Phase 2: z is [μ | S] concatenated → width is 2*d
        z_i = torch.randn(16, 16, requires_grad=True)
        z_j = torch.randn(16, 16, requires_grad=True)
        # Ensure S > 0 so exp → positive covariance component
        z_i = torch.cat([z_i[:, :8], z_i[:, 8:].exp()], dim=-1)
        z_i.requires_grad_(True)
        z_j = torch.cat([z_j[:, :8], z_j[:, 8:].exp()], dim=-1)
        z_j.requires_grad_(True)
        label = torch.rand(16)
        loss = loss_fn(z_i, z_j, label)
        assert loss.shape == ()


# ---- Binner ----------------------------------------------------------------


def _make_dbscan_binner(n: int) -> DBSCANEnsembleBinner:
    names = np.array([f"c{i}" for i in range(n)])
    # Half the contigs share marker "g0", the rest "g1" — gives the
    # marker-F1 selection something to score on without needing prodigal.
    c2m = {f"c{i}": ["g0"] if i < n // 2 else ["g1"] for i in range(n)}
    return DBSCANEnsembleBinner(
        eps_values=[0.1, 0.3, 0.7],
        min_samples=3,
        min_bin_size=3,
        contig_names=names,
        contig_lengths=np.full(n, 1000),
        contig_to_marker=c2m,
        n_total_markers=2,
    )


class TestDBSCANEnsembleBinner:
    def test_cluster_returns_1d_int(self):
        binner = _make_dbscan_binner(40)
        labels = binner.cluster(np.random.randn(40, 10))
        assert labels.ndim == 1
        assert labels.shape[0] == 40
        assert np.issubdtype(labels.dtype, np.integer)

    def test_all_points_assigned(self):
        binner = _make_dbscan_binner(30)
        labels = binner.cluster(np.random.randn(30, 8))
        assert len(labels) == 30
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
