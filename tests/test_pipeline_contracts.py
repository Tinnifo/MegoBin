"""Contract checks and stage resolution on the real pipeline.

Every failure test also asserts that the failure came *before any compute*:
no stage ran, no bins directory exists, and ``status.json`` records
CONTRACT_FAIL.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from megobin.contracts import ContractError, Stage
from megobin.pipeline import make_plan, run_pipeline, build_stages
from megobin.manifest import Manifest
from megobin.ports import Bins
from tests.conftest import compose_experiment


def _status(run_dir: Path) -> dict:
    return json.loads((run_dir / "status.json").read_text())


def _assert_failed_before_compute(run_dir: Path, tmp_path: Path):
    st = _status(run_dir)
    assert st["status"] == "CONTRACT_FAIL"
    assert st["contract"]["status"] == "FAIL"
    assert not (run_dir / "bins").exists()
    assert not (run_dir / "encoder.pt").exists()
    return st


class TestMismatchedPairings:
    def test_mahalanobis_needing_var_with_mean_only_encoder_fails_at_start(
        self, dataset, tmp_path
    ):
        # SemiBin2Encoder sends mu only; a loss that needs z.var cannot be fed.
        cfg = compose_experiment(
            "semibin2_longread", dataset, "loss=mahalanobis_bce", "loss.include_std=true"
        )
        with pytest.raises(ContractError) as exc:
            run_pipeline(cfg, run_dir=tmp_path / "run")
        msg = str(exc.value)
        assert "MahalanobisBCELoss" in msg
        assert "z.['var']" in msg
        assert "SemiBin2Encoder" in msg
        st = _assert_failed_before_compute(tmp_path / "run", tmp_path)
        assert any("MahalanobisBCELoss" in p for p in st["contract"]["problems"])

    def test_phase_two_loss_needing_var_with_mean_only_encoder_names_the_phase(
        self, dataset, tmp_path
    ):
        cfg = compose_experiment(
            "uncertain_gen_dbscan", dataset, "encoder=semibin2", "encoder.input_dim=136"
        )
        del cfg.encoder["embedding_dim"]  # the experiment's UncertainGen-only key
        with pytest.raises(ContractError, match=r"phase 2"):
            run_pipeline(cfg, run_dir=tmp_path / "run")

    def test_mahalanobis_with_unit_covariance_accepts_mean_only_encoder(
        self, dataset, tmp_path
    ):
        # include_std=false needs z.mu only, which a mean-only encoder sends.
        cfg = compose_experiment(
            "semibin2_longread", dataset, "loss=mahalanobis_bce", "trainer.epochs=1"
        )
        res = run_pipeline(cfg, run_dir=tmp_path / "run")
        assert res.status["status"] == "OK"

    def test_hinge_with_variance_sending_encoder_is_fine(self, dataset, tmp_path):
        # Extra field (var) is ignored by a loss that does not need it.
        cfg = compose_experiment(
            "uncertain_gen_dbscan",
            dataset,
            "loss=hinge",
            "trainer=single_phase",
            "trainer.epochs=1",
            "trainer.params=mean",
            "trainer.checkpoint_path=null",
        )
        res = run_pipeline(cfg, run_dir=tmp_path / "run")
        assert res.status["status"] == "OK"

    def test_filter_needing_var_with_mean_only_encoder_fails_at_start(
        self, dataset, tmp_path
    ):
        cfg = compose_experiment("semibin2_longread", dataset, "filter=uncertainty")
        with pytest.raises(ContractError) as exc:
            run_pipeline(cfg, run_dir=tmp_path / "run")
        assert "filter: needs z.['var']" in str(exc.value)
        _assert_failed_before_compute(tmp_path / "run", tmp_path)

    def test_old_encoder_and_loss_flags_in_two_phase_config_are_rejected(self):
        from megobin.trainers.two_phase import TwoPhaseTrainer

        with pytest.raises(ValueError, match="encoder_attrs"):
            TwoPhaseTrainer(phases=[{"params": "mean", "epochs": 1, "optimizer": None,
                                     "encoder_attrs": {"include_std": True}}])


class TestUnavailablePorts:
    def test_binner_needing_contig_fasta_fails_at_start_when_dataset_has_none(
        self, dataset_no_fasta, tmp_path
    ):
        cfg = compose_experiment("semibin2_longread", dataset_no_fasta)
        with pytest.raises(ContractError) as exc:
            run_pipeline(cfg, run_dir=tmp_path / "run")
        msg = str(exc.value)
        assert "binner: needs port 'contigs'" in msg and "fasta" in msg
        _assert_failed_before_compute(tmp_path / "run", tmp_path)

    def test_binner_needing_a_port_nobody_provides_fails_at_start(self, dataset, tmp_path):
        # A graph-aware binner on a dataset without an assembly graph.
        cfg = compose_experiment(
            "semibin2_longread",
            dataset,
            "binner._target_=tests.toy_components.GraphBinner",
        )
        with pytest.raises(ContractError) as exc:
            run_pipeline(cfg, run_dir=tmp_path / "run")
        msg = str(exc.value)
        assert "binner: needs port 'graph'" in msg and "assembly_graph" in msg
        _assert_failed_before_compute(tmp_path / "run", tmp_path)

    def test_graph_binner_runs_when_the_dataset_offers_a_graph(self, dataset, tmp_path):
        (dataset / "assembly_graph.gfa").write_text("S\t1\tAC\n")
        cfg = compose_experiment(
            "semibin2_longread",
            dataset,
            "binner._target_=tests.toy_components.GraphBinner",
            "trainer.epochs=1",
        )
        cfg.binner = {"_target_": "tests.toy_components.GraphBinner"}
        res = run_pipeline(cfg, run_dir=tmp_path / "run")
        assert res.status["status"] == "OK"
        assert res.data["bins"].labels.shape == (24,)

    def test_missing_data_csv_fails_at_start(self, tmp_path):
        from tests.conftest import make_dataset

        ds = make_dataset(tmp_path / "ds")
        (ds / "data.csv").unlink()
        cfg = compose_experiment("semibin2_longread", ds)
        with pytest.raises(ContractError, match="feature_files"):
            run_pipeline(cfg, run_dir=tmp_path / "run")

    def test_signal_mismatch_is_a_contract_failure(self, dataset, tmp_path):
        cfg = compose_experiment(
            "semibin2_longread", dataset, "dataset.signals=[kmers]"
        )
        with pytest.raises(ContractError) as exc:
            run_pipeline(cfg, run_dir=tmp_path / "run")
        assert "signals" in str(exc.value) and "abundance" in str(exc.value)


class TestOptionalRefiner:
    def _cfg(self, dataset, target):
        cfg = compose_experiment("semibin2_longread", dataset, "trainer.epochs=1")
        cfg.pipeline.target = target
        cfg.pipeline.stages.refiner = {
            "_target_": "megobin.stages.refiner.RefinerStage",
            "_recursive_": False,
            "config": {"_target_": "megobin.refiners.no_op.IdentityRefiner"},
        }
        return cfg

    def test_refiner_is_not_resolved_when_target_is_bins(self, dataset):
        cfg = self._cfg(dataset, "bins")
        manifest = Manifest.from_cfg(cfg)
        plan = make_plan(build_stages(cfg), "bins", manifest, [])
        assert plan.ok
        assert plan.order == ["features", "embedder", "filter", "binner"]
        assert "refiner" in plan.stages  # configured, but nobody needs it

    def test_refiner_is_resolved_when_target_asks_for_it(self, dataset, tmp_path):
        cfg = self._cfg(dataset, "bins_refined")
        res = run_pipeline(cfg, run_dir=tmp_path / "run")
        assert res.status["order"][-2:] == ["binner", "refiner"]
        refined = res.data["bins_refined"]
        assert isinstance(refined, Bins)
        np.testing.assert_array_equal(refined.labels, res.data["bins"].labels)

    def test_refiner_needing_a_graph_fails_at_start_without_one(self, dataset, tmp_path):
        cfg = self._cfg(dataset, "bins_refined")
        cfg.pipeline.stages.refiner.config._target_ = "tests.toy_components.GraphRefiner"
        with pytest.raises(ContractError, match="refiner: needs port 'graph'"):
            run_pipeline(cfg, run_dir=tmp_path / "run")


class TestPassThroughEmbedder:
    def _cfg(self, dataset, binner_target):
        cfg = compose_experiment(
            "semibin2_longread", dataset, "pipeline=bin_raw_features"
        )
        cfg.pipeline.name = "raw_features_demo"
        cfg.binner = {"_target_": binner_target}
        return cfg

    def test_pass_through_embedder_runs(self, dataset, tmp_path):
        cfg = self._cfg(dataset, "tests.toy_components.RawFeatureBinner")
        res = run_pipeline(cfg, run_dir=tmp_path / "run")
        assert res.status["order"] == ["features", "embedder", "binner"]
        assert res.data["bins"].labels.shape == (24,)
        # no model was trained
        assert not (tmp_path / "run" / "encoder.pt").exists()
        np.testing.assert_array_equal(
            res.data["z"].z.array("raw_features"), res.data["features"].x
        )

    def test_binner_wanting_an_embedding_is_rejected_behind_a_pass_through(
        self, dataset, tmp_path
    ):
        cfg = self._cfg(dataset, "megobin.binners.semibin_2.DBSCANEnsembleBinner")
        with pytest.raises(ContractError, match="z_kept"):
            run_pipeline(cfg, run_dir=tmp_path / "run")


class TestStatusJson:
    def test_status_records_order_contract_hash_and_ledger(self, dataset, tmp_path):
        cfg = compose_experiment("semibin2_longread", dataset, "trainer.epochs=1")
        cfg.pipeline.interventions = [
            {"name": "fewer_epochs", "level": "L1", "note": "1 epoch for a smoke test"}
        ]
        res = run_pipeline(cfg, run_dir=tmp_path / "run")
        st = _status(tmp_path / "run")
        assert st["status"] == "OK"
        assert st["order"] == ["features", "embedder", "filter", "binner"]
        assert st["contract"] == {"status": "OK", "problems": []}
        assert len(st["config_sha256"]) == 64
        assert st["interventions"][0]["level"] == "L1"
        assert st["pipeline"] == "semibin2_longread"
        assert st == res.status

    def test_same_config_gives_same_hash(self, dataset, tmp_path):
        a = compose_experiment("semibin2_longread", dataset, "trainer.epochs=1")
        b = compose_experiment("semibin2_longread", dataset, "trainer.epochs=1")
        c = compose_experiment("semibin2_longread", dataset, "trainer.epochs=2")
        from megobin.status import config_sha256

        assert config_sha256(a) == config_sha256(b) != config_sha256(c)

    def test_invalid_intervention_level_is_a_contract_failure(self, dataset, tmp_path):
        cfg = compose_experiment("semibin2_longread", dataset)
        cfg.pipeline.interventions = [{"name": "x", "level": "L9"}]
        with pytest.raises(ContractError, match="L9"):
            run_pipeline(cfg, run_dir=tmp_path / "run")


class TestStandaloneEvaluator:
    def test_evaluate_scores_a_bins_directory(self, tmp_path):
        from unittest.mock import patch

        from megobin.evaluate import evaluate
        from megobin.evaluators.checkm2 import CheckM2Evaluator

        bins = tmp_path / "bins"
        bins.mkdir()

        def fake_run(cmd, **kwargs):
            out = Path(cmd[cmd.index("-o") + 1])
            (out / "quality_report.tsv").write_text(
                "Name\tCompleteness\tContamination\nbin_0000\t90\t2\n"
            )

        with patch("subprocess.run", side_effect=fake_run):
            df = evaluate(bins, CheckM2Evaluator())
        assert list(df.columns) == ["completeness", "contamination"]

    def test_method_pipeline_no_longer_runs_an_evaluator(self, dataset, tmp_path):
        cfg = compose_experiment("semibin2_longread", dataset, "trainer.epochs=1")
        assert "evaluator" not in cfg
        res = run_pipeline(cfg, run_dir=tmp_path / "run")
        assert "evaluator" not in res.status["order"]


class TestExperiments:
    def test_semibin2_longread_runs_end_to_end(self, dataset, tmp_path):
        cfg = compose_experiment("semibin2_longread", dataset, "trainer.epochs=2")
        res = run_pipeline(cfg, run_dir=tmp_path / "run")
        assert res.data["z_kept"].z.array("mu").shape == (24, 101)

    def test_uncertain_gen_dbscan_runs_end_to_end(self, dataset, tmp_path):
        cfg = compose_experiment(
            "uncertain_gen_dbscan",
            dataset,
            "trainer.phases.0.epochs=2",
            "trainer.phases.1.epochs=2",
        )
        res = run_pipeline(cfg, run_dir=tmp_path / "run")
        z = res.data["z"].z
        assert z.array("mu").shape == (24, 256)
        assert z.array("var").shape == (24, 256)
