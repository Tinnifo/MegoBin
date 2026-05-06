"""Dataset capability descriptor + fail-fast signal check.

Covers:
- The placeholder ``example`` dataset config parses and has the expected keys.
- Every feature config declares `required_signals`.
- All shipped experiment configs pass the compatibility check.
- A deliberately mismatched (dataset, features) pair raises with a clear error.
- Absent `required_signals` or `signals` is a no-op (backwards compatible).
"""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from megobin.pipeline import _check_signal_compatibility

CONFIG_DIR = str((Path(__file__).parent.parent / "configs").resolve())


class TestDatasetConfigs:
    @pytest.mark.parametrize("name", ["example"])
    def test_dataset_has_required_keys(self, name):
        with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
            cfg = compose(config_name=f"dataset/{name}")
        assert "name" in cfg.dataset
        assert "path" in cfg.dataset
        assert "signals" in cfg.dataset
        assert isinstance(list(cfg.dataset.signals), list)


class TestFeatureConfigs:
    @pytest.mark.parametrize(
        "name,expected_signals",
        [
            ("canonical_kmer", ["kmers"]),
            ("canonical_kmer_abundance", ["kmers", "abundance"]),
        ],
    )
    def test_feature_has_required_signals(self, name, expected_signals):
        with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
            cfg = compose(config_name=f"features/{name}")
        assert list(cfg.features.required_signals) == expected_signals


class TestExperimentCompatibility:
    @pytest.mark.parametrize("name", ["uncertain_gen_dbscan"])
    def test_experiment_passes_check(self, name):
        with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
            cfg = compose(config_name=f"experiment/{name}")
        _check_signal_compatibility(cfg)


class TestCompatibilityFailures:
    def test_missing_signal_raises(self):
        cfg = OmegaConf.create(
            {
                "dataset": {"name": "no_abundance", "signals": ["kmers"]},
                "features": {"required_signals": ["kmers", "abundance"]},
            }
        )
        with pytest.raises(ValueError, match="abundance"):
            _check_signal_compatibility(cfg)

    def test_all_missing_listed(self):
        cfg = OmegaConf.create(
            {
                "dataset": {"name": "minimal", "signals": []},
                "features": {"required_signals": ["kmers", "abundance", "taxonomy"]},
            }
        )
        with pytest.raises(ValueError) as exc:
            _check_signal_compatibility(cfg)
        msg = str(exc.value)
        assert "kmers" in msg
        assert "abundance" in msg
        assert "taxonomy" in msg


class TestBackwardsCompat:
    def test_no_required_signals_is_noop(self):
        cfg = OmegaConf.create(
            {
                "dataset": {"name": "ds", "signals": ["kmers"]},
                "features": {"canonical": True},
            }
        )
        _check_signal_compatibility(cfg)

    def test_no_dataset_signals_is_noop(self):
        cfg = OmegaConf.create(
            {
                "dataset": {"name": "ds"},
                "features": {"required_signals": ["kmers"]},
            }
        )
        _check_signal_compatibility(cfg)

    def test_missing_dataset_is_noop(self):
        cfg = OmegaConf.create({"features": {"required_signals": ["kmers"]}})
        _check_signal_compatibility(cfg)
