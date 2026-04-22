"""Logger smoke tests.

Covers:
- NoOpLogger satisfies the Logger Protocol and silently accepts every call
- TensorBoardLogger writes an event file, checkpoints, and run_meta.json
- A trainer run with a TB logger produces a non-empty event file
"""

import json
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.losses.hinge_contrastive import HingeContrastiveLoss
from src.representations.semibin_encoder import SemiBinEncoder
from src.trainers.single_phase import SinglePhaseTrainer
from src.utils.logger import Logger
from src.utils.no_op_logger import NoOpLogger
from src.utils.tensorboard_logger import TensorBoardLogger


class TestNoOpLogger:
    def test_satisfies_protocol(self):
        assert isinstance(NoOpLogger(), Logger)

    def test_calls_are_noops(self, tmp_path):
        logger = NoOpLogger()
        logger.log_scalars({"loss": 1.0}, step=0)
        logger.log_config({"seed": 42})
        logger.log_text("note", "hello")
        logger.log_dataframe("df", pd.DataFrame({"a": [1, 2]}))
        logger.log_checkpoint(tmp_path / "fake.pt")
        logger.finish()


class TestTensorBoardLogger:
    def test_satisfies_protocol(self, tmp_path):
        assert isinstance(TensorBoardLogger(logdir=tmp_path), Logger)

    def test_writes_event_file(self, tmp_path):
        logger = TensorBoardLogger(logdir=tmp_path / "run1")
        logger.log_scalars({"loss": 0.5, "acc": 0.9}, step=1)
        logger.log_scalars({"loss": 0.3, "acc": 0.95}, step=2)
        logger.finish()

        event_files = list((tmp_path / "run1").glob("events.out.tfevents.*"))
        assert len(event_files) == 1
        assert event_files[0].stat().st_size > 0

    def test_log_config_writes_run_meta(self, tmp_path):
        logger = TensorBoardLogger(logdir=tmp_path / "run2")
        logger.log_config({"seed": 42, "trainer": {"epochs": 10}})
        logger.finish()

        meta_path = tmp_path / "run2" / "run_meta.json"
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text())
        assert meta["config"]["seed"] == 42
        assert "git_sha" in meta
        assert "env_hash" in meta

    def test_log_dataframe_adds_text_without_error(self, tmp_path):
        logger = TensorBoardLogger(logdir=tmp_path / "run3")
        df = pd.DataFrame(
            {"completeness": [95.0, 80.0], "contamination": [1.5, 2.0]}
        )
        logger.log_dataframe("checkm2_results", df)
        logger.finish()

    def test_log_checkpoint_copies_and_records(self, tmp_path):
        ckpt_src = tmp_path / "model.pt"
        ckpt_src.write_bytes(b"fake weights")

        logger = TensorBoardLogger(logdir=tmp_path / "run4")
        logger.log_checkpoint(ckpt_src, name="best")
        logger.finish()

        ckpt_dest = tmp_path / "run4" / "checkpoints" / "best.pt"
        assert ckpt_dest.exists()
        assert ckpt_dest.read_bytes() == b"fake weights"

        meta = json.loads((tmp_path / "run4" / "run_meta.json").read_text())
        assert len(meta["checkpoints"]) == 1
        assert meta["checkpoints"][0]["name"] == "best"


class _ToyPairs(Dataset):
    def __init__(self, n=32, d=16):
        rng = np.random.default_rng(0)
        self.x_i = torch.from_numpy(rng.standard_normal((n, d)).astype("float32"))
        self.x_j = torch.from_numpy(rng.standard_normal((n, d)).astype("float32"))
        self.y = torch.from_numpy((rng.random(n) > 0.5).astype("float32"))

    def __len__(self):
        return len(self.x_i)

    def __getitem__(self, i):
        return self.x_i[i], self.x_j[i], self.y[i]


class TestTrainerWithTensorBoardLogger:
    def test_single_phase_writes_events(self, tmp_path):
        logger = TensorBoardLogger(logdir=tmp_path / "tb")
        encoder = SemiBinEncoder(input_dim=16, embedding_dim=4, dropout=0.0)
        loss_fn = HingeContrastiveLoss()
        sampler = _ToyPairs(n=32, d=16)

        trainer = SinglePhaseTrainer(
            optimizer=partial(torch.optim.Adam, lr=1e-2),
            epochs=2,
            batch_size=8,
            device="cpu",
            log_every=1,
            logger=logger,
        )
        trainer.fit(encoder, sampler, loss_fn)
        logger.finish()

        event_files = list((tmp_path / "tb").glob("events.out.tfevents.*"))
        assert len(event_files) == 1
        assert event_files[0].stat().st_size > 0
