"""Embedder stages: ``features`` -> ``z``.

``TrainedEmbedder`` hides an inner training loop (sampler + encoder + loss +
trainer). The inner loop has its own contracts, checked by ``check_inner``:

* the loss's required ``batch`` labels must be offered by the sampler,
* the loss's required ``z`` labels must be sendable by the encoder, for every
  phase of the trainer,
* the sampler's feature needs (``split``) join the stage's needs.

``PassThroughEmbedder`` is for tools with no model: it hands the raw features
on, labelled as such.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import hydra
import numpy as np
from hydra.utils import get_class
from omegaconf import DictConfig

from megobin.contracts import Problem, Stage, bind_kwargs
from megobin.encoders.base import Encoder
from megobin.losses.base import ContrastiveLoss
from megobin.ports import Embedding, Features
from megobin.trainers.base import Trainer
from megobin.utils.checkpoints import load_checkpoint
from megobin.z import Z

log = logging.getLogger(__name__)


class TrainedEmbedder(Stage):
    """Sampler + encoder + loss + trainer behind one outer stage.

    Construction order matters for reproducibility: the encoder is built first
    (its weight initialisation draws from the global torch RNG, as it did in
    the pre-contract pipeline), then the loss and trainer (no RNG use).
    """

    def __init__(
        self,
        encoder: DictConfig,
        loss: DictConfig,
        pair_sampler: DictConfig | None = None,
        trainer: DictConfig | None = None,
        resume_from: str | None = None,
    ):
        self.encoder: Encoder = hydra.utils.instantiate(encoder)
        self.loss: ContrastiveLoss = hydra.utils.instantiate(loss)
        self.resume_from = resume_from or None
        self.trainer: Trainer | None = None
        self.sampler_cfg = pair_sampler
        self.sampler_cls: Any = (
            get_class(pair_sampler._target_) if pair_sampler is not None else None
        )
        if trainer is not None and pair_sampler is not None:
            self.trainer = hydra.utils.instantiate(trainer)

        log.info("Encoder:        %s", type(self.encoder).__name__)
        log.info("Loss:           %s", type(self.loss).__name__)

        needs = {k: set(v) for k, v in self.encoder.needs.items()}
        if self.trains and self.sampler_cls is not None:
            sampler_cls: Any = self.sampler_cls
            for port, lbls in sampler_cls.needs.items():
                needs.setdefault(port, set()).update(lbls)
        super().__init__(needs=needs, provides={"z": set(self.encoder.provides)})

    @property
    def trains(self) -> bool:
        return not self.resume_from and self.trainer is not None

    def check_inner(self) -> list[Problem]:
        if not self.trains or self.sampler_cls is None or self.trainer is None:
            return []
        problems: list[Problem] = []
        enc = type(self.encoder).__name__
        smp = self.sampler_cls.__name__
        losses: Sequence[ContrastiveLoss] = self.trainer.phase_losses(self.loss)
        for i, loss in enumerate(losses, start=1):
            who = f"{self.name or 'embedder'}/{type(loss).__name__}"
            if len(losses) > 1:
                who += f" (phase {i})"
            for port, offered, provider in (
                ("batch", self.sampler_cls.provides, f"sampler {smp}"),
                ("z", self.encoder.provides, f"encoder {enc}"),
            ):
                missing = frozenset(loss.needs[port]) - frozenset(offered)
                if missing:
                    problems.append(
                        Problem(
                            who,
                            port,
                            tuple(sorted(missing)),
                            tuple(sorted(offered)),
                            provider,
                        )
                    )
        return problems

    def run(self, data: dict[str, Any], ctx: Any) -> dict[str, Any]:
        feats: Features = data["features"]
        view = self.encoder.feature_view()
        x = view.fit_transform(feats.x, feats.kmer_dim)

        if self.resume_from:
            log.info("resume_from set — skipping training, loading %s", self.resume_from)
            load_checkpoint(self.encoder, self.resume_from)
        elif self.trainer is not None and self.sampler_cls is not None:
            self.trainer.logger = ctx.logger  # type: ignore[attr-defined]
            log.info("Trainer:        %s", type(self.trainer).__name__)
            split = view.transform_split(feats.load_split())
            log.info("Loaded sampler input 'features_split' from %s", feats.split_path)
            values = {"features_split": split}
            kwargs = bind_kwargs(
                self.sampler_cls.bindings, values, f"sampler/{self.sampler_cls.__name__}"
            )
            sampler = hydra.utils.instantiate(self.sampler_cfg, **kwargs)
            log.info(
                "Sampler:        %s (size=%d)", type(sampler).__name__, len(sampler)
            )
            self.trainer.fit(encoder=self.encoder, sampler=sampler, loss_fn=self.loss)
        else:
            log.warning(
                "No trainer/pair_sampler configured — skipping training "
                "(encoder will run with its current weights)."
            )

        z = self.encoder.encode_z(x)
        log.info(
            "Embeddings: %s%s",
            z.array("mu").shape,
            f" (with covariance {z.var.shape})" if z.var is not None else "",
        )
        return {"z": Embedding(z=z, names=feats.names)}


class PassThroughEmbedder(Stage):
    """No model: hand the raw feature matrix on as ``z``.

    ``labels`` is what the output is declared to be. The default
    ``raw_features`` only satisfies binners that ask for raw features. Declaring
    ``mu`` asserts that raw features may stand in for an embedding; record that
    in the run's ``interventions`` list.
    """

    def __init__(self, labels: Sequence[str] = ("raw_features",)):
        self.labels = tuple(str(label) for label in labels)
        super().__init__(
            needs={"features": {"kmer"}}, provides={"z": set(self.labels)}
        )

    def run(self, data: dict[str, Any], ctx: Any) -> dict[str, Any]:
        feats: Features = data["features"]
        x = np.asarray(feats.x)
        extra = {k: x for k in self.labels if k not in ("mu", "var")}
        mu = x if "mu" in self.labels else None
        return {"z": Embedding(z=Z(mu=mu, extra=extra), names=feats.names)}
