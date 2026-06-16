import logging
from pathlib import Path
from typing import Any, Callable, Iterable, TYPE_CHECKING, cast

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from megobin.data.base import PairSampler
from megobin.encoders.base import Encoder
from megobin.losses.base import ContrastiveLoss
from megobin.utils.checkpoints import save_checkpoint

if TYPE_CHECKING:
    from megobin.utils.logger import Logger

log = logging.getLogger(__name__)


class TwoPhaseTrainer:
    def __init__(
        self,
        phases: list[dict[str, Any]],
        device: str | None = None,
        log_every: int = 50,
        logger: "Logger | None" = None,
        checkpoint_path: str | Path | None = None,
        checkpoint_per_phase: bool = False,
    ):
        if not phases:
            raise ValueError("TwoPhaseTrainer needs at least one phase.")
        self.phases = phases
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.log_every = log_every
        self.logger = logger
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.checkpoint_per_phase = checkpoint_per_phase

    def fit(
        self,
        encoder: Encoder,
        sampler: PairSampler,
        loss_fn: ContrastiveLoss,
    ) -> None:
        device = torch.device(self.device)
        encoder.to(device)
        loss_fn.to(device)

        global_step = 0
        try:
            for phase_idx, phase in enumerate(self.phases):
                self._run_phase(
                    phase_idx=phase_idx,
                    phase=phase,
                    encoder=encoder,
                    sampler=sampler,
                    loss_fn=loss_fn,
                    device=device,
                    global_step_ref=[global_step],
                )

                if (
                    self.checkpoint_path is not None
                    and self.checkpoint_per_phase
                    and phase_idx < len(self.phases) - 1
                ):
                    interim = self.checkpoint_path.with_name(
                        f"{self.checkpoint_path.stem}_phase{phase_idx + 1}{self.checkpoint_path.suffix}"
                    )
                    save_checkpoint(encoder, interim)
                    if self.logger:
                        self.logger.log_checkpoint(
                            interim, name=f"phase{phase_idx + 1}"
                        )
        finally:
            for p in encoder.parameters():
                p.requires_grad = True

        if self.checkpoint_path is not None:
            save_checkpoint(encoder, self.checkpoint_path)
            if self.logger:
                self.logger.log_checkpoint(self.checkpoint_path, name="final")

    def _run_phase(
        self,
        phase_idx: int,
        phase: dict[str, Any],
        encoder: Encoder,
        sampler: PairSampler,
        loss_fn: ContrastiveLoss,
        device: torch.device,
        global_step_ref: list[int],
    ) -> None:
        params_name = phase["params"]
        epochs = phase["epochs"]
        optimizer_factory: Callable[[Iterable[nn.Parameter]], torch.optim.Optimizer] = (
            phase["optimizer"]
        )
        scheduler_factory = phase.get("scheduler")
        batch_size = phase.get("batch_size", 2048)
        grad_clip = phase.get("grad_clip")
        num_workers = phase.get("num_workers", 0)
        shuffle = phase.get("shuffle", True)

        for k, v in (phase.get("encoder_attrs") or {}).items():
            setattr(encoder, k, v)
        for k, v in (phase.get("loss_attrs") or {}).items():
            setattr(loss_fn, k, v)

        groups = encoder.parameter_groups()
        if params_name not in groups:
            raise ValueError(
                f"Phase {phase_idx + 1}: encoder has no parameter group "
                f"'{params_name}'. Available: {sorted(groups)}"
            )
        for p in encoder.parameters():
            p.requires_grad = False
        for p in groups[params_name]:
            p.requires_grad = True

        trainable = [p for p in groups[params_name] if p.requires_grad]
        if not trainable:
            raise ValueError(
                f"Phase {phase_idx + 1}: parameter group '{params_name}' is empty."
            )

        optimizer = optimizer_factory(trainable)
        scheduler = scheduler_factory(optimizer) if scheduler_factory else None

        loader: DataLoader = DataLoader(
            cast(Dataset, sampler),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
        )

        log.info(
            "phase %d: train '%s' for %d epochs (%d params)",
            phase_idx + 1,
            params_name,
            epochs,
            sum(p.numel() for p in trainable),
        )

        encoder.train()
        resample = getattr(sampler, "set_epoch", None)
        for epoch in range(epochs):
            if callable(resample):
                resample(epoch)
            running = 0.0
            n_batches = 0
            for batch in loader:
                batch = tuple(t.to(device, non_blocking=True) for t in batch)

                loss = encoder.training_step(batch, loss_fn)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip is not None:
                    nn.utils.clip_grad_norm_(trainable, grad_clip)
                optimizer.step()

                running += loss.item()
                n_batches += 1
                global_step_ref[0] += 1

                if self.logger and global_step_ref[0] % self.log_every == 0:
                    self.logger.log_scalars(
                        {
                            f"phase{phase_idx + 1}/loss": loss.item(),
                            f"phase{phase_idx + 1}/lr": optimizer.param_groups[0]["lr"],
                            f"phase{phase_idx + 1}/epoch": epoch,
                        },
                        step=global_step_ref[0],
                    )

            if scheduler is not None:
                scheduler.step()

            avg = running / max(n_batches, 1)
            log.info(
                "phase %d epoch %d/%d — loss %.4f",
                phase_idx + 1,
                epoch + 1,
                epochs,
                avg,
            )
            if self.logger:
                self.logger.log_scalars(
                    {f"phase{phase_idx + 1}/epoch_loss": avg},
                    step=epoch + 1,
                )
