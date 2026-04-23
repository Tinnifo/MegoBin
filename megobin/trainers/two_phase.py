import logging
from pathlib import Path
from typing import Any, Callable, Iterable, TYPE_CHECKING

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from megobin.utils.checkpoints import save_checkpoint

if TYPE_CHECKING:
    from megobin.utils.logger import Logger

log = logging.getLogger(__name__)


class TwoPhaseTrainer:
    """Multi-phase optimization loop.

    Each entry in ``phases`` is a dict describing one contiguous phase.
    The trainer runs them in order, and for each phase it:

    1. Applies ``encoder_attrs`` / ``loss_attrs`` via ``setattr`` — used
       to toggle things like ``include_std`` on the encoder and loss in
       lockstep. Kept generic so any stateful flag can be flipped
       without adding encoder-specific knowledge to the trainer.
    2. Freezes every encoder parameter, then unfreezes the named
       ``params`` group (resolved via ``encoder.parameter_groups()``).
    3. Builds a fresh optimizer from the phase's ``optimizer`` partial
       over the now-trainable parameters. Optional fresh scheduler.
    4. Iterates the sampler via DataLoader for ``epochs`` epochs,
       calling ``encoder.training_step(batch, loss_fn)``.

    The name "two_phase" reflects the primary use case (UncertainGen's
    mean-then-cov schedule) but nothing in the implementation caps the
    count at two — you can configure N phases for curricula or
    pretrain/finetune regimes.

    After all phases complete, ``requires_grad`` is restored on every
    encoder parameter so downstream ``encode`` / eval paths behave
    normally.

    Per-phase keys (all optional unless noted):
        params (str, required):    parameter-group name to unfreeze
        epochs (int, required):    number of epochs in this phase
        optimizer (Callable):      partial optimizer factory — required
        scheduler (Callable|None): partial scheduler factory
        batch_size (int):          DataLoader batch size
        grad_clip (float|None):    optional grad-norm clip
        num_workers (int):         DataLoader workers
        shuffle (bool):            DataLoader shuffle flag
        encoder_attrs (dict):      attrs to setattr on encoder
        loss_attrs (dict):         attrs to setattr on loss_fn
    """

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
        encoder: nn.Module,
        sampler: Dataset,
        loss_fn: nn.Module,
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
                        self.logger.log_checkpoint(interim, name=f"phase{phase_idx + 1}")
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
        encoder: nn.Module,
        sampler: Dataset,
        loss_fn: nn.Module,
        device: torch.device,
        global_step_ref: list[int],
    ) -> None:
        params_name = phase["params"]
        epochs = phase["epochs"]
        optimizer_factory: Callable[[Iterable[nn.Parameter]], torch.optim.Optimizer] = phase["optimizer"]
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

        loader = DataLoader(
            sampler,
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
        for epoch in range(epochs):
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
