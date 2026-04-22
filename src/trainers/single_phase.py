import logging
from pathlib import Path
from typing import Callable, Iterable, TYPE_CHECKING

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.utils.checkpoints import save_checkpoint

if TYPE_CHECKING:
    from src.utils.logger import Logger

log = logging.getLogger(__name__)


class SinglePhaseTrainer:
    """Vanilla single-phase optimization loop.

    Every epoch: iterate the ``PairSampler`` via a DataLoader, call
    ``encoder.training_step(batch, loss_fn)``, backprop, step the
    optimizer, optionally step the scheduler. Gradient clipping and
    a logger are optional.

    ``optimizer`` and ``scheduler`` are injected as *partial* callables
    (Hydra ``_partial_: true``) so the trainer can call them with the
    encoder's parameter group at ``fit`` time. This keeps the trainer
    agnostic to which optimizer/scheduler is used.

    ``params`` selects which of ``encoder.parameter_groups()`` to
    optimize — defaults to ``"all"``. Phase-based trainers override
    this to train only a subset.
    """

    def __init__(
        self,
        optimizer: Callable[[Iterable[nn.Parameter]], torch.optim.Optimizer],
        epochs: int = 10,
        batch_size: int = 2048,
        scheduler: Callable[[torch.optim.Optimizer], torch.optim.lr_scheduler.LRScheduler] | None = None,
        grad_clip: float | None = None,
        params: str = "all",
        num_workers: int = 0,
        shuffle: bool = True,
        device: str | None = None,
        log_every: int = 50,
        logger: "Logger | None" = None,
        checkpoint_path: str | Path | None = None,
        checkpoint_every: int | None = None,
    ):
        self.optimizer_factory = optimizer
        self.scheduler_factory = scheduler
        self.epochs = epochs
        self.batch_size = batch_size
        self.grad_clip = grad_clip
        self.params = params
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.log_every = log_every
        self.logger = logger
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.checkpoint_every = checkpoint_every

    def fit(
        self,
        encoder: nn.Module,
        sampler: Dataset,
        loss_fn: nn.Module,
    ) -> None:
        device = torch.device(self.device)
        encoder.to(device)
        loss_fn.to(device)

        groups = encoder.parameter_groups()
        if self.params not in groups:
            raise ValueError(
                f"Encoder has no parameter group '{self.params}'. "
                f"Available: {sorted(groups)}"
            )
        trainable = [p for p in groups[self.params] if p.requires_grad]
        if not trainable:
            raise ValueError(
                f"Parameter group '{self.params}' has no trainable parameters."
            )

        optimizer = self.optimizer_factory(trainable)
        scheduler = self.scheduler_factory(optimizer) if self.scheduler_factory else None

        loader = DataLoader(
            sampler,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=(device.type == "cuda"),
        )

        encoder.train()
        global_step = 0
        for epoch in range(self.epochs):
            running = 0.0
            n_batches = 0
            for batch in loader:
                batch = tuple(t.to(device, non_blocking=True) for t in batch)

                loss = encoder.training_step(batch, loss_fn)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if self.grad_clip is not None:
                    nn.utils.clip_grad_norm_(trainable, self.grad_clip)
                optimizer.step()

                running += loss.item()
                n_batches += 1
                global_step += 1

                if self.logger and global_step % self.log_every == 0:
                    self.logger.log_scalars(
                        {
                            "train/loss": loss.item(),
                            "train/lr": optimizer.param_groups[0]["lr"],
                            "train/epoch": epoch,
                        },
                        step=global_step,
                    )

            if scheduler is not None:
                scheduler.step()

            avg = running / max(n_batches, 1)
            log.info("epoch %d/%d — loss %.4f", epoch + 1, self.epochs, avg)
            if self.logger:
                self.logger.log_scalars(
                    {"train/epoch_loss": avg}, step=epoch + 1
                )

            if (
                self.checkpoint_path is not None
                and self.checkpoint_every is not None
                and (epoch + 1) % self.checkpoint_every == 0
                and (epoch + 1) < self.epochs
            ):
                interim = self.checkpoint_path.with_name(
                    f"{self.checkpoint_path.stem}_epoch{epoch + 1}{self.checkpoint_path.suffix}"
                )
                save_checkpoint(encoder, interim)
                if self.logger:
                    self.logger.log_checkpoint(interim, name=f"epoch{epoch + 1}")

        if self.checkpoint_path is not None:
            save_checkpoint(encoder, self.checkpoint_path)
            if self.logger:
                self.logger.log_checkpoint(self.checkpoint_path, name="final")
