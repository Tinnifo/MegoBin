import logging
from pathlib import Path

import torch

from megobin.encoders.base import Encoder

log = logging.getLogger(__name__)


def save_checkpoint(encoder: Encoder, path: str | Path) -> Path:
    """Serialize an encoder's state dict to ``path``.

    Saves the state dict only (not the whole module); loading requires
    an encoder instance with the same architecture. Creates parent
    directories if missing.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(encoder.state_dict(), path)
    log.info("checkpoint saved → %s", path)
    return path


def load_checkpoint(encoder: Encoder, path: str | Path) -> Encoder:
    """Load weights into ``encoder`` in place. Returns the encoder.

    ``map_location="cpu"`` so a GPU-saved checkpoint loads on CPU-only
    machines without error; move to device after loading if needed.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    state_dict = torch.load(path, map_location="cpu")
    encoder.load_state_dict(state_dict)
    log.info("checkpoint loaded ← %s", path)
    return encoder
