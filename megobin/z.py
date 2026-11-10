"""Labelled ``z``: what an encoder sends and what a loss or binner reads.

The encoder owns *what it can send* (``provides``), the consumer owns *what it
needs* (``needs``); the labels of a :class:`Z` are the fields that are set.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Generic, Iterable, Mapping, TypeVar

from megobin.contracts import ContractError, Problem

T = TypeVar("T")

MU = "mu"
VAR = "var"


@dataclass(frozen=True, eq=False)
class Z(Generic[T]):
    """Embedding with labelled content.

    ``mu``    — the point estimate (labelled ``mu``).
    ``var``   — per-dimension variance/covariance, ``None`` if not sent
                (labelled ``var``).
    ``extra`` — further labelled arrays (e.g. ``raw_features``).

    Works for torch tensors (training) and numpy arrays (inference).
    """

    mu: T | None
    var: T | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    @property
    def labels(self) -> frozenset[str]:
        out = set(self.extra)
        if self.mu is not None:
            out.add(MU)
        if self.var is not None:
            out.add(VAR)
        return frozenset(out)

    def array(self, label: str) -> Any:
        if label == MU:
            value = self.mu
        elif label == VAR:
            value = self.var
        else:
            value = self.extra.get(label)
        if value is None:
            raise ContractError(
                [Problem("Z", "z", (label,), tuple(sorted(self.labels)), detail="")]
            )
        return value

    def require(self, needed: Iterable[str], who: str) -> None:
        """Loud runtime guard; the pipeline-start check normally fires first."""
        missing = frozenset(needed) - self.labels
        if missing:
            raise ContractError(
                [Problem(who, "z", tuple(sorted(missing)), tuple(sorted(self.labels)))]
            )


def as_z(x: Any) -> "Z[Any]":
    """Accept a bare tensor as ``Z(mu=x)`` (convenience for mean-only callers)."""
    return x if isinstance(x, Z) else Z(mu=x)
