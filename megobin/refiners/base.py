from typing import Any, Mapping, Protocol, runtime_checkable

from megobin.ports import Bins


@runtime_checkable
class Refiner(Protocol):
    """Optional post-binner stage (e.g. graph-based refinement).

    Only resolved into the pipeline when the target port is ``bins_refined``.
    """

    @property
    def needs(self) -> Mapping[str, frozenset[str]]:
        """Inputs by port, e.g. ``{"bins": {"one_label_per_contig"}, "graph": {"assembly_graph"}}``."""
        ...

    @property
    def provides(self) -> Mapping[str, frozenset[str]]:
        """Outputs by port, e.g. ``{"bins_refined": {"one_label_per_contig"}}``."""
        ...

    @property
    def bindings(self) -> Mapping[str, str]:
        """Constructor kwargs -> named input ports handed over by the refiner stage."""
        ...

    def refine(self, bins: Bins, **inputs: Any) -> Bins: ...
