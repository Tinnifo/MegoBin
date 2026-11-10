from typing import Any

from megobin.ports import Bins


class IdentityRefiner:
    """Hands the bins on unchanged. Useful for wiring tests and as a template."""

    needs = {"bins": frozenset({"one_label_per_contig"})}
    provides = {"bins_refined": frozenset({"one_label_per_contig"})}
    bindings: dict[str, str] = {}

    def refine(self, bins: Bins, **inputs: Any) -> Bins:
        return Bins(labels=bins.labels.copy(), names=bins.names)
