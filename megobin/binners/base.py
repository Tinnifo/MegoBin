from typing import Mapping, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Binner(Protocol):
    @property
    def needs(self) -> Mapping[str, frozenset[str]]:
        """Inputs by port: ``{"z_kept": {"mu"}, "contigs": {"fasta"}, ...}``."""
        ...

    @property
    def provides(self) -> Mapping[str, frozenset[str]]:
        """Outputs by port: ``{"bins": {"one_label_per_contig"}}``."""
        ...

    @property
    def embedding(self) -> tuple[str, str]:
        """Which array is clustered: ``(port, label)``."""
        ...

    @property
    def bindings(self) -> Mapping[str, str]:
        """Constructor kwargs -> named runtime values handed over by the binner
        stage (``contig_fasta``, ``contig_names``, ``contig_lengths``,
        ``markers_dir``, plus any other input port by name)."""
        ...

    def cluster(self, embeddings: np.ndarray) -> np.ndarray: ...
