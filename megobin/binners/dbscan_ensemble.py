# This module has been adapted from SemiBin's DBSCAN ensemble binning flow:
# https://github.com/BigDataBiology/SemiBin/blob/main/SemiBin/long_read_cluster.py
# https://github.com/BigDataBiology/SemiBin/blob/main/SemiBin/utils.py

import logging
import multiprocessing
import os
import tempfile
from collections import defaultdict

import numpy as np
from sklearn.cluster import dbscan
from sklearn.neighbors import kneighbors_graph


class DBSCANEnsembleBinner:
    """SemiBin 2 DBSCAN ensemble binner.

    Pipeline:
      1. Build a k-NN distance graph on the encoder embeddings.
      2. Sweep DBSCAN over `eps_values` (with sample weights = contig
         lengths when available) using the precomputed graph.
      3. Iteratively pick the highest marker-F1 bin across the sweep,
         relaxing contamination, exactly like SemiBin's `get_best_bin`.
      4. Any contig not assigned to an extracted bin is given its own
         singleton id, so every row gets a label >= 0.

    Markers may be supplied either pre-computed (`contig_to_marker`) or
    by passing a `contig_fasta`; in the latter case `estimate_seeds` +
    `get_marker` from `megobin.utils.markers` are run on first call.
    """

    def __init__(
        self,
        eps_values=(
            0.01, 0.05, 0.1, 0.15, 0.2, 0.25,
            0.3, 0.35, 0.4, 0.45, 0.5, 0.55,
        ),
        min_samples: int = 5,
        min_bin_size: int = 1,
        k_neighbours: int = 200,
        n_jobs: int = -1,
        n_total_markers: int = 107,
        contig_fasta=None,
        contig_names=None,
        contig_lengths=None,
        contig_to_marker=None,
        output_dir=None,
        min_contig_len=None,
        num_process: int = 0,
        orf_finder: str = "fast-naive",
        prodigal_output_faa=None,
    ):
        self.eps_values = list(eps_values)
        self.min_samples = min_samples
        self.min_bin_size = min_bin_size
        self.k_neighbours = k_neighbours
        self.n_jobs = n_jobs
        self.n_total_markers = n_total_markers

        self.contig_fasta = (
            None if contig_fasta is None else os.fspath(contig_fasta)
        )
        self.contig_names = (
            None if contig_names is None else np.asarray(contig_names)
        )
        self.contig_lengths = (
            None if contig_lengths is None else np.asarray(contig_lengths)
        )
        self.contig_to_marker = contig_to_marker
        self.output_dir = (
            None if output_dir is None else os.fspath(output_dir)
        )
        self.min_contig_len = min_contig_len
        # Mirror SemiBin's argparse normalization (validate_normalize_args),
        # which the binner API bypasses: 0 means "use all CPUs", and the
        # count is clamped to the number of available CPUs. Without this the
        # default num_process=0 crashes ORF finding (Pool needs >= 1 process;
        # prodigal divides by num_process).
        cpu_count = multiprocessing.cpu_count()
        self.num_process = min(num_process or cpu_count, cpu_count)
        self.orf_finder = orf_finder
        self.prodigal_output_faa = prodigal_output_faa

        self._logger = logging.getLogger(__name__)
        self._marker_per_row_cache: list[list[str]] | None = None

    # ------------------------------------------------------------------
    # Marker resolution
    # ------------------------------------------------------------------

    def _resolve_markers(self, n_rows: int) -> list[list[str]]:
        """Return a list of marker-id lists, one per embedding row.

        Cached on the instance after the first call so the (potentially
        expensive) HMM step runs at most once per binner.
        """
        if self._marker_per_row_cache is not None:
            return self._marker_per_row_cache

        if self.contig_to_marker is not None:
            if self.contig_names is None:
                raise ValueError(
                    "DBSCANEnsembleBinner: `contig_names` is required when "
                    "`contig_to_marker` is supplied so the dict can be "
                    "aligned to embedding rows."
                )
            if len(self.contig_names) != n_rows:
                raise ValueError(
                    f"DBSCANEnsembleBinner: contig_names length "
                    f"({len(self.contig_names)}) does not match the number "
                    f"of embedding rows ({n_rows})."
                )
            per_row = [
                list(self.contig_to_marker.get(str(name), []))
                for name in self.contig_names
            ]
        elif self.contig_fasta is not None:
            per_row = self._call_markers_from_fasta(n_rows)
        else:
            raise ValueError(
                "DBSCANEnsembleBinner needs either `contig_to_marker` or "
                "`contig_fasta` (with `contig_names`) at construction time."
            )

        self._marker_per_row_cache = per_row
        return per_row

    def _call_markers_from_fasta(self, n_rows: int) -> list[list[str]]:
        from megobin.utils.markers import estimate_seeds, get_marker
        from megobin.utils.SemiBin_utils import load_fasta

        if self.contig_names is None:
            raise ValueError(
                "DBSCANEnsembleBinner: `contig_names` is required alongside "
                "`contig_fasta` to align markers with embedding rows."
            )
        if len(self.contig_names) != n_rows:
            raise ValueError(
                f"DBSCANEnsembleBinner: contig_names length "
                f"({len(self.contig_names)}) does not match the number "
                f"of embedding rows ({n_rows})."
            )

        if self.min_contig_len is None:
            assert self.contig_fasta is not None
            computed_min, _, _ = load_fasta(self.contig_fasta, ratio=0.05)
            binned_length = computed_min
        else:
            binned_length = self.min_contig_len

        out = self.output_dir
        if out is not None:
            os.makedirs(out, exist_ok=True)

        with tempfile.TemporaryDirectory() as tdir:
            target = out if out is not None else tdir
            estimate_seeds(
                self.contig_fasta,
                binned_length,
                self.num_process,
                output=target,
                orf_finder=self.orf_finder,
                prodigal_output_faa=self.prodigal_output_faa,
            )
            c2m = get_marker(
                os.path.join(target, "markers.hmmout"),
                fasta_path=self.contig_fasta,
                min_contig_len=binned_length,
                orf_finder=self.orf_finder,
                contig_to_marker=True,
            )
        return [list(c2m.get(str(name), [])) for name in self.contig_names]

    # ------------------------------------------------------------------
    # DBSCAN sweep
    # ------------------------------------------------------------------

    def _dbscan_sweep(self, embeddings: np.ndarray) -> list[np.ndarray]:
        n = embeddings.shape[0]
        k = min(self.k_neighbours, max(n - 1, 1))
        dist_matrix = kneighbors_graph(
            embeddings,
            n_neighbors=k,
            mode="distance",
            p=2,
            n_jobs=self.n_jobs,
        )
        sample_weight = (
            self.contig_lengths.astype(float)
            if self.contig_lengths is not None
            else None
        )
        results = []
        for eps in self.eps_values:
            _, labels = dbscan(
                dist_matrix,
                eps=eps,
                min_samples=self.min_samples,
                n_jobs=self.n_jobs,
                metric="precomputed",
                sample_weight=sample_weight,
            )
            results.append(np.asarray(labels))
        return results

    # ------------------------------------------------------------------
    # Marker-F1 bin selection (port of SemiBin's get_best_bin to indices)
    # ------------------------------------------------------------------

    def _select_best_bin(
        self,
        results: list[np.ndarray],
        marker_per_row: list[list[str]],
        weights: np.ndarray,
    ) -> list[int] | None:
        """Pick the highest-F1 bin across the eps sweep, relaxing contamination."""
        for max_contamination in (0.1, 0.2, 0.3, 0.4, 0.5, 1.0):
            max_f1 = 0.0
            weight_of_max = float("inf")
            best_bin: list[int] | None = None
            for labels in results:
                clusters: dict[int, list[int]] = defaultdict(list)
                for row, label in enumerate(labels):
                    if label != -1:
                        clusters[int(label)].append(row)
                for rows in clusters.values():
                    if len(rows) < self.min_bin_size:
                        continue
                    cluster_weight = float(sum(weights[r] for r in rows))
                    markers: list[str] = []
                    for r in rows:
                        markers.extend(marker_per_row[r])
                    if not markers:
                        continue
                    unique = set(markers)
                    recall = len(unique) / self.n_total_markers
                    contamination = (len(markers) - len(unique)) / len(markers)
                    if contamination > max_contamination:
                        continue
                    denom = recall + (1 - contamination)
                    if denom <= 0:
                        continue
                    f1 = 2 * recall * (1 - contamination) / denom
                    if f1 > max_f1 or (
                        f1 == max_f1 and cluster_weight <= weight_of_max
                    ):
                        max_f1 = f1
                        weight_of_max = cluster_weight
                        best_bin = list(rows)
            if max_f1 > 0 and best_bin is not None:
                return best_bin
        return None

    # ------------------------------------------------------------------
    # Binner protocol
    # ------------------------------------------------------------------

    def cluster(self, embeddings: np.ndarray) -> np.ndarray:
        """(N, d) → (N,) integer bin assignments (all >= 0)."""
        embeddings = np.asarray(embeddings)
        n = embeddings.shape[0]
        if n == 0:
            return np.zeros(0, dtype=np.int64)
        if n == 1:
            return np.zeros(1, dtype=np.int64)

        marker_per_row = self._resolve_markers(n)
        weights = (
            self.contig_lengths.astype(float)
            if self.contig_lengths is not None
            else np.ones(n, dtype=float)
        )

        sweep = self._dbscan_sweep(embeddings)

        active = list(range(n))
        extracted: list[list[int]] = []
        while active:
            if len(active) == 1:
                extracted.append(list(active))
                break
            local_marker = [marker_per_row[r] for r in active]
            local_weights = np.array([weights[r] for r in active])
            local_sweep = [
                np.array([sweep_eps[r] for r in active]) for sweep_eps in sweep
            ]
            best_local = self._select_best_bin(
                local_sweep, local_marker, local_weights
            )
            if best_local is None:
                break
            best_global = [active[i] for i in best_local]
            extracted.append(best_global)
            best_set = set(best_global)
            active = [r for r in active if r not in best_set]

        labels = np.full(n, -1, dtype=np.int64)
        for bin_id, rows in enumerate(extracted):
            for r in rows:
                labels[r] = bin_id
        next_id = len(extracted)
        for r in range(n):
            if labels[r] == -1:
                labels[r] = next_id
                next_id += 1
        return labels
