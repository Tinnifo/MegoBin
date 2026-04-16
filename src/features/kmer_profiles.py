from itertools import product
from pathlib import Path

import numpy as np

from .reverse_complement import canonical_kmer


def generate_kmers(k: int, alphabet: str = "ACGT") -> list[str]:
    """Generate all k-mers in order for the given alphabet."""
    return ["".join(combo) for combo in product(alphabet, repeat=k)]


def build_canonical_map(
    k: int, alphabet: str = "ACGT"
) -> tuple[list[str], dict[str, int]]:
    """Build mapping from all k-mers to canonical (RC-collapsed) indices.

    Returns:
        canonical_kmers: ordered list of unique canonical k-mers
        kmer_to_idx: maps every k-mer string to its canonical column index
    """
    all_kmers = generate_kmers(k, alphabet)

    seen: dict[str, int] = {}
    canonical_kmers: list[str] = []

    for kmer in all_kmers:
        canon = canonical_kmer(kmer)
        if canon not in seen:
            seen[canon] = len(canonical_kmers)
            canonical_kmers.append(canon)

    kmer_to_idx = {kmer: seen[canonical_kmer(kmer)] for kmer in all_kmers}
    return canonical_kmers, kmer_to_idx


def compute_kmer_profiles(
    sequences: list[str],
    k: int = 4,
    canonical: bool = False,
    alphabet: str = "ACGT",
    pseudocount: float = 0.0,
) -> np.ndarray:
    """Compute L1-normalized k-mer frequency profiles.

    Args:
        sequences: DNA sequences (non-ACGT bases in k-mer windows are skipped).
        k: K-mer length.
        canonical: If True, collapse reverse-complement k-mers (136 dims for k=4).
        alphabet: Base ordering for column enumeration ("ACGT" for RK, "ATGC" for SemiBin).
        pseudocount: Added to raw counts before normalization.

    Returns:
        (N, dims) float64 array. dims = 4^k (full) or number of canonical k-mers.
    """
    if canonical:
        _, kmer_to_idx = build_canonical_map(k, alphabet)
        num_features = max(kmer_to_idx.values()) + 1
    else:
        all_kmers = generate_kmers(k, alphabet)
        kmer_to_idx = {kmer: i for i, kmer in enumerate(all_kmers)}
        num_features = len(all_kmers)

    valid_bases = set("ACGT")
    profiles = np.zeros((len(sequences), num_features), dtype=np.float64)

    for i, seq in enumerate(sequences):
        seq_upper = seq.upper()
        for j in range(len(seq_upper) - k + 1):
            kmer = seq_upper[j : j + k]
            if all(b in valid_bases for b in kmer):
                idx = kmer_to_idx.get(kmer)
                if idx is not None:
                    profiles[i, idx] += 1

    if pseudocount > 0:
        profiles += pseudocount

    # L1 normalize: each row sums to 1
    row_sums = profiles.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0  # avoid div-by-zero for empty sequences
    profiles /= row_sums

    return profiles


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def read_fasta(path: Path) -> tuple[list[str], list[str]]:
    """Parse a FASTA file into (names, sequences)."""
    names: list[str] = []
    sequences: list[str] = []
    current_name: str | None = None
    current_seq: list[str] = []

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_name is not None:
                    names.append(current_name)
                    sequences.append("".join(current_seq))
                current_name = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line)

    if current_name is not None:
        names.append(current_name)
        sequences.append("".join(current_seq))

    return names, sequences


def compute_profiles_from_fasta(
    fasta_path: Path,
    k: int = 4,
    canonical: bool = False,
    alphabet: str = "ACGT",
    pseudocount: float = 0.0,
    min_length: int = 0,
    max_length: int | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Load FASTA, filter contigs by length, and compute k-mer profiles.

    Returns:
        (profiles, contig_names) — profiles shape (N_filtered, dims).
    """
    names, sequences = read_fasta(fasta_path)

    filtered_names: list[str] = []
    filtered_seqs: list[str] = []
    for name, seq in zip(names, sequences):
        if len(seq) < min_length:
            continue
        if max_length is not None and len(seq) > max_length:
            continue
        filtered_names.append(name)
        filtered_seqs.append(seq)

    profiles = compute_kmer_profiles(
        filtered_seqs,
        k=k,
        canonical=canonical,
        alphabet=alphabet,
        pseudocount=pseudocount,
    )

    return profiles, filtered_names
