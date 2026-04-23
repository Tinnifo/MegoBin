_COMPLEMENT = str.maketrans("ACGT", "TGCA")


def reverse_complement(seq: str) -> str:
    """Return the reverse complement of a DNA sequence."""
    return seq.upper().translate(_COMPLEMENT)[::-1]


def canonical_kmer(kmer: str) -> str:
    """Return the lexicographically smaller of a k-mer and its reverse complement."""
    rc = reverse_complement(kmer)
    return min(kmer, rc)
