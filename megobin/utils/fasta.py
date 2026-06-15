# This script has been adapted from the file available at the following address:
# https://github.com/BigDataBiology/SemiBin/blob/main/SemiBin/fasta.py


from typing import Iterator, Tuple, Union, TextIO


def fasta_iter(
    fname: Union[str, TextIO],
    full_header: bool = False,
) -> Iterator[Tuple[str, str]]:
    """Iterate over a (possibly compressed) FASTA file or open handle.

    Parameters
    ----------
    fname : str or file-like object
        Either a path to the FASTA file or an already-open text handle.
        When a path is given, the compression is inferred from its suffix:
        ``.gz`` -> gzip, ``.bz2`` -> bzip2, ``.xz`` -> lzma; otherwise the
        file is read as plain text. When a file-like object (anything with a
        ``readline`` method) is given, it is read as-is and no compression
        handling is applied.
    full_header : boolean (optional)
        If True, yields the full header. Otherwise (the default), only the
        first word

    Yields
    ------
    (h, seq) : tuple of (str, str)
        The header (see ``full_header``) and the concatenated sequence.
    """
    header = None
    chunks = []
    if hasattr(fname, "readline"):
        op = lambda f, _: f
    elif fname.endswith(".gz"):
        import gzip

        op = gzip.open
    elif fname.endswith(".bz2"):
        import bz2

        op = bz2.open
    elif fname.endswith(".xz"):
        import lzma

        op = lzma.open
    else:
        op = open
    with op(fname, "rt") as f:
        for line in f:
            if line[0] == ">":
                if header is not None:
                    yield header, "".join(chunks)
                line = line[1:].strip()
                if not line:
                    header = ""
                elif full_header:
                    header = line.strip()
                else:
                    header = line.split()[0]
                chunks = []
            else:
                chunks.append(line.strip())
        if header is not None:
            yield header, "".join(chunks)
