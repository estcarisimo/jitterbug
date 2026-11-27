"""
CSV-like readers: ``csv`` (comma, tab or semicolon, detected) and ``tsv`` (tab).
"""

from pathlib import Path

import pandas as pd

from ...models.config import InputConfig
from ..compression import open_text
from .base import RawData, Reader, first_line, register_reader
from .json_records import looks_like_json

DELIMITERS = (",", "\t", ";")
"""Separators ``detect_delimiter`` chooses from, in order of preference on ties."""


def _looks_tab_separated(line: str) -> bool:
    """Sniffer: a header with a tab that is not JSON."""
    return "\t" in line and not looks_like_json(line)


def _looks_comma_separated(line: str) -> bool:
    """Sniffer: a header with a comma or semicolon that is not JSON."""
    return ("," in line or ";" in line) and not looks_like_json(line)


def detect_delimiter(header: str) -> str:
    """
    Pick the field separator of a header line.

    Parameters
    ----------
    header : str
        First non-empty line of the file.

    Returns
    -------
    str
        The one of ``,``, tab and ``;`` that occurs most often (comma on ties); comma
        when none occurs (a one-column file).
    """
    counts = {d: header.count(d) for d in DELIMITERS}
    best = max(DELIMITERS, key=lambda d: counts[d])
    return best if counts[best] else ","


def _read_delimited(path: Path, config: InputConfig, label: str, default: str | None) -> RawData:
    """Read a delimited file; ``default`` is the separator used when none is configured
    (``None`` detects it from the header)."""
    try:
        sep = config.delimiter or default or detect_delimiter(first_line(path))
        with open_text(path, newline="") as f:
            frame = pd.read_csv(f, sep=sep)
    except ImportError:
        raise
    except Exception as e:
        raise ValueError(f"Failed to load {label} file {path}: {e}") from e
    return RawData(frame=frame, metadata={"delimiter": sep})


def read_csv(path: Path, config: InputConfig) -> RawData:
    """
    Read a delimited file with a header row; the separator is ``config.delimiter`` or
    detected from the header (comma, tab or semicolon).

    Parameters
    ----------
    path : Path
        File path (possibly compressed).
    config : InputConfig
        Input settings; only ``delimiter`` is used here.

    Returns
    -------
    RawData
        The raw frame; ``metadata["delimiter"]`` is the separator used.
    """
    return _read_delimited(path, config, "CSV", None)


def read_tsv(path: Path, config: InputConfig) -> RawData:
    """
    Read a tab-separated file with a header row (``config.delimiter`` overrides the tab).

    Parameters
    ----------
    path : Path
        File path (possibly compressed).
    config : InputConfig
        Input settings; only ``delimiter`` is used here.

    Returns
    -------
    RawData
        The raw frame; ``metadata["delimiter"]`` is the separator used.
    """
    return _read_delimited(path, config, "TSV", "\t")


register_reader(
    Reader(
        name="csv",
        label="CSV",
        read=read_csv,
        extensions=(".csv",),
        sniff=_looks_comma_separated,
    )
)
register_reader(
    Reader(
        name="tsv",
        label="TSV",
        read=read_tsv,
        extensions=(".tsv", ".tab"),
        sniff=_looks_tab_separated,
    )
)
