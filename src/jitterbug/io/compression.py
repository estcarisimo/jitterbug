"""
Transparent compression for input and output files.

A path whose last suffix is ``.zst``, ``.gz``, ``.bz2`` or ``.xz`` (``rtts.csv.gz``,
``results.json.zst``) is read and written through the matching codec; any other path is
a plain file. gzip, bzip2 and xz use the standard library. The Zstandard codec is the
standard library's ``compression.zstd`` on Python 3.14 and later, otherwise the
``zstandard`` package (the ``zstd`` extra).
"""

from __future__ import annotations

import bz2
import gzip
import importlib
import logging
import lzma
from pathlib import Path
from typing import IO, Any, cast

logger = logging.getLogger(__name__)

ZSTD_SUFFIX = ".zst"
COMPRESSION_SUFFIXES = {".zst": "zstd", ".gz": "gzip", ".bz2": "bzip2", ".xz": "xz"}
"""Compressed-file suffixes and the codec each one names."""

INSTALL_HINT = (
    "Reading or writing .zst files needs Python 3.14+ or the zstandard package. "
    "Install the extra: pip install 'jitterbug-inference[zstd]'"
)


def is_zstd(path: str | Path) -> bool:
    """
    Return whether ``path`` names a Zstandard-compressed file (last suffix ``.zst``).

    Parameters
    ----------
    path : str | Path
        File path.

    Returns
    -------
    bool
        ``True`` for ``*.zst``, case-insensitive.
    """
    return Path(path).suffix.lower() == ZSTD_SUFFIX


def compression_of(path: str | Path) -> str | None:
    """
    Return the codec named by the last suffix of ``path``.

    Parameters
    ----------
    path : str | Path
        File path.

    Returns
    -------
    str | None
        ``"zstd"``, ``"gzip"``, ``"bzip2"`` or ``"xz"`` (case-insensitive suffix match);
        ``None`` for an uncompressed path.
    """
    return COMPRESSION_SUFFIXES.get(Path(path).suffix.lower())


def is_compressed(path: str | Path) -> bool:
    """
    Return whether ``path`` names a compressed file (see ``COMPRESSION_SUFFIXES``).

    Parameters
    ----------
    path : str | Path
        File path.

    Returns
    -------
    bool
        ``True`` for ``*.zst``, ``*.gz``, ``*.bz2`` and ``*.xz``, case-insensitive.
    """
    return compression_of(path) is not None


def inner_suffix(path: str | Path) -> str:
    """
    Return the suffix that describes the content, ignoring a compression suffix.

    Parameters
    ----------
    path : str | Path
        File path.

    Returns
    -------
    str
        Lower-case suffix, e.g. ``".csv"`` for ``rtts.csv``, ``rtts.csv.zst`` and
        ``rtts.csv.gz``; empty when there is none.
    """
    path = Path(path)
    if is_compressed(path):
        path = path.with_suffix("")
    return path.suffix.lower()


def _zstd_module() -> Any:
    """Return a module with a gzip-style ``open`` for Zstandard, or raise ImportError."""
    for name in ("compression.zstd", "zstandard"):
        try:
            return importlib.import_module(name)
        except ImportError:
            continue
    raise ImportError(INSTALL_HINT)


def open_text(path: str | Path, mode: str = "r", newline: str | None = None) -> IO[str]:
    """
    Open a text file, through a decompressing stream when ``path`` is compressed.

    Parameters
    ----------
    path : str | Path
        File path.
    mode : str
        ``"r"`` or ``"w"``.
    newline : str | None
        Passed to the text layer, as in :func:`open`. Use ``""`` when handing the stream
        to the ``csv`` module or pandas.

    Returns
    -------
    IO[str]
        Text stream; use it as a context manager.

    Raises
    ------
    ValueError
        If ``mode`` is not ``"r"`` or ``"w"``.
    ImportError
        If ``path`` ends in ``.zst`` and no Zstandard codec is available.
    """
    if mode not in ("r", "w"):
        raise ValueError(f"Unsupported mode: {mode!r} (expected 'r' or 'w')")
    path = Path(path)
    codec = compression_of(path)
    if codec is None:
        return path.open(mode, encoding="utf-8", newline=newline)
    text_mode = f"{mode}t"
    if codec != "zstd":
        logger.debug(f"Opening {path} through {codec}")
    if codec == "gzip":
        return cast(IO[str], gzip.open(path, text_mode, encoding="utf-8", newline=newline))
    if codec == "bzip2":
        return cast(IO[str], bz2.open(path, text_mode, encoding="utf-8", newline=newline))
    if codec == "xz":
        return cast(IO[str], lzma.open(path, text_mode, encoding="utf-8", newline=newline))
    zstd = _zstd_module()
    logger.debug(f"Opening {path} through {zstd.__name__}")
    return cast(IO[str], zstd.open(path, text_mode, encoding="utf-8", newline=newline))
