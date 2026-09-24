"""
Transparent Zstandard compression for input and output files.

A path whose last suffix is ``.zst`` (``rtts.csv.zst``, ``results.json.zst``) is read
and written through a Zstandard stream; any other path is a plain file. The codec is the
standard library's ``compression.zstd`` on Python 3.14 and later, otherwise the
``zstandard`` package (the ``zstd`` extra).
"""

from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import IO, Any, cast

logger = logging.getLogger(__name__)

ZSTD_SUFFIX = ".zst"

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


def inner_suffix(path: str | Path) -> str:
    """
    Return the suffix that describes the content, ignoring a trailing ``.zst``.

    Parameters
    ----------
    path : str | Path
        File path.

    Returns
    -------
    str
        Lower-case suffix, e.g. ``".csv"`` for both ``rtts.csv`` and ``rtts.csv.zst``;
        empty when there is none.
    """
    path = Path(path)
    if is_zstd(path):
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
    Open a text file, through a Zstandard stream when ``path`` ends in ``.zst``.

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
    if not is_zstd(path):
        return path.open(mode, encoding="utf-8", newline=newline)
    zstd = _zstd_module()
    logger.debug(f"Opening {path} through {zstd.__name__}")
    return cast(IO[str], zstd.open(path, f"{mode}t", encoding="utf-8", newline=newline))
