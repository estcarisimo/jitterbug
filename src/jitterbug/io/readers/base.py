"""
The reader registry: one small reader per file format.

A reader turns a file into a raw ``pandas.DataFrame`` and nothing more; column mapping,
units and row policies are applied afterwards by :func:`jitterbug.io.normalize.normalize`.
Adding a format is one module that calls :func:`register_reader`.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from ...models import RTTDataset
from ...models.config import InputConfig
from ..compression import inner_suffix, open_text

logger = logging.getLogger(__name__)


@dataclass
class RawData:
    """
    What a reader returns.

    Attributes
    ----------
    frame : pd.DataFrame
        Raw rows, column names as in the file.
    metadata : dict[str, Any]
        Merged into the dataset metadata after normalization (``source``, ``format``,
        ``delimiter``, ...).
    schema : dict[str, Any]
        ``InputConfig`` fields the reader fixes because the format defines them (scamper
        writes ``epoch`` seconds and ``rtt`` milliseconds); they replace the user's
        values for this file. Row policies are never fixed.
    finish : Callable[[RTTDataset], None] | None
        Called on the normalized dataset, e.g. to add format-specific metadata.
    """

    frame: pd.DataFrame
    metadata: dict[str, Any] = field(default_factory=dict)
    schema: dict[str, Any] = field(default_factory=dict)
    finish: Callable[[RTTDataset], None] | None = None


ReadFunction = Callable[[Path, InputConfig], RawData]
SniffFunction = Callable[[str], bool]


@dataclass(frozen=True)
class Reader:
    """
    A registered file format.

    Attributes
    ----------
    name : str
        Format name, as passed to ``--format`` / ``file_format``.
    label : str
        Human-readable name for messages (``"CSV"``).
    read : ReadFunction
        ``read(path, input_config) -> RawData``. Raises ``ValueError`` for unreadable
        content; lets ``ImportError`` (a missing codec) through.
    extensions : tuple[str, ...]
        Lower-case suffixes that select this format (``(".csv",)``); a compression
        suffix after them is ignored.
    sniff : SniffFunction | None
        Given the first non-empty line of a file with an unknown extension, whether it
        looks like this format. Readers are tried in registration order.
    """

    name: str
    label: str
    read: ReadFunction
    extensions: tuple[str, ...] = ()
    sniff: SniffFunction | None = None


_REGISTRY: dict[str, Reader] = {}


def register_reader(reader: Reader) -> Reader:
    """
    Add ``reader`` to the registry.

    Parameters
    ----------
    reader : Reader
        The format to register.

    Returns
    -------
    Reader
        ``reader``, unchanged.

    Raises
    ------
    ValueError
        If the name or one of the extensions is already registered.
    """
    if reader.name in _REGISTRY:
        raise ValueError(f"A reader named '{reader.name}' is already registered")
    for other in _REGISTRY.values():
        shared = set(reader.extensions) & set(other.extensions)
        if shared:
            raise ValueError(f"Extension(s) {sorted(shared)} already belong to '{other.name}'")
    _REGISTRY[reader.name] = reader
    return reader


def available_formats() -> list[str]:
    """
    Return the registered format names, in registration order.

    Returns
    -------
    list[str]
        E.g. ``["json", "scamper", "jsonl", "tsv", "csv"]``.
    """
    return list(_REGISTRY)


def get_reader(name: str) -> Reader:
    """
    Return the reader registered as ``name``.

    Parameters
    ----------
    name : str
        Format name.

    Returns
    -------
    Reader
        The reader.

    Raises
    ------
    ValueError
        If no reader has that name.
    """
    try:
        return _REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"Unsupported file format: {name} (available: {', '.join(available_formats())})"
        ) from None


def first_line(path: Path, max_lines: int = 100) -> str:
    """
    Return the first non-empty line of a (possibly compressed) text file, stripped.

    Parameters
    ----------
    path : Path
        File path.
    max_lines : int
        Give up after this many blank lines.

    Returns
    -------
    str
        The line, or ``""`` if there is none.
    """
    with open_text(path) as f:
        for _, line in zip(range(max_lines), f, strict=False):
            if line.strip():
                return line.strip()
    return ""


def infer_format(path: str | Path) -> str:
    """
    Pick the format of ``path``: by extension (ignoring a compression suffix), otherwise
    by sniffing its first non-empty line.

    Parameters
    ----------
    path : str | Path
        File path.

    Returns
    -------
    str
        A registered format name.

    Raises
    ------
    ValueError
        If the file cannot be read as text or no reader recognizes it.
    ImportError
        If the file is compressed with a codec that is not installed.
    """
    path = Path(path)
    extension = inner_suffix(path)
    for reader in _REGISTRY.values():
        if extension in reader.extensions:
            return reader.name
    try:
        line = first_line(path)
    except ImportError:
        raise
    except Exception as e:
        raise ValueError(f"Cannot infer format for file: {path}") from e
    for reader in _REGISTRY.values():
        if reader.sniff is not None and reader.sniff(line):
            logger.debug(f"Format of {path} sniffed as {reader.name}")
            return reader.name
    raise ValueError(
        f"Cannot infer format for file: {path} (expected a delimited header or a JSON "
        f"object on the first line; pass the format explicitly, one of "
        f"{', '.join(available_formats())})"
    )
