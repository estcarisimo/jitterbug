"""
Parsing shared by the JSON readers: a file is either one JSON value per line (JSON
lines) or a single JSON array of records (what the RIPE Atlas API returns, possibly
spread over many lines). Nothing is registered here.
"""

import json
import logging
from pathlib import Path
from typing import Any

from ..compression import open_text

logger = logging.getLogger(__name__)


def looks_like_json(line: str) -> bool:
    """
    Return whether a first non-empty line starts a JSON object or a JSON array of objects.

    Parameters
    ----------
    line : str
        First non-empty line of a file, stripped.

    Returns
    -------
    bool
        ``True`` for ``{...``, ``[``, ``[{...`` and ``[]``; ``False`` for ``[1700000000.1]
        64 bytes from ...`` and other text that merely starts with a bracket.
    """
    if line.startswith("{"):
        return True
    return line.startswith("[") and line[1:].lstrip()[:1] in ("", "{", "]")


def _first_character(path: Path) -> str:
    """First non-whitespace character of a (possibly compressed) text file, or ``""``."""
    with open_text(path) as f:
        while chunk := f.read(4096):
            stripped = chunk.lstrip()
            if stripped:
                return stripped[0]
    return ""


def parse_json_records(path: Path, label: str) -> tuple[list[Any], int]:
    """
    Parse a JSON-lines file or a file holding one JSON array.

    A file whose first non-whitespace character is ``[`` is parsed as one JSON document
    and must be an array; its elements are the records. Otherwise every non-empty line is
    one record, and lines that are not valid JSON are skipped with a warning.

    Parameters
    ----------
    path : Path
        File path (possibly compressed).
    label : str
        Format name for messages (``"JSON"``).

    Returns
    -------
    tuple[list[Any], int]
        The parsed records and the number of invalid lines skipped (always 0 for an
        array).

    Raises
    ------
    ValueError
        If the file cannot be read, or an array file is not valid JSON.
    ImportError
        If the file is compressed with a codec that is not installed.
    """
    try:
        if _first_character(path) == "[":
            with open_text(path) as f:
                document = json.load(f)
            if not isinstance(document, list):  # pragma: no cover - "[" starts an array
                raise ValueError("the top-level value is not an array")
            return document, 0
        records: list[Any] = []
        invalid = 0
        with open_text(path) as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning(f"Skipping invalid JSON line: {line[:120]!r}")
                    invalid += 1
    except ImportError:
        raise
    except Exception as e:
        raise ValueError(f"Failed to load {label} file {path}: {e}") from e
    return records, invalid
