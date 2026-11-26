"""
JSON-lines readers: ``scamper`` (``scamper -O json`` output), ``jsonl`` (one flat object
per line, mapped through ``InputConfig``), and ``json``, which looks at the records and
picks one of the two.

Telling them apart: a file is scamper output when at least one record is a scamper
``ping`` (``"type": "ping"`` with a ``responses`` field) or one of scamper's bookkeeping
records (``"type"`` ``list``, ``cycle-def``, ``cycle-start`` or ``cycle-stop``). Every
other file is generic JSONL. A ``type`` field alone is not enough: other tools (RIPE
Atlas, for one) also write ``"type": "ping"``.
"""

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from ...models import RTTDataset
from ...models.config import InputConfig
from ..compression import open_text
from .base import RawData, Reader, register_reader

logger = logging.getLogger(__name__)

SCAMPER_BOOKKEEPING_TYPES = frozenset({"list", "cycle-def", "cycle-start", "cycle-stop"})
"""Record types scamper writes around its measurements."""

SCAMPER_SCHEMA = {
    "time_column": "epoch",
    "time_unit": "s",
    "rtt_column": "rtt",
    "rtt_unit": "ms",
    "source_column": "source",
    "destination_column": "destination",
}
"""The frame the scamper reader builds is in this fixed schema, whatever the user's
column and unit settings are."""


def is_scamper_record(record: Any) -> bool:
    """
    Return whether a parsed JSON line is a scamper record (see the module docstring).

    Parameters
    ----------
    record : Any
        One parsed line.

    Returns
    -------
    bool
        ``True`` for a scamper ping or bookkeeping record.
    """
    if not isinstance(record, dict):
        return False
    kind = record.get("type")
    if not isinstance(kind, str):
        return False
    return (kind == "ping" and "responses" in record) or kind in SCAMPER_BOOKKEEPING_TYPES


def _parse_lines(path: Path, label: str) -> tuple[list[Any], int]:
    """Parse every non-empty line; invalid JSON lines are skipped with a warning."""
    records: list[Any] = []
    invalid = 0
    try:
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


def _scamper_data(records: list[Any], invalid: int, path: Path) -> RawData:
    """One row per ping response with an ``rtt`` and a ``tx`` time."""
    rows = []
    try:
        for data in records:
            if not (isinstance(data, dict) and data.get("type") == "ping"):
                continue
            if "responses" not in data:
                continue
            source = data.get("src", "")
            destination = data.get("dst", "")
            for response in data["responses"]:
                if "rtt" not in response:
                    continue  # timeout
                tx = response.get("tx", {})
                if "sec" in tx and "usec" in tx:
                    epoch = tx["sec"] + tx["usec"] / 1e6
                    rows.append((epoch, float(response["rtt"]), source, destination))
    except Exception as e:
        raise ValueError(f"Failed to load JSON file {path}: {e}") from e
    if not rows:
        raise ValueError(f"No valid RTT measurements found in JSON file {path}")

    def finish(dataset: RTTDataset) -> None:
        dropped = dataset.metadata["dropped_rows"]
        dataset.metadata["dropped_responses"] = dropped["non_positive"] + dropped["too_large"]
        dataset.metadata["total_measurements"] = len(dataset)

    return RawData(
        frame=pd.DataFrame(rows, columns=["epoch", "rtt", "source", "destination"]),
        metadata={"source": "json", "format": "scamper_warts", "invalid_lines": invalid},
        schema=SCAMPER_SCHEMA,
        finish=finish,
    )


def _jsonl_data(records: list[Any], invalid: int, path: Path) -> RawData:
    """One row per record; the columns are the records' top-level fields."""
    not_objects = sum(not isinstance(r, dict) for r in records)
    if not_objects:
        raise ValueError(
            f"Failed to load JSONL file {path}: {not_objects} line(s) are not JSON objects"
        )
    if not records:
        raise ValueError(f"No records found in JSONL file {path}")
    return RawData(
        frame=pd.DataFrame.from_records(records),
        metadata={"source": "jsonl", "format": "jsonl", "invalid_lines": invalid},
    )


def read_scamper(path: Path, config: InputConfig) -> RawData:
    """
    Read scamper JSON output (``scamper -O json``, ``sc_warts2json``).

    Only ``ping`` records are used: each response with an ``rtt`` (milliseconds) and a
    ``tx`` time becomes a row, with ``src``/``dst`` as source and destination. Other
    records, timeouts and invalid JSON lines are skipped.

    Parameters
    ----------
    path : Path
        File path (possibly compressed).
    config : InputConfig
        Unused: the scamper schema is fixed (``SCAMPER_SCHEMA``).

    Returns
    -------
    RawData
        The raw frame in ``SCAMPER_SCHEMA``.

    Raises
    ------
    ValueError
        If the file cannot be read or holds no usable response.
    """
    return _scamper_data(*_parse_lines(path, "JSON"), path)


def read_jsonl(path: Path, config: InputConfig) -> RawData:
    """
    Read generic JSON lines: one flat object per line, fields mapped through
    ``InputConfig`` like CSV columns. Invalid JSON lines are skipped with a warning.

    Parameters
    ----------
    path : Path
        File path (possibly compressed).
    config : InputConfig
        Unused here; the mapping happens during normalization.

    Returns
    -------
    RawData
        The raw frame.

    Raises
    ------
    ValueError
        If the file cannot be read, a line is not a JSON object, or there is no record.
    """
    return _jsonl_data(*_parse_lines(path, "JSONL"), path)


def read_json(path: Path, config: InputConfig) -> RawData:
    """
    Read JSON lines as scamper output or generic JSONL, decided by content (see the
    module docstring). A file with no records at all is treated as scamper output.

    Parameters
    ----------
    path : Path
        File path (possibly compressed).
    config : InputConfig
        Input settings.

    Returns
    -------
    RawData
        The raw frame; ``metadata["format"]`` is ``scamper_warts`` or ``jsonl``.
    """
    records, invalid = _parse_lines(path, "JSON")
    if not records or any(is_scamper_record(r) for r in records):
        return _scamper_data(records, invalid, path)
    logger.info(f"{path} has no scamper records; reading it as generic JSONL")
    return _jsonl_data(records, invalid, path)


register_reader(
    Reader(
        name="json",
        label="JSON",
        read=read_json,
        extensions=(".json", ".jsonl", ".ndjson"),
        sniff=lambda line: line.startswith("{"),
    )
)
register_reader(Reader(name="scamper", label="scamper JSON", read=read_scamper))
register_reader(Reader(name="jsonl", label="JSONL", read=read_jsonl))
