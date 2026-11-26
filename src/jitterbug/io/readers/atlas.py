"""
RIPE Atlas ping results: format ``atlas``.

Accepts what the RIPE Atlas API returns for a measurement's results (one JSON array) and
one result per line (JSON lines, as streamed or saved by tools such as
``ripe-atlas-cousteau``). A result is a RIPE Atlas record when it has ``prb_id``,
``msm_id`` and ``result`` fields; ``.json`` files holding such records are read with this
reader by the ``json`` format (see :mod:`jitterbug.io.readers.jsonl`).

Only ``"type": "ping"`` results are used. Each entry of a result's ``result`` list with
an ``rtt`` (milliseconds) becomes one sample at the result's ``timestamp`` (Unix
seconds; every packet of one result shares it). The source is ``from`` (the probe's
public address), or ``src_addr`` when ``from`` is empty; the destination is
``dst_addr``, or ``dst_name``. ``prb_id``, ``msm_id`` and ``src_addr`` are kept as
columns.
"""

import logging
import math
from pathlib import Path
from typing import Any

import pandas as pd

from ...models import RTTDataset
from ...models.config import InputConfig
from .base import RawData, Reader, register_reader
from .json_records import parse_json_records

logger = logging.getLogger(__name__)

ATLAS_KEYS = ("prb_id", "msm_id", "result")
"""Fields every RIPE Atlas result has and that together identify one."""

ATLAS_SCHEMA = {
    "time_column": "epoch",
    "time_unit": "s",
    "rtt_column": "rtt",
    "rtt_unit": "ms",
    "source_column": "source",
    "destination_column": "destination",
}
"""The frame the RIPE Atlas reader builds is in this fixed schema."""

ATLAS_COLUMNS = ["epoch", "rtt", "source", "destination", "prb_id", "msm_id", "src_addr"]


def is_atlas_record(record: Any) -> bool:
    """
    Return whether a parsed JSON record is a RIPE Atlas result (of any measurement type).

    Parameters
    ----------
    record : Any
        One parsed record.

    Returns
    -------
    bool
        ``True`` for an object with ``prb_id``, ``msm_id`` and ``result`` fields.
    """
    return isinstance(record, dict) and all(key in record for key in ATLAS_KEYS)


def _number(value: Any) -> float | None:
    """A finite int or float (not a bool), else ``None``."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value) if math.isfinite(value) else None


def atlas_data(records: list[Any], invalid: int, path: Path) -> RawData:
    """
    Build the raw frame from parsed RIPE Atlas results.

    Parameters
    ----------
    records : list[Any]
        Parsed records; those that are not RIPE Atlas ping results are skipped.
    invalid : int
        Invalid JSON lines skipped while parsing, recorded in the metadata.
    path : Path
        File path, for messages.

    Returns
    -------
    RawData
        One row per reply, in ``ATLAS_SCHEMA``; ``metadata`` counts ``ping_results``,
        ``skipped_records`` (other measurement types and non-Atlas records),
        ``timeouts``, ``errors`` (error entries, and results without a usable timestamp
        or result list) and ``duplicate_replies``; after normalization
        ``dropped_responses`` counts replies outside ``(0, MAX_RTT_MS]``.

    Raises
    ------
    ValueError
        If there is no ping result, or no reply in any of them.
    """
    rows: list[tuple[Any, ...]] = []
    counts = {"ping_results": 0, "skipped": 0, "timeouts": 0, "errors": 0, "duplicates": 0}
    for record in records:
        if not is_atlas_record(record) or record.get("type") != "ping":
            counts["skipped"] += 1
            continue
        counts["ping_results"] += 1
        epoch = _number(record.get("timestamp"))
        entries = record["result"]
        if epoch is None or not isinstance(entries, list):
            counts["errors"] += 1
            continue
        source = record.get("from") or record.get("src_addr")
        destination = record.get("dst_addr") or record.get("dst_name")
        for entry in entries:
            rtt = _number(entry.get("rtt")) if isinstance(entry, dict) else None
            if rtt is not None:
                if entry.get("dup"):
                    counts["duplicates"] += 1
                    continue
                rows.append(
                    (
                        epoch,
                        rtt,
                        source,
                        destination,
                        record.get("prb_id"),
                        record.get("msm_id"),
                        record.get("src_addr"),
                    )
                )
            elif isinstance(entry, dict) and "x" in entry:
                counts["timeouts"] += 1
            else:
                counts["errors"] += 1
    if not counts["ping_results"]:
        raise ValueError(
            f"No RIPE Atlas ping results found in {path} ({counts['skipped']} other record(s))"
        )
    if not rows:
        raise ValueError(
            f"No RTT measurements found in RIPE Atlas file {path}: "
            f"{counts['ping_results']} ping result(s), {counts['timeouts']} timeout(s), "
            f"{counts['errors']} error(s)"
        )
    if counts["skipped"]:
        logger.info(f"Skipped {counts['skipped']} record(s) that are not RIPE Atlas pings")

    def finish(dataset: RTTDataset) -> None:
        dropped = dataset.metadata["dropped_rows"]
        dataset.metadata["dropped_responses"] = dropped["non_positive"] + dropped["too_large"]
        dataset.metadata["total_measurements"] = len(dataset)

    return RawData(
        frame=pd.DataFrame(rows, columns=ATLAS_COLUMNS),
        metadata={
            "format": "ripe_atlas",
            "invalid_lines": invalid,
            "ping_results": counts["ping_results"],
            "skipped_records": counts["skipped"],
            "timeouts": counts["timeouts"],
            "errors": counts["errors"],
            "duplicate_replies": counts["duplicates"],
        },
        schema=ATLAS_SCHEMA,
        finish=finish,
    )


def read_atlas(path: Path, config: InputConfig) -> RawData:
    """
    Read RIPE Atlas ping results (a JSON array or one result per line).

    Parameters
    ----------
    path : Path
        File path (possibly compressed).
    config : InputConfig
        Unused: the schema is fixed (``ATLAS_SCHEMA``); the row policies apply.

    Returns
    -------
    RawData
        See :func:`atlas_data`.

    Raises
    ------
    ValueError
        If the file cannot be read or holds no usable ping result.
    """
    return atlas_data(*parse_json_records(path, "RIPE Atlas"), path)


register_reader(Reader(name="atlas", label="RIPE Atlas", read=read_atlas))
