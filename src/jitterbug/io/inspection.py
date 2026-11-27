"""
A dry run of ingestion: what loading a file would do, without analyzing anything.

:func:`inspect_file` reads a file with the same reader, column mapping, units and row
policies as :meth:`jitterbug.io.DataLoader.load_from_file`, but instead of building a
dataset it reports the format, the mapping in effect, how many rows each policy would
drop or reorder, and the series the file holds. It never raises for a policy set to
``error``: it says the load would fail and why. ``jitterbug inspect`` prints this report.
"""

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..models import MAX_RTT_MS
from ..models.config import InputConfig
from .compression import compression_of
from .data_loader import DataLoader
from .normalize import (
    RTT_UNIT_SCALES,
    epoch_seconds,
    numeric_column,
    resolve_rtt_column,
)
from .normalize import _check_time_range as check_time_range
from .series import SeriesKey, filter_rows, path_key_names, series_key_dict, split_series

logger = logging.getLogger(__name__)


def inspect_file(
    file_path: str | Path,
    file_format: str | None = None,
    input_config: InputConfig | None = None,
) -> dict[str, Any]:
    """
    Report what loading ``file_path`` would do, without analyzing it.

    Parameters
    ----------
    file_path : str | Path
        Input file (possibly compressed).
    file_format : str | None
        A registered format, or ``None`` to infer it as the loader does.
    input_config : InputConfig | None
        Mapping, units, row policies, ``series`` and ``group_by``; ``None`` uses the
        defaults.

    Returns
    -------
    dict[str, Any]
        JSON-serializable report with the keys:

        - ``file``, ``format``, ``format_detail`` (e.g. ``scamper_warts``),
          ``compression`` (codec or ``None``), ``reader`` (reader metadata such as the
          delimiter and invalid JSON lines), ``columns`` (raw column names);
        - ``mapping``: the columns and units in effect, ``time_unit_applied`` (what
          ``auto`` inferred), ``fixed_by_format`` (settings the format imposes);
        - ``rows``: ``read``, ``selected`` (after ``series`` filters), ``missing``,
          ``non_positive``, ``too_large``, ``valid``;
        - ``policies``: per policy, its setting, the rows concerned and the effect
          (``dropped``, ``sorted``, ``kept`` or ``error``), counted per series as the
          loader applies them;
        - ``series_by``: the key names the ``series`` list is split by (``group_by``,
          or, when ``group_by`` is empty, the format's path keys present: source and
          destination, or e.g. ``prb_id``, ``msm_id``, ``destination`` for RIPE Atlas);
          ``grouped``: whether the analysis would separate them;
        - ``series``: per series, its ``key``, ``rows``, ``samples`` (after the
          policies), ``start``/``end`` (ISO 8601, UTC), ``duration_seconds``,
          ``median_interval_seconds`` and ``max_gap_seconds`` (between distinct
          timestamps), ``duplicate_timestamps`` and ``out_of_order_rows``;
        - ``errors``: why loading would fail (a missing column, an unreadable value,
          a policy set to ``error``); empty when it would succeed.

        When a ``series`` filter, the mapping or a ``group_by`` column fails, the report
        stops after ``columns`` with the reason in ``errors``.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the format is unknown or the file cannot be read at all.
    """
    config = input_config if input_config is not None else InputConfig()
    path = Path(file_path)
    reader, raw, config = DataLoader(config).read_raw(path, file_format, apply_series_filter=False)
    report: dict[str, Any] = {
        "file": str(path),
        "format": reader.name,
        "format_detail": raw.metadata.get("format"),
        "compression": compression_of(path),
        "reader": {k: v for k, v in raw.metadata.items() if k not in ("source", "format")},
        "columns": [str(c) for c in raw.frame.columns],
        "errors": [],
    }
    errors: list[str] = report["errors"]
    rows_read = len(raw.frame)
    try:
        frame = filter_rows(raw.frame, config).reset_index(drop=True)
        mapping, epochs, rtts = _map(frame, config)
        # Policies are applied per analyzed series: per group_by key, or the whole input.
        units = split_series(frame, config)
    except ValueError as e:
        errors.append(str(e))
        return report
    mapping["fixed_by_format"] = dict(raw.schema)
    report["mapping"] = mapping

    missing = (epochs.isna() | rtts.isna()).to_numpy()
    non_positive = ~missing & (rtts <= 0).to_numpy()
    too_large = ~missing & (rtts > MAX_RTT_MS).to_numpy()
    valid = ~(missing | non_positive | too_large)
    report["rows"] = {
        "read": rows_read,
        "selected": len(frame),
        "missing": int(missing.sum()),
        "non_positive": int(non_positive.sum()),
        "too_large": int(too_large.sum()),
        "valid": int(valid.sum()),
    }
    epoch_values = epochs.to_numpy(dtype=float)

    unit_stats = [
        _series_stats(epoch_values, valid, rows.index.to_numpy()) for rows in units.values()
    ]
    report["policies"] = _policies(config, report["rows"], unit_stats, errors)

    # Series listed: the group_by series, or the paths present (the format's path keys,
    # else source and destination).
    present = path_key_names(frame, config, raw.path_keys)
    listed: dict[SeriesKey, pd.DataFrame]
    if config.group_by:
        report["series_by"], report["grouped"] = list(config.group_by), True
        listed, listed_stats = units, unit_stats
    else:
        report["series_by"], report["grouped"] = present, False
        listed = split_series(frame, config.model_copy(update={"group_by": present}))
        listed_stats = [
            _series_stats(epoch_values, valid, rows.index.to_numpy()) for rows in listed.values()
        ]
    report["series"] = []
    for key, stats in zip(listed, listed_stats, strict=True):
        samples = stats["valid"] - (
            stats["duplicate_timestamps"] if config.duplicates in ("first", "min") else 0
        )
        entry = {"key": series_key_dict(key), "rows": stats["rows"], "samples": samples}
        entry.update({k: v for k, v in stats.items() if k not in ("rows", "valid")})
        report["series"].append(entry)
    if config.group_by:
        empty = [e for e in report["series"] if e["samples"] == 0]
        if empty and len(empty) == len(report["series"]):
            errors.append("No valid RTT rows in any series")
    elif report["rows"]["valid"] == 0:
        errors.append("No valid RTT rows: every row was missing, non-positive or too large")
    return report


def _map(frame: pd.DataFrame, config: InputConfig) -> tuple[dict[str, Any], pd.Series, pd.Series]:
    """Apply the column mapping and units as the normalizer does; raises ValueError."""
    if config.time_column not in frame.columns:
        raise ValueError(
            f"Input has no '{config.time_column}' column (time_column); "
            f"columns: {list(frame.columns)}"
        )
    rtt_column = resolve_rtt_column(frame, config)
    epochs, unit = epoch_seconds(frame[config.time_column], config.time_unit, config.time_column)
    check_time_range(epochs, config.time_column, unit)
    rtts = numeric_column(frame[rtt_column], rtt_column)
    multiplier, divisor = RTT_UNIT_SCALES[config.rtt_unit]
    rtts = rtts * multiplier / divisor
    mapping = {
        "time_column": config.time_column,
        "time_unit": config.time_unit,
        "time_unit_applied": unit,
        "rtt_column": rtt_column,
        "rtt_column_configured": config.rtt_column,
        "rtt_unit": config.rtt_unit,
        "source_column": config.source_column if config.source_column in frame.columns else None,
        "destination_column": (
            config.destination_column if config.destination_column in frame.columns else None
        ),
        "group_by": list(config.group_by),
        "series": dict(config.series),
    }
    return mapping, epochs, rtts


def _iso(epoch: float) -> str:
    """Unix seconds as ISO 8601 (UTC)."""
    return datetime.fromtimestamp(epoch, tz=UTC).isoformat()


def _series_stats(epochs: np.ndarray, valid: np.ndarray, positions: np.ndarray) -> dict[str, Any]:
    """Statistics of the rows at ``positions`` (file order) that survive the drops."""
    kept = epochs[positions[valid[positions]]]
    stats: dict[str, Any] = {
        "rows": len(positions),
        "valid": len(kept),
        "out_of_order_rows": int(np.sum(kept[1:] < kept[:-1])),
        "duplicate_timestamps": 0,
        "start": None,
        "end": None,
        "duration_seconds": None,
        "median_interval_seconds": None,
        "max_gap_seconds": None,
    }
    if len(kept) == 0:
        return stats
    ordered = np.sort(kept, kind="stable")
    stats["duplicate_timestamps"] = int(np.sum(ordered[1:] == ordered[:-1]))
    distinct = np.unique(ordered)
    stats["start"], stats["end"] = _iso(float(distinct[0])), _iso(float(distinct[-1]))
    stats["duration_seconds"] = float(distinct[-1] - distinct[0])
    if len(distinct) > 1:
        steps = np.diff(distinct)
        stats["median_interval_seconds"] = float(np.median(steps))
        stats["max_gap_seconds"] = float(np.max(steps))
    return stats


def _policies(
    config: InputConfig,
    rows: dict[str, int],
    units: list[dict[str, Any]],
    errors: list[str],
) -> dict[str, Any]:
    """What each row policy would do; appends to ``errors`` for policies set to error."""
    out_of_range = rows["non_positive"] + rows["too_large"]
    if config.out_of_range == "error" and out_of_range:
        errors.append(
            f"{out_of_range} RTT value(s) in the file outside (0, {MAX_RTT_MS:g}] ms "
            "(out_of_range: error)"
        )
    unsorted_series = sum(1 for u in units if u["out_of_order_rows"])
    out_of_order = sum(u["out_of_order_rows"] for u in units)
    if config.unsorted == "error" and out_of_order:
        errors.append(
            f"{unsorted_series} series not in time order, {out_of_order} row(s) earlier than "
            "the row before them (unsorted: error)"
        )
    duplicates = sum(u["duplicate_timestamps"] for u in units)
    if config.duplicates == "error" and duplicates:
        errors.append(
            f"{duplicates} row(s) share a timestamp with an earlier row (duplicates: error)"
        )
    dropping = config.duplicates in ("first", "min")
    return {
        "missing": {"rows": rows["missing"], "effect": "dropped"},
        "out_of_range": {
            "setting": config.out_of_range,
            "rows": out_of_range,
            "effect": "error" if config.out_of_range == "error" and out_of_range else "dropped",
        },
        "unsorted": {
            "setting": config.unsorted,
            "series_affected": unsorted_series,
            "rows": out_of_order,
            "effect": "error" if config.unsorted == "error" and out_of_order else "sorted",
        },
        "duplicates": {
            "setting": config.duplicates,
            "rows": duplicates,
            "dropped": duplicates if dropping else 0,
            "effect": (
                "error"
                if config.duplicates == "error" and duplicates
                else "dropped"
                if dropping
                else "kept"
            ),
        },
    }
