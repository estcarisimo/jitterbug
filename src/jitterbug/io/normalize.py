"""
The single normalization path from raw rows to an ``RTTDataset``.

Every reader (CSV, TSV, JSONL, scamper, ...) and ``DataLoader.load_from_dataframe``
end here, so the input contract is enforced in one place. The steps, in order:

1. **Mapping.** Find the time and RTT columns named by ``InputConfig`` (the RTT column
   falls back to the synonyms ``values``, ``rtt_value``, ``rtt``, ``latency``).
2. **Units.** Convert times to Unix seconds (from ``s``, ``ms``, ``us``, ``ns``, ISO 8601
   strings, or a unit inferred by ``auto``) and RTTs to milliseconds. A value that cannot
   be read in the given unit is an error.
3. **Row policies.** Drop rows with a missing time or RTT; drop (or reject) RTTs outside
   ``(0, MAX_RTT_MS]``; sort (or reject) rows out of time order; keep, thin out or reject
   rows that share a timestamp. Nothing is imputed or resampled.
"""

import logging
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from ..models import MAX_RTT_MS, RTTDataset, RTTMeasurement
from ..models.config import RTT_COLUMN_SYNONYMS, InputConfig

logger = logging.getLogger(__name__)

TIME_UNIT_DIVISORS = {"s": 1.0, "ms": 1e3, "us": 1e6, "ns": 1e9}
"""Divide a numeric time in this unit by the value to get Unix seconds."""

RTT_UNIT_SCALES = {"ms": (1.0, 1.0), "s": (1e3, 1.0), "us": (1.0, 1e3), "ns": (1.0, 1e6)}
"""``(multiplier, divisor)`` that turn an RTT in this unit into milliseconds. Dividing,
rather than multiplying by a fraction, keeps e.g. 1500 us at exactly 1.5 ms."""

AUTO_TIME_UNIT_LIMITS = ((1e11, "s"), (1e14, "ms"), (1e17, "us"))
"""``time_unit: auto`` picks the first unit whose limit exceeds the median absolute time:
Unix seconds stay below 1e11 until the year 5138, milliseconds since 1973 are above it,
and each finer unit is 1000 times larger. Anything at or above 1e17 is nanoseconds."""


def normalize(df: pd.DataFrame, config: InputConfig | None = None) -> RTTDataset:
    """
    Turn a raw frame into a validated, time-ordered ``RTTDataset``.

    Parameters
    ----------
    df : pd.DataFrame
        Raw rows, one RTT sample each, with column names as in the input.
    config : InputConfig | None
        Column mapping, units and row policies; ``None`` uses the defaults (``epoch`` in
        seconds, RTT in milliseconds, duplicates kept, out-of-range RTTs dropped,
        unsorted rows sorted).

    Returns
    -------
    RTTDataset
        The samples, with metadata describing what was done: ``original_columns``,
        ``rtt_column``, ``time_column``, ``time_unit`` (the unit used, inferred or not),
        ``time_unit_inferred``, ``rtt_unit``, ``total_rows``, ``dropped_rows`` (counts by
        reason), ``sorted_on_load``, ``duplicate_rows`` (rows sharing a timestamp with
        an earlier one) and ``duplicates_removed``.

    Raises
    ------
    ValueError
        If a column is missing, a value cannot be read in its unit, a policy set to
        ``error`` is violated, or no valid rows remain.
    """
    config = config or InputConfig()
    time_column = config.time_column
    if time_column not in df.columns:
        raise ValueError(
            f"Input has no '{time_column}' column (time_column); columns: {list(df.columns)}"
        )
    rtt_column = resolve_rtt_column(df, config)

    epochs, time_unit = epoch_seconds(df[time_column], config.time_unit, time_column)
    _check_time_range(epochs, time_column, time_unit)
    rtts = numeric_column(df[rtt_column], rtt_column)
    multiplier, divisor = RTT_UNIT_SCALES[config.rtt_unit]
    if multiplier != 1.0:
        rtts = rtts * multiplier
    if divisor != 1.0:
        rtts = rtts / divisor

    # Row filters, applied together so the counts refer to the original frame
    missing = epochs.isna() | rtts.isna()
    non_positive = ~missing & (rtts <= 0)
    too_large = ~missing & (rtts > MAX_RTT_MS)
    if config.out_of_range == "error" and (non_positive.any() or too_large.any()):
        bad = non_positive | too_large
        raise ValueError(
            f"{int(bad.sum())} RTT value(s) outside (0, {MAX_RTT_MS:g}] ms, e.g. "
            f"{float(rtts[bad].iloc[0])!r} ms in column '{rtt_column}' (out_of_range: error)"
        )
    dropped = {
        "missing": int(missing.sum()),
        "non_positive": int(non_positive.sum()),
        "too_large": int(too_large.sum()),
    }
    keep = ~(missing | non_positive | too_large)
    for reason, count in dropped.items():
        if count:
            logger.warning(f"Dropping {count} row(s) with {reason.replace('_', '-')} RTT")
    if not keep.any():
        raise ValueError("No valid RTT rows: every row was missing, non-positive or too large")

    frame = df.loc[keep]
    epoch_values = epochs[keep].to_numpy(dtype=float)
    rtt_values = rtts[keep].to_numpy(dtype=float)

    needs_sort = bool(np.any(epoch_values[:-1] > epoch_values[1:]))
    if needs_sort:
        if config.unsorted == "error":
            position = int(np.argmax(epoch_values[:-1] > epoch_values[1:])) + 1
            raise ValueError(
                f"Rows are not in time order: time {float(epoch_values[position])!r} s comes "
                f"after {float(epoch_values[position - 1])!r} s (unsorted: error)"
            )
        logger.warning(f"Rows are not in time order; sorting by {time_column}")
        order = np.argsort(epoch_values, kind="stable")
        frame = frame.iloc[order]
        epoch_values, rtt_values = epoch_values[order], rtt_values[order]

    selected, duplicate_rows = _apply_duplicates_policy(epoch_values, rtt_values, config.duplicates)
    if selected is not None:
        frame = frame.iloc[selected]
        epoch_values, rtt_values = epoch_values[selected], rtt_values[selected]
        logger.warning(
            f"Removed {duplicate_rows} row(s) sharing a timestamp (duplicates: {config.duplicates})"
        )

    sources = _optional_column(frame, config.source_column)
    destinations = _optional_column(frame, config.destination_column)
    measurements = [
        RTTMeasurement(
            timestamp=datetime.fromtimestamp(epoch, tz=UTC),
            epoch=epoch,
            rtt_value=rtt,
            source=src,
            destination=dst,
        )
        for epoch, rtt, src, dst in zip(
            epoch_values.tolist(), rtt_values.tolist(), sources, destinations, strict=True
        )
    ]

    metadata: dict[str, Any] = {
        "source": "dataframe",
        "original_columns": list(df.columns),
        "rtt_column": rtt_column,
        "time_column": time_column,
        "time_unit": time_unit,
        "time_unit_inferred": config.time_unit == "auto",
        "rtt_unit": config.rtt_unit,
        "total_rows": len(df),
        "dropped_rows": dropped,
        "sorted_on_load": needs_sort,
        "duplicate_rows": duplicate_rows,
        "duplicates_removed": duplicate_rows if selected is not None else 0,
    }
    return RTTDataset(measurements=measurements, metadata=metadata)


EPOCH_RANGE = (
    datetime(1, 1, 2, tzinfo=UTC).timestamp(),
    datetime(9999, 12, 30, tzinfo=UTC).timestamp(),
)
"""Unix seconds a timestamp can hold (``datetime`` years 1 to 9999, a day of margin on
each side for local-time conversions)."""


def _check_time_range(epochs: pd.Series, name: str, time_unit: str) -> None:
    """Reject infinite times and times ``datetime`` cannot represent, naming the column;
    such values usually mean the wrong ``time_unit``."""
    present = epochs.dropna()
    bad = ~np.isfinite(present) | (present < EPOCH_RANGE[0]) | (present > EPOCH_RANGE[1])
    if bad.any():
        raise ValueError(
            f"Column '{name}' has {int(bad.sum())} time(s) outside the years 1-9999 when read "
            f"as {time_unit}, e.g. {float(present[bad].iloc[0])!r} s; check time_unit"
        )


def resolve_rtt_column(df: pd.DataFrame, config: InputConfig) -> str:
    """
    Return the name of the RTT column of ``df`` under ``config``.

    Parameters
    ----------
    df : pd.DataFrame
        Raw rows.
    config : InputConfig
        ``rtt_column`` if set, otherwise the first synonym present.

    Returns
    -------
    str
        Column name.

    Raises
    ------
    ValueError
        If the configured column, or every synonym, is missing.
    """
    if config.rtt_column is not None:
        if config.rtt_column not in df.columns:
            raise ValueError(
                f"Input has no '{config.rtt_column}' column (rtt_column); "
                f"columns: {list(df.columns)}"
            )
        return config.rtt_column
    found = next((c for c in RTT_COLUMN_SYNONYMS if c in df.columns), None)
    if found is None:
        raise ValueError(
            "Input has no RTT column named one of "
            + ", ".join(f"'{c}'" for c in RTT_COLUMN_SYNONYMS)
            + f" (set rtt_column to use another name); columns: {list(df.columns)}"
        )
    return found


def epoch_seconds(column: pd.Series, time_unit: str, name: str) -> tuple[pd.Series, str]:
    """
    Convert a time column to Unix seconds (floats, NaN where missing).

    Parameters
    ----------
    column : pd.Series
        Raw time values.
    time_unit : str
        One of ``InputConfig.time_unit``'s values.
    name : str
        Column name, for messages.

    Returns
    -------
    tuple[pd.Series, str]
        Seconds since the epoch, and the unit that was applied (the inferred one for
        ``auto``).

    Raises
    ------
    ValueError
        If a value cannot be read in the unit.
    """
    if time_unit == "auto":
        time_unit = infer_time_unit(column, name)
    if time_unit == "iso":
        return _datetime_seconds(column, name), time_unit
    if pd.api.types.is_datetime64_any_dtype(column):
        raise ValueError(
            f"Column '{name}' holds datetimes, not {time_unit} numbers; set time_unit to iso "
            "or auto"
        )
    hint = " (set time_unit to iso or auto for datetime strings)"
    values = numeric_column(column, name, hint=hint)
    divisor = TIME_UNIT_DIVISORS[time_unit]
    if divisor != 1.0:
        values = values / divisor
    return values, time_unit


def infer_time_unit(column: pd.Series, name: str) -> str:
    """
    Guess the unit of a time column (``time_unit: auto``).

    Datetime values, and strings that are not numbers, are ``iso``. Numbers are judged
    by the median absolute value against ``AUTO_TIME_UNIT_LIMITS``. The choice is logged
    at ``INFO`` level.

    Parameters
    ----------
    column : pd.Series
        Raw time values.
    name : str
        Column name, for the log message.

    Returns
    -------
    str
        ``s``, ``ms``, ``us``, ``ns`` or ``iso``.
    """
    if pd.api.types.is_datetime64_any_dtype(column):
        unit = "iso"
        logger.info(f"Time column '{name}' holds datetimes; time_unit auto -> iso")
        return unit
    numbers = pd.to_numeric(column, errors="coerce")
    present = column.notna() & ~_blank(column)
    if (numbers.isna() & present).any():
        logger.info(f"Time column '{name}' holds non-numeric strings; time_unit auto -> iso")
        return "iso"
    magnitudes = numbers.abs().dropna()
    if magnitudes.empty:
        return "s"
    median = float(magnitudes.median())
    unit = next((u for limit, u in AUTO_TIME_UNIT_LIMITS if median < limit), "ns")
    logger.info(f"Time column '{name}' median {median:.6g}; time_unit auto -> {unit}")
    return unit


def datetimes_to_epoch(times: pd.Series) -> pd.Series:
    """
    Seconds since the Unix epoch for tz-aware UTC datetimes, whatever their resolution.

    Rounded to microseconds like ``pd.Timestamp.timestamp()``: a float64 epoch cannot
    hold nanoseconds. ``NaT`` becomes NaN.

    Parameters
    ----------
    times : pd.Series
        ``datetime64[*, UTC]`` values.

    Returns
    -------
    pd.Series
        Float seconds.
    """
    nanos = (times - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(nanoseconds=1)
    return (nanos.astype(float) / 1e9).round(6)


def _datetime_seconds(column: pd.Series, name: str) -> pd.Series:
    """ISO 8601 strings or datetimes to Unix seconds; naive values are UTC."""
    if pd.api.types.is_numeric_dtype(column) and not pd.api.types.is_bool_dtype(column):
        raise ValueError(
            f"Column '{name}' holds numbers, not datetimes; set time_unit to s, ms, us, ns or auto"
        )
    if pd.api.types.is_datetime64_any_dtype(column):
        times = pd.to_datetime(column, utc=True)
    else:
        cleaned = column.where(~_blank(column))
        times = pd.to_datetime(cleaned, utc=True, errors="coerce", format="ISO8601")
        bad = times.isna() & cleaned.notna()
        if bad.any():
            example = column[bad].iloc[0]
            raise ValueError(
                f"Column '{name}' has {int(bad.sum())} value(s) that are not ISO 8601 "
                f"datetimes, e.g. {example!r}"
            )
    return datetimes_to_epoch(times)


def numeric_column(column: pd.Series, name: str, hint: str = "") -> pd.Series:
    """
    Return ``column`` as floats; NaN and blank strings become NaN.

    Parameters
    ----------
    column : pd.Series
        Raw values.
    name : str
        Column name, for messages.
    hint : str
        Appended to the error message.

    Returns
    -------
    pd.Series
        Float values.

    Raises
    ------
    ValueError
        If any other value is not numeric.
    """
    values = pd.to_numeric(column, errors="coerce")
    bad = values.isna() & column.notna() & ~_blank(column)
    if bad.any():
        example = column[bad].iloc[0]
        raise ValueError(
            f"Column '{name}' has {int(bad.sum())} non-numeric value(s), e.g. {example!r}{hint}"
        )
    return values.astype(float)


def _blank(column: pd.Series) -> pd.Series:
    """Mask of empty or whitespace-only strings."""
    return column.map(lambda v: isinstance(v, str) and v.strip() == "").astype(bool)


def _apply_duplicates_policy(
    epochs: np.ndarray, rtts: np.ndarray, policy: str
) -> tuple[np.ndarray | None, int]:
    """
    Apply ``InputConfig.duplicates`` to time-sorted arrays.

    Returns the positions to keep (``None`` keeps every row) and the number of rows that
    share a timestamp with an earlier row.
    """
    repeated = np.zeros(len(epochs), dtype=bool)
    repeated[1:] = epochs[1:] == epochs[:-1]
    count = int(repeated.sum())
    if count == 0 or policy == "keep":
        return None, count
    if policy == "error":
        raise ValueError(
            f"{count} row(s) share a timestamp with an earlier row, e.g. "
            f"{float(epochs[repeated][0])!r} s (duplicates: error)"
        )
    if policy == "first":
        return np.flatnonzero(~repeated), count
    # "min": the lowest RTT of each timestamp, the first one on ties (lexsort is stable)
    groups = np.cumsum(~repeated)
    order = np.lexsort((rtts, groups))
    first_in_group = np.ones(len(order), dtype=bool)
    first_in_group[1:] = groups[order][1:] != groups[order][:-1]
    return np.sort(order[first_in_group]), count


def _optional_column(df: pd.DataFrame, column: str) -> list[str | None]:
    """Per-row values of an optional string column, ``None`` where absent or missing."""
    if column not in df.columns:
        return [None] * len(df)
    return [None if pd.isna(v) else str(v) for v in df[column].tolist()]
