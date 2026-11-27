"""
Series selection on raw rows: the ``series`` filters and ``group_by`` keys of
``InputConfig``.

Both work on the raw frame a reader returns, before normalization, so that the row
policies (duplicates, sorting) apply to each series on its own. A name in ``group_by``
or ``series`` is resolved by :func:`resolve_key_column`: ``source`` and ``destination``
stand for the mapped ``source_column`` and ``destination_column``; any other name is a
raw column. Values are compared as text (:func:`key_values`).
"""

import logging
import math
from collections.abc import Sequence
from typing import Any

import pandas as pd

from ..models.config import InputConfig

logger = logging.getLogger(__name__)

SeriesKey = tuple[tuple[str, str | None], ...]
"""Identifies one series: ``(name, value)`` pairs in ``group_by`` order, e.g.
``(("destination", "8.8.8.8"),)``. A missing value is ``None``. The empty tuple is the
whole input (no ``group_by``)."""

MAX_LISTED_VALUES = 20
"""Error messages list at most this many available values or keys."""


def resolve_key_column(name: str, config: InputConfig) -> str:
    """
    Return the raw column a ``group_by`` or ``series`` name refers to.

    Parameters
    ----------
    name : str
        ``source``, ``destination`` or a raw column name.
    config : InputConfig
        Supplies ``source_column`` and ``destination_column``.

    Returns
    -------
    str
        The column name in the raw frame.
    """
    if name == "source":
        return config.source_column
    if name == "destination":
        return config.destination_column
    return name


def _text(value: Any) -> str | None:
    """One key value as text: ``None`` when missing, integral floats without ``.0``
    (a probe id column with gaps is read as floats)."""
    if value is None or (isinstance(value, float) and math.isnan(value)) or value is pd.NA:
        return None
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def key_values(df: pd.DataFrame, name: str, config: InputConfig) -> pd.Series:
    """
    Return the values of a key column as text (``None`` where missing).

    Parameters
    ----------
    df : pd.DataFrame
        Raw rows.
    name : str
        ``source``, ``destination`` or a raw column name.
    config : InputConfig
        Column mapping.

    Returns
    -------
    pd.Series
        Object series aligned with ``df``.

    Raises
    ------
    ValueError
        If the column does not exist.
    """
    column = resolve_key_column(name, config)
    if column not in df.columns:
        mapped = f" (mapped to column '{column}')" if column != name else ""
        raise ValueError(
            f"Input has no column for series key '{name}'{mapped}; columns: {list(df.columns)}"
        )
    return pd.Series([_text(v) for v in df[column].tolist()], index=df.index, dtype=object)


def format_series_key(key: SeriesKey) -> str:
    """
    Render a series key as ``name=value`` pairs, the syntax of ``--series``.

    Parameters
    ----------
    key : SeriesKey
        The key.

    Returns
    -------
    str
        E.g. ``"destination=8.8.8.8,source=10.0.0.1"``; ``"(all rows)"`` for the empty
        key; a missing value is written ``<missing>``.
    """
    if not key:
        return "(all rows)"
    return ",".join(f"{name}={'<missing>' if value is None else value}" for name, value in key)


def series_key_dict(key: SeriesKey) -> dict[str, str | None]:
    """
    Return a series key as a ``{name: value}`` mapping (the form written to JSON).

    Parameters
    ----------
    key : SeriesKey
        The key.

    Returns
    -------
    dict[str, str | None]
        Names in ``group_by`` order.
    """
    return dict(key)


def _listing(values: list[str]) -> str:
    """Comma-separated, truncated to ``MAX_LISTED_VALUES``."""
    shown = ", ".join(values[:MAX_LISTED_VALUES])
    if len(values) > MAX_LISTED_VALUES:
        shown += f", ... ({len(values)} in all)"
    return shown


def filter_rows(df: pd.DataFrame, config: InputConfig) -> pd.DataFrame:
    """
    Keep the rows that match every ``config.series`` filter.

    Parameters
    ----------
    df : pd.DataFrame
        Raw rows.
    config : InputConfig
        ``series`` filters and the column mapping.

    Returns
    -------
    pd.DataFrame
        The matching rows (``df`` itself when there is no filter).

    Raises
    ------
    ValueError
        If a filter names a column that does not exist, or no row matches; the message
        lists the values the column does have among the rows the earlier filters kept.
    """
    if not config.series:
        return df
    frame = df
    for name, wanted in config.series.items():
        values = key_values(frame, name, config)
        mask = values == wanted
        if not mask.any():
            available = sorted({v for v in values.tolist() if v is not None})
            raise ValueError(
                f"No rows match series filter {name}={wanted}; available {name} values: "
                f"{_listing(available) or '(none)'}"
            )
        frame = frame.loc[mask]
    logger.info(
        f"Series filter {format_series_key(tuple(config.series.items()))} keeps "
        f"{len(frame)} of {len(df)} rows"
    )
    return frame


def split_series(df: pd.DataFrame, config: InputConfig) -> dict[SeriesKey, pd.DataFrame]:
    """
    Split raw rows by ``config.group_by``.

    Parameters
    ----------
    df : pd.DataFrame
        Raw rows (already filtered by :func:`filter_rows`).
    config : InputConfig
        ``group_by`` names and the column mapping.

    Returns
    -------
    dict[SeriesKey, pd.DataFrame]
        One frame per distinct key, in order of first appearance; rows keep their file
        order. Without ``group_by``, the single key ``()`` maps to ``df``.

    Raises
    ------
    ValueError
        If a ``group_by`` column does not exist.
    """
    names = config.group_by
    if not names:
        return {(): df}
    columns = [key_values(df, name, config).tolist() for name in names]
    positions: dict[tuple[str | None, ...], list[int]] = {}
    for position, values in enumerate(zip(*columns, strict=True)):
        positions.setdefault(values, []).append(position)
    return {
        tuple(zip(names, values, strict=True)): df.iloc[rows] for values, rows in positions.items()
    }


def describe_keys(keys: list[SeriesKey]) -> str:
    """
    List series keys for an error or log message.

    Parameters
    ----------
    keys : list[SeriesKey]
        Keys.

    Returns
    -------
    str
        ``name=value`` keys separated by ``;``, truncated to ``MAX_LISTED_VALUES``.
    """
    rendered = [format_series_key(k) for k in keys]
    shown = "; ".join(rendered[:MAX_LISTED_VALUES])
    if len(rendered) > MAX_LISTED_VALUES:
        shown += f"; ... ({len(rendered)} in all)"
    return shown


def count_source_destination_pairs(df: pd.DataFrame, config: InputConfig) -> int:
    """
    Count the distinct (source, destination) pairs of raw rows.

    Parameters
    ----------
    df : pd.DataFrame
        Raw rows.
    config : InputConfig
        Column mapping; an absent source or destination column counts as one value.

    Returns
    -------
    int
        Number of distinct pairs (0 for an empty frame).
    """
    columns = [c for c in (config.source_column, config.destination_column) if c in df.columns]
    if df.empty:
        return 0
    if not columns:
        return 1
    return len(df[columns].drop_duplicates())


def path_key_names(
    df: pd.DataFrame, config: InputConfig, path_keys: Sequence[str] = ()
) -> list[str]:
    """
    Return the key names that tell the paths of raw rows apart.

    Parameters
    ----------
    df : pd.DataFrame
        Raw rows.
    config : InputConfig
        Column mapping, for ``source`` and ``destination``.
    path_keys : Sequence[str]
        Names a format declares for its paths (``RawData.path_keys``, e.g. ``prb_id``,
        ``msm_id`` and ``destination`` for RIPE Atlas); empty means ``source`` and
        ``destination``.

    Returns
    -------
    list[str]
        The names among those whose column is present, usable as ``group_by``.
    """
    names = list(path_keys) or ["source", "destination"]
    return [name for name in names if resolve_key_column(name, config) in df.columns]


def count_paths(df: pd.DataFrame, config: InputConfig, names: Sequence[str]) -> int:
    """
    Count the distinct combinations of the ``names`` key columns among raw rows.

    Parameters
    ----------
    df : pd.DataFrame
        Raw rows.
    config : InputConfig
        Column mapping, for ``source`` and ``destination``.
    names : Sequence[str]
        Key names whose columns are present (see :func:`path_key_names`).

    Returns
    -------
    int
        Number of distinct combinations (0 for an empty frame, 1 without names).
    """
    if df.empty:
        return 0
    if not names:
        return 1
    return len(df[[resolve_key_column(name, config) for name in names]].drop_duplicates())
