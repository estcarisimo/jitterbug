"""
The source registry: one small module per database back end.

A back end runs a query and returns a raw ``pandas.DataFrame``, nothing more, like a
file reader. :func:`fetch_raw` wraps it: it reads the secret from the environment, runs
the query, redacts any error, reads datetime columns as datetimes, applies the back
end's default column names, and returns :class:`~jitterbug.io.readers.RawData` that
then goes through the same series selection and normalization as a file.
"""

import importlib
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import pandas as pd

from ...models.config import InputConfig, SourceConfig
from ..readers.base import RawData
from .credentials import SourceError, read_secret, redact

logger = logging.getLogger(__name__)

FetchFunction = Callable[[SourceConfig, str, str | None], pd.DataFrame]


@dataclass(frozen=True)
class QuerySource:
    """
    A registered database back end.

    Attributes
    ----------
    kind : str
        ``SourceConfig.kind`` value.
    label : str
        Human-readable name for messages (``"ClickHouse"``).
    fetch : FetchFunction
        ``fetch(source, query, secret) -> DataFrame``. Imports its client lazily
        (raising an ``ImportError`` that names the extra), closes the connection, and
        may raise anything else: :func:`fetch_raw` redacts it.
    time_column : str | None
        Name the back end gives the sample time (``_time`` for Flux, ``time`` for
        InfluxDB 3); used when ``input.time_column`` is the default ``epoch`` and the
        result has no such column.
    rtt_column : str | None
        Column used as the RTT when ``input.rtt_column`` is unset and present (``_value``
        for Flux), before the usual synonyms.
    path_keys : tuple[str, ...]
        Key names that tell paths apart in the back end's results (``RawData.path_keys``).
    """

    kind: str
    label: str
    fetch: FetchFunction
    time_column: str | None = None
    rtt_column: str | None = None
    path_keys: tuple[str, ...] = ()


_REGISTRY: dict[str, QuerySource] = {}


def register_source(source: QuerySource) -> QuerySource:
    """
    Add a back end to the registry.

    Parameters
    ----------
    source : QuerySource
        The back end.

    Returns
    -------
    QuerySource
        ``source``, unchanged.

    Raises
    ------
    ValueError
        If the kind is already registered.
    """
    if source.kind in _REGISTRY:
        raise ValueError(f"A source of kind '{source.kind}' is already registered")
    _REGISTRY[source.kind] = source
    return source


def available_sources() -> list[str]:
    """
    Return the registered source kinds.

    Returns
    -------
    list[str]
        E.g. ``["clickhouse", "influxdb2", "influxdb3", "duckdb"]``.
    """
    return list(_REGISTRY)


def get_source(kind: str) -> QuerySource:
    """
    Return the back end registered as ``kind``.

    Parameters
    ----------
    kind : str
        Source kind.

    Returns
    -------
    QuerySource
        The back end.

    Raises
    ------
    ValueError
        If no back end has that kind.
    """
    try:
        return _REGISTRY[kind]
    except KeyError:
        raise ValueError(
            f"Unsupported source kind: {kind} (available: {', '.join(available_sources())})"
        ) from None


def import_client(module: str, extra: str, package: str) -> ModuleType:
    """
    Import a client library, or raise an ``ImportError`` naming the extra to install.

    Parameters
    ----------
    module : str
        Import name (``clickhouse_connect``).
    extra : str
        The ``jitterbug-inference`` extra that installs it.
    package : str
        Its PyPI distribution, for the message.

    Returns
    -------
    ModuleType
        The module.

    Raises
    ------
    ImportError
        If it is not installed.
    """
    try:
        return importlib.import_module(module)
    except ImportError as e:
        raise ImportError(
            f"This source needs {package}. Install the extra: "
            f"pip install 'jitterbug-inference[{extra}]'"
        ) from e


def query_text(source: SourceConfig) -> str:
    """
    The query to run: ``source.query``, or the content of ``source.query_file``.

    Parameters
    ----------
    source : SourceConfig
        The source.

    Returns
    -------
    str
        Query text, stripped.

    Raises
    ------
    ValueError
        If neither is set, the query is empty, or the file cannot be read.
    """
    if source.query_file is not None:
        try:
            text = Path(source.query_file).read_text(encoding="utf-8")
        except OSError as e:
            raise ValueError(f"Cannot read query file {source.query_file}: {e}") from e
    elif source.query is not None:
        text = source.query
    else:
        raise ValueError(
            f"A {source.kind} source needs a query: --query or --query-file on the command "
            "line, or query / query_file in the source profile"
        )
    text = text.strip()
    if not text:
        raise ValueError("The query is empty")
    return text


def describe_source(source: SourceConfig) -> dict[str, Any]:
    """
    Connection details of ``source`` safe to record and display (never a secret).

    Parameters
    ----------
    source : SourceConfig
        The source.

    Returns
    -------
    dict[str, Any]
        ``kind`` plus the settings that apply to it and are set: ``host``, ``port``,
        ``secure``, ``database``, ``org``, ``language`` or ``path``, and ``query_file``.
    """
    fields = {
        "clickhouse": ("host", "port", "secure", "database"),
        "influxdb2": ("host", "port", "secure", "org"),
        "influxdb3": ("host", "port", "secure", "database", "language"),
        "duckdb": ("path", "read_only"),
    }[source.kind]
    details: dict[str, Any] = {"kind": source.kind}
    for name in fields:
        value = getattr(source, name)
        if value is not None:
            details[name] = value
    if source.kind == "duckdb" and source.path is None:
        details["path"] = ":memory:"
        details.pop("read_only", None)
    if source.kind in ("clickhouse", "influxdb2", "influxdb3"):
        details.setdefault("host", "localhost")
    if source.query_file is not None:
        details["query_file"] = str(source.query_file)
    return details


def fetch_raw(source: SourceConfig, config: InputConfig) -> RawData:
    """
    Run the query of ``source`` and return its rows as raw data.

    Parameters
    ----------
    source : SourceConfig
        Back end, connection settings and query.
    config : InputConfig
        Input settings: the default column names of the back end apply only where these
        are left at their defaults, and a datetime time column is read as datetimes.

    Returns
    -------
    RawData
        ``frame`` with the result rows; ``metadata`` with ``source`` (the kind),
        ``source_details`` (:func:`describe_source`) and ``query``; ``schema`` with the
        ``InputConfig`` fields the result decides (time column, ``time_unit: iso`` for
        datetime columns, the RTT column of a Flux result).

    Raises
    ------
    ValueError
        If there is no query or it cannot be read.
    ImportError
        If the back end's client library is not installed.
    SourceError
        If the connection or the query fails; the message is redacted.
    """
    backend = get_source(source.kind)
    query = query_text(source)
    secret = read_secret(source)
    details = describe_source(source)
    logger.info(f"Querying {backend.label} source {details}")
    frame = run_redacted(backend.label, lambda: backend.fetch(source, query, secret), [secret])
    metadata = {"source": source.kind, "source_details": details, "query": query}
    return raw_from_result(frame, config, backend, metadata)


def run_redacted(
    label: str, call: Callable[[], pd.DataFrame], secrets: list[str | None]
) -> pd.DataFrame:
    """
    Run a client call; any error but a missing library becomes a redacted ``SourceError``.

    Parameters
    ----------
    label : str
        Back end name for the message (``"ClickHouse"``).
    call : Callable[[], pd.DataFrame]
        Connects, queries and closes.
    secrets : list[str | None]
        Values to scrub from the message, besides the generic credential patterns.

    Returns
    -------
    pd.DataFrame
        What ``call`` returned.

    Raises
    ------
    ImportError
        If the client library is missing (the message names the extra).
    SourceError
        For any other failure. The client's exception is not chained, since it may
        quote the secret.
    """
    error: str | None = None
    try:
        return call()
    except (ImportError, SourceError):
        raise
    except Exception as e:  # client errors of every kind
        error = redact(f"{type(e).__name__}: {e}", secrets)
    # Raised outside the except block so that the client's exception is not attached
    # as __context__ (a traceback would print it).
    raise SourceError(f"{label} query failed: {error}")


def raw_from_result(
    frame: pd.DataFrame, config: InputConfig, backend: QuerySource, metadata: dict[str, Any]
) -> RawData:
    """
    Wrap a query result as raw data, with the schema the result decides.

    Parameters
    ----------
    frame : pd.DataFrame
        Result rows.
    config : InputConfig
        Input settings.
    backend : QuerySource
        The back end, for its default column names.
    metadata : dict[str, Any]
        Recorded in the dataset metadata.

    Returns
    -------
    RawData
        The rows, metadata, schema and path keys.
    """
    logger.info(
        f"{backend.label} query returned {len(frame)} row(s), columns {list(frame.columns)}"
    )
    frame, schema = _result_schema(frame, config, backend)
    return RawData(frame=frame, metadata=metadata, schema=schema, path_keys=list(backend.path_keys))


def _is_datetime(column: pd.Series) -> bool:
    """Datetime dtype, or Python datetimes stored as objects."""
    if pd.api.types.is_datetime64_any_dtype(column):
        return True
    if column.dtype == object:
        return pd.api.types.infer_dtype(column, skipna=True) in ("datetime", "datetime64", "date")
    return False


def _result_schema(
    frame: pd.DataFrame, config: InputConfig, backend: QuerySource
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Default column names of the back end, and datetimes read as datetimes."""
    schema: dict[str, Any] = {}
    columns = set(frame.columns)
    if (
        backend.time_column is not None
        and config.time_column == InputConfig.model_fields["time_column"].default
        and config.time_column not in columns
        and backend.time_column in columns
    ):
        schema["time_column"] = backend.time_column
    if config.rtt_column is None and backend.rtt_column in columns:
        schema["rtt_column"] = backend.rtt_column
    time_column = schema.get("time_column", config.time_column)
    if time_column in columns and _is_datetime(frame[time_column]):
        if not pd.api.types.is_datetime64_any_dtype(frame[time_column]):
            frame = frame.assign(**{time_column: pd.to_datetime(frame[time_column], utc=True)})
        if config.time_unit not in ("iso", "auto"):
            logger.info(
                f"Time column '{time_column}' holds datetimes; reading them as such "
                f"instead of time_unit {config.time_unit}"
            )
            schema["time_unit"] = "iso"
    return frame, schema
