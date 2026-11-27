"""
ClickHouse source (``clickhouse`` extra: ``clickhouse-connect``, HTTP interface).

The query is SQL; the result is fetched with ``query_df``. ``DateTime`` and
``DateTime64`` columns arrive as datetimes and are read as such whatever
``time_unit`` says. The password comes from ``CLICKHOUSE_PASSWORD`` (or the variable
named by ``password_env``), the user from the source or ``CLICKHOUSE_USER``.
"""

import os
from typing import Any

import pandas as pd

from ...models.config import SourceConfig
from .base import QuerySource, import_client, register_source


def create_client(**kwargs: Any) -> Any:
    """
    Build a ``clickhouse-connect`` client; the seam tests replace with a fake.

    Parameters
    ----------
    **kwargs : Any
        ``clickhouse_connect.get_client`` arguments.

    Returns
    -------
    Any
        The client.
    """
    clickhouse_connect = import_client("clickhouse_connect", "clickhouse", "clickhouse-connect")
    return clickhouse_connect.get_client(**kwargs)


def client_arguments(source: SourceConfig, secret: str | None) -> dict[str, Any]:
    """
    ``get_client`` arguments for ``source``.

    Parameters
    ----------
    source : SourceConfig
        A ``clickhouse`` source.
    secret : str | None
        The password, if any.

    Returns
    -------
    dict[str, Any]
        ``host``, ``secure`` and, when set, ``port``, ``database``, ``username``,
        ``password``.
    """
    kwargs: dict[str, Any] = {"host": source.host or "localhost", "secure": source.secure}
    if source.port is not None:
        kwargs["port"] = source.port
    if source.database:
        kwargs["database"] = source.database
    user = source.user or os.environ.get("CLICKHOUSE_USER")
    if user:
        kwargs["username"] = user
    if secret is not None:
        kwargs["password"] = secret
    return kwargs


def fetch(source: SourceConfig, query: str, secret: str | None) -> pd.DataFrame:
    """
    Run ``query`` on ClickHouse and return the result.

    Parameters
    ----------
    source : SourceConfig
        Connection settings.
    query : str
        SQL query.
    secret : str | None
        The password, if any.

    Returns
    -------
    pd.DataFrame
        Result rows.
    """
    client = create_client(**client_arguments(source, secret))
    try:
        frame: pd.DataFrame = client.query_df(query)
        return frame
    finally:
        client.close()


SOURCE = register_source(QuerySource(kind="clickhouse", label="ClickHouse", fetch=fetch))
