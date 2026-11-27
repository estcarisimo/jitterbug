"""
InfluxDB sources: version 2 with Flux (``influx`` extra: ``influxdb-client``) and
version 3 with SQL or InfluxQL (``influxdb3`` extra: ``influxdb3-python``).

Flux results name the time ``_time`` and the value ``_value``; InfluxDB 3 names the
time ``time``. These are used when ``input`` leaves the time column at ``epoch`` (and
the RTT column unset, for Flux), so a plain query needs no mapping. The token comes
from ``INFLUXDB_V2_TOKEN`` or ``INFLUX_TOKEN`` (or the variable named by
``token_env``).
"""

from typing import Any

import pandas as pd

from ...models.config import SourceConfig
from .base import QuerySource, import_client, register_source
from .credentials import SourceError

DEFAULT_PORTS = {"influxdb2": 8086, "influxdb3": 8181}
"""Ports of a local InfluxDB 2 and InfluxDB 3 server."""


def server_url(source: SourceConfig) -> str:
    """
    ``http(s)://host:port`` of an InfluxDB source.

    Parameters
    ----------
    source : SourceConfig
        An ``influxdb2`` or ``influxdb3`` source. Without a port, the default port of
        the version is used over HTTP, and the scheme's default (443) over HTTPS.

    Returns
    -------
    str
        The server URL.
    """
    host = source.host or "localhost"
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"  # IPv6 literal
    port = source.port
    if port is None and not source.secure:
        port = DEFAULT_PORTS[source.kind]
    scheme = "https" if source.secure else "http"
    return f"{scheme}://{host}" + (f":{port}" if port is not None else "")


def create_v2_client(url: str, token: str | None, org: str | None) -> Any:
    """
    Build an ``influxdb_client.InfluxDBClient``; the seam tests replace with a fake.

    Parameters
    ----------
    url : str
        Server URL.
    token : str | None
        API token.
    org : str | None
        Organization.

    Returns
    -------
    Any
        The client.
    """
    module = import_client("influxdb_client", "influx", "influxdb-client")
    return module.InfluxDBClient(url=url, token=token, org=org)


def query_v2(url: str, token: str | None, org: str | None, query: str) -> pd.DataFrame:
    """
    Run a Flux query; several result tables are concatenated.

    Parameters
    ----------
    url : str
        Server URL.
    token : str | None
        API token.
    org : str | None
        Organization.
    query : str
        Flux query.

    Returns
    -------
    pd.DataFrame
        Result rows.
    """
    client = create_v2_client(url=url, token=token, org=org)
    try:
        result = client.query_api().query_data_frame(query)
        if isinstance(result, list):
            return pd.concat(result, ignore_index=True) if result else pd.DataFrame()
        frame: pd.DataFrame = result
        return frame
    finally:
        client.close()


def fetch_v2(source: SourceConfig, query: str, secret: str | None) -> pd.DataFrame:
    """
    Run a Flux query on InfluxDB 2.

    Parameters
    ----------
    source : SourceConfig
        Connection settings.
    query : str
        Flux query.
    secret : str | None
        The token, if any.

    Returns
    -------
    pd.DataFrame
        Result rows.
    """
    return query_v2(server_url(source), secret, source.org, query)


def create_v3_client(host: str, token: str | None, database: str | None) -> Any:
    """
    Build an ``influxdb_client_3.InfluxDBClient3``; the seam tests replace with a fake.

    Parameters
    ----------
    host : str
        Server URL.
    token : str | None
        API token.
    database : str | None
        Database.

    Returns
    -------
    Any
        The client.
    """
    module = import_client("influxdb_client_3", "influxdb3", "influxdb3-python")
    return module.InfluxDBClient3(host=host, token=token, database=database)


def fetch_v3(source: SourceConfig, query: str, secret: str | None) -> pd.DataFrame:
    """
    Run an SQL or InfluxQL query on InfluxDB 3.

    Parameters
    ----------
    source : SourceConfig
        Connection settings, including ``language``.
    query : str
        Query text.
    secret : str | None
        The token, if any.

    Returns
    -------
    pd.DataFrame
        Result rows.
    """
    if secret is None:
        # influxdb3-python refuses a client without a token
        raise SourceError(
            f"InfluxDB 3 needs a token: set {source.secret_env()} (any value for a server "
            "started without authentication)"
        )
    client = create_v3_client(host=server_url(source), token=secret, database=source.database)
    try:
        frame: pd.DataFrame = client.query(query=query, language=source.language, mode="pandas")
        return frame
    finally:
        client.close()


INFLUXDB2 = register_source(
    QuerySource(
        kind="influxdb2",
        label="InfluxDB 2",
        fetch=fetch_v2,
        time_column="_time",
        rtt_column="_value",
    )
)
INFLUXDB3 = register_source(
    QuerySource(kind="influxdb3", label="InfluxDB 3", fetch=fetch_v3, time_column="time")
)
