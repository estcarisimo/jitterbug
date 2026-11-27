"""
Source URIs: a database source written where a file path would go.

Forms (credentials are rejected; see ``docs/DATA_SOURCES.md``):

- ``clickhouse://[user@]host[:port][/database]``; ``clickhouses://`` connects over HTTPS;
- ``influxdb2://host[:port]?org=ORG`` (Flux); ``influxdb2s://`` over HTTPS;
- ``influxdb3://host[:port]/database[?language=influxql]`` (SQL by default);
  ``influxdb3s://`` over HTTPS;
- ``duckdb:`` or ``duckdb://`` (in-memory), ``duckdb:///absolute/path.db``,
  ``duckdb:relative/path.db``; ``?read_only=false`` opens a file for writing.
"""

import re
from typing import Any
from urllib.parse import parse_qsl, unquote, urlsplit

from ...models.config import SECRET_KEYS, SourceConfig, secret_rejection

SCHEMES = {
    "clickhouse": ("clickhouse", False),
    "clickhouses": ("clickhouse", True),
    "influxdb2": ("influxdb2", False),
    "influxdb2s": ("influxdb2", True),
    "influxdb3": ("influxdb3", False),
    "influxdb3s": ("influxdb3", True),
    "duckdb": ("duckdb", False),
}
"""URI scheme -> (source kind, secure)."""

_PARAMETERS = {
    "clickhouse": {"user"},
    "influxdb2": {"org"},
    "influxdb3": {"language"},
    "duckdb": {"read_only"},
}
"""Query-string parameters each kind accepts."""

_SCHEME = re.compile(r"^(?P<scheme>[A-Za-z][A-Za-z0-9+.-]*):")


def uri_scheme(text: str) -> str | None:
    """
    The lower-case scheme of ``text`` if it is a source URI scheme, else ``None``.

    ``influxdb`` is reported too, so that it can be rejected with a helpful message.

    Parameters
    ----------
    text : str
        A command-line input argument.

    Returns
    -------
    str | None
        The scheme, or ``None`` for anything else (file paths included).
    """
    match = _SCHEME.match(text)
    if match is None:
        return None
    scheme = match["scheme"].lower()
    return scheme if scheme in SCHEMES or scheme == "influxdb" else None


def is_source_uri(text: str) -> bool:
    """
    Whether ``text`` is written as a database source URI rather than a file path.

    Parameters
    ----------
    text : str
        A command-line input argument.

    Returns
    -------
    bool
        ``True`` for the schemes of :data:`SCHEMES` (and ``influxdb``).
    """
    return uri_scheme(text) is not None


def _bool(value: str, name: str) -> bool:
    lowered = value.strip().lower()
    if lowered in ("1", "true", "yes", "on"):
        return True
    if lowered in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"{name} must be true or false, got {value!r}")


def parse_source_uri(uri: str) -> SourceConfig:
    """
    Turn a source URI into a ``SourceConfig`` without a query.

    Parameters
    ----------
    uri : str
        A URI of one of the forms listed in this module's docstring.

    Returns
    -------
    SourceConfig
        Connection settings; the query is set separately (``--query``).

    Raises
    ------
    ValueError
        If the scheme is not a source scheme, the URI holds a password or token, or a
        part does not apply to the kind. Messages never repeat a secret.
    """
    scheme = uri_scheme(uri)
    if scheme == "influxdb":
        raise ValueError(
            "influxdb:// is ambiguous; use influxdb2:// for InfluxDB 2 (Flux) or "
            "influxdb3:// for InfluxDB 3 (SQL, InfluxQL)"
        )
    if scheme is None:
        raise ValueError(f"Not a source URI: {uri!r} (schemes: {', '.join(SCHEMES)})")
    kind, secure = SCHEMES[scheme]
    if kind == "duckdb":
        return _parse_duckdb(uri)

    parts = urlsplit(uri)
    if parts.password is not None:
        raise ValueError(secret_rejection("the URI (user:password@)", kind))
    params = _params(parts.query, kind)
    data: dict[str, Any] = {"kind": kind, "secure": secure}
    if parts.hostname:
        data["host"] = parts.hostname
    try:
        port = parts.port
    except ValueError as e:
        raise ValueError(f"Invalid port in source URI ({e})") from None
    if port is not None:
        data["port"] = port
    if parts.username:
        if kind != "clickhouse":
            raise ValueError(f"A {kind} URI takes no user; the token identifies the caller")
        data["user"] = unquote(parts.username)
    path = unquote(parts.path.strip("/"))
    if path:
        if kind == "influxdb2":
            raise ValueError(
                "An influxdb2 URI has no path: the bucket is named in the Flux query, "
                "the organization with ?org=ORG"
            )
        data["database"] = path
    data.update(params)
    return SourceConfig.model_validate(data)


def _params(query: str, kind: str) -> dict[str, Any]:
    """Query-string parameters, checked against the kind; secrets rejected."""
    params: dict[str, Any] = {}
    for name, value in parse_qsl(query, keep_blank_values=True):
        if name.lower() in SECRET_KEYS:
            raise ValueError(secret_rejection(f"the URI (?{name}=)", kind))
        if name not in _PARAMETERS[kind]:
            accepted = ", ".join(sorted(_PARAMETERS[kind]))
            raise ValueError(f"Unknown parameter {name!r} in a {kind} URI (accepted: {accepted})")
        params[name] = _bool(value, name) if name == "read_only" else value
    return params


def _parse_duckdb(uri: str) -> SourceConfig:
    """``duckdb:``, ``duckdb://``, ``duckdb:///abs.db``, ``duckdb:rel.db``, ``?read_only=``."""
    rest = uri.split(":", 1)[1]
    rest, _, query = rest.partition("?")
    params = _params(query, "duckdb")
    if rest.startswith("//"):
        rest = rest[2:]
        if rest and not rest.startswith("/"):
            raise ValueError(
                "A duckdb URI takes no host: use duckdb:///absolute/path.db or "
                "duckdb:relative/path.db"
            )
    path = unquote(rest)
    data: dict[str, Any] = {"kind": "duckdb", **params}
    if path and path != ":memory:":
        data["path"] = path
    return SourceConfig.model_validate(data)
