"""
Database sources, registered by kind: ``clickhouse``, ``influxdb2``, ``influxdb3`` and
``duckdb``.

A source runs a query and returns raw rows, like a file reader; they then go through the
same ``series`` filters, ``group_by`` split and normalization as a file. Client libraries
are optional extras, imported only when a query runs. To add a back end, write a module
with a ``fetch(source, query, secret) -> DataFrame`` function, call
:func:`register_source`, import the module here, and add the kind to
``jitterbug.models.config.SourceConfig``.
"""

# Importing registers the back ends.
from . import clickhouse, influxdb, duckdb  # noqa: F401  # isort: skip
from .base import (
    QuerySource,
    available_sources,
    describe_source,
    fetch_raw,
    get_source,
    query_text,
    register_source,
)
from .credentials import (
    RedactingFilter,
    SourceError,
    install_redacting_filter,
    read_secret,
    redact,
)
from .uri import is_source_uri, parse_source_uri

__all__ = [
    "QuerySource",
    "RedactingFilter",
    "SourceError",
    "available_sources",
    "describe_source",
    "fetch_raw",
    "get_source",
    "install_redacting_filter",
    "is_source_uri",
    "parse_source_uri",
    "query_text",
    "read_secret",
    "redact",
    "register_source",
]
