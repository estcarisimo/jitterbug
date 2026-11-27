"""
DuckDB source (``duckdb`` extra): SQL in an in-memory database or on a database file.

This is the universal adapter: the query can read Parquet, CSV or JSON files and globs
(``read_parquet('rtts/*.parquet')``), attach SQLite, PostgreSQL or MySQL databases, and
read object storage through DuckDB's own extensions. Several statements separated by
``;`` run in order and the last one gives the rows. A database file is opened read-only
unless ``read_only`` is false.
"""

from typing import Any

import pandas as pd

from ...models.config import SourceConfig
from .base import QuerySource, import_client, register_source

MEMORY = ":memory:"
"""DuckDB's name for an in-memory database."""


def connect(database: str, read_only: bool) -> Any:
    """
    Open a DuckDB connection; the seam tests may replace.

    Parameters
    ----------
    database : str
        Database file, or ``:memory:``.
    read_only : bool
        Open the file read-only.

    Returns
    -------
    Any
        The connection.
    """
    duckdb = import_client("duckdb", "duckdb", "duckdb")
    return duckdb.connect(database=database, read_only=read_only)


def fetch(source: SourceConfig, query: str, secret: str | None) -> pd.DataFrame:
    """
    Run ``query`` in DuckDB and return the rows of its last statement.

    Parameters
    ----------
    source : SourceConfig
        ``path`` (``None`` for in-memory) and ``read_only``.
    query : str
        SQL, possibly several statements.
    secret : str | None
        Unused: DuckDB takes no password (object storage credentials go through
        DuckDB's own secrets, see ``docs/DATA_SOURCES.md``).

    Returns
    -------
    pd.DataFrame
        Result rows.
    """
    path = source.path or MEMORY
    connection = connect(path, source.read_only and path != MEMORY)
    try:
        frame: pd.DataFrame = connection.execute(query).df()
        return frame
    finally:
        connection.close()


SOURCE = register_source(QuerySource(kind="duckdb", label="DuckDB", fetch=fetch))
