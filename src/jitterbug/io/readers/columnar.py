"""
Columnar readers: ``parquet`` (Apache Parquet) and ``arrow`` (Arrow IPC files and
streams, including Feather).

Both need ``pyarrow`` (the ``parquet`` extra) and are imported lazily, so the package
works without it. The columns of the file are the raw columns, mapped through
``InputConfig`` like CSV columns; a time column with a timestamp type is read as
datetimes even when ``time_unit`` names a numeric unit. Both formats compress their own
data, so a compression suffix on top (``rtts.parquet.gz``) is rejected.
"""

import importlib
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from ...models.config import InputConfig
from ..compression import is_compressed, read_head
from .base import RawData, Reader, register_reader

logger = logging.getLogger(__name__)

INSTALL_HINT = (
    "Reading Parquet or Arrow files needs pyarrow. "
    "Install the extra: pip install 'jitterbug-inference[parquet]'"
)

PARQUET_MAGIC = b"PAR1"
"""First bytes of a Parquet file."""
ARROW_FILE_MAGIC = b"ARROW1"
"""First bytes of an Arrow IPC file (Feather version 2)."""
FEATHER_V1_MAGIC = b"FEA1"
"""First bytes of a Feather version 1 file."""
ARROW_STREAM_MAGIC = b"\xff\xff\xff\xff"
"""Continuation marker that starts an Arrow IPC stream (Arrow 0.15 and later)."""


def _pyarrow(module: str) -> Any:
    """Import ``pyarrow.<module>``, or raise an ImportError that names the extra."""
    try:
        return importlib.import_module(f"pyarrow.{module}")
    except ImportError as e:
        raise ImportError(INSTALL_HINT) from e


def _reject_compressed(path: Path, label: str) -> None:
    """Raise ValueError for ``*.parquet.gz`` and the like: the format compresses itself."""
    if is_compressed(path):
        raise ValueError(
            f"{label} files are already compressed; drop the {path.suffix} suffix and "
            f"decompress the file: {path}"
        )


def _raw_data(frame: pd.DataFrame, config: InputConfig, kind: str) -> RawData:
    """Wrap ``frame``; a datetime-typed time column is read as datetimes (``iso``)."""
    schema: dict[str, Any] = {}
    column = config.time_column
    if (
        column in frame.columns
        and pd.api.types.is_datetime64_any_dtype(frame[column])
        and config.time_unit not in ("iso", "auto")
    ):
        logger.info(
            f"Time column '{column}' has a timestamp type; reading it as datetimes "
            f"instead of time_unit {config.time_unit}"
        )
        schema["time_unit"] = "iso"
    return RawData(frame=frame, metadata={"format": kind}, schema=schema)


def read_parquet(path: Path, config: InputConfig) -> RawData:
    """
    Read an Apache Parquet file.

    Parameters
    ----------
    path : Path
        File path; a compression suffix is rejected.
    config : InputConfig
        Input settings; ``time_column`` and ``time_unit`` decide whether a timestamp
        column is read as datetimes.

    Returns
    -------
    RawData
        The raw frame; ``metadata["format"]`` is ``parquet``.

    Raises
    ------
    ImportError
        If pyarrow is not installed.
    ValueError
        If the path has a compression suffix or the file is not valid Parquet.
    """
    _reject_compressed(path, "Parquet")
    parquet = _pyarrow("parquet")
    try:
        frame = parquet.read_table(str(path)).to_pandas()
    except Exception as e:
        raise ValueError(f"Failed to load Parquet file {path}: {e}") from e
    return _raw_data(frame, config, "parquet")


def read_arrow(path: Path, config: InputConfig) -> RawData:
    """
    Read an Arrow IPC file (Feather version 2), a Feather version 1 file, or an Arrow IPC
    stream.

    Parameters
    ----------
    path : Path
        File path; a compression suffix is rejected.
    config : InputConfig
        Input settings; ``time_column`` and ``time_unit`` decide whether a timestamp
        column is read as datetimes.

    Returns
    -------
    RawData
        The raw frame; ``metadata["format"]`` is ``arrow``.

    Raises
    ------
    ImportError
        If pyarrow is not installed.
    ValueError
        If the path has a compression suffix or the file is not valid Arrow data.
    """
    _reject_compressed(path, "Arrow")
    feather = _pyarrow("feather")
    ipc = _pyarrow("ipc")
    try:
        head = read_head(path, len(ARROW_FILE_MAGIC))
        if head.startswith((ARROW_FILE_MAGIC, FEATHER_V1_MAGIC)):
            table = feather.read_table(str(path))
        else:
            with ipc.open_stream(str(path)) as reader:
                table = reader.read_all()
        frame = table.to_pandas()
    except Exception as e:
        raise ValueError(f"Failed to load Arrow file {path}: {e}") from e
    return _raw_data(frame, config, "arrow")


register_reader(
    Reader(
        name="parquet",
        label="Parquet",
        read=read_parquet,
        extensions=(".parquet", ".pq"),
        magic=(PARQUET_MAGIC,),
    )
)
register_reader(
    Reader(
        name="arrow",
        label="Arrow",
        read=read_arrow,
        extensions=(".arrow", ".feather", ".ipc", ".arrows"),
        magic=(ARROW_FILE_MAGIC, FEATHER_V1_MAGIC, ARROW_STREAM_MAGIC),
    )
)
