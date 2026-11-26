"""
File readers, registered by format name and extension.

Built-in formats: ``json`` (JSON lines, scamper or generic, decided by content),
``scamper``, ``jsonl``, ``tsv`` and ``csv``. To add one, write a module with a
``read(path, input_config) -> RawData`` function, call :func:`register_reader`, and
import the module here.
"""

from . import delimited, jsonl  # noqa: F401  (importing registers the readers)
from .base import (
    RawData,
    Reader,
    available_formats,
    get_reader,
    infer_format,
    register_reader,
)

__all__ = [
    "RawData",
    "Reader",
    "available_formats",
    "get_reader",
    "infer_format",
    "register_reader",
]
