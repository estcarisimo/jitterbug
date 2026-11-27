"""
File readers, registered by format name and extension.

Built-in formats: ``parquet`` and ``arrow`` (columnar, need pyarrow), ``ping`` and
``fping`` (command output), ``csv`` and ``tsv``, ``json`` (JSON lines or an array:
scamper, RIPE Atlas or generic, decided by content), ``scamper``, ``jsonl`` and ``atlas``.
To add one, write a module with a ``read(path, input_config) -> RawData`` function, call
:func:`register_reader`, and import the module here. Content sniffers run in import
order, so the specific ones (ping, fping) come before the CSV sniffer, which would accept
any line with a comma.
"""

# Importing registers the readers; the order is the sniffing order.
from . import columnar, ping, delimited, jsonl, atlas  # noqa: F401  # isort: skip
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
