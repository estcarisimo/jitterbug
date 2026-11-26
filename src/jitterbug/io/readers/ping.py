"""
Text output of the ``ping`` and ``fping`` commands: formats ``ping`` and ``fping``.

Each reply line with a Unix timestamp becomes one sample. The timestamp is the
``[1700000000.123456]`` prefix that ``ping -D`` (Linux iputils) and ``fping -D`` print,
or that a line-timestamping tool adds (``ping host | ts '[%.s]'`` with moreutils, which
is how BSD and macOS ``ping``, whose ``-D`` means something else, can be recorded).
Reply lines without one cannot be placed in time: a file of them is an error, and
``jitterbug stream`` skips them.

Supported line shapes:

- ``ping``: ``64 bytes from HOST: icmp_seq=1 ttl=57 time=12.3 ms`` (Linux iputils;
  BSD and macOS print the same shape with ``icmp_seq`` starting at 0; BusyBox writes
  ``seq=``), optionally with ``HOST`` as ``name (address)``. ``(DUP!)`` replies,
  ``no answer yet`` / ``Request timeout`` lines and ICMP errors (``From HOST
  icmp_seq=3 Destination Host Unreachable``) are skipped and counted.
- ``fping``: ``HOST : [0], 64 bytes, 12.3 ms (12.3 avg, 0% loss)`` as printed by
  ``fping -l``, ``-c`` and ``-C``. ``timed out`` lines, ``duplicate for`` replies and ICMP
  error lines are skipped and counted; per-host summaries are ignored.

The line parsers (:func:`parse_ping_line`, :func:`parse_fping_line`) are shared with
``jitterbug stream --format ping|fping``.
"""

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

from ...models.config import InputConfig
from ..compression import open_text
from .base import RawData, Reader, register_reader

logger = logging.getLogger(__name__)

ProbeKind = Literal["reply", "duplicate", "timeout", "error"]

_EPOCH = r"(?:\[(?P<epoch>\d+(?:\.\d+)?)\]\s+)?"
_NUMBER = r"\d+(?:\.\d+)?"

PING_REPLY = re.compile(
    rf"^{_EPOCH}\d+ bytes from (?P<host>.+?): (?:icmp_)?seq=(?P<seq>\d+)\b"
    rf".*?\btime=(?P<rtt>{_NUMBER}) ?ms\b(?P<rest>.*)$"
)
PING_TIMEOUT = re.compile(
    rf"^{_EPOCH}(?:no answer yet for icmp_seq=|Request timeout for icmp_seq[= ])(?P<seq>\d+)"
)
PING_ERROR = re.compile(
    rf"^{_EPOCH}(?:From (?P<host>\S+)(?: \([^)]*\))? (?:icmp_)?seq=(?P<seq>\d+)"
    rf"|\d+ bytes from (?P<host2>[^:]+): (?!(?:icmp_)?seq=))"
)
PING_HEADER = re.compile(rf"^{_EPOCH}PING6?\b.*\bbytes\b")

FPING_REPLY = re.compile(
    rf"^{_EPOCH}(?P<host>\S+)\s+: \[(?P<seq>\d+)\], \d+ bytes, (?P<rtt>{_NUMBER}) ms\b"
)
FPING_TIMEOUT = re.compile(rf"^{_EPOCH}(?P<host>\S+)\s+: \[(?P<seq>\d+)\], timed out\b")
FPING_DUPLICATE = re.compile(rf"^{_EPOCH}(?P<host>\S+)\s+: duplicate for \[(?P<seq>\d+)\]")
FPING_ERROR = re.compile(rf"^{_EPOCH}ICMP .+ for ICMP Echo sent to (?P<host>\S+)")

_FPING_OTHER: tuple[tuple[ProbeKind, re.Pattern[str]], ...] = (
    ("timeout", FPING_TIMEOUT),
    ("duplicate", FPING_DUPLICATE),
    ("error", FPING_ERROR),
)

_NAMED_HOST = re.compile(r"^\S+ \((?P<address>[^)]+)\)$")


@dataclass(frozen=True)
class ProbeLine:
    """
    One recognized line of ``ping`` or ``fping`` output.

    Attributes
    ----------
    kind : Literal['reply', 'duplicate', 'timeout', 'error']
        A reply with an RTT, a duplicate reply, a probe without a reply, or an ICMP error.
    epoch : float | None
        Unix seconds from the ``[epoch]`` prefix; ``None`` without one.
    rtt : float | None
        Round-trip time in milliseconds (replies and duplicates).
    destination : str | None
        The probed host as printed (the address when ``ping`` prints ``name (address)``).
    seq : int | None
        Sequence number of the probe.
    """

    kind: ProbeKind
    epoch: float | None = None
    rtt: float | None = None
    destination: str | None = None
    seq: int | None = None


def _epoch(match: re.Match[str]) -> float | None:
    value = match.group("epoch")
    return float(value) if value is not None else None


def _host(value: str | None) -> str | None:
    """``name (address)`` -> ``address``; anything else unchanged."""
    if value is None:
        return None
    named = _NAMED_HOST.match(value)
    return named.group("address") if named else value


def _seq(match: re.Match[str]) -> int | None:
    value = match.groupdict().get("seq")
    return int(value) if value is not None else None


def parse_ping_line(line: str) -> ProbeLine | None:
    """
    Parse one line of ``ping`` output.

    Parameters
    ----------
    line : str
        A line, with or without its newline.

    Returns
    -------
    ProbeLine | None
        The recognized line, or ``None`` for headers, statistics and anything else.
    """
    text = line.strip()
    reply = PING_REPLY.match(text)
    if reply:
        kind: ProbeKind = "duplicate" if "DUP!" in reply.group("rest") else "reply"
        return ProbeLine(
            kind=kind,
            epoch=_epoch(reply),
            rtt=float(reply.group("rtt")),
            destination=_host(reply.group("host")),
            seq=_seq(reply),
        )
    timeout = PING_TIMEOUT.match(text)
    if timeout:
        return ProbeLine(kind="timeout", epoch=_epoch(timeout), seq=_seq(timeout))
    error = PING_ERROR.match(text)
    if error:
        host = error.group("host") or error.group("host2")
        return ProbeLine(kind="error", epoch=_epoch(error), destination=host, seq=_seq(error))
    return None


def parse_fping_line(line: str) -> ProbeLine | None:
    """
    Parse one line of ``fping`` output.

    Parameters
    ----------
    line : str
        A line, with or without its newline.

    Returns
    -------
    ProbeLine | None
        The recognized line, or ``None`` for summaries and anything else.
    """
    text = line.strip()
    reply = FPING_REPLY.match(text)
    if reply:
        return ProbeLine(
            kind="reply",
            epoch=_epoch(reply),
            rtt=float(reply.group("rtt")),
            destination=reply.group("host"),
            seq=_seq(reply),
        )
    for kind, pattern in _FPING_OTHER:
        match = pattern.match(text)
        if match:
            return ProbeLine(
                kind=kind,
                epoch=_epoch(match),
                destination=match.group("host"),
                seq=_seq(match),
            )
    return None


LineParser = Callable[[str], ProbeLine | None]

LINE_PARSERS: dict[str, LineParser] = {"ping": parse_ping_line, "fping": parse_fping_line}
"""Line parser of each text probe format, by format name."""

PROBE_SCHEMA = {
    "time_column": "epoch",
    "time_unit": "s",
    "rtt_column": "rtt",
    "rtt_unit": "ms",
    "source_column": "source",
    "destination_column": "destination",
}
"""The frame the ping and fping readers build is in this fixed schema."""

NO_TIMESTAMP_HINT = {
    "ping": (
        "rerun with ping -D (Linux iputils) to print a Unix timestamp on each line, or, "
        "where ping has no such option (BSD, macOS), pipe it through moreutils' "
        "ts '[%.s]'"
    ),
    "fping": "rerun with fping -D to print a Unix timestamp on each line",
}
"""What to do about reply lines without a timestamp, by format name."""


def _read_probe_output(path: Path, name: str, label: str) -> RawData:
    """One row per timestamped reply; other recognized lines are counted."""
    parse = LINE_PARSERS[name]
    rows: list[tuple[float, float, str | None, int | None]] = []
    counts = {"timeout": 0, "error": 0, "duplicate": 0, "untimestamped": 0}
    try:
        with open_text(path) as f:
            for raw in f:
                parsed = parse(raw)
                if parsed is None:
                    continue
                if parsed.kind != "reply":
                    counts[parsed.kind] += 1
                elif parsed.epoch is None or parsed.rtt is None:
                    counts["untimestamped"] += 1
                else:
                    rows.append((parsed.epoch, parsed.rtt, parsed.destination, parsed.seq))
    except ImportError:
        raise
    except Exception as e:
        raise ValueError(f"Failed to load {label} file {path}: {e}") from e
    if not rows:
        if counts["untimestamped"]:
            raise ValueError(
                f"{label} file {path} has {counts['untimestamped']} reply line(s) but no "
                f"timestamps; {NO_TIMESTAMP_HINT[name]}"
            )
        raise ValueError(f"No {label} reply lines found in {path}")
    if counts["untimestamped"]:
        logger.warning(
            f"Skipping {counts['untimestamped']} {label} reply line(s) without a timestamp"
        )
    for kind in ("timeout", "error", "duplicate"):
        if counts[kind]:
            logger.info(f"Skipped {counts[kind]} {label} {kind} line(s) in {path}")

    return RawData(
        frame=pd.DataFrame(rows, columns=["epoch", "rtt", "destination", "seq"]),
        metadata={
            "format": name,
            "timeouts": counts["timeout"],
            "errors": counts["error"],
            "duplicate_replies": counts["duplicate"],
            "untimestamped_replies": counts["untimestamped"],
        },
        schema=PROBE_SCHEMA,
    )


def read_ping(path: Path, config: InputConfig) -> RawData:
    """
    Read ``ping -D`` output (see the module docstring for the line shapes).

    Parameters
    ----------
    path : Path
        File path (possibly compressed).
    config : InputConfig
        Unused: the schema is fixed (``PROBE_SCHEMA``); the row policies apply.

    Returns
    -------
    RawData
        One row per timestamped reply, with ``destination`` and ``seq`` columns;
        ``metadata`` counts ``timeouts``, ``errors``, ``duplicate_replies`` and
        ``untimestamped_replies``.

    Raises
    ------
    ValueError
        If the file holds no reply line, or reply lines without any timestamp.
    """
    return _read_probe_output(path, "ping", "ping")


def read_fping(path: Path, config: InputConfig) -> RawData:
    """
    Read ``fping -D`` output (``-l``, ``-c`` or ``-C``; see the module docstring).

    Parameters
    ----------
    path : Path
        File path (possibly compressed).
    config : InputConfig
        Unused: the schema is fixed (``PROBE_SCHEMA``); the row policies apply.

    Returns
    -------
    RawData
        One row per timestamped reply, with ``destination`` and ``seq`` columns;
        ``metadata`` counts ``timeouts``, ``errors``, ``duplicate_replies`` and
        ``untimestamped_replies``.

    Raises
    ------
    ValueError
        If the file holds no reply line, or reply lines without any timestamp.
    """
    return _read_probe_output(path, "fping", "fping")


def _looks_like_ping(line: str) -> bool:
    """Sniffer: a ``PING host ...`` header or a ping reply, timeout or error line."""
    return bool(PING_HEADER.match(line)) or parse_ping_line(line) is not None


def _looks_like_fping(line: str) -> bool:
    """Sniffer: an fping reply, timeout, duplicate or ICMP error line."""
    return parse_fping_line(line) is not None


register_reader(Reader(name="ping", label="ping", read=read_ping, sniff=_looks_like_ping))
register_reader(Reader(name="fping", label="fping", read=read_fping, sniff=_looks_like_fping))
