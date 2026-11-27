"""
Secrets of database sources: read from the environment, kept out of every message.

Passwords and tokens are never part of a ``SourceConfig``; :func:`read_secret` takes
them from the environment when a query runs. Errors raised by the client libraries can
quote connection strings or headers, so :func:`redact` scrubs the secret itself and the
usual credential patterns from any text before it reaches a log, an exception or a
report.
"""

import logging
import os
import re
from collections.abc import Iterable

from ...models.config import SourceConfig

REDACTED = "***"
"""What a secret is replaced with."""

_PATTERNS = (
    # user:password@ in a URL or DSN; the password may contain spaces (not on one line
    # with another '@' or '/' though, which ends the user information)
    (re.compile(r"(?P<keep>://[^/\s:@]*:)[^@/\n]{1,256}(?=@)"), r"\g<keep>" + REDACTED),
    # Authorization: Token abc / Bearer abc / Basic abc
    (
        re.compile(r"(?P<keep>authorization['\"]?\s*[:=]\s*['\"]?(?:\w+\s+)?)[^\s'\",}]+", re.I),
        r"\g<keep>" + REDACTED,
    ),
    # password=..., token: ..., "api_key": "..." (query strings, dicts, JSON)
    (
        re.compile(
            r"(?P<keep>\b(?:password|passwd|pwd|token|access_token|api_key|apikey|api_token|"
            r"secret)['\"]?\s*[:=]\s*['\"]?)[^\s'\",&;}]+",
            re.I,
        ),
        r"\g<keep>" + REDACTED,
    ),
    # Token <value> / Bearer <value> in a header dump
    (re.compile(r"(?P<keep>\b(?:Token|Bearer)\s+)[A-Za-z0-9._~+/=-]{8,}"), r"\g<keep>" + REDACTED),
)


class SourceError(RuntimeError):
    """A database source could not be read. The message has been redacted, and the
    client library's exception is not chained (it may hold the secret)."""


_SECRETS_READ: set[str] = set()
"""Every secret :func:`read_secret` returned in this process, for :class:`RedactingFilter`."""


def read_secret(source: SourceConfig) -> str | None:
    """
    Read the password or token of ``source`` from its environment variable.

    Parameters
    ----------
    source : SourceConfig
        The source; ``SourceConfig.secret_env`` names the variable.

    Returns
    -------
    str | None
        The secret, or ``None`` when the back end takes none (DuckDB) or the variable is
        unset or empty.
    """
    name = source.secret_env()
    if name is None:
        return None
    secret = os.environ.get(name) or None
    if secret is not None:
        _SECRETS_READ.add(secret)
    return secret


def redact(text: str, secrets: Iterable[str | None] = ()) -> str:
    """
    Remove secrets from ``text``.

    Parameters
    ----------
    text : str
        A message, typically from a client library's exception.
    secrets : Iterable[str | None]
        Known secret values to replace wherever they appear; ``None`` and empty values
        are ignored.

    Returns
    -------
    str
        ``text`` with each known secret, and anything that looks like a password or
        token (URL user information, ``password=``, ``Authorization:`` headers...),
        replaced by ``***``.
    """
    for secret in secrets:
        if secret:
            text = text.replace(secret, REDACTED)
    for pattern, replacement in _PATTERNS:
        text = pattern.sub(replacement, text)
    return text


class RedactingFilter(logging.Filter):
    """
    A logging filter that redacts records before a handler writes them.

    Client libraries log on their own (``clickhouse-connect`` logs transport errors with
    their traceback); attached to a handler, this filter scrubs every secret read by
    :func:`read_secret` and the generic credential patterns of :func:`redact` from the
    message and the formatted traceback. ``jitterbug`` commands install it on the root
    handlers; library users can do the same.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        """Redact ``record`` in place; never drops it."""
        message = record.getMessage()
        cleaned = redact(message, _SECRETS_READ)
        if cleaned != message:
            record.msg, record.args = cleaned, None
        if record.exc_info:
            text = logging.Formatter().formatException(record.exc_info)
            record.exc_text = redact(text, _SECRETS_READ)
            record.exc_info = None
        return True


def install_redacting_filter(logger: logging.Logger | None = None) -> None:
    """
    Add a :class:`RedactingFilter` to every handler of ``logger`` (the root by default,
    plus ``logging.lastResort``) that does not have one yet.

    Parameters
    ----------
    logger : logging.Logger | None
        Logger whose handlers to filter.
    """
    handlers = list((logger or logging.getLogger()).handlers)
    if logger is None and logging.lastResort is not None:
        handlers.append(logging.lastResort)  # used while the root has no handler
    for handler in handlers:
        if not any(isinstance(f, RedactingFilter) for f in handler.filters):
            handler.addFilter(RedactingFilter())
