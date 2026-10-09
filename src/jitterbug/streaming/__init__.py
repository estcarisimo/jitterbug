"""
Online (streaming) congestion inference.

``OnlineJitterbug`` consumes RTT samples one at a time and emits change points and
congestion verdicts as ``StreamingEvent`` objects; ``replay`` and ``score`` run a finite
dataset through it and summarize the result. Settings live in ``JitterbugConfig.streaming``
(``StreamingConfig``). Requires the ``bcp`` extra.
"""

from ..models import StreamingConfig
from .online_analyzer import OnlineJitterbug, StreamingEvent
from .replay import events_to_json, load_reference, replay, score

__all__ = [
    "OnlineJitterbug",
    "StreamingConfig",
    "StreamingEvent",
    "events_to_json",
    "load_reference",
    "replay",
    "score",
]
