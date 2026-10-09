"""
Online (streaming) congestion inference.

Two back ends consume RTT samples one at a time and emit change points and congestion
verdicts as ``StreamingEvent`` objects: ``OnlineJitterbug`` (``backend: bocpd``, incremental
Bayesian online change point detection; needs the ``bcp`` extra) and
``SlidingWindowJitterbug`` (``backend: window``, the offline pipeline rerun on a trailing
window). ``create_online_analyzer`` picks one from ``JitterbugConfig.streaming``; ``replay``
and ``score`` run a finite dataset through it and summarize the result.
"""

from typing import Protocol

from ..models import JitterbugConfig, StreamingConfig
from .online_analyzer import OnlineJitterbug, StreamingEvent
from .replay import events_to_json, load_reference, replay, score
from .verdict import PeriodVerdict, two_period_verdict
from .window_backend import SlidingWindowJitterbug


class OnlineAnalyzer(Protocol):
    """What every online back end offers."""

    events: list[StreamingEvent]
    change_points: list[float]

    def push(self, epoch: float, rtt: float) -> list[StreamingEvent]: ...

    def flush(self) -> list[StreamingEvent]: ...


def create_online_analyzer(config: JitterbugConfig | None = None) -> OnlineAnalyzer:
    """Return the back end that ``config.streaming.backend`` names."""
    config = config or JitterbugConfig()
    if config.streaming.backend == "window":
        return SlidingWindowJitterbug(config)
    return OnlineJitterbug(config)


__all__ = [
    "OnlineAnalyzer",
    "OnlineJitterbug",
    "PeriodVerdict",
    "SlidingWindowJitterbug",
    "StreamingConfig",
    "StreamingEvent",
    "create_online_analyzer",
    "events_to_json",
    "load_reference",
    "replay",
    "score",
    "two_period_verdict",
]
