"""
Online (streaming) congestion inference.

Two back ends consume RTT samples one at a time and emit change points and congestion
verdicts as ``StreamingEvent`` objects: ``OnlineJitterbug`` (``backend: bocpd``, incremental
Bayesian online change point detection; needs the ``bcp`` extra) and
``SlidingWindowJitterbug`` (``backend: window``, the offline pipeline rerun on a trailing
window). ``create_online_analyzer`` picks one from ``JitterbugConfig.streaming``; ``replay``
and ``score`` run a finite dataset through it and summarize the result.
``StateChangeTracker`` follows either back end and reports state changes of the path
(``baseline`` <-> ``congested``): suspected early from the latency jump, then confirmed or
retracted by the verdicts.
"""

from typing import Protocol

from ..models import JitterbugConfig, StreamingConfig
from .online_analyzer import OnlineJitterbug, OpenPeriodBins, StreamingEvent
from .replay import (
    events_to_json,
    load_reference,
    replay,
    replay_state_changes,
    score,
    score_state_changes,
)
from .state_changes import CommandHook, StateChangeEvent, StateChangeTracker
from .verdict import PeriodVerdict, two_period_verdict
from .window_backend import SlidingWindowJitterbug


class OnlineAnalyzer(Protocol):
    """What every online back end offers."""

    events: list[StreamingEvent]
    change_points: list[float]

    def push(self, epoch: float, rtt: float) -> list[StreamingEvent]: ...

    def flush(self) -> list[StreamingEvent]: ...

    def open_period_bins(self) -> OpenPeriodBins | None: ...


def create_online_analyzer(config: JitterbugConfig | None = None) -> OnlineAnalyzer:
    """Return the back end that ``config.streaming.backend`` names."""
    config = config or JitterbugConfig()
    if config.streaming.backend == "window":
        return SlidingWindowJitterbug(config)
    return OnlineJitterbug(config)


__all__ = [
    "CommandHook",
    "OnlineAnalyzer",
    "OnlineJitterbug",
    "OpenPeriodBins",
    "PeriodVerdict",
    "SlidingWindowJitterbug",
    "StateChangeEvent",
    "StateChangeTracker",
    "StreamingConfig",
    "StreamingEvent",
    "create_online_analyzer",
    "events_to_json",
    "load_reference",
    "replay",
    "replay_state_changes",
    "score",
    "score_state_changes",
    "two_period_verdict",
]
