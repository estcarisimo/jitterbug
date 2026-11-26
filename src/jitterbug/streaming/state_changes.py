"""
State changes of the path, reported early and corrected later.

The online back ends judge a period twice: a *provisional* verdict once the open period
holds enough jitter observations (100 samples for the KS test, about 45 min; 12 dispersion
values, 3 h, for dispersion) and a *final* verdict when the next change point closes it.
``StateChangeTracker`` adds an earlier, weaker signal on top of those events, without
changing them.

The path is in one of two states, ``baseline`` or ``congested``; the stream's first period
is the baseline. A *state change* is a transition between them at a change point, and it
has a status:

``suspected``
    Raised from the latency jump alone, before the jitter test has enough observations:
    once the period a change point opens holds ``streaming.state_change_bins`` closed
    minimum-RTT bins, each of them is compared with the mean minimum RTT of the previous
    period. If every one is more than ``latency_jump.threshold`` above it and the path is
    in ``baseline``, a change to ``congested`` is suspected. If none is above it and the
    path is ``congested``, a return to ``baseline`` is suspected: the two-period rule
    declares a period without a latency jump not congested, so the early rule is the
    latency-jump half of that rule in both directions. Mixed bins raise nothing; the
    verdicts decide.
``confirmed``
    A provisional or final verdict puts the period in the suspected state. A verdict can
    also confirm a change nobody suspected (the first event of that change is then
    ``confirmed``).
``retracted``
    A verdict leaves the path where it was. A final verdict that contradicts the
    provisional one corrects it: ``retracted`` after ``confirmed``, or ``confirmed``
    after ``retracted``.

Each change has a stable ``id`` (period start and direction), so every event about it
can be matched. The state before a period is the latest evidence on the period before it
(final verdict, else provisional verdict, else a suspected change, else the state before
that); when that evidence changes, the changes after it are re-derived, so a change whose
direction no longer holds is retracted.

Agreement with the verdicts is not ground truth: the verdicts are themselves the offline
method's rule applied online. Users who want notifications can build them on these events;
``CommandHook`` runs a command per event, as ``jitterbug stream --on-state-change`` does.
"""

from __future__ import annotations

import json
import logging
import math
import shlex
import subprocess
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from ..models import JitterbugConfig
from .online_analyzer import OpenPeriodBins, StreamingEvent

if TYPE_CHECKING:
    from . import OnlineAnalyzer

logger = logging.getLogger(__name__)

PathState = Literal["baseline", "congested"]
Status = Literal["suspected", "confirmed", "retracted"]
EvidenceStage = Literal["latency_jump", "provisional", "final"]

MAX_TRACKED_PERIODS = 256
"""Periods kept for reconciliation; older ones are frozen (their changes can no longer move)."""


@dataclass
class StateChangeEvent:
    """One status update of a state change of the path."""

    id: str
    """Stable identifier of the change: ``<period start epoch>:<from>-><to>``."""
    from_state: PathState
    to_state: PathState
    status: Status
    change_epoch: float
    """Start of the period the change opens (its change point)."""
    emitted_at: float
    """Epoch of the sample whose arrival triggered this update."""
    suspected_at: float | None
    """When the change was first suspected; ``None`` if it never was."""
    confirmed_at: float | None
    """When a verdict first confirmed the change; ``None`` if none has."""
    stage: EvidenceStage
    """Evidence behind this update: the early latency jump or a verdict stage."""
    jump_magnitude: float | None
    """Mean minimum RTT of the bins used minus that of the previous period (ms)."""
    jitter_method: Literal["ks_test", "jitter_dispersion"]
    jitter_metric: float | None = None
    """The verdict's KS statistic or dispersion change; ``None`` for the early stage."""
    has_jitter: bool | None = None
    p_value: float | None = None
    n_prev: int = 0
    """Observations of the previous period behind the evidence: minimum-RTT bins for the
    ``latency_jump`` stage, jitter observations for a verdict."""
    n_curr: int = 0
    """Same for the period the change opens."""
    kind: Literal["state_change"] = "state_change"

    @property
    def delay(self) -> float:
        """Seconds between the change point and this update."""
        return self.emitted_at - self.change_epoch

    def to_dict(self) -> dict[str, Any]:
        """Return the event as a plain dictionary (JSON-serializable)."""
        return dict(self.__dict__)


@dataclass
class _Early:
    """Outcome of the early rule on one period."""

    rise: bool | None  # True: every bin above the threshold; False: none; None: mixed
    magnitude: float
    n_prev: int
    n_curr: int
    at: float


@dataclass
class _Period:
    start: float
    early: _Early | None = None
    early_done: bool = False
    provisional: StreamingEvent | None = None
    final: StreamingEvent | None = None
    emitted: dict[str, StateChangeEvent] = field(default_factory=dict)


def _state(is_congested: bool | None) -> PathState:
    return "congested" if is_congested else "baseline"


class StateChangeTracker:
    """
    Follow an online back end and report state changes of the path.

    Wraps ``OnlineJitterbug`` or ``SlidingWindowJitterbug``: ``push`` and ``flush`` feed
    the back end and return its events together with the state-change events they
    caused. The back end's events are passed through unchanged.

    Parameters
    ----------
    analyzer : OnlineAnalyzer
        The back end to follow; it must expose ``open_period_bins()``.
    config : JitterbugConfig, optional
        ``streaming.state_change_bins``, ``latency_jump.threshold``,
        ``jitter_analysis.method`` and ``data_processing.minimum_interval_minutes`` are
        used. Defaults to the back end's own configuration.
    """

    def __init__(self, analyzer: OnlineAnalyzer, config: JitterbugConfig | None = None) -> None:
        self.analyzer = analyzer
        self.config = config or getattr(analyzer, "config", None) or JitterbugConfig()
        self._interval_s = self.config.data_processing.minimum_interval_minutes * 60
        self._bins = self.config.streaming.state_change_bins
        self._threshold = self.config.latency_jump.threshold
        self._method = self.config.jitter_analysis.method
        self._periods: list[_Period] = []
        self._anchor: PathState = "baseline"  # state before self._periods[0]
        self._bin_id: int | None = None
        self._last_epoch = 0.0
        self.state_changes: list[StateChangeEvent] = []

    # ------------------------------------------------------------------ input
    @property
    def state(self) -> PathState:
        """Current state of the path, including suspected changes."""
        return self._states()[-1] if self._periods else self._anchor

    def push(self, epoch: float, rtt: float) -> tuple[list[StreamingEvent], list[StateChangeEvent]]:
        """
        Feed one sample to the back end.

        Parameters
        ----------
        epoch : float
            Unix timestamp of the sample, seconds.
        rtt : float
            Round-trip time in milliseconds.

        Returns
        -------
        tuple[list[StreamingEvent], list[StateChangeEvent]]
            The back end's events for this sample and the state changes they caused.
        """
        events = self.analyzer.push(epoch, rtt)
        bin_closed = False
        # Track bin closes as the back ends do: they drop invalid and out-of-order samples.
        if rtt > 0 and math.isfinite(rtt) and math.isfinite(epoch) and epoch >= self._last_epoch:
            self._last_epoch = epoch
            bin_id = int(epoch // self._interval_s)
            bin_closed = self._bin_id is not None and bin_id > self._bin_id
            if self._bin_id is None or bin_id > self._bin_id:
                self._bin_id = bin_id
        return events, self.observe(events, epoch, check_early=bin_closed)

    def flush(self) -> tuple[list[StreamingEvent], list[StateChangeEvent]]:
        """Close the back end's open bin and report what that changed."""
        events = self.analyzer.flush()
        now = events[-1].emitted_at if events else self._last_epoch
        return events, self.observe(events, now, check_early=True)

    def observe(
        self, events: list[StreamingEvent], now: float, check_early: bool = True
    ) -> list[StateChangeEvent]:
        """
        Update the state from a batch of back-end events.

        Parameters
        ----------
        events : list[StreamingEvent]
            Events the back end emitted for one sample (or ``flush``).
        now : float
            Epoch of that sample; the ``emitted_at`` of the resulting updates.
        check_early : bool
            Whether to evaluate the early rule on the open period (``push`` does so
            when a bin closes, since the rule only looks at closed bins).

        Returns
        -------
        list[StateChangeEvent]
            The state-change updates, in order.
        """
        for event in events:
            if event.kind == "change_point" and (
                not self._periods or event.start_epoch > self._periods[-1].start
            ):
                self._periods.append(_Period(start=event.start_epoch))
                check_early = True
        for event in events:
            if event.kind != "verdict":
                continue
            period = self._period(event.start_epoch)
            if period is None:
                continue
            if event.stage == "final":
                period.final = event
            else:
                period.provisional = event
        if check_early:
            self._check_early(now)
        out = self._reconcile(now)
        self._prune()
        self.state_changes.extend(out)
        return out

    # ------------------------------------------------------------------ internals
    def _period(self, start: float) -> _Period | None:
        for period in reversed(self._periods):
            if period.start == start:
                return period
        return None

    def _check_early(self, now: float) -> None:
        """Evaluate the early rule on the open period once it holds enough closed bins."""
        if len(self._periods) < 2:
            return  # the stream's first period is the baseline
        period = self._periods[-1]
        if period.early_done or period.provisional or period.final:
            return
        bins: OpenPeriodBins | None = self.analyzer.open_period_bins()
        if bins is None or bins.start_epoch != period.start:
            return
        if len(bins.current) < self._bins:
            return
        period.early_done = True
        if len(bins.previous) == 0:
            return
        prev_mean = float(np.mean(bins.previous))
        cur = np.asarray(bins.current[: self._bins], dtype=float)
        above = cur - prev_mean > self._threshold
        rise: bool | None = True if above.all() else False if not above.any() else None
        period.early = _Early(
            rise=rise,
            magnitude=float(np.mean(cur) - prev_mean),
            n_prev=len(bins.previous),
            n_curr=len(cur),
            at=now,
        )

    def _claim(self, period: _Period, before: PathState) -> tuple[PathState, EvidenceStage | None]:
        """State of ``period`` given the state before it, and the evidence behind it."""
        if period.final is not None:
            return _state(period.final.is_congested), "final"
        if period.provisional is not None:
            return _state(period.provisional.is_congested), "provisional"
        early = period.early
        if early is not None:
            if before == "baseline" and early.rise is True:
                return "congested", "latency_jump"
            if before == "congested" and early.rise is False:
                return "baseline", "latency_jump"
        return before, None

    def _states(self) -> list[PathState]:
        states: list[PathState] = []
        before = self._anchor
        for period in self._periods:
            before, _ = self._claim(period, before)
            states.append(before)
        return states

    def _reconcile(self, now: float) -> list[StateChangeEvent]:
        """Re-derive every tracked change and emit the status updates that differ."""
        out: list[StateChangeEvent] = []
        before = self._anchor
        for period in self._periods:
            after, stage = self._claim(period, before)
            tid = f"{period.start:.0f}:{before}->{after}" if after != before else None
            for old in list(period.emitted.values()):
                if old.status != "retracted" and old.id != tid:
                    out.append(
                        self._update(period, old.from_state, old.to_state, "retracted", stage, now)
                    )
            if tid is not None and stage is not None:
                status: Status = "suspected" if stage == "latency_jump" else "confirmed"
                last = period.emitted.get(tid)
                if (
                    last is None
                    or last.status == "retracted"
                    or (last.status == "suspected" and status == "confirmed")
                ):
                    out.append(self._update(period, before, after, status, stage, now))
            before = after
        return out

    def _update(
        self,
        period: _Period,
        from_state: PathState,
        to_state: PathState,
        status: Status,
        stage: EvidenceStage | None,
        now: float,
    ) -> StateChangeEvent:
        tid = f"{period.start:.0f}:{from_state}->{to_state}"
        last = period.emitted.get(tid)
        suspected_at = last.suspected_at if last else None
        confirmed_at = last.confirmed_at if last else None
        if status == "suspected" and suspected_at is None:
            suspected_at = now
        if status == "confirmed" and confirmed_at is None:
            confirmed_at = now
        verdict = (
            period.final
            if stage == "final"
            else period.provisional
            if stage == "provisional"
            else None
        )
        event = StateChangeEvent(
            id=tid,
            from_state=from_state,
            to_state=to_state,
            status=status,
            change_epoch=period.start,
            emitted_at=now,
            suspected_at=suspected_at,
            confirmed_at=confirmed_at,
            stage=stage or "latency_jump",
            jump_magnitude=None,
            jitter_method=self._method,
        )
        if verdict is not None:
            event.jump_magnitude = verdict.jump_magnitude
            event.jitter_method = verdict.jitter_method or self._method
            event.jitter_metric = verdict.jitter_metric
            event.has_jitter = verdict.has_jitter
            event.p_value = verdict.p_value
            event.n_prev = verdict.n_prev
            event.n_curr = verdict.n_curr
        elif period.early is not None:
            event.jump_magnitude = period.early.magnitude
            event.n_prev = period.early.n_prev
            event.n_curr = period.early.n_curr
        period.emitted[tid] = event
        return event

    def _prune(self) -> None:
        """
        Freeze periods whose changes can no longer move.

        Both back ends emit final verdicts in time order, so once a period has its final
        verdict the period before it gets no more evidence; its state, and the change
        into the next period, are settled.
        """
        while len(self._periods) > 2 and (
            self._periods[1].final is not None or len(self._periods) > MAX_TRACKED_PERIODS
        ):
            self._anchor, _ = self._claim(self._periods[0], self._anchor)
            self._periods.pop(0)


class CommandHook:
    """
    Run a command once per state-change event, with the event as JSON on its stdin.

    The command line is split with ``shlex.split`` and run without a shell, so nothing in
    the event can be interpreted by one. Commands run in the background: the stream does
    not wait for them. A command that cannot start or exits with a non-zero status is
    logged as a warning; one still running after ``timeout`` seconds is killed when
    ``poll`` (called before each new command, and by the CLI once per sample) or
    ``close`` next runs. Its standard output is discarded, since the stream's own
    standard output carries JSON; its standard error is inherited.

    Parameters
    ----------
    command : str
        Command line, for example ``"notify-send-wrapper --path A"``.
    timeout : float
        Seconds a command may run before it is killed.
    """

    def __init__(self, command: str, timeout: float = 60.0) -> None:
        self.argv = shlex.split(command)
        if not self.argv:
            raise ValueError("Empty state-change command")
        self.timeout = timeout
        self._running: list[tuple[subprocess.Popen[bytes], float]] = []
        self.failures = 0

    def __call__(self, event: StateChangeEvent) -> None:
        """Start the command for ``event`` and reap finished ones."""
        self.poll()
        payload = (json.dumps(event.to_dict()) + "\n").encode()
        try:
            proc = subprocess.Popen(
                self.argv,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=None,
            )
        except OSError as e:
            self.failures += 1
            logger.warning("State-change command %s could not start: %s", self.argv[0], e)
            return
        try:
            assert proc.stdin is not None
            proc.stdin.write(payload)
            proc.stdin.close()
        except OSError as e:  # the command exited without reading its input
            logger.debug("State-change command closed its input early: %s", e)
        self._running.append((proc, time.monotonic()))

    def poll(self) -> None:
        """Reap finished commands, log failures, kill the ones past the timeout."""
        still: list[tuple[subprocess.Popen[bytes], float]] = []
        now = time.monotonic()
        for proc, started in self._running:
            code = proc.poll()
            if code is None:
                if now - started > self.timeout:
                    proc.kill()
                    proc.wait()
                    self.failures += 1
                    logger.warning(
                        "State-change command %s killed after %.0f s", self.argv[0], self.timeout
                    )
                else:
                    still.append((proc, started))
            elif code != 0:
                self.failures += 1
                logger.warning("State-change command %s exited with status %d", self.argv[0], code)
        self._running = still

    def close(self) -> None:
        """Wait for running commands (up to the timeout overall), killing the stragglers."""
        deadline = time.monotonic() + self.timeout
        for proc, _ in self._running:
            try:
                code = proc.wait(timeout=max(0.0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
                self.failures += 1
                logger.warning(
                    "State-change command %s killed after %.0f s", self.argv[0], self.timeout
                )
                continue
            if code != 0:
                self.failures += 1
                logger.warning("State-change command %s exited with status %d", self.argv[0], code)
        self._running = []
