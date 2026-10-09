"""
Sliding-window back end: rerun the offline pipeline as the stream grows.

The baseline the incremental detector is measured against. Every ``rerun_every_bins``
closed minimum-RTT bins, the sequential pipeline (``JitterbugAnalyzer``, with the
configured detector and jitter method) runs on the trailing ``window_hours`` of raw
samples. A change point or a closed period's verdict is emitted once it has come out
identical, to within one bin, in ``stable_runs`` consecutive reruns; emitted events are
never retracted. The open period after the last stable change point gets a provisional
verdict from the shared two-period rule once it holds ``min_period_samples`` jitter
samples, as in the incremental back end.

The window start opens the baseline period, as the stream start does in the incremental
back end: when the detector reports no change point within one bin of the first bin, one
is inserted there, so the first period after the window start can be judged (the
sequential pipeline needs three change points per verdict). Once the window has moved
past the stream start, change points and periods touching its left edge are ignored:
they are cut by the window and were emitted long before.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ..analyzer import JitterbugAnalyzer
from ..models import ChangePoint, CongestionInference, JitterbugConfig
from .online_analyzer import StreamingEvent, verdict_event
from .verdict import two_period_verdict

logger = logging.getLogger(__name__)


class SlidingWindowJitterbug:
    """
    Streaming congestion inference by rerunning the offline pipeline on a window.

    Same interface as ``OnlineJitterbug``: ``push(epoch, rtt)`` returns the events the
    sample triggered, ``flush()`` closes the open bin and reruns once, ``events`` holds
    everything emitted. Samples must arrive in non-decreasing epoch order.

    Parameters
    ----------
    config : JitterbugConfig, optional
        ``streaming.window_hours``, ``streaming.rerun_every_bins`` and
        ``streaming.stable_runs`` drive the back end; the detector and jitter method are
        those of the sequential mode (``analysis_mode`` is forced to ``sequential``).
    """

    def __init__(self, config: JitterbugConfig | None = None) -> None:
        self.config = config or JitterbugConfig()
        self.streaming = self.config.streaming
        offline = self.config.model_copy(deep=True)
        offline.analysis_mode = "sequential"
        offline.verbose = False
        self._analyzer = JitterbugAnalyzer(offline)

        self._interval_s = self.config.data_processing.minimum_interval_minutes * 60
        self._window_s = self.streaming.window_hours * 3600
        self._min_sep_s = self.streaming.min_time_elapsed

        self._raw_epochs: list[float] = []
        self._raw_rtts: list[float] = []
        self._bin_id: int | None = None
        self._bin_min_rtt = math.inf
        self._bin_min_epoch = math.inf
        self._bins_since_rerun = 0
        self._stream_start: float | None = None

        self._cp_streak: dict[int, int] = {}
        self._period_streak: dict[tuple[int, int], tuple[bool, int]] = {}
        self._last_final_start = -math.inf
        self._provisional_done: set[int] = set()
        self._congestion_state = False

        self.change_points: list[float] = []
        self.events: list[StreamingEvent] = []

    # ------------------------------------------------------------------ stream input
    def push(self, epoch: float, rtt: float) -> list[StreamingEvent]:
        """Consume one RTT sample (seconds, milliseconds) and return the events it triggered."""
        events: list[StreamingEvent] = []
        if not (rtt > 0) or not math.isfinite(rtt) or not math.isfinite(epoch):
            return events
        if self._raw_epochs and epoch < self._raw_epochs[-1]:
            logger.debug("Out-of-order sample at %s dropped", epoch)
            return events
        bin_id = int(epoch // self._interval_s)
        if self._bin_id is None:
            self._bin_id = bin_id
            self._stream_start = epoch
        elif bin_id > self._bin_id:
            events.extend(self._close_bin(closing_epoch=epoch))
            self._bin_id = bin_id
        self._bin_min_rtt = min(self._bin_min_rtt, rtt)
        self._bin_min_epoch = min(self._bin_min_epoch, epoch)
        self._raw_epochs.append(epoch)
        self._raw_rtts.append(rtt)
        self.events.extend(events)
        return events

    def flush(self) -> list[StreamingEvent]:
        """Close the open bin and rerun once, whatever the cadence."""
        if self._bin_id is None or not math.isfinite(self._bin_min_rtt):
            return []
        self._bins_since_rerun = self.streaming.rerun_every_bins  # force a rerun
        events = self._close_bin(closing_epoch=self._raw_epochs[-1])
        self._bin_id = None
        self.events.extend(events)
        return events

    # ------------------------------------------------------------------ reruns
    def _close_bin(self, closing_epoch: float) -> list[StreamingEvent]:
        events: list[StreamingEvent] = []
        self._bin_min_rtt = math.inf
        self._bin_min_epoch = math.inf
        if not self.change_points and self._stream_start is not None:
            # The stream start opens the baseline period, as in the incremental back end.
            self.change_points.append(self._stream_start)
            events.append(
                StreamingEvent(
                    kind="change_point", emitted_at=closing_epoch, start_epoch=self._stream_start
                )
            )
        self._bins_since_rerun += 1
        if self._bins_since_rerun >= self.streaming.rerun_every_bins:
            self._bins_since_rerun = 0
            events.extend(self._rerun(closing_epoch))
        return events

    def _window(self, now: float) -> tuple[np.ndarray, np.ndarray, float]:
        """Raw samples in the trailing window (pruning older ones) and the window start."""
        start = max(now - self._window_s, self._stream_start or -math.inf)
        epochs = np.asarray(self._raw_epochs)
        first = int(np.searchsorted(epochs, start, side="left"))
        if first > 0:
            del self._raw_epochs[:first]
            del self._raw_rtts[:first]
        return np.asarray(self._raw_epochs), np.asarray(self._raw_rtts), start

    def _offline(
        self, epochs: np.ndarray, rtts: np.ndarray
    ) -> tuple[list[ChangePoint], list[CongestionInference]]:
        """The sequential pipeline on the window, with the window start as a change point."""
        analyzer = self._analyzer
        dataset = analyzer.data_loader.load_from_dataframe(
            pd.DataFrame({"epoch": epochs, "rtt_value": rtts})
        )
        interval = self.config.data_processing.minimum_interval_minutes
        min_rtt = dataset.compute_minimum_intervals(interval)
        if len(min_rtt) < 2:
            return [], []
        cps = analyzer.change_point_detector.detect(min_rtt)
        first_epoch = min_rtt.measurements[0].epoch
        if not cps or cps[0].epoch > first_epoch + self._interval_s:
            cps.insert(
                0,
                ChangePoint(
                    timestamp=datetime.fromtimestamp(first_epoch, tz=timezone.utc),
                    epoch=first_epoch,
                    confidence=1.0,
                    algorithm="window_start",
                ),
            )
        if len(cps) < 3:
            return cps, []
        jumps = analyzer.latency_jump_analyzer.analyze(min_rtt, cps)
        if self.config.jitter_analysis.method == "jitter_dispersion":
            jitter = analyzer.jitter_analyzer.analyze_jitter_dispersion(min_rtt, cps)
        else:
            jitter = analyzer.jitter_analyzer.analyze_ks_test(dataset, cps)
        return cps, analyzer.congestion_inference_analyzer.infer(jumps, jitter)

    def _rerun(self, now: float) -> list[StreamingEvent]:
        events: list[StreamingEvent] = []
        epochs, rtts, window_start = self._window(now)
        if len(epochs) < 2:
            return events
        offline_cps, inferences = self._offline(epochs, rtts)
        truncated = window_start > (self._stream_start or -math.inf)
        edge = window_start + self._interval_s if truncated else -math.inf

        # Change points: stable across reruns, after the window edge, spaced out.
        seen: dict[int, tuple[float, float]] = {}
        for cp in offline_cps:
            if cp.epoch <= edge:
                continue
            seen[int(cp.epoch // self._interval_s)] = (cp.epoch, cp.confidence)
        self._cp_streak = {k: self._cp_streak.get(k, 0) + 1 for k in seen}
        for cp_key in sorted(seen):
            cp_epoch, confidence = seen[cp_key]
            if self._cp_streak[cp_key] < self.streaming.stable_runs:
                continue
            if cp_epoch - self.change_points[-1] < self._min_sep_s:
                continue
            self.change_points.append(cp_epoch)
            events.append(
                StreamingEvent(
                    kind="change_point",
                    emitted_at=now,
                    start_epoch=cp_epoch,
                    detector_probability=confidence,
                )
            )

        # Final verdicts: closed periods whose verdict is stable, in time order, once.
        current: dict[tuple[int, int], tuple[bool, CongestionInference]] = {}
        for period in inferences:
            if period.start_epoch <= edge:
                continue
            period_key = (
                int(period.start_epoch // self._interval_s),
                int(period.end_epoch // self._interval_s),
            )
            current[period_key] = (bool(period.is_congested), period)
        streak: dict[tuple[int, int], tuple[bool, int]] = {}
        for period_key, (is_congested, _) in current.items():
            prev = self._period_streak.get(period_key)
            runs = prev[1] + 1 if prev is not None and prev[0] == is_congested else 1
            streak[period_key] = (is_congested, runs)
        self._period_streak = streak
        for period_key in sorted(current):
            is_congested, period = current[period_key]
            if streak[period_key][1] < self.streaming.stable_runs:
                continue
            if period.start_epoch <= self._last_final_start:
                continue
            self._last_final_start = period.start_epoch
            self._congestion_state = is_congested
            jump = period.latency_jump
            jitter = period.jitter_analysis
            events.append(
                StreamingEvent(
                    kind="verdict",
                    emitted_at=now,
                    start_epoch=period.start_epoch,
                    end_epoch=period.end_epoch,
                    stage="final",
                    is_congested=is_congested,
                    confidence=float(period.confidence),
                    has_jump=None if jump is None else jump.has_jump,
                    jump_magnitude=None if jump is None else jump.magnitude,
                    has_jitter=None if jitter is None else jitter.has_significant_jitter,
                    ks_statistic=None if jitter is None else jitter.jitter_metric,
                    p_value=None if jitter is None else jitter.p_value,
                )
            )

        # Provisional verdict for the open period after the last emitted change point.
        if len(self.change_points) >= 2:
            cur_start = self.change_points[-1]
            open_key = int(cur_start // self._interval_s)
            if open_key not in self._provisional_done and cur_start > self._last_final_start:
                prev_start = max(self.change_points[-2], window_start)
                jitter_epochs, jitter_values = epochs[1:], np.diff(rtts)
                cur_jitter = jitter_values[jitter_epochs >= cur_start]
                if len(cur_jitter) >= self.streaming.min_period_samples:
                    prev_jitter = jitter_values[
                        (jitter_epochs >= prev_start) & (jitter_epochs < cur_start)
                    ]
                    bins = pd.Series(rtts).groupby(epochs // self._interval_s).min()
                    bin_ids = bins.index.to_numpy()
                    bin_values = bins.to_numpy()
                    prev_id, cur_id = prev_start // self._interval_s, cur_start // self._interval_s
                    prev_bins = bin_values[(bin_ids >= prev_id) & (bin_ids < cur_id)]
                    cur_bins = bin_values[bin_ids >= cur_id]
                    provisional = two_period_verdict(
                        prev_bins,
                        cur_bins,
                        prev_jitter,
                        cur_jitter,
                        self._congestion_state,
                        self.config,
                    )
                    self._provisional_done.add(open_key)
                    events.append(verdict_event(provisional, "provisional", now, cur_start, None))
        return events
