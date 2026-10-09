"""
Sliding-window back end: rerun the offline pipeline as the stream grows.

The baseline the incremental detector is measured against. Every ``rerun_every_bins``
closed minimum-RTT bins, the sequential pipeline (``JitterbugAnalyzer``, with the
configured detector and jitter method) runs on the trailing ``window_hours`` of raw
samples. A change point is emitted once the detector has placed one within
``TOLERANCE_BINS`` of the same bin in ``stable_runs`` consecutive reruns; change points are
emitted in time order and never retracted. Final verdicts form a contiguous chain: the next
one is emitted once the offline pipeline has judged the period at the last emitted boundary
identically (same verdict, boundaries within the tolerance) in ``stable_runs`` consecutive
reruns and the change point that closes it has been emitted; its boundaries are those
emitted change points, so change points and verdicts never disagree. If the window moves
past a period before it stabilizes, that period is skipped with a warning and the chain
resumes at the next emitted change point. A change point that only stabilizes after a
later one was emitted is dropped (time order is kept), which can leave the chain waiting
for the next emitted boundary for up to a window length. The open period after the last
emitted change point gets a provisional verdict from the shared two-period rule, with the
configured jitter method (trailing filters for dispersion), once it holds
``min_period_samples`` jitter samples, as in the incremental back end.

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

TOLERANCE_BINS = 2
"""Two boundaries this many bins apart or closer are the same boundary (30 min at 15-min bins).

The offline detector moves a change point by a bin or two as the window slides; the
sequential pipeline keeps change points at least ``change_point_detection.min_time_elapsed``
apart (two bins by default), so a larger tolerance would merge real boundaries."""


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

        self._raw_epochs: list[float] = []
        self._raw_rtts: list[float] = []
        self._bin_id: int | None = None
        self._bin_min_rtt = math.inf
        self._bin_min_epoch = math.inf
        self._bins_since_rerun = 0
        self._stream_start: float | None = None

        self._cp_streak: dict[int, int] = {}
        self._period_streak: dict[tuple[int, int], tuple[bool, int]] = {}
        self._next_final_start: float | None = None  # boundary the next final must start at
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

    def _bin(self, epoch: float) -> int:
        return int(epoch // self._interval_s)

    def _emitted_near(self, epoch: float) -> float | None:
        """The emitted change point within ``TOLERANCE_BINS`` of ``epoch``, if any."""
        target = self._bin(epoch)
        for cp in reversed(self.change_points):
            if abs(self._bin(cp) - target) <= TOLERANCE_BINS:
                return cp
        return None

    def _rerun(self, now: float) -> list[StreamingEvent]:
        events: list[StreamingEvent] = []
        epochs, rtts, window_start = self._window(now)
        if len(epochs) < 2:
            return events
        offline_cps, inferences = self._offline(epochs, rtts)
        truncated = window_start > (self._stream_start or -math.inf)
        edge = window_start + self._interval_s if truncated else -math.inf
        stable = self.streaming.stable_runs

        # Change points: within the tolerance of the same bin for `stable` reruns, after
        # the window edge, later than and not within the tolerance of an emitted one. The
        # spacing between change points is the offline detector's
        # (change_point_detection.min_time_elapsed); re-filtering here would leave offline
        # periods without an emitted boundary.
        neighbors = range(-TOLERANCE_BINS, TOLERANCE_BINS + 1)
        seen: dict[int, tuple[float, float]] = {}
        for cp in offline_cps:
            if cp.epoch > edge:
                seen[self._bin(cp.epoch)] = (cp.epoch, cp.confidence)
        self._cp_streak = {
            k: 1 + max(self._cp_streak.get(k + d, 0) for d in neighbors) for k in seen
        }
        for cp_key in sorted(seen):
            cp_epoch, confidence = seen[cp_key]
            if self._cp_streak[cp_key] < stable:
                continue
            if cp_epoch <= self.change_points[-1] or self._emitted_near(cp_epoch) is not None:
                continue
            self.change_points.append(cp_epoch)
            if self._next_final_start is None:
                self._next_final_start = cp_epoch  # the first period that can be judged
            events.append(
                StreamingEvent(
                    kind="change_point",
                    emitted_at=now,
                    start_epoch=cp_epoch,
                    detector_probability=confidence,
                )
            )

        # Period streaks: same verdict with both boundaries within the tolerance.
        current: dict[tuple[int, int], tuple[bool, CongestionInference]] = {}
        for period in inferences:
            if period.start_epoch > edge:
                key = (self._bin(period.start_epoch), self._bin(period.end_epoch))
                current[key] = (bool(period.is_congested), period)
        streak: dict[tuple[int, int], tuple[bool, int]] = {}
        for (sb, eb), (is_congested, _) in current.items():
            best = 0
            for ds in neighbors:
                for de in neighbors:
                    prev = self._period_streak.get((sb + ds, eb + de))
                    if prev is not None and prev[0] == is_congested:
                        best = max(best, prev[1])
            streak[(sb, eb)] = (is_congested, best + 1)
        self._period_streak = streak

        # Final verdicts: a contiguous chain whose boundaries are emitted change points.
        jitter_epochs = epochs[1:]
        while self._next_final_start is not None:
            start_cp = self._next_final_start
            # The offline period at our boundary: one that starts within the tolerance, or
            # failing that one that contains it (the detector moved the boundary; ours stands).
            ordered = sorted(current.items())
            match = next(
                (
                    (key, v, period)
                    for key, (v, period) in ordered
                    if abs(key[0] - self._bin(start_cp)) <= TOLERANCE_BINS
                ),
                None,
            ) or next(
                (
                    (key, v, period)
                    for key, (v, period) in ordered
                    if period.start_epoch < start_cp < period.end_epoch
                ),
                None,
            )
            if match is None:
                if start_cp <= edge:
                    # The window left this period behind before it stabilized.
                    later = [cp for cp in self.change_points if cp > edge]
                    logger.warning(
                        "Period starting at %s never stabilized before leaving the window; skipped",
                        start_cp,
                    )
                    self._next_final_start = later[0] if later else None
                    if self._next_final_start == start_cp:
                        break
                    continue
                break
            key, is_congested, period = match
            if streak[key][1] < stable:
                break
            end_cp = self._emitted_near(period.end_epoch)
            if end_cp is None or end_cp <= start_cp:
                break  # wait for the closing change point to be emitted
            prev_cp = next(
                (cp for cp in reversed(self.change_points) if cp < start_cp), window_start
            )
            prev_start = max(prev_cp, window_start)
            in_prev = (jitter_epochs >= prev_start) & (jitter_epochs < start_cp)
            in_cur = (jitter_epochs >= start_cp) & (jitter_epochs <= end_cp)
            self._congestion_state = is_congested
            self._next_final_start = end_cp
            jump = period.latency_jump
            jitter = period.jitter_analysis
            ks = jitter is not None and jitter.method == "ks_test"
            events.append(
                StreamingEvent(
                    kind="verdict",
                    emitted_at=now,
                    start_epoch=start_cp,
                    end_epoch=end_cp,
                    stage="final",
                    is_congested=is_congested,
                    confidence=float(period.confidence),
                    has_jump=None if jump is None else jump.has_jump,
                    jump_magnitude=None if jump is None else jump.magnitude,
                    has_jitter=None if jitter is None else jitter.has_significant_jitter,
                    ks_statistic=jitter.jitter_metric if ks and jitter is not None else None,
                    p_value=jitter.p_value if ks and jitter is not None else None,
                    n_prev=int(np.count_nonzero(in_prev)),
                    n_curr=int(np.count_nonzero(in_cur)),
                    jitter_method=None if jitter is None else jitter.method,
                    jitter_metric=None if jitter is None else jitter.jitter_metric,
                )
            )

        # Provisional verdict for the open period after the last emitted change point.
        if len(self.change_points) >= 2:
            cur_start = self.change_points[-1]
            open_key = self._bin(cur_start)
            if open_key not in self._provisional_done:
                prev_start = max(self.change_points[-2], window_start)
                jitter_values = np.diff(rtts)
                cur_jitter = jitter_values[jitter_epochs >= cur_start]
                if len(cur_jitter) >= self.streaming.min_period_samples:
                    prev_jitter = jitter_values[
                        (jitter_epochs >= prev_start) & (jitter_epochs < cur_start)
                    ]
                    grouped = pd.DataFrame({"epoch": epochs, "rtt": rtts}).groupby(
                        epochs // self._interval_s
                    )
                    bin_ids = grouped["rtt"].min().index.to_numpy()
                    bin_values = grouped["rtt"].min().to_numpy()
                    bin_epochs = grouped["epoch"].min().to_numpy()
                    prev_id, cur_id = self._bin(prev_start), self._bin(cur_start)
                    prev_bins = bin_values[(bin_ids >= prev_id) & (bin_ids < cur_id)]
                    cur_bins = bin_values[bin_ids >= cur_id]
                    if self.config.jitter_analysis.method == "jitter_dispersion":
                        d_epochs, d_values = (
                            self._analyzer.jitter_analyzer.compute_causal_jitter_dispersion(
                                bin_epochs, bin_values
                            )
                        )
                        prev_jitter = d_values[(d_epochs >= prev_start) & (d_epochs < cur_start)]
                        cur_jitter = d_values[d_epochs >= cur_start]
                        needed = (
                            self._analyzer.jitter_analyzer.causal_dispersion_delay
                            + self.config.jitter_analysis.moving_average_order
                        )
                        if len(cur_jitter) < needed:
                            return events  # wait until values reflect the new period
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
