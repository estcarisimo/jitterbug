"""
Online (streaming) Jitterbug pipeline.

The sequential pipeline is retrospective twice over: the offline Bayesian detector sees
the whole series, and the verdict for period ``i`` needs change point ``i + 1`` to close
the period. This module runs the same decision rule one RTT sample at a time:

1. Minimum-RTT bins close when a sample from a later bin arrives (causal binning).
2. Each closed bin feeds a Bayesian online change point detector (Adams & MacKay 2007,
   ``bayesian_changepoint_detection.streaming.OnlineChangepointDetector``). A change
   point is declared when the MAP run length drops (default), or when the posterior that
   a segment started ``lag`` bins ago (``lag``) or within the last ``lag`` bins
   (``window``) exceeds a threshold. On the paper dataset the MAP rule recovers 14 of the
   15 reference congestion periods with no spurious one, like the offline pipeline, and
   reports congestion onsets one bin (15 min) after they start. The fixed-delay rules
   recover 5 to 12 (``lag``) or 5 to 15 (``window``) of the 15 depending on the setting but
   always with 3 to 6 spurious congested periods, because the posterior mass of a change
   spreads over several run lengths (see ``docs/ONLINE_MODE.md``).
3. The stream start opens the baseline period. When a change point opens a new period, a
   *provisional* verdict is emitted as soon as the open period holds ``min_period_samples``
   jitter samples. When the next change point closes the period, the *final* verdict is
   emitted with the full period, which is exactly what the sequential pipeline computes.

Only the Kolmogorov-Smirnov jitter test is supported: it uses consecutive RTT
differences and is causal. The jitter-dispersion filters are centered and are not.

Settings live in ``JitterbugConfig.streaming`` (``models.config.StreamingConfig``); the bin
width, latency jump threshold, significance level and device come from the shared
sections of ``JitterbugConfig``.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from functools import partial
from typing import Any, Literal

import numpy as np

from ..models import JitterbugConfig
from .verdict import PeriodVerdict, two_period_verdict

logger = logging.getLogger(__name__)


@dataclass
class StreamingEvent:
    """Something the online pipeline decided while consuming the stream."""

    kind: Literal["change_point", "verdict"]
    emitted_at: float
    """Epoch of the sample whose arrival triggered the event."""
    start_epoch: float
    """Change point epoch, or start of the period a verdict is about."""
    end_epoch: float | None = None
    stage: Literal["provisional", "final"] | None = None
    is_congested: bool | None = None
    confidence: float | None = None
    has_jump: bool | None = None
    jump_magnitude: float | None = None
    has_jitter: bool | None = None
    ks_statistic: float | None = None
    p_value: float | None = None
    n_prev: int = 0
    n_curr: int = 0
    detector_probability: float | None = None
    jitter_method: Literal["ks_test", "jitter_dispersion"] | None = None
    """Jitter test behind ``has_jitter``; the KS fields are set for ``ks_test`` only."""
    jitter_metric: float | None = None
    """The jitter test's metric: the KS statistic, or the dispersion change."""

    @property
    def delay(self) -> float:
        """Seconds between the event's reference time and its emission."""
        return self.emitted_at - (
            self.end_epoch if self.end_epoch is not None else self.start_epoch
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the event as a plain dictionary (JSON-serializable)."""
        return dict(self.__dict__)


def verdict_event(
    verdict: PeriodVerdict,
    stage: Literal["provisional", "final"],
    emitted_at: float,
    start_epoch: float,
    end_epoch: float | None,
) -> StreamingEvent:
    """Wrap a ``PeriodVerdict`` as a verdict event."""
    return StreamingEvent(
        kind="verdict",
        emitted_at=emitted_at,
        start_epoch=start_epoch,
        end_epoch=end_epoch,
        stage=stage,
        is_congested=verdict.is_congested,
        confidence=verdict.confidence,
        has_jump=verdict.has_jump,
        jump_magnitude=verdict.jump_magnitude,
        has_jitter=verdict.has_jitter,
        ks_statistic=verdict.ks_statistic,
        p_value=verdict.p_value,
        n_prev=verdict.n_prev,
        n_curr=verdict.n_curr,
        jitter_method="ks_test",
        jitter_metric=verdict.ks_statistic,
    )


class OnlineJitterbug:
    """
    Streaming congestion inference: feed RTT samples, get events.

    Samples must arrive in non-decreasing epoch order; a sample earlier than the last one
    accepted is dropped (logged at DEBUG), whatever bin it falls in. Bins are indexed by
    the samples that arrive, so a gap in the stream does not produce empty bins: the
    detector sees the bins before and after a gap as consecutive, as the offline pipeline
    does after ``dropna`` in ``RTTDataset.compute_minimum_intervals``.

    Parameters
    ----------
    config : JitterbugConfig, optional
        Full configuration; ``config.streaming`` holds the online-specific settings.
    """

    def __init__(self, config: JitterbugConfig | None = None) -> None:
        self.config = config or JitterbugConfig()
        self.streaming = self.config.streaming
        try:
            from bayesian_changepoint_detection.hazard_functions import constant_hazard
            from bayesian_changepoint_detection.online_likelihoods import StudentT
            from bayesian_changepoint_detection.streaming import OnlineChangepointDetector
        except ImportError as e:  # pragma: no cover - depends on the extra
            raise ImportError(
                "The bcp extra is required for online detection: "
                "uv sync --extra bcp / pip install 'jitterbug-inference[bcp]'"
            ) from e
        self._constant_hazard = constant_hazard
        self._StudentT = StudentT
        self._OnlineDetector = OnlineChangepointDetector
        self._detector: Any = None

        self._interval_s = self.config.data_processing.minimum_interval_minutes * 60
        self._min_sep_bins = max(1, math.ceil(self.streaming.min_time_elapsed / self._interval_s))

        # Current (open) minimum-RTT bin.
        self._bin_id: int | None = None
        self._bin_min_rtt = math.inf
        self._bin_min_epoch = math.inf
        # Closed bins and raw samples since the start of the previous period.
        self._bin_epochs: list[float] = []
        self._bin_values: list[float] = []
        self._raw_epochs: list[float] = []
        self._raw_rtts: list[float] = []
        self._pruned_bins = 0  # bins dropped from the front, to keep indices global
        self._t = 0  # bins fed to the detector
        self._map_start = 0  # last MAP-implied segment start (map rule)

        self.change_points: list[float] = []
        self._cp_bin_index: list[int] = []
        self._congestion_state = False
        self._awaiting_provisional = False
        self.events: list[StreamingEvent] = []

    # ------------------------------------------------------------------ stream input
    def push(self, epoch: float, rtt: float) -> list[StreamingEvent]:
        """
        Consume one RTT sample and return the events it triggered.

        Parameters
        ----------
        epoch : float
            Unix timestamp of the sample, seconds.
        rtt : float
            Round-trip time in milliseconds.
        """
        events: list[StreamingEvent] = []
        if not (rtt > 0) or not math.isfinite(rtt):
            return events
        if self._raw_epochs and epoch < self._raw_epochs[-1]:
            logger.debug("Out-of-order sample at %s dropped", epoch)
            return events
        bin_id = int(epoch // self._interval_s)
        if self._bin_id is None:
            self._bin_id = bin_id
        elif bin_id > self._bin_id:
            events.extend(self._close_bin(closing_epoch=epoch))
            self._bin_id = bin_id

        if rtt < self._bin_min_rtt:
            self._bin_min_rtt = rtt
        if epoch < self._bin_min_epoch:
            self._bin_min_epoch = epoch
        self._raw_epochs.append(epoch)
        self._raw_rtts.append(rtt)

        if self._awaiting_provisional and self._open_period_jitter_count() >= (
            self.streaming.min_period_samples
        ):
            events.append(self._verdict("provisional", emitted_at=epoch, end_epoch=None))
            self._awaiting_provisional = False

        self.events.extend(events)
        return events

    def flush(self) -> list[StreamingEvent]:
        """Close the open bin at the end of a finite stream."""
        if self._bin_id is None or not math.isfinite(self._bin_min_rtt):
            return []
        events = self._close_bin(closing_epoch=self._raw_epochs[-1])
        self._bin_id = None
        self.events.extend(events)
        return events

    # ------------------------------------------------------------------ bins and detector
    def _close_bin(self, closing_epoch: float) -> list[StreamingEvent]:
        events: list[StreamingEvent] = []
        self._bin_epochs.append(self._bin_min_epoch)
        self._bin_values.append(self._bin_min_rtt)
        value = self._bin_min_rtt
        self._bin_min_rtt = math.inf
        self._bin_min_epoch = math.inf

        if self._detector is None:
            mu = self.streaming.prior_mu if self.streaming.prior_mu is not None else value
            self._detector = self._OnlineDetector(
                partial(self._constant_hazard, self.streaming.hazard_lambda),
                self._StudentT(
                    alpha=self.streaming.prior_alpha,
                    beta=self.streaming.prior_beta,
                    kappa=self.streaming.prior_kappa,
                    mu=mu,
                    device=self.config.change_point_detection.bcp_device,
                ),
                max_run_length=self.streaming.max_run_length,
                device=self.config.change_point_detection.bcp_device,
            )
        posterior = self._detector.update(value)
        self._t += 1
        if self._t == 1:
            # The stream start opens the baseline period, as the offline detector's
            # change point at index 0 does; the first real change point closes it.
            events.extend(self._accept_change_point(0, None, emitted_at=closing_epoch))
            return events

        candidate: tuple[int, float] | None = None  # (global bin index, probability)
        if self.streaming.decision == "lag":
            lag = self.streaming.lag
            if self._t > lag:
                p = float(self._detector.changepoint_probability(lag))
                if p > self.streaming.threshold:
                    candidate = (self._t - lag, p)
        elif self.streaming.decision == "window":
            lag = self.streaming.lag
            if self._t > lag:
                head = posterior[: lag + 1].detach().cpu().numpy()
                head[0] = 0.0  # run length 0 is the hazard, not evidence
                if head.sum() > self.streaming.threshold:
                    r = int(head.argmax())
                    candidate = (self._t - r, float(head.sum()))
        else:
            # Run length 0 means "a change at this very observation"; under a constant
            # hazard its mass is the hazard rate, not evidence, so the MAP is taken over
            # run lengths >= 1 (it is also one past the last bin we hold).
            r = int(posterior[1:].argmax()) + 1
            start = self._t - r
            if start > self._map_start:
                self._map_start = start
                candidate = (start, float(posterior[r]))

        if candidate is not None:
            idx, prob = candidate
            last = self._cp_bin_index[-1] if self._cp_bin_index else -self._min_sep_bins
            # Index is the first bin of the new segment; the detector counts from 1.
            if idx - last >= self._min_sep_bins:
                events.extend(self._accept_change_point(idx, prob, emitted_at=closing_epoch))
        return events

    def _accept_change_point(
        self, bin_index: int, prob: float | None, emitted_at: float
    ) -> list[StreamingEvent]:
        events: list[StreamingEvent] = []
        cp_epoch = self._bin_epochs[bin_index - self._pruned_bins]
        events.append(
            StreamingEvent(
                kind="change_point",
                emitted_at=emitted_at,
                start_epoch=cp_epoch,
                detector_probability=prob,
            )
        )
        # The period that this change point closes gets its final verdict first.
        if len(self.change_points) >= 2:
            events.append(self._verdict("final", emitted_at=emitted_at, end_epoch=cp_epoch))
        self.change_points.append(cp_epoch)
        self._cp_bin_index.append(bin_index)
        self._awaiting_provisional = len(self.change_points) >= 2
        self._prune()
        return events

    def _prune(self) -> None:
        """Drop samples and bins before the start of the previous period."""
        if len(self.change_points) < 2:
            return
        keep_from = self.change_points[-2]
        n_raw = int(np.searchsorted(np.asarray(self._raw_epochs), keep_from, side="left"))
        # Keep one extra raw sample so the first jitter value of the period exists.
        n_raw = max(0, n_raw - 1)
        del self._raw_epochs[:n_raw]
        del self._raw_rtts[:n_raw]
        n_bins = int(np.searchsorted(np.asarray(self._bin_epochs), keep_from, side="left"))
        del self._bin_epochs[:n_bins]
        del self._bin_values[:n_bins]
        self._pruned_bins += n_bins

    # ------------------------------------------------------------------ verdicts
    def _jitter(self) -> tuple[np.ndarray, np.ndarray]:
        epochs = np.asarray(self._raw_epochs)
        rtts = np.asarray(self._raw_rtts)
        if len(epochs) < 2:
            return np.empty(0), np.empty(0)
        return epochs[1:], np.diff(rtts)

    def _open_period_jitter_count(self) -> int:
        if not self.change_points:
            return 0
        jitter_epochs, _ = self._jitter()
        return int(np.count_nonzero(jitter_epochs >= self.change_points[-1]))

    def _verdict(
        self, stage: Literal["provisional", "final"], emitted_at: float, end_epoch: float | None
    ) -> StreamingEvent:
        prev_start, cur_start = self.change_points[-2], self.change_points[-1]
        end = end_epoch if end_epoch is not None else emitted_at

        bin_epochs = np.asarray(self._bin_epochs)
        bin_values = np.asarray(self._bin_values)
        prev_bins = bin_values[(bin_epochs >= prev_start) & (bin_epochs < cur_start)]
        cur_bins = bin_values[(bin_epochs >= cur_start) & (bin_epochs <= end)]
        if stage == "provisional" and math.isfinite(self._bin_min_rtt):
            cur_bins = np.append(cur_bins, self._bin_min_rtt)  # the open bin counts too

        jitter_epochs, jitter_values = self._jitter()
        prev_jitter = jitter_values[(jitter_epochs >= prev_start) & (jitter_epochs < cur_start)]
        cur_jitter = jitter_values[(jitter_epochs >= cur_start) & (jitter_epochs <= end)]

        verdict = two_period_verdict(
            prev_bins, cur_bins, prev_jitter, cur_jitter, self._congestion_state, self.config
        )
        if stage == "final":
            self._congestion_state = verdict.is_congested
        return verdict_event(verdict, stage, emitted_at, cur_start, end_epoch)
