"""
Prototype of an online (streaming) Jitterbug pipeline.

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
   reports congestion onsets one bin (15 min) after they start; the fixed-lag rules miss
   most returns to baseline, because the posterior mass spreads over several run lengths.
3. The stream start opens the baseline period. When a change point opens a new period, a
   *provisional* verdict is emitted as soon as the open period holds ``min_period_samples``
   jitter samples. When the next change point closes the period, the *final* verdict is
   emitted with the full period, which is exactly what the sequential pipeline computes.

Only the Kolmogorov-Smirnov jitter test is supported: it uses consecutive RTT
differences and is causal. The jitter-dispersion filters are centered and are not.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from functools import partial
from typing import Any, Literal

import numpy as np
import scipy.stats
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class StreamingConfig(BaseModel):
    """Settings of the online pipeline (prototype; will move to ``models/config.py``)."""

    interval_minutes: int = Field(default=15, gt=0, description="Minimum-RTT bin width")
    hazard_lambda: float = Field(
        default=50.0, ge=1, description="Expected run length in bins (constant hazard)"
    )
    decision: Literal["lag", "window", "map"] = Field(
        default="map",
        description=(
            "Change point rule: 'lag' thresholds P(run length = lag); 'window' thresholds "
            "P(run length <= lag) and places the change at the most probable run length; "
            "'map' fires when the MAP run length drops"
        ),
    )
    lag: int = Field(default=4, ge=1, description="Decision delay in bins for the lag rule")
    threshold: float = Field(default=0.5, gt=0, le=1, description="Posterior threshold (lag rule)")
    max_run_length: int | None = Field(default=1000, description="Run-length truncation")
    min_time_elapsed: int = Field(
        default=3600,
        gt=0,
        description="Seconds between change points; the MAP rule refines a boundary a few bins "
        "after first reporting it, and this spacing absorbs the refinement",
    )
    prior_alpha: float = Field(default=0.1, gt=0)
    prior_beta: float = Field(default=0.1, gt=0)
    prior_kappa: float = Field(default=1.0, gt=0)
    prior_mu: float | None = Field(
        default=None, description="Prior mean; None uses the first minimum RTT seen"
    )
    latency_jump_threshold: float = Field(default=0.5, gt=0, description="Mean min-RTT jump, ms")
    significance_level: float = Field(default=0.05, gt=0, lt=1)
    min_ks_statistic: float = Field(default=0.0, ge=0, le=1, description="Effect-size guard")
    min_period_samples: int = Field(
        default=100,
        ge=2,
        description="Jitter samples in the open period before a provisional verdict (about "
        "three 15-minute bins on the paper dataset; 30 gives noisy KS p-values)",
    )
    device: Literal["cpu", "cuda", "mps"] = "cpu"


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

    @property
    def delay(self) -> float:
        """Seconds between the event's reference time and its emission."""
        return self.emitted_at - (
            self.end_epoch if self.end_epoch is not None else self.start_epoch
        )

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


class OnlineJitterbug:
    """
    Streaming congestion inference: feed RTT samples, get events.

    Parameters
    ----------
    config : StreamingConfig
        Settings; see the class for the meaning of each field.
    """

    def __init__(self, config: StreamingConfig | None = None) -> None:
        self.config = config or StreamingConfig()
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

        self._interval_s = self.config.interval_minutes * 60
        self._min_sep_bins = max(1, math.ceil(self.config.min_time_elapsed / self._interval_s))

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
        bin_id = int(epoch // self._interval_s)
        if self._bin_id is None:
            self._bin_id = bin_id
        elif bin_id > self._bin_id:
            events.extend(self._close_bin(closing_epoch=epoch))
            self._bin_id = bin_id
        elif bin_id < self._bin_id:
            logger.debug("Out-of-order sample at %s dropped", epoch)
            return events

        if rtt < self._bin_min_rtt:
            self._bin_min_rtt = rtt
        if epoch < self._bin_min_epoch:
            self._bin_min_epoch = epoch
        self._raw_epochs.append(epoch)
        self._raw_rtts.append(rtt)

        if self._awaiting_provisional and self._open_period_jitter_count() >= (
            self.config.min_period_samples
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
            mu = self.config.prior_mu if self.config.prior_mu is not None else value
            self._detector = self._OnlineDetector(
                partial(self._constant_hazard, self.config.hazard_lambda),
                self._StudentT(
                    alpha=self.config.prior_alpha,
                    beta=self.config.prior_beta,
                    kappa=self.config.prior_kappa,
                    mu=mu,
                    device=self.config.device,
                ),
                max_run_length=self.config.max_run_length,
                device=self.config.device,
            )
        posterior = self._detector.update(value)
        self._t += 1
        if self._t == 1:
            # The stream start opens the baseline period, as the offline detector's
            # change point at index 0 does; the first real change point closes it.
            events.extend(self._accept_change_point(0, None, emitted_at=closing_epoch))
            return events

        candidate: tuple[int, float] | None = None  # (global bin index, probability)
        if self.config.decision == "lag":
            lag = self.config.lag
            if self._t > lag:
                p = float(self._detector.changepoint_probability(lag))
                if p > self.config.threshold:
                    candidate = (self._t - lag, p)
        elif self.config.decision == "window":
            lag = self.config.lag
            if self._t > lag:
                head = posterior[: lag + 1].detach().cpu().numpy()
                head[0] = 0.0  # run length 0 is the hazard, not evidence
                if head.sum() > self.config.threshold:
                    r = int(head.argmax())
                    candidate = (self._t - r, float(head.sum()))
        else:
            r = int(posterior.argmax())
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

        has_jump = has_jitter = None
        magnitude = ks_stat = p_value = None
        if len(prev_bins) and len(cur_bins):
            magnitude = float(np.mean(cur_bins) - np.mean(prev_bins))
            has_jump = magnitude > self.config.latency_jump_threshold
        if len(prev_jitter) >= 2 and len(cur_jitter) >= 2:
            ks_stat, p_value = (float(v) for v in scipy.stats.ks_2samp(prev_jitter, cur_jitter))
            has_jitter = (
                p_value < self.config.significance_level and ks_stat >= self.config.min_ks_statistic
            )

        state = self._congestion_state
        if has_jump is not None and has_jitter is not None:
            if has_jump and has_jitter:
                state = True
            elif not has_jump:
                state = False
        confidence = 0.0
        if state and magnitude is not None:
            confidence = 0.8 + (0.1 if magnitude > 2 * self.config.latency_jump_threshold else 0.0)
        if stage == "final":
            self._congestion_state = state

        return StreamingEvent(
            kind="verdict",
            emitted_at=emitted_at,
            start_epoch=cur_start,
            end_epoch=end_epoch,
            stage=stage,
            is_congested=state,
            confidence=confidence,
            has_jump=has_jump,
            jump_magnitude=magnitude,
            has_jitter=has_jitter,
            ks_statistic=ks_stat,
            p_value=p_value,
            n_prev=int(len(prev_jitter)),
            n_curr=int(len(cur_jitter)),
        )
