"""
The two-period decision rule shared by the online back ends.

Given the minimum-RTT bins and the jitter observations of a period and of the period
before it, decide whether the period is congested the way the sequential pipeline does: a
latency jump is a rise of the mean minimum RTT above ``latency_jump.threshold``; a jitter
change is, with ``jitter_analysis.method: ks_test``, a significant Kolmogorov-Smirnov test
on the raw jitter samples with a statistic of at least ``streaming.min_ks_statistic``, and
with ``jitter_dispersion`` a rise of the mean dispersion (trailing filters, see
``JitterAnalyzer.compute_causal_jitter_dispersion``) above ``jitter_analysis.threshold``;
the congestion state carries over: both → congested, no jump → not congested, jump without
jitter change → unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import scipy.stats

from ..models import JitterbugConfig


@dataclass(frozen=True)
class PeriodVerdict:
    """Outcome of ``two_period_verdict``."""

    is_congested: bool
    confidence: float
    has_jump: bool | None
    jump_magnitude: float | None
    has_jitter: bool | None
    ks_statistic: float | None
    p_value: float | None
    n_prev: int
    n_curr: int
    jitter_method: Literal["ks_test", "jitter_dispersion"] = "ks_test"
    jitter_metric: float | None = None


def two_period_verdict(
    prev_bins: np.ndarray,
    cur_bins: np.ndarray,
    prev_jitter: np.ndarray,
    cur_jitter: np.ndarray,
    previous_state: bool,
    config: JitterbugConfig,
) -> PeriodVerdict:
    """
    Classify a period against the one before it.

    Parameters
    ----------
    prev_bins, cur_bins : np.ndarray
        Minimum RTT per bin of the previous and the current period (ms).
    prev_jitter, cur_jitter : np.ndarray
        Jitter observations of the previous and the current period: consecutive RTT
        differences (ms) for ``ks_test``, causal dispersion values for
        ``jitter_dispersion``.
    previous_state : bool
        Congestion state after the previous period's final verdict.
    config : JitterbugConfig
        ``latency_jump.threshold``, ``jitter_analysis.method``, ``jitter_analysis.threshold``,
        ``jitter_analysis.significance_level`` and ``streaming.min_ks_statistic`` are used.

    Returns
    -------
    PeriodVerdict
        The decision and the statistics behind it. Tests that could not run (too few
        observations) leave their fields ``None`` and the state unchanged.
    """
    threshold = config.latency_jump.threshold
    method = config.jitter_analysis.method
    has_jump = has_jitter = None
    magnitude = ks_stat = p_value = metric = None
    if len(prev_bins) and len(cur_bins):
        magnitude = float(np.mean(cur_bins) - np.mean(prev_bins))
        has_jump = magnitude > threshold
    if method == "ks_test":
        if len(prev_jitter) >= 2 and len(cur_jitter) >= 2:
            ks_stat, p_value = (float(v) for v in scipy.stats.ks_2samp(prev_jitter, cur_jitter))
            significant = p_value < config.jitter_analysis.significance_level
            has_jitter = significant and ks_stat >= config.streaming.min_ks_statistic
            metric = ks_stat
    elif len(prev_jitter) >= 1 and len(cur_jitter) >= 2:
        # Same rule as JitterAnalyzer.analyze_jitter_dispersion: mean dispersion rises by
        # more than the threshold.
        metric = float(np.mean(cur_jitter) - np.mean(prev_jitter))
        has_jitter = metric > config.jitter_analysis.threshold

    state = previous_state
    if has_jump is not None and has_jitter is not None:
        if has_jump and has_jitter:
            state = True
        elif not has_jump:
            state = False
    confidence = 0.0
    if state and magnitude is not None:
        confidence = 0.8 + (0.1 if magnitude > 2 * threshold else 0.0)
    return PeriodVerdict(
        is_congested=state,
        confidence=confidence,
        has_jump=has_jump,
        jump_magnitude=magnitude,
        has_jitter=has_jitter,
        ks_statistic=ks_stat,
        p_value=p_value,
        n_prev=int(len(prev_jitter)),
        n_curr=int(len(cur_jitter)),
        jitter_method=method,
        jitter_metric=metric,
    )
