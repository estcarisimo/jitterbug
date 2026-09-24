"""
Non-sequential congestion inference: cluster minimum-RTT intervals by (latency, jitter).

The sequential method (PAM 2022) splits the series at change points and compares each
period with the one before it. This mode drops the time order: every minimum-RTT
interval is a point with two features (its minimum RTT and the interquartile range of the
jitter samples inside it), the points are clustered, and each cluster is compared with
the baseline cluster (lowest median minimum RTT), no matter whether their intervals are
adjacent in time. A cluster is congested when both signals of the paper hold: its median
minimum RTT exceeds the baseline's by more than the latency threshold, and the raw jitter
samples of the two clusters differ (a significant Kolmogorov-Smirnov test whose statistic
reaches a minimum effect size).
Consecutive intervals with the same verdict are merged into periods, after an optional
temporal smoothing, so the output has the same shape as the sequential mode.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, cast

import numpy as np
import pandas as pd
import scipy.stats

from ..models import (
    CongestionInference,
    JitterAnalysis,
    LatencyJump,
    RTTDataset,
)
from ..models.config import ClusteringConfig

logger = logging.getLogger(__name__)

INSTALL_HINT = (
    "The clustering mode needs scikit-learn. Install the extra: "
    "pip install 'jitterbug-inference[clustering]'"
)


@dataclass(frozen=True)
class IntervalFeatures:
    """
    Minimum-RTT intervals with the features used for clustering.

    Attributes
    ----------
    starts : np.ndarray
        Start of each interval (epoch seconds, UTC), sorted.
    min_rtt : np.ndarray
        Minimum RTT of each interval (ms).
    jitter_iqr : np.ndarray
        Interquartile range of the jitter samples of each interval (ms).
    jitter : list[np.ndarray]
        Raw jitter samples (differences of consecutive RTTs) of each interval.
    interval_seconds : float
        Interval length in seconds.
    skipped : int
        Non-empty intervals left out because they have fewer than two jitter samples.
    """

    starts: np.ndarray
    min_rtt: np.ndarray
    jitter_iqr: np.ndarray
    jitter: list[np.ndarray]
    interval_seconds: float
    skipped: int = 0

    def __len__(self) -> int:
        return len(self.starts)


@dataclass(frozen=True)
class ClusterSummary:
    """
    Statistics and verdict for one cluster.

    Attributes
    ----------
    cluster : int
        Cluster index; clusters are numbered by increasing median minimum RTT, so ``0``
        is always the baseline.
    size : int
        Number of intervals in the cluster.
    median_min_rtt : float
        Median minimum RTT of its intervals (ms).
    median_jitter_iqr : float
        Median jitter IQR of its intervals (ms).
    latency_jump : float
        ``median_min_rtt`` minus the baseline's (ms).
    ks_statistic : float | None
        Kolmogorov-Smirnov statistic against the baseline's jitter (``None`` for the
        baseline).
    p_value : float | None
        p-value of that test (``None`` for the baseline).
    is_congested : bool
        Whether the cluster is classified as congested.
    """

    cluster: int
    size: int
    median_min_rtt: float
    median_jitter_iqr: float
    latency_jump: float
    ks_statistic: float | None
    p_value: float | None
    is_congested: bool


@dataclass
class ClusteringResult:
    """
    Output of :meth:`ClusteringCongestionAnalyzer.analyze`.

    Attributes
    ----------
    inferences : list[CongestionInference]
        Periods of consecutive intervals with the same verdict, in time order.
    clusters : list[ClusterSummary]
        One summary per cluster, baseline first.
    labels : np.ndarray
        Cluster of each interval.
    selection_scores : dict[int, float]
        Model-selection score per number of clusters tried (BIC for ``gmm``, lower is
        better; silhouette for ``kmeans_silhouette``, higher is better); empty for
        ``kmeans``.
    intervals : int
        Intervals clustered.
    intervals_skipped : int
        Non-empty intervals left out for having fewer than two jitter samples.
    """

    inferences: list[CongestionInference]
    clusters: list[ClusterSummary]
    labels: np.ndarray
    selection_scores: dict[int, float] = field(default_factory=dict)
    intervals: int = 0
    intervals_skipped: int = 0

    def metadata(self, algorithm: str) -> dict[str, Any]:
        """Return a JSON-serializable summary for ``CongestionInferenceResult.metadata``."""
        return {
            "algorithm": algorithm,
            "n_clusters": len(self.clusters),
            "selection_scores": {str(k): v for k, v in self.selection_scores.items()},
            "intervals": self.intervals,
            "intervals_skipped": self.intervals_skipped,
            "clusters": [asdict(c) for c in self.clusters],
        }


def compute_interval_features(rtt_data: RTTDataset, interval_minutes: int) -> IntervalFeatures:
    """
    Bin raw RTTs into minimum-RTT intervals and compute the clustering features.

    The bins are the same as ``RTTDataset.compute_minimum_intervals``. Jitter is the
    difference between consecutive RTTs, assigned to the interval of the later sample.

    Parameters
    ----------
    rtt_data : RTTDataset
        Raw RTT measurements.
    interval_minutes : int
        Interval length in minutes.

    Returns
    -------
    IntervalFeatures
        Intervals with at least two jitter samples, sorted by start time.
    """
    epochs, rtts = rtt_data.to_arrays()
    order = np.argsort(epochs, kind="stable")
    df = pd.DataFrame({"epoch": epochs[order], "rtt": rtts[order]})
    df["jitter"] = df["rtt"].diff()
    df["datetime"] = pd.to_datetime(df["epoch"], unit="s")

    starts, min_rtt, jitter_iqr, jitter = [], [], [], []
    skipped = 0
    for bin_start, group in df.groupby(pd.Grouper(key="datetime", freq=f"{interval_minutes}min")):
        if group.empty:
            continue
        samples = group["jitter"].dropna().to_numpy()
        if len(samples) < 2:
            skipped += 1
            continue
        q75, q25 = np.percentile(samples, [75, 25])
        starts.append((cast(pd.Timestamp, bin_start) - pd.Timestamp(0)) / pd.Timedelta(seconds=1))
        min_rtt.append(float(group["rtt"].min()))
        jitter_iqr.append(float(q75 - q25))
        jitter.append(samples)

    if skipped:
        logger.info(f"Skipped {skipped} intervals with fewer than two jitter samples")
    return IntervalFeatures(
        starts=np.asarray(starts, dtype=float),
        min_rtt=np.asarray(min_rtt, dtype=float),
        jitter_iqr=np.asarray(jitter_iqr, dtype=float),
        jitter=jitter,
        interval_seconds=interval_minutes * 60.0,
        skipped=skipped,
    )


def smooth_labels(
    congested: np.ndarray, starts: np.ndarray, step: float, window: int
) -> np.ndarray:
    """
    Remove short gaps and short bursts from per-interval congestion labels.

    Within each block of contiguous intervals, runs of at most ``window`` non-congested
    intervals that lie between congested ones are relabeled congested; then congested
    runs of at most ``window`` intervals are relabeled non-congested. A missing interval
    (a gap in the data) always ends a block.

    Parameters
    ----------
    congested : np.ndarray
        Boolean label per interval, in time order.
    starts : np.ndarray
        Interval start times (epoch seconds), same order.
    step : float
        Interval length in seconds.
    window : int
        Largest run length to fill or drop; ``0`` returns a copy of ``congested``.

    Returns
    -------
    np.ndarray
        Smoothed boolean labels.
    """
    labels = np.asarray(congested, dtype=bool).copy()
    if window <= 0 or len(labels) == 0:
        return labels
    for lo, hi in _contiguous_blocks(starts, step):
        block = labels[lo:hi]
        for start, end, value in _runs(block):
            if not value and end - start <= window and start > 0 and end < len(block):
                block[start:end] = True
        for start, end, value in _runs(block):
            if value and end - start <= window:
                block[start:end] = False
        labels[lo:hi] = block
    return labels


def _contiguous_blocks(starts: np.ndarray, step: float) -> list[tuple[int, int]]:
    """Index ranges ``[lo, hi)`` of intervals that follow each other without gaps."""
    if len(starts) == 0:
        return []
    breaks = np.flatnonzero(np.diff(starts) > step * 1.5) + 1
    edges = [0, *breaks.tolist(), len(starts)]
    return list(zip(edges[:-1], edges[1:], strict=True))


def _runs(values: np.ndarray) -> list[tuple[int, int, bool]]:
    """Maximal runs of equal values as ``(start, end, value)`` with ``end`` exclusive."""
    runs: list[tuple[int, int, bool]] = []
    start = 0
    for i in range(1, len(values) + 1):
        if i == len(values) or values[i] != values[start]:
            runs.append((start, i, bool(values[start])))
            start = i
    return runs


def _standardize(features: np.ndarray) -> np.ndarray:
    std = features.std(axis=0)
    std[std == 0] = 1.0
    return (features - features.mean(axis=0)) / std


class ClusteringCongestionAnalyzer:
    """
    Infer congestion by clustering minimum-RTT intervals (non-sequential mode).

    Parameters
    ----------
    config : ClusteringConfig
        Clustering options.
    latency_threshold : float
        Minimum excess of a cluster's median minimum RTT over the baseline's (ms), from
        ``clustering.latency_threshold`` or else ``latency_jump.threshold``.
    significance_level : float
        Significance level of the KS test, from ``jitter_analysis.significance_level``.
    interval_minutes : int
        Minimum-RTT interval length, from ``data_processing.minimum_interval_minutes``.
    """

    def __init__(
        self,
        config: ClusteringConfig,
        latency_threshold: float,
        significance_level: float,
        interval_minutes: int,
    ) -> None:
        self.config = config
        self.latency_threshold = latency_threshold
        self.significance_level = significance_level
        self.interval_minutes = interval_minutes

    def analyze(self, rtt_data: RTTDataset) -> ClusteringResult:
        """
        Cluster the intervals of ``rtt_data`` and classify each cluster.

        Parameters
        ----------
        rtt_data : RTTDataset
            Raw RTT measurements.

        Returns
        -------
        ClusteringResult
            Periods, per-cluster summaries and model-selection scores. With fewer than
            two usable intervals the result is empty.

        Raises
        ------
        ImportError
            If scikit-learn is not installed.
        """
        features = compute_interval_features(rtt_data, self.interval_minutes)
        if len(features) < 2:
            logger.warning("Need at least two intervals with jitter samples for clustering")
            return ClusteringResult(
                inferences=[],
                clusters=[],
                labels=np.array([], dtype=int),
                intervals=len(features),
                intervals_skipped=features.skipped,
            )

        points = _standardize(np.column_stack([features.min_rtt, features.jitter_iqr]))
        raw_labels, scores = self._cluster(points)
        labels = self._order_by_latency(raw_labels, features.min_rtt)
        clusters = self._classify(labels, features)
        logger.info(
            f"{self.config.algorithm}: {len(clusters)} clusters, "
            f"{sum(c.is_congested for c in clusters)} congested"
        )

        verdict = np.array([clusters[c].is_congested for c in labels])
        smoothed = smooth_labels(
            verdict, features.starts, features.interval_seconds, self.config.min_period_intervals
        )
        inferences = self._periods(features, labels, verdict, smoothed, clusters)
        return ClusteringResult(
            inferences=inferences,
            clusters=clusters,
            labels=labels,
            selection_scores=scores,
            intervals=len(features),
            intervals_skipped=features.skipped,
        )

    def _cluster(self, points: np.ndarray) -> tuple[np.ndarray, dict[int, float]]:
        """Cluster standardized points; return labels and model-selection scores."""
        try:
            from sklearn.cluster import KMeans
            from sklearn.metrics import silhouette_score
            from sklearn.mixture import GaussianMixture
        except ImportError as e:
            raise ImportError(INSTALL_HINT) from e

        n = len(points)
        seed = self.config.random_state
        algorithm = self.config.algorithm

        if algorithm == "gmm":
            scores: dict[int, float] = {}
            models = {}
            for k in range(1, min(self.config.max_clusters, n) + 1):
                model = GaussianMixture(n_components=k, random_state=seed, n_init=5).fit(points)
                scores[k] = float(model.bic(points))
                models[k] = model
            best = min(scores, key=lambda k: scores[k])
            return models[best].predict(points), scores

        if algorithm == "kmeans":
            k = min(self.config.n_clusters, n)
            return KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(points), {}

        # kmeans_silhouette: the silhouette needs 2 <= k <= n - 1
        candidates = range(2, min(self.config.max_clusters, n - 1) + 1)
        if not candidates:
            return np.zeros(n, dtype=int), {}
        fits = {
            k: KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(points)
            for k in candidates
        }
        scores = {
            k: float(silhouette_score(points, labels))
            for k, labels in fits.items()
            if len(np.unique(labels)) > 1
        }
        if not scores:
            return np.zeros(n, dtype=int), {}
        best = max(scores, key=lambda k: scores[k])
        return fits[best], scores

    @staticmethod
    def _order_by_latency(labels: np.ndarray, min_rtt: np.ndarray) -> np.ndarray:
        """Renumber clusters by increasing median minimum RTT (0 = baseline)."""
        present = np.unique(labels)
        medians = [np.median(min_rtt[labels == c]) for c in present]
        mapping = {
            int(c): rank for rank, c in enumerate(present[np.argsort(medians, kind="stable")])
        }
        return np.array([mapping[int(c)] for c in labels], dtype=int)

    def _classify(self, labels: np.ndarray, features: IntervalFeatures) -> list[ClusterSummary]:
        """Compare every cluster with the baseline (cluster 0)."""

        def pooled_jitter(cluster: int) -> np.ndarray:
            return np.concatenate([features.jitter[i] for i in np.flatnonzero(labels == cluster)])

        baseline_rtt = float(np.median(features.min_rtt[labels == 0]))
        baseline_jitter = pooled_jitter(0)
        summaries = []
        for cluster in range(int(labels.max()) + 1):
            members = labels == cluster
            median_rtt = float(np.median(features.min_rtt[members]))
            jump = median_rtt - baseline_rtt
            ks_statistic: float | None = None
            p_value: float | None = None
            congested = False
            if cluster > 0:
                ks = scipy.stats.ks_2samp(baseline_jitter, pooled_jitter(cluster))
                ks_statistic, p_value = float(ks.statistic), float(ks.pvalue)
                congested = jump > self.latency_threshold and self._jitter_differs(
                    ks_statistic, p_value
                )
            summaries.append(
                ClusterSummary(
                    cluster=cluster,
                    size=int(members.sum()),
                    median_min_rtt=median_rtt,
                    median_jitter_iqr=float(np.median(features.jitter_iqr[members])),
                    latency_jump=jump,
                    ks_statistic=ks_statistic,
                    p_value=p_value,
                    is_congested=congested,
                )
            )
        return summaries

    def _jitter_differs(self, ks_statistic: float | None, p_value: float | None) -> bool:
        """Whether a KS comparison with the baseline counts as a jitter change."""
        if ks_statistic is None or p_value is None:
            return False
        return p_value < self.significance_level and ks_statistic >= self.config.min_ks_statistic

    def _periods(
        self,
        features: IntervalFeatures,
        labels: np.ndarray,
        verdict: np.ndarray,
        smoothed: np.ndarray,
        clusters: list[ClusterSummary],
    ) -> list[CongestionInference]:
        """Merge consecutive intervals with the same smoothed verdict into periods."""
        step = features.interval_seconds
        inferences = []
        for lo, hi in _contiguous_blocks(features.starts, step):
            for start, end, congested in _runs(smoothed[lo:hi]):
                idx = np.arange(lo + start, lo + end)
                # The cluster that describes the period: the most common congested cluster
                # of a congested period, the most common cluster otherwise.
                pool = idx[verdict[idx]] if congested and verdict[idx].any() else idx
                dominant = clusters[int(np.bincount(labels[pool]).argmax())]
                start_epoch = float(features.starts[idx[0]])
                end_epoch = float(features.starts[idx[-1]] + step)
                start_ts = datetime.fromtimestamp(start_epoch, tz=timezone.utc)
                end_ts = datetime.fromtimestamp(end_epoch, tz=timezone.utc)
                inferences.append(
                    CongestionInference(
                        start_timestamp=start_ts,
                        end_timestamp=end_ts,
                        start_epoch=start_epoch,
                        end_epoch=end_epoch,
                        is_congested=congested,
                        # share of the period's intervals that were congested before smoothing
                        confidence=float(verdict[idx].mean()) if congested else 0.0,
                        latency_jump=LatencyJump(
                            start_timestamp=start_ts,
                            end_timestamp=end_ts,
                            start_epoch=start_epoch,
                            end_epoch=end_epoch,
                            has_jump=dominant.latency_jump > self.latency_threshold,
                            magnitude=dominant.latency_jump,
                            threshold=self.latency_threshold,
                        ),
                        jitter_analysis=JitterAnalysis(
                            start_timestamp=start_ts,
                            end_timestamp=end_ts,
                            start_epoch=start_epoch,
                            end_epoch=end_epoch,
                            has_significant_jitter=self._jitter_differs(
                                dominant.ks_statistic, dominant.p_value
                            ),
                            jitter_metric=dominant.ks_statistic or 0.0,
                            method="ks_test",
                            threshold=self.significance_level,
                            p_value=dominant.p_value,
                        ),
                    )
                )
        return inferences
