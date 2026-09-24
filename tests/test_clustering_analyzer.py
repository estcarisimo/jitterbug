"""Tests for the non-sequential (clustering) analysis mode."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose
from pydantic import ValidationError
from typer.testing import CliRunner

from jitterbug import JitterbugAnalyzer, JitterbugConfig
from jitterbug.analysis.clustering_analyzer import (
    ClusteringCongestionAnalyzer,
    compute_interval_features,
    smooth_labels,
)
from jitterbug.cli.main import app
from jitterbug.models import ClusteringConfig, RTTDataset

from .conftest import T0, make_measurements

needs_sklearn = pytest.mark.skipif(
    importlib.util.find_spec("sklearn") is None, reason="clustering extra not installed"
)

STEP = 10.0  # seconds between raw samples
INTERVAL = 15  # minutes
PER_INTERVAL = int(INTERVAL * 60 / STEP)
# Intervals of each segment: quiet, congested, quiet, congested, quiet (hours × 4).
SEGMENTS = [(32, False), (16, True), (32, False), (16, True), (16, False)]


def _series(rng: np.random.Generator, jump: float = 20.0) -> np.ndarray:
    """Quiet segments at 20 ms with small jitter; congested ones ``jump`` ms higher and noisier."""
    parts = []
    for intervals, congested in SEGMENTS:
        n = intervals * PER_INTERVAL
        if congested:
            parts.append(20.0 + jump + np.abs(rng.normal(0, 4.0, n)))
        else:
            parts.append(20.0 + np.abs(rng.normal(0, 0.3, n)))
    return np.concatenate(parts)


def _expected_congested() -> list[tuple[float, float]]:
    """(start, end) epochs of the congested segments."""
    out, cursor = [], T0.timestamp()
    for intervals, congested in SEGMENTS:
        end = cursor + intervals * INTERVAL * 60
        if congested:
            out.append((cursor, end))
        cursor = end
    return out


@pytest.fixture
def two_episode_raw(rng: np.random.Generator) -> RTTDataset:
    """28 hours of raw RTTs, one sample every 10 s, with two 4-hour congestion episodes."""
    return RTTDataset(measurements=make_measurements(_series(rng), step_seconds=STEP))


def _analyzer(algorithm: str = "gmm", **overrides: float) -> ClusteringCongestionAnalyzer:
    config = ClusteringConfig(algorithm=algorithm, **overrides)  # type: ignore[arg-type]
    return ClusteringCongestionAnalyzer(
        config, latency_threshold=0.5, significance_level=0.05, interval_minutes=INTERVAL
    )


# --- smoothing --------------------------------------------------------------------------


def _labels(pattern: str) -> np.ndarray:
    return np.array([c == "x" for c in pattern])


@pytest.mark.parametrize(
    ("pattern", "window", "expected"),
    [
        ("xx.xx", 1, "xxxxx"),  # a one-interval gap inside congestion is filled
        ("xx..xx", 1, "xx..xx"),  # a two-interval gap is longer than the window
        ("xx..xx", 2, "xxxxxx"),
        ("..x..", 1, "....."),  # a one-interval burst is dropped
        ("..xx...xxx", 2, ".......xxx"),  # bursts up to the window are dropped, longer kept
        ("..xx..xxx", 2, "..xxxxxxx"),  # gaps are filled before bursts are judged
        (".xx.", 2, "...."),  # gaps at the edges are not filled, so this burst is dropped
        ("x.x", 0, "x.x"),  # window 0 leaves the labels alone
    ],
)
def test_smooth_labels(pattern: str, window: int, expected: str) -> None:
    starts = np.arange(len(pattern)) * 60.0
    result = smooth_labels(_labels(pattern), starts, step=60.0, window=window)
    assert "".join("x" if v else "." for v in result) == expected


def test_smoothing_never_bridges_a_gap_in_the_data() -> None:
    # Intervals 0-1 and 2-3 are congested, but interval 2 starts an hour after interval 1.
    starts = np.array([0.0, 60.0, 3660.0, 3720.0])
    result = smooth_labels(_labels("x.x."), starts, step=60.0, window=1)
    assert not result.any()  # each block holds a single-interval burst, dropped


# --- features ---------------------------------------------------------------------------


def test_interval_features_follow_the_minimum_rtt_bins(two_episode_raw: RTTDataset) -> None:
    features = compute_interval_features(two_episode_raw, INTERVAL)
    min_rtt = two_episode_raw.compute_minimum_intervals(INTERVAL)

    assert len(features) == len(min_rtt) == sum(n for n, _ in SEGMENTS)
    assert_allclose(features.min_rtt, [m.rtt_value for m in min_rtt.measurements])
    assert_allclose(np.diff(features.starts), INTERVAL * 60.0)
    assert features.starts[0] == T0.timestamp()
    # Congested intervals have far more jitter than quiet ones
    quiet, loud = features.jitter_iqr[:32], features.jitter_iqr[32:48]
    assert loud.min() > 5 * quiet.max()


def test_intervals_with_fewer_than_two_jitter_samples_are_skipped() -> None:
    # Two samples in the first interval (one jitter sample), plenty in the second.
    values = np.array([20.0, 21.0] + [20.0 + 0.1 * i for i in range(10)])
    epochs = [0.0, 10.0] + [900.0 + 10 * i for i in range(10)]
    dataset = RTTDataset(
        measurements=[
            m.model_copy(update={"epoch": T0.timestamp() + e})
            for m, e in zip(make_measurements(values), epochs, strict=True)
        ]
    )
    features = compute_interval_features(dataset, INTERVAL)
    assert len(features) == 1
    assert features.skipped == 1


# --- clustering -------------------------------------------------------------------------


@needs_sklearn
@pytest.mark.parametrize("algorithm", ["gmm", "kmeans", "kmeans_silhouette"])
def test_every_algorithm_recovers_the_two_episodes(
    algorithm: str, two_episode_raw: RTTDataset
) -> None:
    result = _analyzer(algorithm).analyze(two_episode_raw)

    congested = [(p.start_epoch, p.end_epoch) for p in result.inferences if p.is_congested]
    assert congested == _expected_congested()
    baseline = result.clusters[0]
    assert not baseline.is_congested
    assert baseline.ks_statistic is None
    for cluster in result.clusters[1:]:
        if cluster.is_congested:
            assert cluster.latency_jump > 0.5
            assert cluster.p_value is not None and cluster.p_value < 0.05
    # periods tile the series in time order
    assert all(
        a.end_epoch == b.start_epoch
        for a, b in zip(result.inferences, result.inferences[1:], strict=False)
    )


@needs_sklearn
def test_clusters_are_numbered_by_increasing_latency(two_episode_raw: RTTDataset) -> None:
    result = _analyzer("kmeans", n_clusters=3).analyze(two_episode_raw)
    medians = [c.median_min_rtt for c in result.clusters]
    assert medians == sorted(medians)
    assert sum(c.size for c in result.clusters) == len(result.labels)


@needs_sklearn
def test_gmm_finds_no_congestion_in_a_flat_series(rng: np.random.Generator) -> None:
    flat = 20.0 + np.abs(rng.normal(0, 0.3, 64 * PER_INTERVAL))
    dataset = RTTDataset(measurements=make_measurements(flat, step_seconds=STEP))

    result = _analyzer("gmm").analyze(dataset)

    # BIC may still split the (non-Gaussian) features, but no cluster sits above the
    # baseline by more than the latency threshold.
    assert set(result.selection_scores) == set(range(1, 7))
    assert all(abs(c.latency_jump) < 0.5 for c in result.clusters)
    assert len(result.inferences) == 1
    assert not result.inferences[0].is_congested


@needs_sklearn
def test_a_jitter_change_without_a_latency_jump_is_not_congestion(
    rng: np.random.Generator,
) -> None:
    # Same jitter pattern with no level shift: the noisy segments' minimum RTT stays at
    # the 20 ms floor, so they differ from the baseline in jitter only.
    dataset = RTTDataset(measurements=make_measurements(_series(rng, jump=0.0), step_seconds=STEP))
    result = _analyzer("kmeans").analyze(dataset)
    assert not any(p.is_congested for p in result.inferences)
    assert result.clusters[1].p_value is not None and result.clusters[1].p_value < 0.05


@needs_sklearn
def test_min_ks_statistic_bounds_the_jitter_effect_size(two_episode_raw: RTTDataset) -> None:
    default = _analyzer("kmeans").analyze(two_episode_raw)
    statistic = default.clusters[1].ks_statistic
    assert statistic is not None and statistic >= ClusteringConfig().min_ks_statistic
    assert default.clusters[1].is_congested

    # A significant test whose statistic is below the minimum is not a jitter change.
    strict = _analyzer("kmeans", min_ks_statistic=min(1.0, statistic + 0.01)).analyze(
        two_episode_raw
    )
    assert strict.clusters[1].p_value is not None and strict.clusters[1].p_value < 0.05
    assert not strict.clusters[1].is_congested
    assert not any(p.is_congested for p in strict.inferences)
    assert not any(p.jitter_analysis.has_significant_jitter for p in strict.inferences)


@needs_sklearn
def test_smoothing_window_zero_keeps_per_interval_verdicts(two_episode_raw: RTTDataset) -> None:
    raw = _analyzer("gmm", min_period_intervals=0).analyze(two_episode_raw)
    smoothed = _analyzer("gmm").analyze(two_episode_raw)
    assert len(raw.inferences) >= len(smoothed.inferences)
    assert all(p.confidence == 1.0 for p in raw.inferences if p.is_congested)


def test_missing_scikit_learn_raises_with_the_install_hint(
    monkeypatch: pytest.MonkeyPatch, two_episode_raw: RTTDataset
) -> None:
    for module in ("sklearn", "sklearn.cluster", "sklearn.metrics", "sklearn.mixture"):
        monkeypatch.setitem(sys.modules, module, None)
    with pytest.raises(ImportError, match=r"jitterbug-inference\[clustering\]"):
        _analyzer("gmm").analyze(two_episode_raw)


def test_too_few_intervals_give_an_empty_result() -> None:
    dataset = RTTDataset(measurements=make_measurements(np.full(5, 20.0), step_seconds=STEP))
    result = _analyzer("gmm").analyze(dataset)
    assert result.inferences == []
    assert result.intervals == 1


# --- configuration ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"algorithm": "dbscan"},
        {"n_clusters": 1},
        {"max_clusters": 1},
        {"min_period_intervals": -1},
        {"latency_threshold": 0.0},
        {"min_ks_statistic": -0.1},
        {"min_ks_statistic": 1.5},
    ],
)
def test_invalid_clustering_options_are_rejected(kwargs: dict) -> None:
    with pytest.raises(ValidationError):
        ClusteringConfig(**kwargs)


@pytest.mark.parametrize(("own", "expected"), [(None, 0.5), (3.0, 3.0)])
def test_clustering_latency_threshold_falls_back_to_the_sequential_one(
    own: float | None, expected: float
) -> None:
    config = JitterbugConfig(analysis_mode="clustering")
    config.clustering.latency_threshold = own
    assert JitterbugAnalyzer(config).clustering_analyzer.latency_threshold == expected


def test_sequential_is_the_default_mode() -> None:
    assert JitterbugConfig().analysis_mode == "sequential"
    with pytest.raises(ValidationError):
        JitterbugConfig(analysis_mode="random")  # type: ignore[arg-type]


# --- analyzer and CLI -------------------------------------------------------------------


@needs_sklearn
def test_analyzer_clustering_mode_end_to_end(two_episode_raw: RTTDataset, tmp_path: Path) -> None:
    config = JitterbugConfig(analysis_mode="clustering")
    config.data_processing.minimum_interval_minutes = INTERVAL
    analyzer = JitterbugAnalyzer(config)

    result = analyzer.analyze(two_episode_raw)

    assert len(result.get_congested_periods()) == 2
    assert result.metadata["analysis_mode"] == "clustering"
    assert result.metadata["change_points"] == 0
    summary = result.metadata["clustering"]
    assert summary["algorithm"] == "gmm"
    assert summary["n_clusters"] == len(summary["clusters"])
    assert analyzer.change_points == []
    # the metadata survives the JSON exporter
    out = tmp_path / "results.json"
    analyzer.save_results(result, out, "json")
    assert json.loads(out.read_text())["metadata"]["clustering"]["algorithm"] == "gmm"


@needs_sklearn
def test_cli_mode_clustering(two_episode_raw: RTTDataset, tmp_path: Path) -> None:
    csv = tmp_path / "rtts.csv"
    epochs, values = two_episode_raw.to_arrays()
    csv.write_text(
        "epoch,values\n" + "".join(f"{e},{v}\n" for e, v in zip(epochs, values, strict=True))
    )
    out = tmp_path / "out.json"

    result = CliRunner().invoke(
        app,
        ["analyze", str(csv), "--mode", "clustering", "--clustering-algorithm", "kmeans"]
        + ["--output", str(out)],
    )

    assert result.exit_code == 0, result.output
    data = json.loads(out.read_text())
    assert data["metadata"]["clustering"]["algorithm"] == "kmeans"
    assert sum(p["is_congested"] for p in data["inferences"]) == 2


def test_cli_rejects_an_unknown_mode(tmp_path: Path) -> None:
    csv = tmp_path / "rtts.csv"
    csv.write_text("epoch,values\n0,20\n10,21\n")
    result = CliRunner().invoke(app, ["analyze", str(csv), "--mode", "random"])
    assert result.exit_code != 0


@needs_sklearn
def test_clustering_latency_threshold_above_the_episodes_finds_no_congestion(
    two_episode_raw: RTTDataset,
) -> None:
    config = JitterbugConfig(analysis_mode="clustering")
    config.data_processing.minimum_interval_minutes = INTERVAL
    config.clustering.latency_threshold = 50.0  # the episodes are 20 ms above the floor
    result = JitterbugAnalyzer(config).analyze(two_episode_raw)
    assert result.get_congested_periods() == []
    assert config.latency_jump.threshold == 0.5  # the sequential threshold is untouched
