"""
Regression tests against the PAM 2022 dataset bundled in ``examples/network_analysis``.

Two kinds of assertion, deliberately kept apart:

* **Golden values** pin what the current implementation produces (period and congestion
  counts). They exist to catch unintended behavior changes; if a change is intended,
  update the numbers and say why in the changelog.
* **Agreement with the paper's reference output** (``expected_results/*.csv``, produced
  by the 1.x scripts) is measured by interval overlap, because the 2.x minimum-RTT binning
  does not reproduce the 1.x interval boundaries exactly. The thresholds are the current
  recall with zero spurious detections; they must not go down.
"""

import csv
import importlib.util
from pathlib import Path

import pytest
from numpy.testing import assert_allclose

from jitterbug import JitterbugAnalyzer, JitterbugConfig
from jitterbug.models import CongestionInferenceResult

DATA_DIR = Path(__file__).resolve().parents[1] / "examples" / "network_analysis"
RAW_CSV = DATA_DIR / "data" / "raw.csv"
REFERENCE = {
    "jitter_dispersion": DATA_DIR / "expected_results" / "jd_inferences.csv",
    "ks_test": DATA_DIR / "expected_results" / "kstest_inferences.csv",
}

pytestmark = pytest.mark.skipif(not RAW_CSV.exists(), reason="PAM 2022 dataset not present")

Interval = tuple[float, float]


def _reference_congested(method: str) -> list[Interval]:
    with REFERENCE[method].open() as f:
        rows = list(csv.DictReader(f))
    return [(float(r["starts"]), float(r["ends"])) for r in rows if float(r["congestion"]) == 1]


def _congested(results: CongestionInferenceResult) -> list[Interval]:
    return [(p.start_epoch, p.end_epoch) for p in results.inferences if p.is_congested]


def _overlap(a: Interval, b: Interval) -> float:
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def _covers(needle: Interval, haystack: list[Interval]) -> bool:
    """True when some interval in ``haystack`` overlaps more than half of ``needle``."""
    return any(_overlap(needle, other) > 0.5 * (needle[1] - needle[0]) for other in haystack)


def _agreement(results: CongestionInferenceResult, method: str) -> tuple[int, int, int]:
    """Return (reference periods recovered, reference periods, spurious detections)."""
    ours, theirs = _congested(results), _reference_congested(method)
    recovered = sum(_covers(ref, ours) for ref in theirs)
    spurious = sum(not _covers(mine, theirs) for mine in ours)
    return recovered, len(theirs), spurious


def _run(algorithm: str, method: str) -> CongestionInferenceResult:
    config = JitterbugConfig()
    config.change_point_detection.algorithm = algorithm  # type: ignore[assignment]
    config.jitter_analysis.method = method  # type: ignore[assignment]
    return JitterbugAnalyzer(config).analyze_from_file(RAW_CSV)


@pytest.fixture(scope="module")
def ruptures_jd() -> CongestionInferenceResult:
    return _run("ruptures", "jitter_dispersion")


def test_ruptures_jitter_dispersion_golden(ruptures_jd: CongestionInferenceResult) -> None:
    assert len(ruptures_jd.inferences) == 22
    assert len(_congested(ruptures_jd)) == 11
    assert ruptures_jd.metadata["total_measurements"] == 47163


def test_ruptures_jitter_dispersion_agrees_with_paper(
    ruptures_jd: CongestionInferenceResult,
) -> None:
    recovered, total, spurious = _agreement(ruptures_jd, "jitter_dispersion")
    assert total == 15
    assert recovered >= 11
    assert spurious == 0


@pytest.mark.slow
@pytest.mark.skipif(
    importlib.util.find_spec("bayesian_changepoint_detection") is None,
    reason="bcp extra not installed",
)
class TestBayesianKSTest:
    """The paper's configuration: Bayesian change points + Kolmogorov-Smirnov test."""

    @pytest.fixture(scope="class")
    def bcp_ks(self) -> CongestionInferenceResult:
        return _run("bcp", "ks_test")

    def test_golden(self, bcp_ks: CongestionInferenceResult) -> None:
        assert len(bcp_ks.inferences) == 28
        assert len(_congested(bcp_ks)) == 14

    def test_agrees_with_paper(self, bcp_ks: CongestionInferenceResult) -> None:
        recovered, total, spurious = _agreement(bcp_ks, "ks_test")
        assert total == 15
        assert recovered >= 14
        assert spurious == 0


@pytest.mark.slow
@pytest.mark.skipif(
    importlib.util.find_spec("bayesian_changepoint_detection") is None,
    reason="bcp extra not installed",
)
class TestOnlineReplay:
    """The online mode (MAP rule, defaults) replayed over the paper dataset."""

    @pytest.fixture(scope="class")
    def summary(self) -> dict:
        from jitterbug.io import DataLoader
        from jitterbug.streaming import replay, score

        dataset = DataLoader().load_from_file(RAW_CSV)
        return score(replay(dataset, JitterbugConfig()), REFERENCE["ks_test"])

    def test_golden(self, summary: dict) -> None:
        assert summary["change_points"] == 33
        assert summary["periods"] == 31
        assert summary["congested"] == 15
        assert summary["provisional_pairs"] == 31
        assert summary["provisional_flips"] == 0

    def test_agrees_with_paper(self, summary: dict) -> None:
        assert summary["reference_congested"] == 15
        assert summary["recovered"] >= 14
        assert summary["spurious"] == 0
        assert summary["boundaries_within_30min"] == 20
        assert summary["reference_boundaries"] == 30

    def test_detection_delays(self, summary: dict) -> None:
        # One bin is 15 min; delays are measured to the sample that closes the bin.
        assert_allclose(summary["onset_delay_min_median"], 15.0, atol=0.1)
        assert_allclose(summary["onset_delay_min_max"], 90.0, atol=0.1)
        assert_allclose(summary["return_delay_min_median"], 135.0, atol=0.1)
        assert_allclose(summary["return_delay_min_max"], 255.0, atol=0.1)


@pytest.mark.slow
@pytest.mark.skipif(
    importlib.util.find_spec("bayesian_changepoint_detection") is None,
    reason="bcp extra not installed",
)
class TestSlidingWindowReplay:
    """Sliding-window back end on the paper dataset: BCP + KS, 72 h window, rerun every 4 bins."""

    @pytest.fixture(scope="class")
    def events(self) -> list:
        from jitterbug.io import DataLoader
        from jitterbug.models import StreamingConfig
        from jitterbug.streaming import replay

        config = JitterbugConfig(streaming=StreamingConfig(backend="window", rerun_every_bins=4))
        config.change_point_detection.algorithm = "bcp"  # type: ignore[assignment]
        config.jitter_analysis.method = "ks_test"  # type: ignore[assignment]
        return replay(DataLoader().load_from_file(RAW_CSV), config)

    @pytest.fixture(scope="class")
    def summary(self, events: list) -> dict:
        from jitterbug.streaming import score

        return score(events, REFERENCE["ks_test"])

    def test_golden(self, summary: dict) -> None:
        assert summary["change_points"] == 32
        assert summary["periods"] == 30
        assert summary["congested"] == 15
        assert summary["provisional_pairs"] == 30
        assert summary["provisional_flips"] == 1

    def test_agrees_with_paper(self, summary: dict) -> None:
        assert summary["recovered"] >= 14
        # One 1.75 h period next to a real congestion episode, where the detector moved
        # the boundary by two bins between reruns and the emitted one stood.
        assert summary["spurious"] == 1
        assert summary["boundaries_within_30min"] == 30

    def test_detection_delays_are_bounded_by_the_cadence(self, summary: dict) -> None:
        # Reruns every 4 bins (1 h) and two stable runs: 2 h median onset delay.
        assert_allclose(summary["onset_delay_min_median"], 120.0, atol=0.1)
        assert_allclose(summary["onset_delay_min_max"], 225.0, atol=0.1)
        assert_allclose(summary["return_delay_min_median"], 180.0, atol=0.1)
        assert_allclose(summary["return_delay_min_max"], 360.0, atol=0.1)

    def test_events_are_consistent(self, events: list) -> None:
        """Change points in time order; finals a contiguous chain over emitted change points."""
        cps = [e.start_epoch for e in events if e.kind == "change_point"]
        assert cps == sorted(cps) and len(cps) == len(set(cps))
        finals = [e for e in events if e.stage == "final"]
        for earlier, later in zip(finals, finals[1:], strict=False):
            assert earlier.end_epoch == later.start_epoch
        assert all(e.start_epoch in cps and e.end_epoch in cps for e in finals)
        assert all(e.n_prev > 0 and e.n_curr > 0 for e in finals)


@pytest.mark.skipif(
    importlib.util.find_spec("sklearn") is None, reason="clustering extra not installed"
)
@pytest.mark.parametrize(
    ("algorithm", "min_recovered"),
    [("gmm", 15), ("kmeans", 14), ("kmeans_silhouette", 14)],
)
def test_clustering_mode_agrees_with_paper(algorithm: str, min_recovered: int) -> None:
    """
    Non-sequential mode (default smoothing of two intervals), compared with the KS-test
    reference. The two periods counted as spurious are at the two ends of the data, where
    the minimum RTT is elevated but the sequential reference has no verdict (its first
    period has no predecessor, its last no closing change point). The bound keeps other
    spurious detections from appearing.
    """
    config = JitterbugConfig(analysis_mode="clustering")
    config.clustering.algorithm = algorithm  # type: ignore[assignment]
    results = JitterbugAnalyzer(config).analyze_from_file(RAW_CSV)

    assert len(results.inferences) == 31
    assert len(_congested(results)) == 16
    recovered, total, spurious = _agreement(results, "ks_test")
    assert total == 15
    assert recovered >= min_recovered
    assert spurious <= 2


@pytest.mark.skipif(
    importlib.util.find_spec("sklearn") is None, reason="clustering extra not installed"
)
def test_clustering_ks_statistic_separates_congested_clusters() -> None:
    """
    Every cluster has p < 1e-49 against the baseline, so the p-value cannot tell them
    apart; the KS statistic can. The default ``min_ks_statistic`` (0.1) sits between the
    cluster at the baseline's latency and the congested ones.
    """
    config = JitterbugConfig(analysis_mode="clustering")
    results = JitterbugAnalyzer(config).analyze_from_file(RAW_CSV)
    clusters = results.metadata["clustering"]["clusters"][1:]
    minimum = config.clustering.min_ks_statistic

    assert all(c["p_value"] < 1e-49 for c in clusters)
    same_latency = [c for c in clusters if c["latency_jump"] < 0.1]
    assert same_latency and all(c["ks_statistic"] < minimum for c in same_latency)
    assert all(c["ks_statistic"] > 2 * minimum for c in clusters if c["is_congested"])
