"""Unit tests for the period classifiers in ``jitterbug.analysis``."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from jitterbug.analysis import CongestionInferenceAnalyzer, JitterAnalyzer, LatencyJumpAnalyzer
from jitterbug.models import JitterAnalysisConfig, LatencyJumpConfig, MinimumRTTDataset, RTTDataset

from .conftest import change_point_at, jitter, jump

# Segment boundaries of the three_segment_* fixtures (indices into the min-RTT series).
QUIET_END, LOUD_END, LAST = 30, 60, 89


def _boundaries(dataset: MinimumRTTDataset | RTTDataset, *indices: int) -> list:
    return [change_point_at(dataset, i) for i in indices]


class TestLatencyJumpAnalyzer:
    def test_needs_two_change_points(self, three_segment_min_rtt: MinimumRTTDataset) -> None:
        analyzer = LatencyJumpAnalyzer(LatencyJumpConfig())
        assert analyzer.analyze(three_segment_min_rtt, []) == []
        assert analyzer.analyze(three_segment_min_rtt, _boundaries(three_segment_min_rtt, 0)) == []

    def test_detects_the_rise_and_the_fall(self, three_segment_min_rtt: MinimumRTTDataset) -> None:
        analyzer = LatencyJumpAnalyzer(LatencyJumpConfig(threshold=0.5))
        cps = _boundaries(three_segment_min_rtt, 0, QUIET_END, LOUD_END, LAST)
        jumps = analyzer.analyze(three_segment_min_rtt, cps)

        # The first period has no predecessor, so two results for three periods.
        assert [j.has_jump for j in jumps] == [True, False]
        assert_allclose(jumps[0].magnitude, 20.0, atol=1.0)  # 20 ms -> 40 ms
        assert_allclose(jumps[1].magnitude, -20.0, atol=1.0)  # back down
        assert jumps[0].start_epoch == cps[1].epoch
        assert jumps[0].end_epoch == cps[2].epoch

    def test_threshold_is_respected(self, three_segment_min_rtt: MinimumRTTDataset) -> None:
        strict = LatencyJumpAnalyzer(LatencyJumpConfig(threshold=25.0))
        cps = _boundaries(three_segment_min_rtt, 0, QUIET_END, LOUD_END, LAST)
        assert [j.has_jump for j in strict.analyze(three_segment_min_rtt, cps)] == [False, False]

    def test_statistics(self, three_segment_min_rtt: MinimumRTTDataset) -> None:
        analyzer = LatencyJumpAnalyzer(LatencyJumpConfig())
        cps = _boundaries(three_segment_min_rtt, 0, QUIET_END, LOUD_END, LAST)
        stats = analyzer.get_jump_statistics(analyzer.analyze(three_segment_min_rtt, cps))
        assert stats["total_periods"] == 2
        assert stats["jump_periods"] == 1
        assert_allclose(stats["max_magnitude"], 20.0, atol=1.0)


class TestJitterAnalyzerDispersion:
    def test_needs_two_change_points(self, three_segment_min_rtt: MinimumRTTDataset) -> None:
        analyzer = JitterAnalyzer(JitterAnalysisConfig())
        assert analyzer.analyze_jitter_dispersion(three_segment_min_rtt, []) == []

    def test_flags_the_noisy_segment(self, three_segment_min_rtt: MinimumRTTDataset) -> None:
        analyzer = JitterAnalyzer(JitterAnalysisConfig(method="jitter_dispersion", threshold=0.25))
        cps = _boundaries(three_segment_min_rtt, 0, QUIET_END, LOUD_END, LAST)
        results = analyzer.analyze_jitter_dispersion(three_segment_min_rtt, cps)

        assert [r.method for r in results] == ["jitter_dispersion"] * 2
        assert results[0].has_significant_jitter is True  # quiet -> loud
        assert results[1].has_significant_jitter is False  # loud -> quiet
        assert results[0].jitter_metric > 0 > results[1].jitter_metric
        assert results[0].p_value is None

    def test_filters_have_the_configured_orders(self) -> None:
        analyzer = JitterAnalyzer(JitterAnalysisConfig(moving_iqr_order=4, moving_average_order=6))
        signal = np.arange(20, dtype=float)
        # The IQR window is centred: `order` points on each side.
        iqr = analyzer._moving_iqr_filter(signal, 4)
        assert len(iqr) == len(signal) - 2 * 4
        assert_allclose(iqr, 4.0)  # constant spread on a ramp
        assert analyzer._moving_iqr_filter(signal[:8], 4).size == 0  # too short
        smoothed = analyzer._moving_average_filter(signal, 6)
        assert len(smoothed) == len(signal) - 6 + 1
        assert_allclose(np.diff(smoothed), 1.0)  # a ramp stays a ramp


class TestJitterAnalyzerKS:
    def test_ks_test_finds_the_distribution_change(self, three_segment_raw: RTTDataset) -> None:
        analyzer = JitterAnalyzer(JitterAnalysisConfig(method="ks_test", significance_level=0.05))
        cps = _boundaries(three_segment_raw, 0, 1800, 3600, 5399)
        results = analyzer.analyze_ks_test(three_segment_raw, cps)

        assert len(results) == 2
        assert all(r.method == "ks_test" for r in results)
        assert all(r.p_value is not None and 0 <= r.p_value <= 1 for r in results)
        # Both transitions change the jitter distribution.
        assert all(r.has_significant_jitter for r in results)
        assert all(0 < r.jitter_metric <= 1 for r in results)  # KS statistic

    def test_identical_periods_are_not_significant(self, rng: np.random.Generator) -> None:
        from .conftest import make_measurements

        values = 20.0 + np.abs(rng.normal(0, 0.5, 3000))
        dataset = RTTDataset(measurements=make_measurements(values, step_seconds=1.0))
        analyzer = JitterAnalyzer(JitterAnalysisConfig(method="ks_test"))
        results = analyzer.analyze_ks_test(dataset, _boundaries(dataset, 0, 1000, 2000, 2999))
        assert [r.has_significant_jitter for r in results] == [False, False]


class TestCongestionInferenceAnalyzer:
    """The v1 state machine: jump AND jitter -> congested; no jump -> clear; jump only -> hold."""

    @pytest.fixture
    def analyzer(self) -> CongestionInferenceAnalyzer:
        return CongestionInferenceAnalyzer()

    def test_empty_inputs(self, analyzer: CongestionInferenceAnalyzer) -> None:
        assert analyzer.infer([], []) == []
        assert analyzer.infer([jump(0, 10, True)], []) == []

    def test_both_signals_mark_congestion(self, analyzer: CongestionInferenceAnalyzer) -> None:
        jumps = [jump(0, 10, True), jump(10, 20, False)]
        jitters = [jitter(0, 10, True), jitter(10, 20, False)]
        result = analyzer.infer(jumps, jitters)
        assert [r.is_congested for r in result] == [True, False]
        assert result[0].confidence >= 0.8
        assert result[1].confidence == 0.0

    def test_jump_without_jitter_keeps_previous_state(
        self, analyzer: CongestionInferenceAnalyzer
    ) -> None:
        jumps = [jump(0, 10, True), jump(10, 20, True), jump(20, 30, False), jump(30, 40, True)]
        jitters = [
            jitter(0, 10, True),
            jitter(10, 20, False),
            jitter(20, 30, True),
            jitter(30, 40, False),
        ]
        states = [r.is_congested for r in analyzer.infer(jumps, jitters)]
        # congested, held (jump only), cleared (no jump), stays clear (jump only, was clear)
        assert states == [True, True, False, False]

    def test_large_jump_raises_confidence(self, analyzer: CongestionInferenceAnalyzer) -> None:
        small = analyzer.infer([jump(0, 10, True, magnitude=0.6)], [jitter(0, 10, True)])[0]
        large = analyzer.infer([jump(0, 10, True, magnitude=5.0)], [jitter(0, 10, True)])[0]
        assert large.confidence > small.confidence

    def test_unmatched_periods_are_skipped(self, analyzer: CongestionInferenceAnalyzer) -> None:
        result = analyzer.infer([jump(0, 10, True)], [jitter(100, 110, True)])
        assert result == []

    def test_statistics(self, analyzer: CongestionInferenceAnalyzer) -> None:
        result = analyzer.infer(
            [jump(0, 10, True), jump(10, 20, False)], [jitter(0, 10, True), jitter(10, 20, False)]
        )
        stats = analyzer.get_inference_statistics(result)
        assert stats["total_periods"] == 2
        assert stats["congested_periods"] == 1
