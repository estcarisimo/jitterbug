"""Tests for the online (streaming) prototype."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest
from numpy.testing import assert_allclose

from jitterbug.models import JitterbugConfig, StreamingConfig

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("bayesian_changepoint_detection") is None,
    reason="bcp extra not installed",
)

INTERVAL_S = 15 * 60


def _config(**streaming: object) -> JitterbugConfig:
    """Online config with the KS jitter method (the config default is dispersion)."""
    config = JitterbugConfig(streaming=StreamingConfig(**streaming))  # type: ignore[arg-type]
    config.jitter_analysis.method = "ks_test"
    return config


def _synthetic_dispersive(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Like ``_synthetic`` but the congested minimum RTT also wanders, bin to bin.

    Jitter dispersion works on the minimum RTT per bin; an exponential tail alone barely
    moves that minimum, so the congested segment gets Gaussian noise of 3 ms as well.
    """
    hours = [48, 12, 48]
    bases = [10.0, 30.0, 10.0]
    spreads = [0.5, 8.0, 0.5]
    sigmas = [0.05, 3.0, 0.05]
    epochs, rtts = [], []
    t = 1_700_000_000.0
    for h, base, spread, sigma in zip(hours, bases, spreads, sigmas, strict=True):
        n = int(h * 3600 / 30)
        e = t + np.arange(n) * 30.0
        r = base + rng.exponential(spread, n) + np.abs(rng.normal(0.0, sigma, n)) + 0.1
        epochs.append(e)
        rtts.append(r)
        t = e[-1] + 30.0
    return np.concatenate(epochs), np.concatenate(rtts)


def _synthetic(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Baseline, then 12 h of congestion (higher minimum, wider jitter), then baseline."""
    hours = [48, 12, 48]
    bases = [10.0, 30.0, 10.0]
    spreads = [0.5, 8.0, 0.5]
    epochs, rtts = [], []
    t = 1_700_000_000.0
    for h, base, spread in zip(hours, bases, spreads, strict=True):
        n = int(h * 3600 / 30)  # one sample every 30 s
        e = t + np.arange(n) * 30.0
        r = base + rng.exponential(spread, n) + 0.1
        epochs.append(e)
        rtts.append(r)
        t = e[-1] + 30.0
    return np.concatenate(epochs), np.concatenate(rtts)


@pytest.fixture(scope="module")
def events() -> list:
    from jitterbug.streaming import OnlineJitterbug

    epochs, rtts = _synthetic(np.random.default_rng(0))
    online = OnlineJitterbug(_config())
    for e, r in zip(epochs, rtts, strict=True):
        online.push(float(e), float(r))
    online.flush()
    return online.events


def test_change_points_bracket_the_congested_segment(events: list) -> None:
    cps = [e.start_epoch for e in events if e.kind == "change_point"]
    onset = 1_700_000_000.0 + 48 * 3600
    offset = onset + 12 * 3600
    assert any(abs(cp - onset) <= 2 * INTERVAL_S for cp in cps)
    assert any(abs(cp - offset) <= 4 * INTERVAL_S for cp in cps)


def test_onset_is_reported_within_two_bins(events: list) -> None:
    onset = 1_700_000_000.0 + 48 * 3600
    cp = min(
        (e for e in events if e.kind == "change_point"), key=lambda e: abs(e.start_epoch - onset)
    )
    assert cp.delay <= 2 * INTERVAL_S


def test_final_verdicts_match_the_segments(events: list) -> None:
    finals = [e for e in events if e.kind == "verdict" and e.stage == "final"]
    onset = 1_700_000_000.0 + 48 * 3600
    congested = [e for e in finals if abs(e.start_epoch - onset) <= 2 * INTERVAL_S]
    assert congested and congested[0].is_congested
    assert congested[0].has_jump and congested[0].has_jitter
    # The return to baseline is still open when the stream ends: only provisional.
    after = [e for e in events if e.stage == "provisional" and e.start_epoch > onset + 6 * 3600]
    assert after and not any(e.is_congested for e in after)


def test_stream_start_opens_the_baseline_period(events: list) -> None:
    first = next(e for e in events if e.kind == "change_point")
    assert first.start_epoch == 1_700_000_000.0
    assert first.detector_probability is None


def test_provisional_precedes_final_and_agrees(events: list) -> None:
    provisional = {e.start_epoch: e for e in events if e.stage == "provisional"}
    for final in (e for e in events if e.stage == "final"):
        prov = provisional.get(final.start_epoch)
        assert prov is not None
        assert prov.emitted_at < final.emitted_at
        assert prov.is_congested == final.is_congested


def test_out_of_order_and_invalid_samples_are_ignored() -> None:
    from jitterbug.streaming import OnlineJitterbug

    online = OnlineJitterbug()
    assert online.push(1_700_000_000.0, 10.0) == []
    first = online.push(1_700_000_000.0 + INTERVAL_S, 10.0)  # closes the first bin
    assert [e.kind for e in first] == ["change_point"]  # the stream start
    assert online.push(1_700_000_000.0 - 10, 10.0) == []  # earlier bin: dropped
    assert online.push(1_700_000_000.0 + INTERVAL_S - 1, 10.0) == []  # same bin, earlier
    assert online.push(1_700_000_000.0 + INTERVAL_S, 12.0) == []  # equal epoch: kept
    assert online.push(1_700_000_000.0 + INTERVAL_S + 1, float("nan")) == []
    assert online.push(1_700_000_000.0 + INTERVAL_S + 2, -1.0) == []
    assert online._raw_rtts == [10.0, 10.0, 12.0]
    assert online._raw_epochs == sorted(online._raw_epochs)


def test_flush_on_empty_or_closed_stream_is_a_no_op() -> None:
    from jitterbug.streaming import OnlineJitterbug

    online = OnlineJitterbug()
    assert online.flush() == []
    online.push(1_700_000_000.0, 10.0)
    assert [e.kind for e in online.flush()] == ["change_point"]  # the only bin closes
    assert online.flush() == []


@pytest.mark.parametrize("decision", ["lag", "window"])
def test_fixed_delay_rules_report_the_onset(decision: str) -> None:
    from jitterbug.streaming import OnlineJitterbug

    epochs, rtts = _synthetic(np.random.default_rng(1))
    online = OnlineJitterbug(_config(decision=decision, lag=2, threshold=0.3))
    for e, r in zip(epochs, rtts, strict=True):
        online.push(float(e), float(r))
    onset = 1_700_000_000.0 + 48 * 3600
    cps = [e for e in online.events if e.kind == "change_point"]
    hit = min(cps, key=lambda e: abs(e.start_epoch - onset))
    assert abs(hit.start_epoch - onset) <= 2 * INTERVAL_S
    assert hit.delay <= 3 * INTERVAL_S  # two bins of lag plus the bin that closes
    assert hit.detector_probability is not None and hit.detector_probability > 0.3


@pytest.mark.parametrize("config_kwargs", [{"hazard_lambda": 1}, {"max_run_length": 3}])
def test_map_rule_survives_run_length_zero_being_most_probable(config_kwargs: dict) -> None:
    """Regression: a tiny hazard or truncation made run length 0 the MAP and crashed."""
    from jitterbug.streaming import OnlineJitterbug

    rng = np.random.default_rng(2)
    online = OnlineJitterbug(_config(**config_kwargs))
    t = 1_700_000_000.0
    for i in range(4000):
        online.push(t + 30.0 * i, 10.0 + 20.0 * ((i // 1500) % 2) + rng.exponential(0.5))
    online.flush()
    cps = [e for e in online.events if e.kind == "change_point"]
    assert cps and all(e.start_epoch <= e.emitted_at for e in cps)


def test_lag_larger_than_run_length_is_rejected() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="max_run_length"):
        StreamingConfig(decision="lag", lag=10, max_run_length=5)
    with pytest.raises(ValidationError, match="max_run_length"):
        StreamingConfig(decision="window", lag=10, max_run_length=5)
    with pytest.raises(ValidationError):
        StreamingConfig(max_run_length=0)
    StreamingConfig(lag=10, max_run_length=None)  # unbounded is fine
    StreamingConfig(decision="map", max_run_length=3)  # lag is unused by the MAP rule


def test_buffers_are_pruned_across_many_periods() -> None:
    from jitterbug.streaming import OnlineJitterbug

    rng = np.random.default_rng(3)
    online = OnlineJitterbug()
    t = 1_700_000_000.0
    n = 12 * 1200  # 12 segments of 10 h at one sample per 30 s
    for i in range(n):
        segment = i // 1200
        rtt = 10.0 + 20.0 * (segment % 2) + rng.exponential(0.5 + 6.0 * (segment % 2))
        online.push(t + 30.0 * i, rtt)
    online.flush()
    finals = [e for e in online.events if e.stage == "final"]
    assert len(finals) >= 9
    assert len(online._raw_epochs) < 3 * 1200  # at most the previous and open periods
    assert online._pruned_bins > 0


def test_streaming_settings_come_from_the_shared_config_sections() -> None:
    """Bin width, latency threshold, significance level and device are not duplicated."""
    from jitterbug.streaming import OnlineJitterbug

    config = JitterbugConfig()
    config.data_processing.minimum_interval_minutes = 5
    config.latency_jump.threshold = 2.0
    online = OnlineJitterbug(config)
    assert online._interval_s == 300
    assert online.config.latency_jump.threshold == 2.0
    assert online.streaming is config.streaming


def test_streaming_section_round_trips_through_a_config_file(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("streaming:\n  decision: window\n  lag: 2\n  min_period_samples: 40\n")
    loaded = JitterbugConfig.from_file(path)
    assert loaded.streaming.decision == "window"
    assert loaded.streaming.lag == 2
    assert loaded.streaming.min_period_samples == 40
    assert loaded.streaming.hazard_lambda == 50.0  # default kept
    assert "streaming" in JitterbugConfig().model_dump()


# ---------------------------------------------------------------- sliding-window back end


def _window_config(**streaming: object) -> JitterbugConfig:
    config = _config(backend="window", rerun_every_bins=4, **streaming)
    config.jitter_analysis.method = "ks_test"  # the paper's method, also the causal one
    return config


@pytest.fixture(scope="module")
def window_events() -> list:
    from jitterbug.streaming import SlidingWindowJitterbug

    epochs, rtts = _synthetic(np.random.default_rng(0))
    online = SlidingWindowJitterbug(_window_config())
    for e, r in zip(epochs, rtts, strict=True):
        online.push(float(e), float(r))
    online.flush()
    return online.events


def test_factory_picks_the_backend() -> None:
    from jitterbug.streaming import OnlineJitterbug, SlidingWindowJitterbug, create_online_analyzer

    assert isinstance(create_online_analyzer(), OnlineJitterbug)
    assert isinstance(create_online_analyzer(_config(backend="window")), SlidingWindowJitterbug)


def test_window_backend_brackets_and_judges_the_congested_segment(window_events: list) -> None:
    onset = 1_700_000_000.0 + 48 * 3600
    offset = onset + 12 * 3600
    cps = [e.start_epoch for e in window_events if e.kind == "change_point"]
    assert cps[0] == 1_700_000_000.0  # the stream start opens the baseline period
    assert any(abs(cp - onset) <= 3 * INTERVAL_S for cp in cps)
    assert any(abs(cp - offset) <= 3 * INTERVAL_S for cp in cps)
    finals = [e for e in window_events if e.stage == "final"]
    congested = [e for e in finals if abs(e.start_epoch - onset) <= 3 * INTERVAL_S]
    assert congested and congested[0].is_congested
    assert congested[0].has_jump and congested[0].has_jitter
    assert abs(congested[0].end_epoch - offset) <= 3 * INTERVAL_S
    after = [
        e for e in window_events if e.stage == "provisional" and e.start_epoch > onset + 6 * 3600
    ]
    assert after and not any(e.is_congested for e in after)


def test_window_backend_provisional_precedes_final_and_agrees(window_events: list) -> None:
    provisional = {e.start_epoch: e for e in window_events if e.stage == "provisional"}
    finals = [e for e in window_events if e.stage == "final"]
    assert finals
    for final in finals:
        prov = provisional.get(final.start_epoch)
        assert prov is not None
        assert prov.emitted_at < final.emitted_at
        assert prov.is_congested == final.is_congested


def test_window_backend_waits_for_stable_runs(window_events: list) -> None:
    """A change point is reported only after stable_runs reruns, so at least that late."""
    onset = 1_700_000_000.0 + 48 * 3600
    cp = min(
        (e for e in window_events if e.kind == "change_point" and e.start_epoch > 0),
        key=lambda e: abs(e.start_epoch - onset),
    )
    assert cp.delay >= 2 * 4 * INTERVAL_S - INTERVAL_S  # stable_runs=2, rerun every 4 bins


def _assert_consistent(events: list) -> None:
    """Change points in time order; finals a contiguous chain over emitted change points."""
    cps = [e.start_epoch for e in events if e.kind == "change_point"]
    assert cps == sorted(cps) and len(cps) == len(set(cps))
    finals = [e for e in events if e.stage == "final"]
    for earlier, later in zip(finals, finals[1:], strict=False):
        assert earlier.end_epoch == later.start_epoch
    assert all(e.start_epoch in cps and e.end_epoch in cps for e in finals)


def test_window_backend_events_are_consistent(window_events: list) -> None:
    _assert_consistent(window_events)


def test_window_backend_with_jitter_dispersion(window_events: list) -> None:
    """Final verdicts and provisional ones both follow the configured jitter method."""
    from jitterbug.streaming import SlidingWindowJitterbug

    epochs, rtts = _synthetic_dispersive(np.random.default_rng(0))
    config = _window_config()
    config.jitter_analysis.method = "jitter_dispersion"
    online = SlidingWindowJitterbug(config)
    for e, r in zip(epochs, rtts, strict=True):
        online.push(float(e), float(r))
    online.flush()
    _assert_consistent(online.events)
    finals = [e for e in online.events if e.stage == "final"]
    assert finals and finals[0].is_congested
    assert finals[0].jitter_method == "jitter_dispersion"
    assert finals[0].ks_statistic is None and finals[0].p_value is None
    assert finals[0].jitter_metric is not None
    assert finals[0].n_prev > 0 and finals[0].n_curr > 0
    # Counts are dispersion values (one per bin), not raw samples (one per 30 s).
    assert finals[0].n_curr <= 12 * 4 + 2
    provisional = [e for e in online.events if e.stage == "provisional"]
    assert provisional and all(e.jitter_method == "jitter_dispersion" for e in provisional)
    ks_finals = [e for e in window_events if e.stage == "final"]
    assert ks_finals[0].jitter_method == "ks_test"
    assert ks_finals[0].ks_statistic == ks_finals[0].jitter_metric
    assert ks_finals[0].p_value is not None and ks_finals[0].n_curr > 0


def test_window_backend_prunes_samples_outside_the_window() -> None:
    from jitterbug.streaming import SlidingWindowJitterbug

    online = SlidingWindowJitterbug(_window_config(window_hours=6))
    t = 1_700_000_000.0
    for i in range(24 * 120):  # one day at 30 s
        online.push(t + 30.0 * i, 10.0 + (i % 7) * 0.1)
    assert len(online._raw_epochs) <= 6 * 120 + 120  # the window plus the open bin


def test_window_backend_uses_the_window_start_when_the_detector_has_no_first_change_point():
    """ruptures reports no change point at index 0; the window start stands in for it."""
    from jitterbug.streaming import SlidingWindowJitterbug

    epochs, rtts = _synthetic(np.random.default_rng(0))
    config = _window_config()
    config.change_point_detection.algorithm = "ruptures"
    online = SlidingWindowJitterbug(config)
    for e, r in zip(epochs, rtts, strict=True):
        online.push(float(e), float(r))
    online.flush()
    finals = [e for e in online.events if e.stage == "final"]
    assert finals and finals[0].is_congested


def test_window_settings_are_validated() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        StreamingConfig(backend="window", window_hours=0)
    with pytest.raises(ValidationError):
        StreamingConfig(rerun_every_bins=0)
    with pytest.raises(ValidationError):
        StreamingConfig(stable_runs=0)
    with pytest.raises(ValidationError):
        StreamingConfig(backend="sliding")


# ---------------------------------------------------------------- causal jitter dispersion


def test_causal_dispersion_is_the_offline_series_delayed() -> None:
    from jitterbug.analysis.jitter_analyzer import JitterAnalyzer
    from jitterbug.models import JitterAnalysisConfig

    analyzer = JitterAnalyzer(JitterAnalysisConfig(method="jitter_dispersion"))
    rng = np.random.default_rng(0)
    epochs = np.arange(200) * 900.0
    values = rng.normal(10, 1, 200)
    centered_epochs, centered = analyzer._compute_jitter_dispersion(epochs, values)
    causal_epochs, causal = analyzer.compute_causal_jitter_dispersion(epochs, values)
    assert len(causal) == len(centered)
    np.testing.assert_allclose(causal, centered)
    delay = analyzer.causal_dispersion_delay
    assert delay == 6
    np.testing.assert_allclose(causal_epochs - centered_epochs, delay * 900.0)
    assert len(causal) == len(epochs) - 1 - analyzer.causal_dispersion_lag


def test_causal_dispersion_never_depends_on_later_samples() -> None:
    from jitterbug.analysis.jitter_analyzer import JitterAnalyzer
    from jitterbug.models import JitterAnalysisConfig

    analyzer = JitterAnalyzer(JitterAnalysisConfig(method="jitter_dispersion"))
    rng = np.random.default_rng(1)
    epochs = np.arange(120) * 900.0
    values = rng.normal(10, 1, 120)
    _, before = analyzer.compute_causal_jitter_dispersion(epochs, values)
    changed = values.copy()
    changed[90:] += 50.0  # a change after sample 89
    _, after = analyzer.compute_causal_jitter_dispersion(epochs, changed)
    # The value at jitter index t depends on samples up to t + 1 only.
    first_affected = 89 - analyzer.causal_dispersion_lag
    np.testing.assert_allclose(before[:first_affected], after[:first_affected])
    assert not np.allclose(before[first_affected + 1 :], after[first_affected + 1 :])
    short_epochs, short = analyzer.compute_causal_jitter_dispersion(epochs[:5], values[:5])
    assert len(short) == 0 and len(short_epochs) == 0


def test_two_period_verdict_with_dispersion_uses_the_mean_rise() -> None:
    from jitterbug.streaming import two_period_verdict

    config = JitterbugConfig()
    config.jitter_analysis.method = "jitter_dispersion"
    prev = np.full(20, 0.5)
    verdict = two_period_verdict(
        np.full(10, 10.0), np.full(10, 30.0), prev, np.full(10, 1.0), False, config
    )
    assert verdict.jitter_method == "jitter_dispersion"
    assert verdict.has_jump and verdict.has_jitter and verdict.is_congested
    np.testing.assert_allclose(verdict.jitter_metric, 0.5)
    assert verdict.ks_statistic is None and verdict.p_value is None
    assert verdict.n_prev == 20 and verdict.n_curr == 10
    flat = two_period_verdict(
        np.full(10, 10.0), np.full(10, 30.0), prev, np.full(10, 0.6), False, config
    )
    assert flat.has_jitter is False and flat.is_congested is False
    too_few = two_period_verdict(np.full(10, 10.0), np.full(10, 30.0), prev, prev[:1], True, config)
    assert too_few.has_jitter is None and too_few.is_congested is True  # state carried over


@pytest.fixture(scope="module")
def dispersion_events() -> list:
    from jitterbug.streaming import OnlineJitterbug

    epochs, rtts = _synthetic_dispersive(np.random.default_rng(0))
    config = JitterbugConfig()  # the config default method is jitter_dispersion
    assert config.jitter_analysis.method == "jitter_dispersion"
    online = OnlineJitterbug(config)
    for e, r in zip(epochs, rtts, strict=True):
        online.push(float(e), float(r))
    online.flush()
    return online.events


def test_incremental_backend_with_dispersion_judges_the_segment(dispersion_events: list) -> None:
    onset = 1_700_000_000.0 + 48 * 3600
    finals = [e for e in dispersion_events if e.stage == "final"]
    congested = [e for e in finals if abs(e.start_epoch - onset) <= 2 * INTERVAL_S]
    assert congested and congested[0].is_congested
    assert congested[0].jitter_method == "jitter_dispersion"
    assert congested[0].has_jump and congested[0].has_jitter
    assert congested[0].ks_statistic is None and congested[0].jitter_metric is not None
    later = [
        e
        for e in dispersion_events
        if e.stage == "provisional" and e.start_epoch > onset + 6 * 3600
    ]
    assert later and not any(e.is_congested for e in later)


def test_dispersion_provisional_waits_for_values_that_reflect_the_new_period(
    dispersion_events: list,
) -> None:
    from jitterbug.analysis.jitter_analyzer import JitterAnalyzer
    from jitterbug.models import JitterAnalysisConfig

    analyzer = JitterAnalyzer(JitterAnalysisConfig(method="jitter_dispersion"))
    needed = analyzer.causal_dispersion_delay + analyzer.config.moving_average_order
    assert needed == 12
    onset = 1_700_000_000.0 + 48 * 3600
    provisional = {e.start_epoch: e for e in dispersion_events if e.stage == "provisional"}
    cps = [e for e in dispersion_events if e.kind == "change_point"]
    cp = min(cps, key=lambda e: abs(e.start_epoch - onset))
    prov = provisional[cp.start_epoch]
    assert prov.emitted_at - cp.start_epoch >= (needed - 1) * INTERVAL_S
    assert prov.n_curr >= needed
    finals = {e.start_epoch: e for e in dispersion_events if e.stage == "final"}
    assert prov.is_congested == finals[cp.start_epoch].is_congested


def test_window_backend_provisional_follows_the_dispersion_method() -> None:
    from jitterbug.streaming import SlidingWindowJitterbug

    epochs, rtts = _synthetic_dispersive(np.random.default_rng(0))
    config = _window_config()
    config.jitter_analysis.method = "jitter_dispersion"
    online = SlidingWindowJitterbug(config)
    for e, r in zip(epochs, rtts, strict=True):
        online.push(float(e), float(r))
    online.flush()
    provisional = [e for e in online.events if e.stage == "provisional"]
    assert provisional and all(e.jitter_method == "jitter_dispersion" for e in provisional)
    assert all(e.ks_statistic is None and e.jitter_metric is not None for e in provisional)
    finals = {e.start_epoch: e for e in online.events if e.stage == "final"}
    for prov in provisional:
        if prov.start_epoch in finals:
            assert prov.is_congested == finals[prov.start_epoch].is_congested


# ---------------------------------------------------------------------------- revisions


class ScriptedDetector:
    """Stands in for the online detector: the MAP segment start follows a script."""

    def __init__(self, script: dict[int, int]) -> None:
        self.script = script  # bins fed -> start of the current segment (bin index)
        self.t = 0
        self.start = 0

    def update(self, value: float) -> np.ndarray:
        self.t += 1
        self.start = self.script.get(self.t, self.start)
        posterior = np.zeros(self.t + 1)
        posterior[self.t - self.start] = 1.0  # run length t - start is the MAP
        return posterior


def _bins_series(congested_bins: range, n_bins: int = 72) -> tuple[np.ndarray, np.ndarray]:
    """Samples every 30 s; the bins in ``congested_bins`` are congested."""
    rng = np.random.default_rng(1)
    epochs, rtts = [], []
    t0 = 1_700_000_000.0 - 1_700_000_000.0 % INTERVAL_S  # bins aligned with the epochs
    for b in range(n_bins):
        e = t0 + b * INTERVAL_S + np.arange(30) * 30.0
        if b in congested_bins:
            r = 30.0 + rng.exponential(8.0, 30)
        else:
            r = 10.0 + rng.exponential(0.5, 30)
        epochs.append(e)
        rtts.append(r)
    return np.concatenate(epochs), np.concatenate(rtts) + 0.1


def _scripted_run(
    script: dict[int, int], congested_bins: range, track: bool = False, **streaming: object
) -> tuple[list, list, float]:
    from jitterbug.streaming import OnlineJitterbug, StateChangeTracker

    config = _config(min_time_elapsed=8 * INTERVAL_S, min_period_samples=60, **streaming)
    online = OnlineJitterbug(config)
    online._detector = ScriptedDetector(script)
    tracker = StateChangeTracker(online) if track else None
    epochs, rtts = _bins_series(congested_bins)
    for e, r in zip(epochs, rtts, strict=True):
        (tracker or online).push(float(e), float(r))
    (tracker or online).flush()
    return online.events, tracker.state_changes if tracker else [], float(epochs[0])


# The MAP start moves to bin 41 when bin 41 closes, is refined to bin 44 when bin 46
# closes (3 bins later, within the 8-bin separation), and moves to bin 60 when bin 61 closes.
SCRIPT = {42: 41, 47: 44, 62: 60}


def test_revision_moves_the_open_period_and_changes_its_provisional_verdict() -> None:
    events, _, t0 = _scripted_run(SCRIPT, range(44, 60), revise_change_points=True)
    epoch = {b: t0 + b * INTERVAL_S for b in (41, 44, 60)}
    summary = [(e.kind, e.stage, e.start_epoch, e.is_congested) for e in events[1:]]
    assert summary == [
        ("change_point", None, epoch[41], None),
        ("verdict", "provisional", epoch[41], False),  # bins 41-42: no latency jump
        ("revision", None, epoch[44], None),
        ("verdict", "provisional", epoch[44], True),  # judged again on the revised period
        ("change_point", None, epoch[60], None),
        ("verdict", "final", epoch[44], True),
        ("verdict", "provisional", epoch[60], False),
    ]
    revision = events[3]
    assert revision.revised_from == epoch[41]
    assert revision.emitted_at == epoch[44] + 3 * INTERVAL_S  # first sample of bin 47
    assert revision.to_dict()["revised_from"] == epoch[41]
    assert events[-2].end_epoch == epoch[60]
    # The previous period's final verdict was emitted before the revision and keeps its end.
    assert all(e.revised_from is None for e in events if e.kind != "revision")


def test_without_revisions_the_refinement_is_dropped() -> None:
    events, _, t0 = _scripted_run(SCRIPT, range(44, 60))
    assert not [e for e in events if e.kind == "revision"]
    finals = [e for e in events if e.stage == "final"]
    assert [f.start_epoch for f in finals] == [t0 + 41 * INTERVAL_S]
    provisional = [e for e in events if e.stage == "provisional"]
    assert [(p.start_epoch, p.is_congested) for p in provisional] == [
        (t0 + 41 * INTERVAL_S, False),
        (t0 + 60 * INTERVAL_S, False),
    ]


def test_refinements_beyond_the_separation_or_of_the_stream_start_are_not_revisions() -> None:
    # Bin 3 is within the separation of the stream start, which is never revised; bin 50
    # is 9 bins after bin 41, a change point of its own.
    events, _, t0 = _scripted_run({4: 3, 42: 41, 51: 50}, range(41, 50), revise_change_points=True)
    assert not [e for e in events if e.kind == "revision"]
    cps = [e.start_epoch for e in events if e.kind == "change_point"]
    assert cps == [t0, t0 + 41 * INTERVAL_S, t0 + 50 * INTERVAL_S]


def test_revisions_only_apply_to_the_map_rule() -> None:
    from jitterbug.streaming import OnlineJitterbug

    online = OnlineJitterbug(_config(decision="lag", revise_change_points=True))
    online._cp_bin_index = [0, 10]
    assert not online._revisable(12)
    online.streaming.decision = "map"
    assert online._revisable(12)
    assert not online._revisable(10)  # a refinement only moves a boundary later


def test_revision_moves_a_state_change_and_keeps_its_id() -> None:
    # Congestion starts at bin 41: the early rule suspects it on the first bin, the MAP
    # rule then moves the boundary 3 bins later (a refinement that is wrong here).
    events, changes, t0 = _scripted_run(
        SCRIPT, range(41, 60), track=True, revise_change_points=True
    )
    epoch41, epoch44 = t0 + 41 * INTERVAL_S, t0 + 44 * INTERVAL_S
    onset = [c for c in changes if c.to_state == "congested"]
    assert {c.id for c in onset} == {f"{epoch41:.0f}:baseline->congested"}
    statuses = [(c.status, c.change_epoch, c.revised_from) for c in onset]
    assert statuses[0] == ("suspected", epoch41, None)
    assert ("confirmed", epoch41, None) in statuses  # the provisional verdict, before
    revised = [c for c in onset if c.revised_from is not None]
    assert [(c.status, c.change_epoch, c.revised_from) for c in revised] == [
        ("confirmed", epoch44, epoch41)
    ]
    assert onset[-1].change_epoch == epoch44
    assert [e.kind for e in events].count("revision") == 1


def test_score_applies_revisions(tmp_path) -> None:
    from jitterbug.streaming import score

    events, _, t0 = _scripted_run(SCRIPT, range(44, 60), revise_change_points=True)
    reference = tmp_path / "ref.csv"
    e44, e60 = t0 + 44 * INTERVAL_S, t0 + 60 * INTERVAL_S
    reference.write_text(f"starts,ends,congestion\n{t0},{e44},0\n{e44},{e60},1\n")
    s = score(events, reference)
    assert s["revisions"] == s["revised_change_points"] == 1
    assert_allclose(s["revision_shift_min_median"], 45.0)
    assert_allclose(s["revision_lag_min_median"], 5 * 15.0)  # reported at bin 42, revised at 47
    assert s["provisional_superseded"] == 1
    assert s["provisional_superseded_changed"] == 1
    assert s["provisional_pairs"] == 1
    assert s["provisional_flips"] == 0
    assert s["boundaries_within_5min"] == 3  # the revised change point is at the onset
    assert s["recovered"] == 1
    assert s["spurious"] == 0
