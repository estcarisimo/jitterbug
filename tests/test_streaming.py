"""Tests for the online (streaming) prototype."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("bayesian_changepoint_detection") is None,
    reason="bcp extra not installed",
)

INTERVAL_S = 15 * 60


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
    from jitterbug.streaming import OnlineJitterbug, StreamingConfig

    epochs, rtts = _synthetic(np.random.default_rng(0))
    online = OnlineJitterbug(StreamingConfig())
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
    from jitterbug.streaming import OnlineJitterbug, StreamingConfig

    epochs, rtts = _synthetic(np.random.default_rng(1))
    online = OnlineJitterbug(StreamingConfig(decision=decision, lag=2, threshold=0.3))
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
    from jitterbug.streaming import OnlineJitterbug, StreamingConfig

    rng = np.random.default_rng(2)
    online = OnlineJitterbug(StreamingConfig(lag=1, **config_kwargs))
    t = 1_700_000_000.0
    for i in range(4000):
        online.push(t + 30.0 * i, 10.0 + 20.0 * ((i // 1500) % 2) + rng.exponential(0.5))
    online.flush()
    cps = [e for e in online.events if e.kind == "change_point"]
    assert cps and all(e.start_epoch <= e.emitted_at for e in cps)


def test_lag_larger_than_run_length_is_rejected() -> None:
    from pydantic import ValidationError

    from jitterbug.streaming import StreamingConfig

    with pytest.raises(ValidationError, match="max_run_length"):
        StreamingConfig(decision="lag", lag=10, max_run_length=5)
    with pytest.raises(ValidationError):
        StreamingConfig(max_run_length=0)
    StreamingConfig(lag=10, max_run_length=None)  # unbounded is fine


def test_buffers_are_pruned_across_many_periods() -> None:
    from jitterbug.streaming import OnlineJitterbug, StreamingConfig

    rng = np.random.default_rng(3)
    online = OnlineJitterbug(StreamingConfig())
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
