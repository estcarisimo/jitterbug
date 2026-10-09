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
    assert online.push(1_700_000_000.0 + INTERVAL_S + 1, float("nan")) == []
    assert online.push(1_700_000_000.0 + INTERVAL_S + 2, -1.0) == []
    assert len(online._raw_rtts) == 2
