"""Tests for the state-change layer of the online mode (``jitterbug.streaming.state_changes``)."""

from __future__ import annotations

import importlib.util
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose
from pydantic import ValidationError

from jitterbug.models import JitterbugConfig, StreamingConfig
from jitterbug.streaming import (
    CommandHook,
    OpenPeriodBins,
    StateChangeEvent,
    StateChangeTracker,
    StreamingEvent,
    score_state_changes,
)

T0 = 1_700_000_000.0
BIN = 900.0
HAS_BCP = importlib.util.find_spec("bayesian_changepoint_detection") is not None


class FakeBackend:
    """A scripted back end: the test sets the bins and hands events to ``observe``."""

    def __init__(self, config: JitterbugConfig | None = None) -> None:
        self.config = config or JitterbugConfig()
        self.bins: OpenPeriodBins | None = None
        self.events: list[StreamingEvent] = []
        self.change_points: list[float] = []

    def push(self, epoch: float, rtt: float) -> list[StreamingEvent]:
        return []

    def flush(self) -> list[StreamingEvent]:
        return []

    def open_period_bins(self) -> OpenPeriodBins | None:
        return self.bins


def _cp(start: float, at: float) -> StreamingEvent:
    return StreamingEvent(kind="change_point", emitted_at=at, start_epoch=start)


def _verdict(
    start: float, stage: str, congested: bool, at: float, end: float | None = None
) -> StreamingEvent:
    return StreamingEvent(
        kind="verdict",
        emitted_at=at,
        start_epoch=start,
        end_epoch=end,
        stage=stage,  # type: ignore[arg-type]
        is_congested=congested,
        jump_magnitude=20.0 if congested else 0.1,
        has_jitter=congested,
        jitter_method="ks_test",
        jitter_metric=0.4,
        p_value=0.001,
        n_prev=500,
        n_curr=100,
    )


def _tracker(bins: int = 1) -> tuple[StateChangeTracker, FakeBackend]:
    config = JitterbugConfig(streaming=StreamingConfig(state_change_bins=bins))
    config.jitter_analysis.method = "ks_test"
    backend = FakeBackend(config)
    tracker = StateChangeTracker(backend)  # type: ignore[arg-type]
    tracker.observe([_cp(T0, T0 + BIN)], T0 + BIN)  # the stream start opens the baseline
    return tracker, backend


def _open(
    tracker: StateChangeTracker,
    backend: FakeBackend,
    start: float,
    current: list[float],
    previous: list[float] | None = None,
) -> list[StateChangeEvent]:
    """A change point at ``start`` whose new period holds ``current`` closed bins."""
    at = start + len(current) * BIN
    backend.bins = OpenPeriodBins(
        start_epoch=start,
        previous=np.asarray(previous if previous is not None else [10.0] * 8),
        current=np.asarray(current),
    )
    return tracker.observe([_cp(start, at)], at)


S1 = T0 + 10 * BIN  # first change point after the stream start
S2 = T0 + 30 * BIN


def test_suspected_onset_then_confirmed_by_the_provisional_verdict() -> None:
    tracker, backend = _tracker()
    (suspected,) = _open(tracker, backend, S1, [30.0])
    assert suspected.status == "suspected"
    assert (suspected.from_state, suspected.to_state) == ("baseline", "congested")
    assert suspected.id == f"{S1:.0f}:baseline->congested"
    assert suspected.change_epoch == S1
    assert suspected.emitted_at == suspected.suspected_at == S1 + BIN
    assert suspected.confirmed_at is None
    assert suspected.stage == "latency_jump"
    assert_allclose(suspected.jump_magnitude, 20.0)
    assert (suspected.n_prev, suspected.n_curr) == (8, 1)
    assert tracker.state == "congested"

    (confirmed,) = tracker.observe([_verdict(S1, "provisional", True, S1 + 3 * BIN)], S1 + 3 * BIN)
    assert confirmed.id == suspected.id
    assert confirmed.status == "confirmed"
    assert confirmed.stage == "provisional"
    assert confirmed.suspected_at == S1 + BIN
    assert confirmed.confirmed_at == S1 + 3 * BIN
    assert confirmed.jitter_method == "ks_test"
    assert confirmed.jitter_metric == 0.4
    assert (confirmed.n_prev, confirmed.n_curr) == (500, 100)

    # A final verdict that agrees adds nothing.
    final = _verdict(S1, "final", True, S2, end=S2)
    assert tracker.observe([_cp(S2, S2), final], S2) == []


def test_suspected_onset_retracted_by_the_provisional_verdict() -> None:
    tracker, backend = _tracker()
    (suspected,) = _open(tracker, backend, S1, [30.0])
    (retracted,) = tracker.observe([_verdict(S1, "provisional", False, S1 + 3 * BIN)], S1 + 3 * BIN)
    assert retracted.id == suspected.id
    assert retracted.status == "retracted"
    assert retracted.stage == "provisional"
    assert retracted.suspected_at == S1 + BIN
    assert retracted.confirmed_at is None
    assert tracker.state == "baseline"


def test_final_verdict_contradicting_the_provisional_retracts_the_confirmed_change() -> None:
    tracker, backend = _tracker()
    _open(tracker, backend, S1, [10.2])  # no latency jump: nothing suspected
    assert tracker.state_changes == []
    (confirmed,) = tracker.observe([_verdict(S1, "provisional", True, S1 + 3 * BIN)], S1 + 3 * BIN)
    assert confirmed.status == "confirmed" and confirmed.suspected_at is None
    (retracted,) = tracker.observe([_cp(S2, S2), _verdict(S1, "final", False, S2, S2)], S2)
    assert retracted.id == confirmed.id
    assert retracted.status == "retracted"
    assert retracted.stage == "final"
    assert retracted.confirmed_at == S1 + 3 * BIN
    summary = score_state_changes([], tracker.state_changes)
    assert summary["corrected_by_final"] == 1


def test_final_verdict_confirms_a_change_the_provisional_retracted() -> None:
    tracker, backend = _tracker()
    _open(tracker, backend, S1, [30.0])
    tracker.observe([_verdict(S1, "provisional", False, S1 + 3 * BIN)], S1 + 3 * BIN)
    (confirmed,) = tracker.observe([_cp(S2, S2), _verdict(S1, "final", True, S2, S2)], S2)
    assert confirmed.status == "confirmed"
    assert confirmed.stage == "final"
    assert confirmed.suspected_at == S1 + BIN
    assert [c.status for c in tracker.state_changes] == ["suspected", "retracted", "confirmed"]
    assert score_state_changes([], tracker.state_changes)["corrected_by_final"] == 1


def test_late_confirmation_by_the_final_verdict_alone() -> None:
    """A period closed before it had a provisional verdict or enough bins."""
    tracker, backend = _tracker(bins=3)
    assert _open(tracker, backend, S1, [30.0, 30.0]) == []  # two of three bins
    (confirmed,) = tracker.observe([_cp(S2, S2), _verdict(S1, "final", True, S2, S2)], S2)
    assert confirmed.status == "confirmed"
    assert confirmed.stage == "final"
    assert confirmed.suspected_at is None
    assert confirmed.confirmed_at == S2


def test_suspected_return_to_baseline_then_confirmed() -> None:
    tracker, backend = _tracker()
    _open(tracker, backend, S1, [30.0])
    tracker.observe([_verdict(S1, "provisional", True, S1 + 3 * BIN)], S1 + 3 * BIN)
    tracker.observe([_verdict(S1, "final", True, S2, S2)], S2)
    (back,) = _open(tracker, backend, S2, [10.1], previous=[30.0] * 20)
    assert back.status == "suspected"
    assert (back.from_state, back.to_state) == ("congested", "baseline")
    assert_allclose(back.jump_magnitude, -19.9)
    assert tracker.state == "baseline"
    (confirmed,) = tracker.observe([_verdict(S2, "provisional", False, S2 + 3 * BIN)], S2 + 3 * BIN)
    assert confirmed.status == "confirmed"


def test_rise_while_congested_is_not_a_change() -> None:
    """A further jump keeps the path congested, as the two-period rule does."""
    tracker, backend = _tracker()
    _open(tracker, backend, S1, [30.0])
    tracker.observe([_verdict(S1, "final", True, S2, S2)], S2)
    n = len(tracker.state_changes)
    assert _open(tracker, backend, S2, [50.0], previous=[30.0] * 20) == []
    assert len(tracker.state_changes) == n


def test_guard_waits_for_every_bin_above_the_threshold() -> None:
    tracker, backend = _tracker(bins=2)
    assert _open(tracker, backend, S1, [30.0]) == []  # one bin of two
    backend.bins = OpenPeriodBins(S1, np.full(8, 10.0), np.array([30.0, 31.0]))
    (suspected,) = tracker.observe([], S1 + 2 * BIN)
    assert suspected.status == "suspected"
    assert suspected.n_curr == 2
    assert_allclose(suspected.jump_magnitude, 20.5)


def test_mixed_bins_raise_nothing() -> None:
    tracker, backend = _tracker(bins=2)
    assert _open(tracker, backend, S1, [30.0, 10.0]) == []
    backend.bins = OpenPeriodBins(S1, np.full(8, 10.0), np.array([30.0, 10.0, 30.0]))
    assert tracker.observe([], S1 + 3 * BIN) == []  # evaluated once, on the first bins


def test_threshold_is_the_latency_jump_threshold() -> None:
    tracker, backend = _tracker()
    assert _open(tracker, backend, S1, [10.5]) == []  # exactly the threshold: no jump
    tracker2, backend2 = _tracker()
    assert len(_open(tracker2, backend2, S1, [10.51])) == 1


def test_changes_are_re_derived_when_an_earlier_verdict_changes() -> None:
    """The window back end can judge a period after the next one was opened."""
    tracker, backend = _tracker()
    _open(tracker, backend, S1, [30.0])
    (back,) = _open(tracker, backend, S2, [10.0], previous=[30.0] * 20)
    assert (back.from_state, back.to_state) == ("congested", "baseline")
    # The first period turns out not congested: both changes go.
    out = tracker.observe([_verdict(S1, "final", False, S2 + BIN, S2)], S2 + BIN)
    assert sorted((c.change_epoch, c.status) for c in out) == [
        (S1, "retracted"),
        (S2, "retracted"),
    ]
    assert tracker.state == "baseline"


def test_ids_are_unique_per_change_and_events_serialize() -> None:
    tracker, backend = _tracker()
    _open(tracker, backend, S1, [30.0])
    tracker.observe([_verdict(S1, "final", True, S2, S2)], S2)
    _open(tracker, backend, S2, [10.0], previous=[30.0] * 20)
    ids = {c.id for c in tracker.state_changes}
    assert len(ids) == 2
    for change in tracker.state_changes:
        assert json.loads(json.dumps(change.to_dict()))["kind"] == "state_change"


def test_old_periods_are_frozen() -> None:
    tracker, backend = _tracker()
    start = S1
    for i in range(10):
        congested = i % 2 == 0
        nxt = start + 20 * BIN
        _open(tracker, backend, start, [30.0 if congested else 10.0])
        tracker.observe([_cp(nxt, nxt), _verdict(start, "final", congested, nxt, nxt)], nxt)
        start = nxt
    assert len(tracker._periods) <= 3
    assert tracker.state == "baseline"


def test_state_change_bins_is_validated() -> None:
    with pytest.raises(ValidationError):
        StreamingConfig(state_change_bins=0)


def test_score_counts_outcomes_lead_and_reference(tmp_path: Path) -> None:
    tracker, backend = _tracker()
    _open(tracker, backend, S1, [30.0])
    prov = _verdict(S1, "provisional", True, S1 + 3 * BIN)
    tracker.observe([prov], prov.emitted_at)
    reference = tmp_path / "ref.csv"
    reference.write_text(f"starts,ends,congestion\n{T0},{S1},0\n{S1 + 600},{S2},1\n")
    s = score_state_changes([prov], tracker.state_changes, reference)
    assert s["suspected"] == s["suspected_onsets"] == s["suspected_confirmed"] == 1
    assert s["suspected_retracted"] == 0
    assert_allclose(s["suspected_lead_min_median"], 30.0)
    assert_allclose(s["suspected_delay_min_median"], 15.0)
    assert s["reference_congested_suspected"] == 1
    assert s["suspected_onsets_outside_reference"] == 0


# ---------------------------------------------------------------------------- command hook


def _python(code: str) -> str:
    return f"{sys.executable} -c {json.dumps(code)}"


def _event() -> StateChangeEvent:
    tracker, backend = _tracker()
    (event,) = _open(tracker, backend, S1, [30.0])
    return event


def test_command_hook_gets_the_event_on_stdin_without_a_shell(tmp_path: Path) -> None:
    out = tmp_path / "out.json"
    # A shell would expand $HOME; the argument must arrive verbatim.
    code = f"import sys; open({str(out)!r}, 'w').write(sys.stdin.read() + sys.argv[1])"
    hook = CommandHook(_python(code) + " '$HOME'")
    hook(_event())
    hook.close()
    text = out.read_text()
    assert text.endswith("$HOME")
    assert json.loads(text[: -len("$HOME")])["status"] == "suspected"
    assert hook.failures == 0


def test_command_hook_failures_do_not_raise(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING)
    failing = CommandHook(_python("import sys; sys.exit(3)"))
    failing(_event())
    failing.close()
    missing = CommandHook("/nonexistent/jitterbug-hook")
    missing(_event())
    missing.close()
    assert failing.failures == missing.failures == 1
    assert "status 3" in caplog.text and "could not start" in caplog.text


def test_command_hook_kills_slow_commands_without_blocking() -> None:
    import time

    hook = CommandHook(_python("import time; time.sleep(30)"), timeout=0.5)
    start = time.monotonic()
    hook(_event())
    assert time.monotonic() - start < 0.5  # the stream does not wait
    hook.close()
    assert hook.failures == 1
    assert time.monotonic() - start < 5


def test_command_hook_rejects_an_empty_command() -> None:
    with pytest.raises(ValueError):
        CommandHook("  ")


# ---------------------------------------------------------------------------- real back ends


def _step_series(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """24 h baseline, 12 h congested (higher minimum, wider jitter), 24 h baseline."""
    rng = np.random.default_rng(seed)
    epochs, rtts = [], []
    t = T0
    for hours, base, spread in ((24, 10.0, 0.5), (12, 30.0, 8.0), (24, 10.0, 0.5)):
        n = int(hours * 3600 / 30)
        epochs.append(t + np.arange(n) * 30.0)
        rtts.append(base + rng.exponential(spread, n) + 0.1)
        t = epochs[-1][-1] + 30.0
    return np.concatenate(epochs), np.concatenate(rtts)


def _run(backend_name: str) -> tuple[list[StreamingEvent], list[StreamingEvent], list]:
    from jitterbug.streaming import create_online_analyzer

    streaming = StreamingConfig(backend=backend_name, rerun_every_bins=4)  # type: ignore[arg-type]
    config = JitterbugConfig(streaming=streaming)
    config.jitter_analysis.method = "ks_test"
    epochs, rtts = _step_series()
    plain = create_online_analyzer(config)
    tracker = StateChangeTracker(create_online_analyzer(config))
    for e, r in zip(epochs, rtts, strict=True):
        plain.push(float(e), float(r))
        tracker.push(float(e), float(r))
    plain.flush()
    tracker.flush()
    return plain.events, tracker.analyzer.events, tracker.state_changes


@pytest.mark.skipif(not HAS_BCP, reason="bcp extra not installed")
def test_incremental_backend_suspects_then_confirms_the_congested_segment() -> None:
    plain, tracked, changes = _run("bocpd")
    assert [e.to_dict() for e in tracked] == [e.to_dict() for e in plain]  # events untouched
    onset, back = T0 + 24 * 3600, T0 + 36 * 3600
    first: dict[str, StateChangeEvent] = {}
    for change in changes:
        first.setdefault(change.id, change)
    up = [c for c in first.values() if c.to_state == "congested"]
    down = [c for c in first.values() if c.to_state == "baseline"]
    assert len(up) == 1 and abs(up[0].change_epoch - onset) <= 2 * BIN
    assert len(down) == 1 and abs(down[0].change_epoch - back) <= 2 * BIN
    assert {c.id: c.status for c in changes} == {up[0].id: "confirmed", down[0].id: "confirmed"}
    # Raised within two bins of the onset, before the first verdict on that period.
    assert up[0].status == "suspected"
    assert up[0].delay <= 2 * BIN
    verdicts = [e for e in plain if e.kind == "verdict" and e.start_epoch == up[0].change_epoch]
    assert up[0].emitted_at < verdicts[0].emitted_at


def test_window_backend_feeds_the_tracker_without_changing_its_events() -> None:
    plain, tracked, changes = _run("window")
    assert [e.to_dict() for e in tracked] == [e.to_dict() for e in plain]
    outcome = {c.id: c for c in changes}
    assert any(c.to_state == "congested" and c.status == "confirmed" for c in outcome.values())
    assert any(c.to_state == "baseline" and c.status == "confirmed" for c in outcome.values())


def test_window_backend_open_period_bins_leave_out_the_open_bin() -> None:
    from jitterbug.streaming import SlidingWindowJitterbug

    online = SlidingWindowJitterbug(JitterbugConfig())
    assert online.open_period_bins() is None
    epochs, rtts = _step_series()
    epochs, rtts = epochs[:200], rtts[:200]
    for e, r in zip(epochs, rtts, strict=True):
        online.push(float(e), float(r))
    cut = float(epochs[120])
    online.change_points.append(cut)  # as if a change point had been emitted there
    bins = online.open_period_bins()
    assert bins is not None and bins.start_epoch == cut
    ids = (epochs // BIN).astype(int)
    closed = sorted(set(ids[ids < ids[-1]]))
    starts = [epochs[ids == b].min() for b in closed]
    expected_current = [
        rtts[ids == b].min() for b, s in zip(closed, starts, strict=True) if s >= cut
    ]
    expected_previous = [
        rtts[ids == b].min() for b, s in zip(closed, starts, strict=True) if s < cut
    ]
    assert_allclose(bins.current, expected_current)
    assert_allclose(bins.previous, expected_previous)


def test_samples_the_backends_drop_do_not_close_a_bin() -> None:
    tracker, backend = _tracker()
    backend.bins = OpenPeriodBins(S1, np.full(8, 10.0), np.array([]))
    assert tracker.observe([_cp(S1, S1)], S1) == []  # no closed bin of the new period yet
    tracker.push(S1, 10.0)
    backend.bins = OpenPeriodBins(S1, np.full(8, 10.0), np.array([30.0]))
    _, out = tracker.push(S1 + BIN, float("nan"))  # dropped: no bin closes
    assert out == []
    _, out = tracker.push(S1 - 60, 10.0)  # out of order: dropped too
    assert out == []
    _, out = tracker.push(S1 + BIN, 10.0)
    assert [c.status for c in out] == ["suspected"]
