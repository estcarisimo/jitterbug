"""
Replay a finite RTT dataset through the online pipeline and score the result.

``replay`` feeds every sample of an ``RTTDataset`` to the configured online back end in
timestamp order and returns the events. ``score`` summarizes them: change points, periods,
detection delays, provisional-to-final agreement and, when a reference file is given,
agreement with it using the metric of ``tests/test_paper_regression.py`` (a reference
congested period counts as recovered when one of ours overlaps more than half of it) and
how many reference period boundaries have a change point within 30 min.

Delays are measured from the change point to the sample that reported it. *Onset* is the
change point that opens a congested final period, *return* the one that closes it. With
``streaming.revise_change_points``, ``revision`` events move change points; the scores use
each change point's latest epoch and count the revisions.

``replay_state_changes`` runs the same replay through ``StateChangeTracker`` and
``score_state_changes`` summarizes the state changes: how many were suspected, confirmed and
retracted, how far ahead of the first verdict the suspicion came, and, with a reference,
how many reference congested periods got a suspected onset.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from ..models import JitterbugConfig, RTTDataset
from .online_analyzer import StreamingEvent
from .state_changes import StateChangeEvent, StateChangeTracker

Interval = tuple[float, float]


def replay(dataset: RTTDataset, config: JitterbugConfig | None = None) -> list[StreamingEvent]:
    """
    Feed ``dataset`` sample by sample to the online pipeline.

    Parameters
    ----------
    dataset : RTTDataset
        Samples to replay; they are sorted by epoch (stable) before being fed.
    config : JitterbugConfig, optional
        Configuration; ``config.streaming`` holds the online-specific settings, including
        which back end runs (``backend``).

    Returns
    -------
    list[StreamingEvent]
        Every change point and verdict the pipeline emitted, in order, after ``flush``.
    """
    from . import create_online_analyzer  # the factory lives in the package module

    online = create_online_analyzer(config)
    epochs, rtts = dataset.to_arrays()
    order = np.argsort(epochs, kind="stable")
    for epoch, rtt in zip(epochs[order], rtts[order], strict=True):
        online.push(float(epoch), float(rtt))
    online.flush()
    return online.events


def replay_state_changes(
    dataset: RTTDataset, config: JitterbugConfig | None = None
) -> tuple[list[StreamingEvent], list[StateChangeEvent]]:
    """
    Feed ``dataset`` to the online pipeline through a ``StateChangeTracker``.

    Parameters
    ----------
    dataset : RTTDataset
        Samples to replay; they are sorted by epoch (stable) before being fed.
    config : JitterbugConfig, optional
        Configuration, as for ``replay``; ``streaming.state_change_bins`` sets the early rule.

    Returns
    -------
    tuple[list[StreamingEvent], list[StateChangeEvent]]
        The back end's events (identical to what ``replay`` returns) and every
        state-change update, in order, after ``flush``.
    """
    from . import create_online_analyzer

    tracker = StateChangeTracker(create_online_analyzer(config), config)
    epochs, rtts = dataset.to_arrays()
    order = np.argsort(epochs, kind="stable")
    for epoch, rtt in zip(epochs[order], rtts[order], strict=True):
        tracker.push(float(epoch), float(rtt))
    tracker.flush()
    return tracker.analyzer.events, tracker.state_changes


def _read_reference(path: Path) -> list[tuple[Interval, bool]]:
    with path.open() as f:
        reader = csv.DictReader(f)
        missing = {"starts", "ends", "congestion"} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"Reference {path} lacks the column(s) {sorted(missing)}; "
                "expected starts,ends,congestion"
            )
        rows = list(reader)
    return [((float(r["starts"]), float(r["ends"])), float(r["congestion"]) == 1) for r in rows]


def load_reference(path: Path) -> list[Interval]:
    """Congested periods ``(start, end)`` of a reference CSV with ``starts,ends,congestion``."""
    return [period for period, congested in _read_reference(path) if congested]


def _overlap(a: Interval, b: Interval) -> float:
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def _covers(needle: Interval, haystack: list[Interval]) -> bool:
    return any(_overlap(needle, other) > 0.5 * (needle[1] - needle[0]) for other in haystack)


def score(events: list[StreamingEvent], reference: Path | None = None) -> dict[str, Any]:
    """
    Summarize replay events.

    Parameters
    ----------
    events : list[StreamingEvent]
        Output of ``replay``.
    reference : Path, optional
        CSV with ``starts,ends,congestion`` columns (the paper's ``expected_results``).
        Adds ``recovered``, ``reference_congested``, ``spurious`` and boundary placement:
        reference boundaries with a change point within 30 min and within 5 min (the same
        bin), and the median and mean distance from a reference boundary to the nearest
        change point (``boundary_offset_min_*``).

    Returns
    -------
    dict
        Counts, delays in minutes and provisional-to-final agreement. Change point
        revisions are applied before scoring: boundaries, period ends and delays use the
        latest epoch of each change point. A boundary's delay is that of its first
        report (``emitted_at`` minus the epoch first reported). ``revisions`` counts the
        revision events, ``revised_change_points`` the change points they moved,
        ``revision_shift_min_*`` how far (minutes, later in time), ``revision_lag_min_*``
        how long after the change point was first reported. ``provisional_superseded``
        counts provisional verdicts judged on a boundary that was revised afterward and
        ``provisional_superseded_changed`` those the provisional verdict on the revised
        boundary contradicted.
    """
    cps = [e for e in events if e.kind == "change_point"]
    revisions = [e for e in events if e.kind == "revision"]
    finals = [e for e in events if e.kind == "verdict" and e.stage == "final"]
    provisionals = [e for e in events if e.kind == "verdict" and e.stage == "provisional"]
    provisional = {e.start_epoch: e for e in provisionals}
    congested = [e for e in finals if e.is_congested]

    # Revisions move a change point: follow each epoch to its latest value.
    moved = {e.revised_from: e.start_epoch for e in revisions if e.revised_from is not None}

    def _latest(epoch: float) -> float:
        while epoch in moved:
            epoch = moved[epoch]
        return epoch

    reported_at = {e.start_epoch: e.emitted_at for e in cps}
    for e in revisions:  # a revised change point keeps the time it was first reported
        if e.revised_from in reported_at:
            reported_at.setdefault(e.start_epoch, reported_at[e.revised_from])

    # The delay of a boundary is that of its first report, from the epoch it was reported at.
    cp_delay_min = {_latest(e.start_epoch): e.delay / 60 for e in cps}
    onset = [cp_delay_min[e.start_epoch] for e in congested if e.start_epoch in cp_delay_min]
    back = [
        cp_delay_min[_latest(e.end_epoch)]
        for e in congested
        if e.end_epoch is not None and _latest(e.end_epoch) in cp_delay_min
    ]

    flips = 0
    lead_s = 0.0
    pairs = 0
    for final in finals:
        prov = provisional.get(final.start_epoch)
        if prov is None:
            continue
        pairs += 1
        flips += prov.is_congested != final.is_congested
        lead_s += final.emitted_at - prov.emitted_at

    def _stats(values: list[float]) -> tuple[float | None, float | None]:
        if not values:
            return None, None
        return float(np.median(values)), float(np.max(values))

    onset_med, onset_max = _stats(onset)
    back_med, back_max = _stats(back)
    shift_med, shift_max = _stats(
        [(e.start_epoch - e.revised_from) / 60 for e in revisions if e.revised_from is not None]
    )
    lag_med, lag_max = _stats(
        [
            (e.emitted_at - reported_at[e.revised_from]) / 60
            for e in revisions
            if e.revised_from in reported_at
        ]
    )
    # Provisional verdicts judged on a boundary that was revised afterward, and how many
    # of them the provisional verdict on the revised boundary contradicted.
    superseded = [p for p in provisionals if _latest(p.start_epoch) != p.start_epoch]
    superseded_changed = sum(
        provisional[_latest(p.start_epoch)].is_congested != p.is_congested
        for p in superseded
        if _latest(p.start_epoch) in provisional
    )
    result: dict[str, Any] = {
        "change_points": len(cps),
        "periods": len(finals),
        "congested": len(congested),
        "onset_delay_min_median": onset_med,
        "onset_delay_min_max": onset_max,
        "return_delay_min_median": back_med,
        "return_delay_min_max": back_max,
        "provisional_pairs": pairs,
        "provisional_flips": flips,
        "provisional_lead_h_mean": (lead_s / pairs / 3600) if pairs else None,
        "revisions": len(revisions),
        "revised_change_points": len({_latest(e.start_epoch) for e in revisions}),
        "revision_shift_min_median": shift_med,
        "revision_shift_min_max": shift_max,
        "revision_lag_min_median": lag_med,
        "revision_lag_min_max": lag_max,
        "provisional_superseded": len(superseded),
        "provisional_superseded_changed": superseded_changed,
    }
    if reference is not None:
        periods = _read_reference(reference)
        ref = [period for period, is_congested in periods if is_congested]
        ours = [(e.start_epoch, _latest(e.end_epoch)) for e in congested if e.end_epoch is not None]
        result["recovered"] = sum(_covers(r, ours) for r in ref)
        result["reference_congested"] = len(ref)
        result["spurious"] = sum(not _covers(mine, ref) for mine in ours)
        # Boundary placement: reference period edges with a change point within 30 min.
        bounds = sorted({edge for period, _ in periods for edge in period})
        cp_epochs = np.asarray([_latest(e.start_epoch) for e in cps])
        offsets = [float(np.min(np.abs(cp_epochs - b))) / 60 for b in bounds] if len(cps) else []
        result["boundaries_within_30min"] = sum(o <= 30 for o in offsets)
        result["boundaries_within_5min"] = sum(o <= 5 for o in offsets)
        result["boundary_offset_min_median"] = float(np.median(offsets)) if offsets else None
        result["boundary_offset_min_mean"] = float(np.mean(offsets)) if offsets else None
        result["reference_boundaries"] = len(bounds)
    return result


def _median_max(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    return float(np.median(values)), float(np.max(values))


def score_state_changes(
    events: list[StreamingEvent],
    state_changes: list[StateChangeEvent],
    reference: Path | None = None,
) -> dict[str, Any]:
    """
    Summarize the state changes of a replay.

    A change counts once per ``id``; its outcome is its last status. *Lead* is the time
    from the first ``suspected`` update to the first verdict on the same period (the
    provisional one when there is one, else the final one); *delay* is the time from the
    change point to the suspicion.

    Parameters
    ----------
    events : list[StreamingEvent]
        The back end's events from ``replay_state_changes``.
    state_changes : list[StateChangeEvent]
        The state-change updates from ``replay_state_changes``.
    reference : Path, optional
        CSV with ``starts,ends,congestion`` columns. Adds
        ``reference_congested_suspected``: reference congested periods with a suspected
        onset (baseline to congested) at most 30 min before their start or inside them,
        and ``suspected_onsets_outside_reference``: suspected onsets that fall in none.

    Returns
    -------
    dict
        Counts and times in minutes, keys prefixed ``state_changes_`` or ``suspected_``.
    """
    by_id: dict[str, list[StateChangeEvent]] = {}
    for change in state_changes:
        by_id.setdefault(change.id, []).append(change)
    first_verdict: dict[float, float] = {}
    for e in events:
        if e.kind == "verdict" and e.start_epoch not in first_verdict:
            first_verdict[e.start_epoch] = e.emitted_at

    suspected = {k: v for k, v in by_id.items() if v[0].status == "suspected"}
    outcome = {k: v[-1].status for k, v in by_id.items()}
    leads = [
        (first_verdict[v[0].change_epoch] - v[0].emitted_at) / 60
        for v in suspected.values()
        if v[0].change_epoch in first_verdict
    ]
    delays = [v[0].delay / 60 for v in suspected.values()]
    # A final verdict that reversed the outcome of the provisional one.
    corrected = sum(
        any(
            a.stage == "provisional" and b.stage == "final" and a.status != b.status
            for a, b in zip(v, v[1:], strict=False)
        )
        for v in by_id.values()
    )
    lead_med, lead_max = _median_max(leads)
    delay_med, delay_max = _median_max(delays)
    onsets = [v[0] for v in suspected.values() if v[0].to_state == "congested"]
    result: dict[str, Any] = {
        "state_changes": len(by_id),
        "state_changes_confirmed": sum(s == "confirmed" for s in outcome.values()),
        "state_changes_revised": sum(
            any(c.revised_from is not None for c in v) for v in by_id.values()
        ),
        "suspected": len(suspected),
        "suspected_onsets": len(onsets),
        "suspected_returns": len(suspected) - len(onsets),
        "suspected_confirmed": sum(outcome[k] == "confirmed" for k in suspected),
        "suspected_retracted": sum(outcome[k] == "retracted" for k in suspected),
        "suspected_unresolved": sum(outcome[k] == "suspected" for k in suspected),
        "confirmed_without_suspicion": sum(v[0].status == "confirmed" for v in by_id.values()),
        "corrected_by_final": corrected,
        "suspected_lead_min_median": lead_med,
        "suspected_lead_min_max": lead_max,
        "suspected_delay_min_median": delay_med,
        "suspected_delay_min_max": delay_max,
    }
    if reference is not None:
        ref = load_reference(reference)

        def _hits(period: Interval, epoch: float) -> bool:
            return period[0] - 1800 <= epoch < period[1]

        result["reference_congested_suspected"] = sum(
            any(_hits(r, c.change_epoch) for c in onsets) for r in ref
        )
        result["suspected_onsets_outside_reference"] = sum(
            not any(_hits(r, c.change_epoch) for r in ref) for c in onsets
        )
        result["reference_congested"] = len(ref)
    return result


def events_to_json(events: list[StreamingEvent], path: Path) -> None:
    """Write ``events`` as a JSON array of plain dictionaries."""
    path.write_text(json.dumps([e.to_dict() for e in events], indent=1))
