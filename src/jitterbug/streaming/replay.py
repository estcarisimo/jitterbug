"""
Replay a finite RTT dataset through the online pipeline and score the result.

``replay`` feeds every sample of an ``RTTDataset`` to the configured online back end in
timestamp order and returns the events. ``score`` summarizes them: change points, periods,
detection delays, provisional-to-final agreement and, when a reference file is given,
agreement with it using the metric of ``tests/test_paper_regression.py`` (a reference
congested period counts as recovered when one of ours overlaps more than half of it) and
how many reference period boundaries have a change point within 30 min.

Delays are measured from the change point to the sample that reported it. *Onset* is the
change point that opens a congested final period, *return* the one that closes it.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from ..models import JitterbugConfig, RTTDataset
from .online_analyzer import StreamingEvent

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
        Adds ``recovered``, ``reference_congested`` and ``spurious``.

    Returns
    -------
    dict
        Counts, delays in minutes and provisional-to-final agreement.
    """
    cps = [e for e in events if e.kind == "change_point"]
    finals = [e for e in events if e.kind == "verdict" and e.stage == "final"]
    provisional = {e.start_epoch: e for e in events if e.stage == "provisional"}
    congested = [e for e in finals if e.is_congested]

    cp_delay_min = {e.start_epoch: e.delay / 60 for e in cps}
    onset = [cp_delay_min[e.start_epoch] for e in congested if e.start_epoch in cp_delay_min]
    back = [cp_delay_min[e.end_epoch] for e in congested if e.end_epoch in cp_delay_min]

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
    }
    if reference is not None:
        periods = _read_reference(reference)
        ref = [period for period, is_congested in periods if is_congested]
        ours = [(e.start_epoch, e.end_epoch) for e in congested if e.end_epoch is not None]
        result["recovered"] = sum(_covers(r, ours) for r in ref)
        result["reference_congested"] = len(ref)
        result["spurious"] = sum(not _covers(mine, ref) for mine in ours)
        # Boundary placement: reference period edges with a change point within 30 min.
        bounds = sorted({edge for period, _ in periods for edge in period})
        cp_epochs = np.asarray([e.start_epoch for e in cps])
        result["boundaries_within_30min"] = (
            int(sum(np.min(np.abs(cp_epochs - b)) <= 1800 for b in bounds)) if len(cps) else 0
        )
        result["reference_boundaries"] = len(bounds)
    return result


def events_to_json(events: list[StreamingEvent], path: Path) -> None:
    """Write ``events`` as a JSON array of plain dictionaries."""
    path.write_text(json.dumps([e.to_dict() for e in events], indent=1))
