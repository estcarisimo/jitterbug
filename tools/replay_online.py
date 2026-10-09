"""
Replay the paper dataset through the online prototype and score it.

Feeds ``raw.csv`` one sample at a time, in timestamp order, to
``jitterbug.streaming.OnlineJitterbug`` and compares the final verdicts with the paper's
reference (``kstest_inferences.csv``) using the metric of ``tests/test_paper_regression.py``
(a reference congested period counts as recovered when one of ours overlaps more than
half of it). Also reports detection delay and how often the provisional verdict differs
from the final one.

Usage::

    uv run python tools/replay_online.py                 # defaults
    uv run python tools/replay_online.py --lag 2 --hazard-lambda 25 --threshold 0.3
    uv run python tools/replay_online.py --sweep         # grid over lag, hazard, threshold
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from jitterbug.streaming import OnlineJitterbug, StreamingConfig, StreamingEvent

ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "examples/network_analysis/data/raw.csv"
REFERENCE = ROOT / "examples/network_analysis/expected_results/kstest_inferences.csv"
logging.getLogger("jitterbug").setLevel(logging.ERROR)

Interval = tuple[float, float]


def _reference(path: Path) -> tuple[list[Interval], list[Interval]]:
    with path.open() as f:
        rows = list(csv.DictReader(f))
    congested = [
        (float(r["starts"]), float(r["ends"])) for r in rows if float(r["congestion"]) == 1
    ]
    periods = [(float(r["starts"]), float(r["ends"])) for r in rows]
    return congested, periods


def _overlap(a: Interval, b: Interval) -> float:
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def _covers(needle: Interval, haystack: list[Interval]) -> bool:
    return any(_overlap(needle, other) > 0.5 * (needle[1] - needle[0]) for other in haystack)


def replay(df: pd.DataFrame, config: StreamingConfig) -> list[StreamingEvent]:
    online = OnlineJitterbug(config)
    for epoch, rtt in zip(df["epoch"].to_numpy(), df["values"].to_numpy(), strict=True):
        online.push(float(epoch), float(rtt))
    online.flush()
    return online.events


def score(events: list[StreamingEvent], reference: Path = REFERENCE) -> dict[str, Any]:
    ref_congested, ref_periods = _reference(reference)
    cps = [e for e in events if e.kind == "change_point"]
    finals = [e for e in events if e.kind == "verdict" and e.stage == "final"]
    provisional = {
        e.start_epoch: e for e in events if e.kind == "verdict" and e.stage == "provisional"
    }

    ours = [(e.start_epoch, e.end_epoch) for e in finals if e.is_congested and e.end_epoch]
    recovered = sum(_covers(ref, ours) for ref in ref_congested)
    spurious = sum(not _covers(mine, ref_congested) for mine in ours)

    # Change point placement against the reference period boundaries.
    ref_bounds = sorted({b for p in ref_periods for b in p})
    cp_epochs = np.asarray([e.start_epoch for e in cps])
    near = 0
    if len(cp_epochs):
        near = sum(np.min(np.abs(cp_epochs - b)) <= 1800 for b in ref_bounds)

    flips = lead = 0
    for e in finals:
        p = provisional.get(e.start_epoch)
        if p is None:
            continue
        flips += p.is_congested != e.is_congested
        lead += e.emitted_at - p.emitted_at
    n_pairs = sum(e.start_epoch in provisional for e in finals)

    delays_min = [e.delay / 60 for e in cps]
    return {
        "change_points": len(cps),
        "periods": len(finals),
        "congested": sum(bool(e.is_congested) for e in finals),
        "recovered": recovered,
        "reference_congested": len(ref_congested),
        "spurious": spurious,
        "boundaries_within_30min": f"{near}/{len(ref_bounds)}",
        "detection_delay_min_median": float(np.median(delays_min)) if delays_min else None,
        "detection_delay_min_max": float(np.max(delays_min)) if delays_min else None,
        "provisional_flips": f"{flips}/{n_pairs}",
        "provisional_lead_h_mean": (lead / n_pairs / 3600) if n_pairs else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--csv", type=Path, default=RAW_CSV)
    parser.add_argument("--decision", choices=["lag", "window", "map"], default="map")
    parser.add_argument("--lag", type=int, default=4)
    parser.add_argument("--hazard-lambda", type=float, default=50.0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min-ks-statistic", type=float, default=0.0)
    parser.add_argument("--prior-mu", type=float, default=None)
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--events", type=Path, default=None, help="Write all events as JSON")
    args = parser.parse_args()

    df = pd.read_csv(args.csv).sort_values("epoch", kind="stable")

    if args.sweep:
        grid = itertools.product(
            ["lag", "window", "map"], [2, 4, 8], [25.0, 50.0, 100.0], [0.3, 0.5, 0.7]
        )
        rows = []
        for decision, lag, lam, thr in grid:
            if decision == "map" and (lag != 2 or thr != 0.3):
                continue  # lag/threshold do not apply to the MAP rule
            if decision == "lag" and thr == 0.7:
                continue
            config = StreamingConfig(
                decision=decision,
                lag=lag,
                hazard_lambda=lam,
                threshold=thr,
                min_ks_statistic=args.min_ks_statistic,
                prior_mu=args.prior_mu,
            )
            s = score(replay(df, config))
            rows.append({"decision": decision, "lag": lag, "lambda": lam, "thr": thr, **s})
        table = pd.DataFrame(rows)
        with pd.option_context("display.width", 250, "display.max_columns", 30):
            print(table.to_string(index=False))
        return

    config = StreamingConfig(
        decision=args.decision,
        lag=args.lag,
        hazard_lambda=args.hazard_lambda,
        threshold=args.threshold,
        min_ks_statistic=args.min_ks_statistic,
        prior_mu=args.prior_mu,
    )
    events = replay(df, config)
    for key, value in score(events).items():
        print(f"{key:28s} {value}")
    if args.events:
        args.events.write_text(json.dumps([e.to_dict() for e in events], indent=1))
        print(f"wrote {args.events}")


if __name__ == "__main__":
    main()
