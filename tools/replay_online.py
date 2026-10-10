"""
Replay the paper dataset through the online mode and score it, or sweep its parameters.

Thin wrapper over ``jitterbug.streaming.replay`` / ``score`` (which ``jitterbug replay``
also uses) that adds the parameter sweep and exposes every ``StreamingConfig`` field as
a flag. Unlike the CLI, ``--method`` defaults to ``ks_test`` here: the sweep figures quoted
in ``docs/ONLINE_MODE.md`` are KS figures. The reference file follows the method unless
``--reference`` is given.

Usage::

    uv run python tools/replay_online.py                        # defaults, scored vs the paper
    uv run python tools/replay_online.py --min-time-elapsed 1800 --min-period-samples 30
    uv run python tools/replay_online.py --sweep      # decision x lag x hazard x threshold
"""

from __future__ import annotations

import argparse
import itertools
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from jitterbug.io import DataLoader
from jitterbug.models import JitterbugConfig, RTTDataset, StreamingConfig
from jitterbug.streaming import events_to_json, replay, score

ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "examples/network_analysis/data/raw.csv"
REFERENCES = {
    "ks_test": ROOT / "examples/network_analysis/expected_results/kstest_inferences.csv",
    "jitter_dispersion": ROOT / "examples/network_analysis/expected_results/jd_inferences.csv",
}
logging.getLogger("jitterbug").setLevel(logging.ERROR)

SWEEP_COLUMNS = [
    "change_points",
    "periods",
    "congested",
    "recovered",
    "spurious",
    "boundaries_within_30min",
    "onset_delay_min_median",
    "onset_delay_min_max",
    "return_delay_min_median",
    "return_delay_min_max",
    "provisional_flips",
    "provisional_lead_h_mean",
]
SWEEP_AXES = ("decision", "lag", "hazard_lambda", "threshold")
FIELDS = (
    "backend",
    "window_hours",
    "rerun_every_bins",
    "stable_runs",
    "decision",
    "lag",
    "hazard_lambda",
    "threshold",
    "max_run_length",
    "min_time_elapsed",
    "min_period_samples",
    "min_ks_statistic",
    "prior_mu",
)


def _config(method: str = "ks_test", **streaming: Any) -> JitterbugConfig:
    config = JitterbugConfig(streaming=StreamingConfig(**streaming))
    config.change_point_detection.algorithm = "bcp"  # the paper's detector
    config.jitter_analysis.method = method  # type: ignore[assignment]
    return config


def _sweep(
    dataset: RTTDataset, reference: Path, method: str, fixed: dict[str, Any]
) -> pd.DataFrame:
    grid = itertools.product(
        ["lag", "window", "map"], [2, 4, 8], [25.0, 50.0, 100.0], [0.3, 0.5, 0.7]
    )
    rows = []
    for decision, lag, lam, thr in grid:
        if decision == "map" and (lag != 2 or thr != 0.3):
            continue  # lag and threshold do not apply to the MAP rule
        if decision == "lag" and thr == 0.7:
            continue
        config = _config(
            method, decision=decision, lag=lag, hazard_lambda=lam, threshold=thr, **fixed
        )
        summary = score(replay(dataset, config), reference)
        rows.append(
            {"decision": decision, "lag": lag, "lambda": lam, "thr": thr}
            | {k: summary[k] for k in SWEEP_COLUMNS}
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--csv", type=Path, default=RAW_CSV)
    parser.add_argument("--reference", type=Path, help="Defaults to the paper's file for --method")
    parser.add_argument("--method", choices=["ks_test", "jitter_dispersion"], default="ks_test")
    parser.add_argument("--backend", choices=["bocpd", "window"])
    parser.add_argument("--window-hours", type=float)
    parser.add_argument("--rerun-every-bins", type=int)
    parser.add_argument("--stable-runs", type=int)
    parser.add_argument("--decision", choices=["map", "lag", "window"])
    parser.add_argument("--lag", type=int)
    parser.add_argument("--hazard-lambda", type=float)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--max-run-length", type=int)
    parser.add_argument("--min-time-elapsed", type=int)
    parser.add_argument("--min-period-samples", type=int)
    parser.add_argument("--min-ks-statistic", type=float)
    parser.add_argument("--prior-mu", type=float)
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--events", type=Path, help="Write all events as JSON")
    args = parser.parse_args()

    dataset = DataLoader().load_from_file(args.csv)
    given = {key: getattr(args, key) for key in FIELDS if getattr(args, key) is not None}
    reference = args.reference or REFERENCES[args.method]

    if args.sweep:
        fixed = {k: v for k, v in given.items() if k not in SWEEP_AXES}
        table = _sweep(dataset, reference, args.method, fixed)
        with pd.option_context("display.width", 250, "display.max_columns", 30):
            print(table.to_string(index=False))
        return

    events = replay(dataset, _config(args.method, **given))
    for key, value in score(events, reference).items():
        print(f"{key:28s} {value}")
    if args.events:
        events_to_json(events, args.events)
        print(f"wrote {args.events}")


if __name__ == "__main__":
    main()
