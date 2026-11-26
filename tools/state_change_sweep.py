"""
Measure the state-change layer of the online mode on the paper dataset.

Replays ``examples/network_analysis/data/raw.csv`` through the ``bocpd`` back end and
``StateChangeTracker`` for each jitter method and each ``streaming.state_change_bins``
value, scores the state changes against the paper's reference for that method, and prints
the table of ``docs/ONLINE_MODE.md`` (section "State changes"). About 2 s per row.

    uv sync --extra bcp
    uv run python tools/state_change_sweep.py
    uv run python tools/state_change_sweep.py --bins 1 2 --backend window
"""

from __future__ import annotations

import argparse
from pathlib import Path

from jitterbug.io import DataLoader
from jitterbug.models import JitterbugConfig
from jitterbug.streaming import replay_state_changes, score_state_changes

DATA = Path(__file__).resolve().parents[1] / "examples" / "network_analysis"
REFERENCE = {
    "ks_test": DATA / "expected_results" / "kstest_inferences.csv",
    "jitter_dispersion": DATA / "expected_results" / "jd_inferences.csv",
}
COLUMNS = [
    ("suspected", "Suspected (onset / return)"),
    ("suspected_confirmed", "Confirmed"),
    ("suspected_retracted", "Retracted"),
    ("confirmed_without_suspicion", "Confirmed, never suspected"),
    ("corrected_by_final", "Corrected by final"),
    ("reference_congested_suspected", "Reference congested periods suspected"),
    ("suspected_onsets_outside_reference", "Suspected onsets outside reference"),
    ("suspected_delay_min_median", "Delay after change, median / max (min)"),
    ("suspected_lead_min_median", "Lead over first verdict, median / max (min)"),
]


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.0f}"


def main() -> None:
    """Run the sweep and print a Markdown table."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--bins", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--backend", choices=["bocpd", "window"], default="bocpd")
    args = parser.parse_args()

    dataset = DataLoader().load_from_file(DATA / "data" / "raw.csv")
    print("| Jitter method | Bins | " + " | ".join(label for _, label in COLUMNS) + " |")
    print("|---" * (len(COLUMNS) + 2) + "|")
    for method in ("ks_test", "jitter_dispersion"):
        for bins in args.bins:
            config = JitterbugConfig()
            config.change_point_detection.algorithm = "bcp"  # type: ignore[assignment]
            config.jitter_analysis.method = method  # type: ignore[assignment]
            config.streaming.backend = args.backend  # type: ignore[assignment]
            if args.backend == "window":
                config.streaming.rerun_every_bins = 4
            config.streaming.state_change_bins = bins
            events, changes = replay_state_changes(dataset, config)
            s = score_state_changes(events, changes, REFERENCE[method])
            cells = [
                f"{s['suspected']} ({s['suspected_onsets']} / {s['suspected_returns']})",
                str(s["suspected_confirmed"]),
                str(s["suspected_retracted"]),
                str(s["confirmed_without_suspicion"]),
                str(s["corrected_by_final"]),
                f"{s['reference_congested_suspected']} of {s['reference_congested']}",
                str(s["suspected_onsets_outside_reference"]),
                f"{_fmt(s['suspected_delay_min_median'])} / {_fmt(s['suspected_delay_min_max'])}",
                f"{_fmt(s['suspected_lead_min_median'])} / {_fmt(s['suspected_lead_min_max'])}",
            ]
            print(f"| {method} | {bins} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
