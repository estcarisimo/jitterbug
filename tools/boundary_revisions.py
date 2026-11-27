"""
Measure change point revisions of the ``bocpd`` back end on the paper dataset.

Replays ``examples/network_analysis/data/raw.csv`` through ``OnlineJitterbug`` and
``StateChangeTracker`` with ``streaming.revise_change_points`` off and on, for each jitter
method, scores both against the paper's reference for that method, and prints the table of
``docs/ONLINE_MODE.md`` (section "Change point revisions"). About 2 s per row.

    uv sync --extra bcp
    uv run python tools/boundary_revisions.py
    uv run python tools/boundary_revisions.py --min-time-elapsed 1800 3600
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from jitterbug.io import DataLoader
from jitterbug.models import JitterbugConfig
from jitterbug.streaming import replay_state_changes, score, score_state_changes

DATA = Path(__file__).resolve().parents[1] / "examples" / "network_analysis"
REFERENCE = {
    "ks_test": DATA / "expected_results" / "kstest_inferences.csv",
    "jitter_dispersion": DATA / "expected_results" / "jd_inferences.csv",
}
NAMES = {"ks_test": "KS test", "jitter_dispersion": "dispersion"}
COLUMNS = [
    "Jitter method",
    "Min. separation",
    "Revisions (change points moved)",
    "Change points / periods / congested",
    "Recovered / spurious",
    "Boundaries within 30 min / 5 min",
    "Boundary offset, mean",
    "Provisional flips",
    "Provisional verdicts superseded (changed)",
    "Revision shift, median / max",
    "Revision after first report, median / max",
    "State changes (revised)",
    "Reference congested periods suspected",
]
logging.getLogger("jitterbug").setLevel(logging.ERROR)


def _min(value: float | None) -> str:
    return "-" if value is None else f"{value:.0f}"


def main() -> None:
    """Run the replays and print a Markdown table."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--min-time-elapsed", type=int, nargs="+", default=[3600])
    args = parser.parse_args()

    dataset = DataLoader().load_from_file(DATA / "data" / "raw.csv")
    print("| " + " | ".join(COLUMNS) + " |")
    print("|---" * len(COLUMNS) + "|")
    for method in ("ks_test", "jitter_dispersion"):
        for separation in args.min_time_elapsed:
            for revise in (False, True):
                config = JitterbugConfig()
                config.jitter_analysis.method = method  # type: ignore[assignment]
                config.streaming.min_time_elapsed = separation
                config.streaming.revise_change_points = revise
                events, changes = replay_state_changes(dataset, config)
                s = score(events, REFERENCE[method])
                c = score_state_changes(events, changes, REFERENCE[method])
                cells = [
                    NAMES[method],
                    f"{separation // 60} min",
                    f"{s['revisions']} ({s['revised_change_points']})" if revise else "off",
                    f"{s['change_points']} / {s['periods']} / {s['congested']}",
                    f"{s['recovered']} of {s['reference_congested']} / {s['spurious']}",
                    f"{s['boundaries_within_30min']} / {s['boundaries_within_5min']} "
                    f"of {s['reference_boundaries']}",
                    f"{s['boundary_offset_min_mean']:.1f} min",
                    f"{s['provisional_flips']} of {s['provisional_pairs']}",
                    f"{s['provisional_superseded']} ({s['provisional_superseded_changed']})",
                    f"{_min(s['revision_shift_min_median'])} / "
                    f"{_min(s['revision_shift_min_max'])} min",
                    f"{_min(s['revision_lag_min_median'])} / {_min(s['revision_lag_min_max'])} min",
                    f"{c['state_changes']} ({c['state_changes_revised']})",
                    f"{c['reference_congested_suspected']} of {c['reference_congested']}",
                ]
                print("| " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
