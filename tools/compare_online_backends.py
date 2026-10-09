"""
Compare the online back ends on the paper dataset and print the Markdown table of
``docs/ONLINE_MODE.md``.

Every row replays ``raw.csv`` with BCP + KS (the paper's configuration), scores the final
verdicts against ``kstest_inferences.csv`` with ``jitterbug.streaming.score`` and records
the wall time of the replay. Delays are measured from the change point to the sample that
reported it; *onset* is the change point that opens a congested period, *return* the one
that closes it.

Usage::

    uv run python tools/compare_online_backends.py            # about 4 min
    uv run python tools/compare_online_backends.py --quick    # skip the every-bin window run
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Any

from jitterbug.io import DataLoader
from jitterbug.models import JitterbugConfig, StreamingConfig
from jitterbug.streaming import replay, score

ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "examples/network_analysis/data/raw.csv"
REFERENCE = ROOT / "examples/network_analysis/expected_results/kstest_inferences.csv"
logging.getLogger("jitterbug").setLevel(logging.ERROR)

ROWS: list[tuple[str, dict[str, Any]]] = [
    ("Incremental Bayesian (`bocpd`, MAP rule)", {}),
    ("Sliding window, rerun every bin (`window`)", {"backend": "window", "rerun_every_bins": 1}),
    ("Sliding window, rerun every 4 bins", {"backend": "window", "rerun_every_bins": 4}),
]


def _config(**streaming: Any) -> JitterbugConfig:
    config = JitterbugConfig(streaming=StreamingConfig(**streaming))
    config.change_point_detection.algorithm = "bcp"
    config.jitter_analysis.method = "ks_test"
    return config


def _minutes(value: float | None) -> str:
    return "-" if value is None else f"{value:.0f} min"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--quick", action="store_true", help="Skip the every-bin window run")
    args = parser.parse_args()

    dataset = DataLoader().load_from_file(RAW_CSV)
    header = (
        "| Back end | Periods / congested | Reference periods recovered | Spurious | "
        "Boundaries within 30 min | Onset delay, median / max | Return delay, median / max | "
        "Provisional flips | Replay wall time |"
    )
    print(header)
    print("|" + "---|" * 9)
    for label, streaming in ROWS:
        if args.quick and streaming.get("rerun_every_bins") == 1:
            continue
        start = time.perf_counter()
        events = replay(dataset, _config(**streaming))
        elapsed = time.perf_counter() - start
        s = score(events, REFERENCE)
        print(
            f"| {label} | {s['periods']} / {s['congested']} | "
            f"{s['recovered']} of {s['reference_congested']} | {s['spurious']} | "
            f"{s['boundaries_within_30min']} of {s['reference_boundaries']} | "
            f"{_minutes(s['onset_delay_min_median'])} / {_minutes(s['onset_delay_min_max'])} | "
            f"{_minutes(s['return_delay_min_median'])} / {_minutes(s['return_delay_min_max'])} | "
            f"{s['provisional_flips']} of {s['provisional_pairs']} | {elapsed:.0f} s |"
        )


if __name__ == "__main__":
    main()
