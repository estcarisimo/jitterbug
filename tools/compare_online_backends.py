"""
Compare the online back ends on the paper dataset and print the Markdown table of
``docs/ONLINE_MODE.md``.

Every row replays ``raw.csv`` with the BCP detector and one jitter method, scores the
final verdicts against the paper's reference for that method (``kstest_inferences.csv`` or
``jd_inferences.csv``) with ``jitterbug.streaming.score`` and records the wall time of the
replay. Delays are measured from the change point to the sample that
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
REFERENCES = {
    "ks_test": ROOT / "examples/network_analysis/expected_results/kstest_inferences.csv",
    "jitter_dispersion": ROOT / "examples/network_analysis/expected_results/jd_inferences.csv",
}
logging.getLogger("jitterbug").setLevel(logging.ERROR)

ROWS: list[tuple[str, str, dict[str, Any]]] = [
    ("Incremental Bayesian (`bocpd`, MAP rule)", "ks_test", {}),
    ("Sliding window, rerun every bin (`window`)", "ks_test", {"backend": "window"}),
    ("Sliding window, rerun every 4 bins", "ks_test", {"backend": "window", "rerun_every_bins": 4}),
    ("Incremental Bayesian (`bocpd`, MAP rule)", "jitter_dispersion", {}),
    (
        "Sliding window, rerun every 4 bins",
        "jitter_dispersion",
        {"backend": "window", "rerun_every_bins": 4},
    ),
]


def _config(method: str, **streaming: Any) -> JitterbugConfig:
    config = JitterbugConfig(streaming=StreamingConfig(**streaming))
    config.change_point_detection.algorithm = "bcp"
    config.jitter_analysis.method = method  # type: ignore[assignment]
    return config


def _minutes(value: float | None) -> str:
    return "-" if value is None else f"{value:.0f} min"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--quick", action="store_true", help="Skip the every-bin window run")
    args = parser.parse_args()

    dataset = DataLoader().load_from_file(RAW_CSV)
    header = (
        "| Back end | Jitter method | Periods / congested | Reference periods recovered | "
        "Spurious | Boundaries within 30 min | Onset delay, median / max | "
        "Return delay, median / max | Provisional flips | Replay wall time |"
    )
    print(header)
    print("|" + "---|" * 10)
    names = {"ks_test": "KS test", "jitter_dispersion": "dispersion (causal)"}
    for label, method, streaming in ROWS:
        if (
            args.quick
            and streaming.get("backend") == "window"
            and "rerun_every_bins" not in streaming
        ):
            continue
        start = time.perf_counter()
        events = replay(dataset, _config(method, **streaming))
        elapsed = time.perf_counter() - start
        s = score(events, REFERENCES[method])
        print(
            f"| {label} | {names[method]} | {s['periods']} / {s['congested']} | "
            f"{s['recovered']} of {s['reference_congested']} | {s['spurious']} | "
            f"{s['boundaries_within_30min']} of {s['reference_boundaries']} | "
            f"{_minutes(s['onset_delay_min_median'])} / {_minutes(s['onset_delay_min_max'])} | "
            f"{_minutes(s['return_delay_min_median'])} / {_minutes(s['return_delay_min_max'])} | "
            f"{s['provisional_flips']} of {s['provisional_pairs']} | {elapsed:.0f} s |"
        )


if __name__ == "__main__":
    main()
