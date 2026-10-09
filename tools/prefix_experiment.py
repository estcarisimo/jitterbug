"""
How retrospective is the offline Jitterbug pipeline?

Runs the sequential analysis (BCP + KS by default) on growing prefixes of the paper
dataset and records, for each cutoff, the congestion verdicts for every period that
is already closed at that cutoff. Comparing the verdict at the cutoff with the verdict
of the full run tells how often, and how long after the fact, the offline method
changes its mind. This is the baseline any online mode has to beat.

Two delays are reported for every congested period of the full run, each measured as
the first cutoff at which a prefix run places a period boundary within one bin (15 min)
of the boundary in question, minus the time of that boundary: *onset* for the period's
start and *return* for its end. Cutoffs are ``--step-hours`` apart, so delays are upper
bounds within one step.

Usage::

    uv run python tools/prefix_experiment.py --step-hours 1 --start-hours 24 --output prefix.json
    uv run python tools/prefix_experiment.py --from-json prefix.json   # re-score a saved run
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from jitterbug import JitterbugAnalyzer, JitterbugConfig
from jitterbug.models import CongestionInferenceResult

RAW_CSV = Path(__file__).resolve().parents[1] / "examples/network_analysis/data/raw.csv"
logging.getLogger("jitterbug").setLevel(logging.ERROR)


def _run(df: pd.DataFrame, algorithm: str, method: str) -> CongestionInferenceResult:
    config = JitterbugConfig()
    config.change_point_detection.algorithm = algorithm  # type: ignore[assignment]
    config.jitter_analysis.method = method  # type: ignore[assignment]
    return JitterbugAnalyzer(config).analyze_from_dataframe(df)


def _verdicts(result: CongestionInferenceResult) -> list[tuple[float, float, bool]]:
    return [(p.start_epoch, p.end_epoch, p.is_congested) for p in result.inferences]


def _match(period: tuple[float, float], others: list[tuple[float, float, bool]]) -> bool | None:
    """Verdict of the period in ``others`` that overlaps ``period`` by more than half."""
    best, best_overlap = None, 0.0
    for s, e, c in others:
        overlap = max(0.0, min(e, period[1]) - max(s, period[0]))
        if overlap > best_overlap:
            best, best_overlap = c, overlap
    if best_overlap > 0.5 * (period[1] - period[0]):
        return best
    return None


def _boundary_delays(table: pd.DataFrame, tolerance_h: float = 0.25) -> pd.DataFrame:
    """Onset and return delays of the congested periods of the last (full) cutoff."""
    last = table["cutoff_h"].max()
    full = table[(table["cutoff_h"] == last) & table["verdict"]]
    rows = []
    for _, period in full.iterrows():
        delays = {}
        for name, boundary in (("onset", period["start_h"]), ("return", period["end_h"])):
            seen = table[
                (abs(table["start_h"] - boundary) <= tolerance_h)
                | (abs(table["end_h"] - boundary) <= tolerance_h)
            ]
            delays[name] = seen["cutoff_h"].min() - boundary if len(seen) else np.nan
        rows.append({"start_h": period["start_h"], "end_h": period["end_h"], **delays})
    return pd.DataFrame(rows)


def _report(table: pd.DataFrame) -> None:
    matched = table.dropna(subset=["final"])
    print("\nDisagreement with the final verdict by age of the period at the cutoff:")
    bins = [0, 6, 12, 24, 48, 96, 1e9]
    labels = ["<6 h", "6-12 h", "12-24 h", "1-2 d", "2-4 d", ">4 d"]
    matched = matched.assign(age_bin=pd.cut(matched["age_h"], bins=bins, labels=labels))
    summary = matched.groupby("age_bin", observed=True)["agrees"].agg(["count", "mean"])
    summary["disagree_frac"] = 1 - summary["mean"]
    print(summary[["count", "disagree_frac"]].to_string())
    unmatched = int(table["final"].isna().sum())
    print(f"\nunmatched periods (boundaries differ from the full run): {unmatched}")

    delays = _boundary_delays(table)
    print("\nFirst appearance of each congested period's boundaries, hours after the fact:")
    print(delays.round(2).to_string(index=False))
    for name in ("onset", "return"):
        d = delays[name].dropna()
        print(
            f"{name:6s}: median {d.median():.2f} h, p90 {d.quantile(0.9):.2f} h, "
            f"max {d.max():.2f} h ({len(d)}/{len(delays)} periods)"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--csv", type=Path, default=RAW_CSV)
    parser.add_argument("--algorithm", default="bcp", choices=["bcp", "ruptures"])
    parser.add_argument("--method", default="ks_test", choices=["ks_test", "jitter_dispersion"])
    parser.add_argument("--step-hours", type=float, default=12.0)
    parser.add_argument("--start-hours", type=float, default=48.0)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--from-json", type=Path, default=None, help="Re-score a saved --output")
    args = parser.parse_args()

    if args.from_json:
        _report(pd.DataFrame(json.loads(args.from_json.read_text())))
        return

    df = pd.read_csv(args.csv)
    df = df.rename(columns={"values": "rtt_value"})
    t0, t1 = df["epoch"].min(), df["epoch"].max()

    full = _verdicts(_run(df, args.algorithm, args.method))
    print(f"full run: {len(full)} periods, {sum(c for *_, c in full)} congested")

    cutoffs = np.arange(t0 + args.start_hours * 3600, t1, args.step_hours * 3600)
    rows = []
    for cutoff in cutoffs:
        prefix = df[df["epoch"] <= cutoff]
        now = _verdicts(_run(prefix, args.algorithm, args.method))
        for s, e, c in now:
            final = _match((s, e), full)
            rows.append(
                {
                    "cutoff_h": (cutoff - t0) / 3600,
                    "start_h": (s - t0) / 3600,
                    "end_h": (e - t0) / 3600,
                    "age_h": (cutoff - e) / 3600,
                    "verdict": c,
                    "final": final,
                    "agrees": None if final is None else final == c,
                }
            )
        n_dis = sum(
            1 for r in rows if r["cutoff_h"] == (cutoff - t0) / 3600 and r["agrees"] is False
        )
        n_unm = sum(1 for r in rows if r["cutoff_h"] == (cutoff - t0) / 3600 and r["final"] is None)
        print(
            f"cutoff {(cutoff - t0) / 3600:6.1f} h: {len(now):3d} periods, "
            f"{n_dis} disagree with final, {n_unm} unmatched"
        )

    table = pd.DataFrame(rows)
    _report(table)

    if args.output:
        args.output.write_text(json.dumps(rows, indent=1))
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
