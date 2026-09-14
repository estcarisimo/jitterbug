"""Shared fixtures: small synthetic RTT series with known structure."""

import os
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from jitterbug.models import (
    ChangePoint,
    JitterAnalysis,
    LatencyJump,
    MinimumRTTDataset,
    RTTDataset,
    RTTMeasurement,
)

# Headless backend for the visualization tests, forced (not `setdefault`) so a GUI backend
# in the developer's environment cannot leak in. It has to be in place before any test
# module imports `jitterbug.cli.main`, which imports matplotlib.pyplot at collection time.
os.environ["MPLBACKEND"] = "Agg"

T0 = datetime(2024, 1, 1, tzinfo=timezone.utc)


def make_measurements(
    values: np.ndarray, step_seconds: float = 60.0, start: datetime = T0
) -> list[RTTMeasurement]:
    """One RTTMeasurement per value, ``step_seconds`` apart, starting at ``start``."""
    out = []
    for i, v in enumerate(values):
        ts = start + timedelta(seconds=i * step_seconds)
        out.append(RTTMeasurement(timestamp=ts, epoch=ts.timestamp(), rtt_value=float(v)))
    return out


def change_point_at(dataset: RTTDataset, index: int, confidence: float = 0.9) -> ChangePoint:
    m = dataset.measurements[index]
    return ChangePoint(
        timestamp=m.timestamp, epoch=m.epoch, confidence=confidence, algorithm="test"
    )


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture
def three_segment_min_rtt(rng: np.random.Generator) -> MinimumRTTDataset:
    """
    90 minimum-RTT points, one per minute, in three segments of 30:
    quiet at 20 ms, then a congested-looking segment (mean 40 ms, five times the
    spread), then back to 20 ms.
    """
    quiet = 20.0 + rng.normal(0, 0.2, 30)
    loud = 40.0 + rng.normal(0, 1.0, 30)
    values = np.concatenate([quiet, loud, 20.0 + rng.normal(0, 0.2, 30)])
    return MinimumRTTDataset(measurements=make_measurements(values), interval_minutes=1)


@pytest.fixture
def three_segment_raw(rng: np.random.Generator) -> RTTDataset:
    """Raw samples every second for the same three segments (1 800 points per segment)."""
    quiet = 20.0 + np.abs(rng.normal(0, 0.5, 1800))
    loud = 40.0 + np.abs(rng.normal(0, 6.0, 1800))
    values = np.concatenate([quiet, loud, 20.0 + np.abs(rng.normal(0, 0.5, 1800))])
    return RTTDataset(measurements=make_measurements(values, step_seconds=1.0))


def jump(start: float, end: float, has_jump: bool, magnitude: float = 10.0) -> LatencyJump:
    return LatencyJump(
        start_timestamp=datetime.fromtimestamp(start, tz=timezone.utc),
        end_timestamp=datetime.fromtimestamp(end, tz=timezone.utc),
        start_epoch=start,
        end_epoch=end,
        has_jump=has_jump,
        magnitude=magnitude if has_jump else 0.0,
        threshold=0.5,
    )


def jitter(start: float, end: float, significant: bool) -> JitterAnalysis:
    return JitterAnalysis(
        start_timestamp=datetime.fromtimestamp(start, tz=timezone.utc),
        end_timestamp=datetime.fromtimestamp(end, tz=timezone.utc),
        start_epoch=start,
        end_epoch=end,
        has_significant_jitter=significant,
        jitter_metric=1.0 if significant else 0.0,
        method="jitter_dispersion",
        threshold=0.25,
    )
