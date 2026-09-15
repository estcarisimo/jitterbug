"""Tests for the matplotlib plotter (headless backend set in conftest)."""

from pathlib import Path

import pytest

from jitterbug.models import CongestionInferenceResult

matplotlib = pytest.importorskip("matplotlib")

from jitterbug.visualization import JitterbugPlotter  # noqa: E402


def test_confidence_heatmap_with_no_inferences_still_saves(tmp_path: Path):
    """An analysis with no change points yields no inferences; the plot must not crash."""
    results = CongestionInferenceResult(inferences=[], metadata={})
    out = tmp_path / "heatmap.png"

    fig = JitterbugPlotter().plot_confidence_heatmap(results, save_path=out)
    matplotlib.pyplot.close(fig)

    assert out.stat().st_size > 0


def test_save_all_plots_with_no_inferences_writes_all_files(
    tmp_path: Path, three_segment_raw, three_segment_min_rtt
):
    results = CongestionInferenceResult(inferences=[], metadata={})

    JitterbugPlotter().save_all_plots(
        three_segment_raw, three_segment_min_rtt, results, change_points=[], output_dir=tmp_path
    )
    matplotlib.pyplot.close("all")

    assert sorted(p.name for p in tmp_path.glob("*.png")) == [
        "jitterbug_change_points.png",
        "jitterbug_confidence_heatmap.png",
        "jitterbug_congestion_analysis.png",
        "jitterbug_rtt_timeseries.png",
        "jitterbug_summary_stats.png",
    ]
