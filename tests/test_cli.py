"""Smoke tests for the Typer CLI on the bundled PAM 2022 example dataset."""

import importlib.metadata
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

import jitterbug
from jitterbug.cli.main import app

EXAMPLE_CSV = (
    Path(__file__).resolve().parents[1] / "examples" / "network_analysis" / "data" / "raw.csv"
)

runner = CliRunner()


def test_version_is_derived_from_package_metadata():
    assert jitterbug.__version__ == importlib.metadata.version("jitterbug")


def test_version_command():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0, result.output
    assert jitterbug.__version__ in result.output


@pytest.mark.skipif(not EXAMPLE_CSV.exists(), reason="example dataset not present")
def test_analyze_example_dataset(tmp_path: Path):
    out = tmp_path / "results.json"
    result = runner.invoke(app, ["analyze", str(EXAMPLE_CSV), "--output", str(out)])
    assert result.exit_code == 0, result.output
    payload = json.loads(out.read_text())
    assert "inferences" in payload and "metadata" in payload
    assert payload["metadata"]["total_measurements"] > 0
    assert len(payload["inferences"]) > 0


@pytest.mark.skipif(not EXAMPLE_CSV.exists(), reason="example dataset not present")
def test_visualize_writes_the_standard_plots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    pytest.importorskip("matplotlib")
    monkeypatch.setenv("MPLBACKEND", "Agg")
    out = tmp_path / "plots"
    result = runner.invoke(app, ["visualize", str(EXAMPLE_CSV), "--output-dir", str(out)])
    assert result.exit_code == 0, result.output
    pngs = sorted(p.name for p in out.glob("*.png"))
    assert pngs == [
        "jitterbug_change_points.png",
        "jitterbug_confidence_heatmap.png",
        "jitterbug_congestion_analysis.png",
        "jitterbug_rtt_timeseries.png",
        "jitterbug_summary_stats.png",
    ]
    assert all((out / p).stat().st_size > 10_000 for p in pngs)
