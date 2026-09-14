"""Smoke tests for the Typer CLI on the bundled PAM 2022 example dataset."""

import importlib.metadata
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

import jitterbug
from jitterbug.cli.main import app
from jitterbug.models import JitterbugConfig

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
def test_visualize_writes_the_standard_plots(tmp_path: Path):
    pytest.importorskip("matplotlib")
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


def test_config_file_values_survive_when_flags_are_omitted(tmp_path: Path):
    """Regression: the CLI used to overwrite --config values with its own defaults."""
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "change_point_detection:\n  algorithm: bcp\n  threshold: 0.4\n"
        "jitter_analysis:\n  method: ks_test\noutput_format: csv\n"
    )
    from jitterbug.cli.main import _apply_overrides

    loaded = JitterbugConfig.from_file(cfg)
    untouched = _apply_overrides(loaded)
    assert untouched.change_point_detection.algorithm == "bcp"
    assert untouched.change_point_detection.threshold == 0.4
    assert untouched.jitter_analysis.method == "ks_test"
    assert untouched.output_format == "csv"

    overridden = _apply_overrides(loaded, algorithm="ruptures", threshold=0.1)
    assert overridden.change_point_detection.algorithm == "ruptures"
    assert overridden.change_point_detection.threshold == 0.1
    assert overridden.jitter_analysis.method == "ks_test"  # not touched


def test_invalid_algorithm_flag_is_rejected():
    result = runner.invoke(app, ["analyze", str(EXAMPLE_CSV), "--algorithm", "nope"])
    assert result.exit_code != 0
    assert "algorithm" in result.output


@pytest.mark.skipif(not EXAMPLE_CSV.exists(), reason="example dataset not present")
def test_analyze_honours_a_config_file_end_to_end(tmp_path: Path):
    """The file's `output_format` and `threshold` must reach the analyzer through the CLI."""
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "change_point_detection:\n  algorithm: ruptures\n  threshold: 0.9\n"
        "jitter_analysis:\n  method: jitter_dispersion\noutput_format: csv\n"
    )
    out = tmp_path / "results.csv"
    result = runner.invoke(
        app, ["analyze", str(EXAMPLE_CSV), "--config", str(cfg), "--output", str(out)]
    )
    assert result.exit_code == 0, result.output
    header = out.read_text().splitlines()[0]
    assert header == "starts,ends,congestion", header  # the v1 CSV layout, not JSON


def test_verbose_from_a_config_file_takes_effect(tmp_path: Path):
    """Regression: the CLI installs a logging handler before reading the file, and a
    second `basicConfig` is a no-op, so `verbose: true` never enabled debug output."""
    import logging

    logging.basicConfig(level=logging.INFO)  # what the CLI does before loading the config
    from jitterbug.analyzer import JitterbugAnalyzer

    JitterbugAnalyzer(JitterbugConfig(verbose=True))
    assert logging.getLogger("jitterbug").isEnabledFor(logging.DEBUG)
    logging.getLogger("jitterbug").setLevel(logging.NOTSET)  # do not leak into other tests
