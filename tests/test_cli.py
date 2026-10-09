"""Smoke tests for the Typer CLI on the bundled PAM 2022 example dataset."""

import importlib.metadata
import importlib.util
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

REFERENCE_CSV = EXAMPLE_CSV.parents[1] / "expected_results" / "kstest_inferences.csv"
BCP_MISSING = importlib.util.find_spec("bayesian_changepoint_detection") is None

runner = CliRunner()


def test_version_is_derived_from_package_metadata():
    assert jitterbug.__version__ == importlib.metadata.version("jitterbug-inference")


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
def test_analyze_honors_a_config_file_end_to_end(tmp_path: Path):
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

    from jitterbug.analyzer import JitterbugAnalyzer
    from jitterbug.cli.main import _apply_overrides

    cfg = tmp_path / "config.yaml"
    cfg.write_text("verbose: true\n")
    package_logger = logging.getLogger("jitterbug")
    try:
        logging.basicConfig(level=logging.INFO)  # what the CLI does before loading the file
        config = _apply_overrides(JitterbugConfig.from_file(cfg))  # the CLI's own wiring
        JitterbugAnalyzer(config)
        assert package_logger.isEnabledFor(logging.DEBUG)
    finally:
        package_logger.setLevel(logging.NOTSET)  # do not leak into other tests


def test_analyzer_does_not_downgrade_an_explicit_logger_level():
    import logging

    from jitterbug.analyzer import JitterbugAnalyzer

    package_logger = logging.getLogger("jitterbug")
    try:
        package_logger.setLevel(logging.DEBUG)
        JitterbugAnalyzer(JitterbugConfig())  # verbose=False must leave it alone
        assert package_logger.level == logging.DEBUG
    finally:
        package_logger.setLevel(logging.NOTSET)


# --- validate


@pytest.mark.skipif(not EXAMPLE_CSV.exists(), reason="example dataset not present")
def test_validate_reports_quality_metrics():
    result = runner.invoke(app, ["validate", str(EXAMPLE_CSV)])
    assert result.exit_code == 0, result.output
    assert "Data validation passed" in result.output
    assert "47163" in result.output  # total measurements
    assert "RTT Statistics" not in result.output  # only with --verbose


@pytest.mark.skipif(not EXAMPLE_CSV.exists(), reason="example dataset not present")
def test_validate_verbose_adds_rtt_statistics():
    result = runner.invoke(app, ["validate", str(EXAMPLE_CSV), "--verbose"])
    assert result.exit_code == 0, result.output
    assert "RTT Statistics" in result.output
    assert "Outliers" in result.output


@pytest.mark.parametrize(
    ("rows", "expected"),
    [("1.0,10.0\n2.0,11.0\n2.0,12.0\n", "✓"), ("1.0,10.0\n2.0,11.0\n3.0,12.0\n", "✗")],
)
def test_validate_reports_duplicates(tmp_path: Path, rows: str, expected: str):
    csv = tmp_path / "rtts.csv"
    csv.write_text("epoch,values\n" + rows)
    result = runner.invoke(app, ["validate", str(csv)])
    assert result.exit_code == 0, result.output
    row = next(line for line in result.output.splitlines() if "Has Duplicates" in line)
    assert expected in row and ("✓" if expected == "✗" else "✗") not in row


def test_validate_rejects_a_file_that_does_not_follow_the_contract(tmp_path: Path):
    csv = tmp_path / "rtts.csv"
    csv.write_text("time,ms\n1.0,10.0\n")
    result = runner.invoke(app, ["validate", str(csv)])
    assert result.exit_code == 1
    assert "Validation failed" in result.output and "'epoch' column" in result.output


def test_validate_missing_file_is_a_usage_error(tmp_path: Path):
    result = runner.invoke(app, ["validate", str(tmp_path / "nope.csv")])
    assert result.exit_code == 2  # Typer's `exists=True` check, before our code runs
    assert "nope.csv" in result.output  # a token Click's wrapping cannot split


# --- config


def test_config_template_to_stdout_is_valid_yaml():
    import yaml

    result = runner.invoke(app, ["config", "--template"])
    assert result.exit_code == 0, result.output
    data = yaml.safe_load(result.output)
    assert data == JitterbugConfig().model_dump()


def test_config_template_json_to_stdout():
    result = runner.invoke(app, ["config", "--template", "--format", "json"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == JitterbugConfig().model_dump()


@pytest.mark.parametrize("name", ["config.yaml", "config.yml", "config.json"])
def test_config_template_round_trips_through_a_file(tmp_path: Path, name: str):
    out = tmp_path / name
    result = runner.invoke(app, ["config", "--template", "--output", str(out)])
    assert result.exit_code == 0, result.output
    assert "saved to" in result.output and name in result.output  # Rich wraps the path
    assert JitterbugConfig.from_file(out) == JitterbugConfig()


def test_config_without_template_prints_a_hint():
    result = runner.invoke(app, ["config"])
    assert result.exit_code == 0
    assert "--template" in result.output


# --- analyze error paths


def test_analyze_bad_input_exits_non_zero(tmp_path: Path):
    csv = tmp_path / "rtts.csv"
    csv.write_text("epoch,values\n1.0,ten\n")
    result = runner.invoke(app, ["analyze", str(csv)])
    assert result.exit_code == 1
    assert "Error" in result.output and "'ten'" in result.output


def test_analyze_rejects_unknown_method(tmp_path: Path):
    csv = tmp_path / "rtts.csv"
    csv.write_text("epoch,values\n1.0,10.0\n")
    result = runner.invoke(app, ["analyze", str(csv), "--method", "magic"])
    assert result.exit_code == 1
    assert "jitter_dispersion" in result.output  # Pydantic lists the allowed values


def test_analyze_reads_and_writes_zstandard(tmp_path: Path):
    pytest.importorskip("zstandard")
    from jitterbug.io.compression import open_text

    packed_input = tmp_path / "raw.csv.zst"
    with open_text(packed_input, "w", newline="") as f:
        f.write(EXAMPLE_CSV.read_text())
    out = tmp_path / "results.json.zst"
    result = runner.invoke(app, ["analyze", str(packed_input), "--output", str(out)])
    assert result.exit_code == 0, result.output
    assert packed_input.stat().st_size < EXAMPLE_CSV.stat().st_size / 2
    with open_text(out) as f:
        assert json.load(f)["inferences"]


def test_config_template_has_a_streaming_section():
    result = runner.invoke(app, ["config", "--template"])
    assert result.exit_code == 0
    assert "streaming:" in result.stdout
    assert "hazard_lambda" in result.stdout


def test_streaming_overrides_only_touch_the_flags_given(tmp_path: Path):
    from jitterbug.cli.main import _apply_streaming_overrides

    cfg = tmp_path / "config.yaml"
    cfg.write_text("streaming:\n  decision: window\n  lag: 2\n")
    base = JitterbugConfig.from_file(cfg)
    out = _apply_streaming_overrides(base, min_period_samples=40, min_time_elapsed=1800)
    assert out.streaming.decision == "window"
    assert out.streaming.lag == 2
    assert out.streaming.min_period_samples == 40
    assert out.streaming.min_time_elapsed == 1800
    assert out.streaming.hazard_lambda == 50.0


@pytest.mark.skipif(BCP_MISSING, reason="bcp extra not installed")
def test_stream_reads_stdin_and_emits_json_lines():
    # 400 samples 30 s apart: 3.3 h, so several 15-minute bins close.
    lines = ["epoch,values"] + [f"{1_700_000_000 + 30 * i},{10 + i % 3}" for i in range(400)]
    result = runner.invoke(
        app, ["stream", "--events", "change-points"], input="\n".join(lines) + "\n"
    )
    assert result.exit_code == 0, result.output
    events = [json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")]
    assert events and events[0]["kind"] == "change_point"
    assert events[0]["start_epoch"] == 1_700_000_000.0  # the stream start opens the baseline
    assert all(e["kind"] == "change_point" for e in events)


def test_stream_rejects_an_unknown_event_filter():
    result = runner.invoke(app, ["stream", "--events", "everything"], input="")
    assert result.exit_code == 2


@pytest.mark.skipif(BCP_MISSING, reason="bcp extra not installed")
def test_stream_missing_file_is_an_error(tmp_path: Path):
    result = runner.invoke(app, ["stream", str(tmp_path / "nope.csv")])
    assert result.exit_code == 1
    assert "not found" in result.output


@pytest.mark.slow
@pytest.mark.skipif(
    BCP_MISSING or not EXAMPLE_CSV.exists(), reason="bcp extra or example dataset missing"
)
def test_replay_example_dataset_against_the_reference(tmp_path: Path):
    out = tmp_path / "events.json"
    result = runner.invoke(
        app,
        ["replay", str(EXAMPLE_CSV), "--output", str(out), "--reference", str(REFERENCE_CSV)],
    )
    assert result.exit_code == 0, result.output
    events = json.loads(out.read_text())
    assert any(e["kind"] == "verdict" and e["stage"] == "final" for e in events)
    assert "Reference congested periods recovered" in result.stdout
    assert "Events saved to" in result.stdout


def test_iter_lines_holds_a_half_written_line_until_its_newline(tmp_path: Path):
    import threading
    import time

    from jitterbug.cli.main import _iter_lines

    path = tmp_path / "rtts.csv"
    path.write_text("1700000000,10.5\n1700000030,12")

    def finish_line() -> None:
        time.sleep(0.8)
        with path.open("a") as f:
            f.write(".75\n1700000060,11\n")

    threading.Thread(target=finish_line, daemon=True).start()
    lines = _iter_lines(path, follow=True)
    assert next(lines) == "1700000000,10.5\n"
    assert next(lines) == "1700000030,12.75\n"  # not "12" and then ".75"
    assert next(lines) == "1700000060,11\n"


def test_iter_lines_without_follow_yields_a_trailing_unterminated_line(tmp_path: Path):
    from jitterbug.cli.main import _iter_lines

    path = tmp_path / "rtts.csv"
    path.write_text("1700000000,10.5\n1700000030,12")
    assert list(_iter_lines(path, follow=False)) == ["1700000000,10.5\n", "1700000030,12"]


@pytest.mark.parametrize("line", ["nan,5", "inf,5", "1700000000,nan", "epoch,values", "x", ""])
def test_parse_sample_rejects_non_finite_and_junk_lines(line: str):
    from jitterbug.cli.main import _parse_sample

    assert _parse_sample(line) is None
    assert _parse_sample("1700000000,10.5,extra") == (1700000000.0, 10.5)


@pytest.mark.skipif(BCP_MISSING, reason="bcp extra not installed")
def test_stream_skips_bad_lines_instead_of_aborting():
    lines = ["epoch,values", "1700000000,10", "nan,5", "inf,5", "junk", "1700000930,11"]
    result = runner.invoke(app, ["stream"], input="\n".join(lines) + "\n")
    assert result.exit_code == 0, result.output
    assert "2 samples read, 4 lines skipped" in result.output


@pytest.mark.skipif(BCP_MISSING or not EXAMPLE_CSV.exists(), reason="bcp extra or dataset")
def test_stream_matches_replay_on_a_prefix(tmp_path: Path):
    """Without --follow the open bin is closed at EOF, as replay does."""
    prefix = tmp_path / "prefix.csv"
    with EXAMPLE_CSV.open() as src:
        prefix.write_text("".join(next(src) for _ in range(6000)))
    streamed = tmp_path / "stream.jsonl"
    replayed = tmp_path / "replay.json"
    r1 = runner.invoke(app, ["stream", str(prefix), "--output", str(streamed)])
    r2 = runner.invoke(app, ["replay", str(prefix), "--output", str(replayed)])
    assert r1.exit_code == 0, r1.output
    assert r2.exit_code == 0, r2.output
    from_stream = [json.loads(line) for line in streamed.read_text().splitlines()]
    from_replay = json.loads(replayed.read_text())
    assert from_stream == from_replay
    assert len(from_stream) > 2


@pytest.mark.skipif(BCP_MISSING, reason="bcp extra not installed")
def test_stream_verdicts_filter_and_stdout_stays_json(tmp_path: Path):
    # Two 10 h segments, 30 s apart, with a jump and wider jitter in the second.
    lines = ["epoch,values"]
    for i in range(2400):
        base, spread = (10.0, 0.2) if i < 1200 else (30.0, 6.0)
        lines.append(f"{1_700_000_000 + 30 * i},{base + spread * ((i * 7919) % 97) / 97:.3f}")
    result = runner.invoke(app, ["stream", "--events", "verdicts"], input="\n".join(lines) + "\n")
    assert result.exit_code == 0, result.output
    stdout_lines = [line for line in result.stdout.splitlines() if line.strip()]
    events = [json.loads(line) for line in stdout_lines]  # every stdout line is JSON
    assert events and all(e["kind"] == "verdict" for e in events)
    assert "samples read" not in result.stdout  # the summary goes to stderr


def test_stream_errors_do_not_go_to_stdout():
    result = runner.invoke(app, ["stream", "--events", "everything"], input="")
    assert result.exit_code == 2
    assert result.stdout == ""
