"""CLI tests for multi-series input: ``--group-by``, ``--series`` and ``jitterbug inspect``."""

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from jitterbug.cli.main import app

EXAMPLE_CSV = (
    Path(__file__).resolve().parents[1] / "examples" / "network_analysis" / "data" / "raw.csv"
)
needs_example = pytest.mark.skipif(not EXAMPLE_CSV.exists(), reason="example dataset not present")

runner = CliRunner()


def flat(output: str) -> str:
    """Console output with line wrapping undone: Rich wraps long lines (temporary paths
    on macOS are long), which can split the text a test looks for."""
    return " ".join(output.split())


@pytest.fixture(scope="module")
def two_destinations(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The paper dataset split in two destinations, plus one too short to analyze."""
    if not EXAMPLE_CSV.exists():
        pytest.skip("example dataset not present")
    raw = pd.read_csv(EXAMPLE_CSV)
    frame = pd.concat(
        [
            raw.iloc[:23000].assign(destination="8.8.8.8"),
            raw.iloc[23000:].assign(destination="1.1.1.1"),
            raw.iloc[:5].assign(destination="9.9.9.9"),
        ]
    )
    path = tmp_path_factory.mktemp("multi") / "multi.csv"
    frame.to_csv(path, index=False)
    return path


@pytest.fixture
def small_mixed(tmp_path: Path) -> Path:
    path = tmp_path / "mixed.csv"
    path.write_text("epoch,rtt,destination\n10,5,a\n20,6,a\n10,7,b\n30,8,b\n")
    return path


# --------------------------------------------------------------------------- analyze


@needs_example
def test_single_series_output_is_unchanged_when_the_new_paths_are_engaged(tmp_path: Path):
    """The bundled dataset, as is and with a constant destination column selected by
    --series or grouped by --group-by, gives the same periods; the CSV is byte-identical."""
    tagged = tmp_path / "tagged.csv"
    pd.read_csv(EXAMPLE_CSV).assign(destination="198.51.100.1").to_csv(tagged, index=False)

    def run(path: Path, out: Path, *flags: str) -> None:
        result = runner.invoke(app, ["analyze", str(path), "-o", str(out), *flags])
        assert result.exit_code == 0, result.output

    run(EXAMPLE_CSV, tmp_path / "plain.csv", "--output-format", "csv")
    run(
        tagged,
        tmp_path / "filtered.csv",
        "--output-format",
        "csv",
        "--series",
        "destination=198.51.100.1",
    )
    assert (tmp_path / "plain.csv").read_bytes() == (tmp_path / "filtered.csv").read_bytes()

    run(EXAMPLE_CSV, tmp_path / "plain.json")
    run(tagged, tmp_path / "grouped.json", "--group-by", "destination")
    plain = json.loads((tmp_path / "plain.json").read_text())
    grouped = json.loads((tmp_path / "grouped.json").read_text())
    (entry,) = grouped["series"]
    assert entry["key"] == {"destination": "198.51.100.1"}
    # Only the configuration differs: it records the group_by that was used.
    assert "group_by" not in plain["metadata"]["config"]["input"]
    assert entry["result"]["metadata"]["config"]["input"].pop("group_by") == ["destination"]
    assert entry["result"] == plain


def test_group_by_writes_one_json_document(two_destinations: Path, tmp_path: Path):
    out = tmp_path / "r.json"
    result = runner.invoke(
        app, ["analyze", str(two_destinations), "--group-by", "destination", "-o", str(out)]
    )
    assert result.exit_code == 0, result.output
    assert "3 series, 2 analyzed, 1 skipped" in flat(result.output)
    assert "8.8.8.8" in flat(result.output) and "skipped:" in flat(result.output)
    doc = json.loads(out.read_text())
    assert doc["group_by"] == ["destination"]
    assert [e["key"]["destination"] for e in doc["series"]] == ["8.8.8.8", "1.1.1.1", "9.9.9.9"]
    assert "result" in doc["series"][0] and "skipped_reason" in doc["series"][2]


@pytest.mark.parametrize("fmt", ["csv", "parquet"])
def test_group_by_tables_lead_with_key_columns(two_destinations: Path, tmp_path: Path, fmt: str):
    if fmt == "parquet" and importlib.util.find_spec("pyarrow") is None:
        pytest.skip("pyarrow not installed")
    out = tmp_path / f"r.{fmt}"
    result = runner.invoke(
        app,
        ["analyze", str(two_destinations), "--group-by", "destination", "--output-format", fmt]
        + ["-o", str(out)],
    )
    assert result.exit_code == 0, result.output
    table = pd.read_csv(out) if fmt == "csv" else pd.read_parquet(out)
    assert list(table.columns) == ["destination", "starts", "ends", "congestion"]
    assert set(table["destination"]) == {"8.8.8.8", "1.1.1.1"}


def test_group_by_accepts_repeated_and_comma_separated_names(small_mixed: Path, tmp_path: Path):
    out = tmp_path / "r.json"
    result = runner.invoke(
        app,
        ["analyze", str(small_mixed), "--group-by", "destination,epoch", "--group-by", "rtt"]
        + ["-o", str(out)],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(out.read_text())["group_by"] == ["destination", "epoch", "rtt"]


def test_group_by_from_a_config_file(small_mixed: Path, tmp_path: Path):
    config = tmp_path / "c.yaml"
    config.write_text("input:\n  group_by: [destination]\n")
    out = tmp_path / "r.json"
    result = runner.invoke(app, ["analyze", str(small_mixed), "-c", str(config), "-o", str(out)])
    assert result.exit_code == 0, result.output
    assert len(json.loads(out.read_text())["series"]) == 2


def test_series_filter_that_matches_nothing_lists_the_values(small_mixed: Path):
    result = runner.invoke(app, ["analyze", str(small_mixed), "--series", "destination=z"])
    assert result.exit_code == 1
    assert "available destination values: a, b" in flat(result.output)


def test_malformed_series_flag_is_a_usage_error(small_mixed: Path):
    result = runner.invoke(app, ["analyze", str(small_mixed), "--series", "destination"])
    assert result.exit_code == 2
    assert "COLUMN=VALUE" in flat(result.output)


# --------------------------------------------------------------------------- one-series commands


def test_validate_with_a_series_filter(small_mixed: Path):
    result = runner.invoke(app, ["validate", str(small_mixed), "--series", "destination=b"])
    assert result.exit_code == 0, result.output
    assert "Total Measurements │ 2" in flat(result.output)


@pytest.mark.parametrize("command", ["visualize", "replay", "validate"])
def test_one_series_commands_refuse_several_series(command: str, small_mixed: Path, tmp_path: Path):
    if command == "visualize" and importlib.util.find_spec("matplotlib") is None:
        pytest.skip("matplotlib not installed")
    config = tmp_path / "c.yaml"
    config.write_text("input:\n  group_by: [destination]\n")
    result = runner.invoke(app, [command, str(small_mixed), "-c", str(config)])
    assert result.exit_code == 1
    assert "2 series under group_by" in flat(result.output)
    assert "destination=a; destination=b" in flat(result.output)


def test_replay_with_a_series_filter(tmp_path: Path):
    # Three hours a minute apart per destination; only one destination is replayed.
    epochs = [1_699_999_200 + 60 * i for i in range(180)]
    rows = [f"{e},{20 + (i % 7) / 10},{d}" for d in ("a", "b") for i, e in enumerate(epochs)]
    path = tmp_path / "two.csv"
    path.write_text("epoch,rtt,destination\n" + "\n".join(rows) + "\n")
    out = tmp_path / "events.json"
    args = ["replay", str(path), "--backend", "window", "-o", str(out)]
    result = runner.invoke(app, [*args, "--series", "destination=a"])
    assert result.exit_code == 0, result.output
    assert "Online replay" in flat(result.output) and out.exists()


# --------------------------------------------------------------------------- inspect


def test_inspect_text_lists_pairs_and_warns_about_mixing(small_mixed: Path):
    result = runner.invoke(app, ["inspect", str(small_mixed)])
    assert result.exit_code == 0, result.output
    assert "Format: csv, compression: none" in flat(result.output)
    assert "Source/destination pairs (analyzed as one series): 2" in flat(result.output)
    assert "--group-by source,destination" in flat(result.output)
    assert "Loading would succeed" in flat(result.output)


def test_inspect_json_with_group_by(small_mixed: Path):
    result = runner.invoke(
        app, ["inspect", str(small_mixed), "--group-by", "destination", "--json"]
    )
    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)
    assert report["grouped"] is True
    assert [e["key"] for e in report["series"]] == [{"destination": "a"}, {"destination": "b"}]
    assert report["series"][1]["median_interval_seconds"] == 20.0


def test_inspect_takes_the_input_flags_and_config(tmp_path: Path):
    path = tmp_path / "ms.csv"
    path.write_text("ts,rtt_us\n1700000000000,1500\n1700000060000,2500\n")
    config = tmp_path / "c.yaml"
    config.write_text("input:\n  rtt_unit: us\n")
    result = runner.invoke(
        app,
        ["inspect", str(path), "-c", str(config), "--time-column", "ts", "--time-unit", "auto"]
        + ["--rtt-column", "rtt_us", "--json"],
    )
    assert result.exit_code == 0, result.output
    mapping = json.loads(result.stdout)["mapping"]
    assert (mapping["time_unit_applied"], mapping["rtt_unit"]) == ("ms", "us")


def test_inspect_exits_1_when_loading_would_fail(small_mixed: Path):
    result = runner.invoke(app, ["inspect", str(small_mixed), "--time-column", "ts"])
    assert result.exit_code == 1
    assert "Loading would fail" in flat(result.output) and "no 'ts' column" in flat(result.output)
    as_json = runner.invoke(app, ["inspect", str(small_mixed), "--time-column", "ts", "--json"])
    assert as_json.exit_code == 1
    assert json.loads(as_json.stdout)["errors"]


def test_inspect_unreadable_file_is_an_error(tmp_path: Path):
    path = tmp_path / "x.bin"
    path.write_bytes(b"\x00\x01\x02")
    result = runner.invoke(app, ["inspect", str(path)])
    assert result.exit_code == 1
    assert "Error" in flat(result.output)
